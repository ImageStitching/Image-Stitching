package com.stitching.imageStitching;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;

import java.util.List;

import static com.stitching.imageStitching.CylinderStitcher.ImageNode;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ImageBlender {

    private static final double EPS = 1e-8;

    public static Mat blend(List<ImageNode> nodes) {
        // 1. Tính canvas size
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE;

        for (ImageNode node : nodes) {
            Size s = node.img.size();
            Mat corners = new Mat(4, 1, CV_32FC2);
            FloatPointer ptr = new FloatPointer(corners.data());
            ptr.put(0,0); ptr.put(1,0);
            ptr.put(2,s.width()); ptr.put(3,0);
            ptr.put(4,s.width()); ptr.put(5,s.height());
            ptr.put(6,0); ptr.put(7,s.height());

            Mat dst = new Mat();
            perspectiveTransform(corners, dst, node.globalTransform);
            FloatPointer res = new FloatPointer(dst.data());

            for (int i = 0; i < 4; i++) {
                float x = res.get(2*i);
                float y = res.get(2*i+1);
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }
            corners.release();
            dst.release();
        }

        int W = (int) Math.ceil(maxX - minX);
        int H = (int) Math.ceil(maxY - minY);

        if (W <= 0 || H <= 0 || W > 30000 || H > 30000) {
            System.err.println("Invalid canvas size: " + W + "x" + H);
            return null;
        }

        // 2. Offset matrix
        Mat T_offset = Mat.eye(3, 3, CV_64F).asMat();
        DoublePointer dp = new DoublePointer(T_offset.data());
        dp.put(2, -minX);
        dp.put(5, -minY);

        // ========================================
        // 3. OPENPANO STYLE: Tạo weighted images
        // ========================================

        // Lưu warped images + weight maps
        Mat[] warpedImages = new Mat[nodes.size()];
        Mat[] weightMaps = new Mat[nodes.size()];

        for (int idx = 0; idx < nodes.size(); idx++) {
            ImageNode node = nodes.get(idx);

            // Transform matrix
            Mat H_final = new Mat();
            gemm(T_offset, node.globalTransform, 1.0, new Mat(), 0.0, H_final);

            // Warp image
            Mat warpedImg = new Mat();
            warpPerspective(node.img, warpedImg, H_final, new Size(W, H),
                    INTER_LINEAR, BORDER_CONSTANT, new Scalar(0,0,0,0));

            // Tạo weight map dựa trên khoảng cách từ tâm ảnh (OpenPano style)
            Mat weightMap = new Mat(H, W, CV_32F, new Scalar(0));
            FloatPointer wPtr = new FloatPointer(weightMap.data());

            int imgW = node.img.cols();
            int imgH = node.img.rows();

            // Tính weight cho mỗi pixel
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    // Transform ngược từ canvas về ảnh gốc
                    Mat srcPt = new Mat(1, 1, CV_64FC2);
                    DoublePointer srcPtr = new DoublePointer(srcPt.data());
                    srcPtr.put(0, x);
                    srcPtr.put(1, y);

                    Mat dstPt = new Mat();
                    Mat H_inv = new Mat();
                    invert(H_final, H_inv, DECOMP_LU);
                    perspectiveTransform(srcPt, dstPt, H_inv);

                    DoublePointer dstPtr = new DoublePointer(dstPt.data());
                    double origX = dstPtr.get(0);
                    double origY = dstPtr.get(1);

                    // Check if point is inside original image
                    if (origX >= 0 && origX < imgW && origY >= 0 && origY < imgH) {
                        // Normalize to [-0.5, 0.5]
                        double normX = origX / imgW - 0.5;
                        double normY = origY / imgH - 0.5;

                        // Weight: max at center, decrease towards edges
                        // OpenPano formula: (0.5 - |x|) * (0.5 - |y|)
                        double weight = Math.max(0.0,
                                (0.5 - Math.abs(normX)) * (0.5 - Math.abs(normY))) + EPS;

                        wPtr.put(y * W + x, (float) weight);
                    }

                    srcPt.release();
                    dstPt.release();
                    H_inv.release();
                }
            }

            warpedImages[idx] = warpedImg;
            weightMaps[idx] = weightMap;
            H_final.release();
        }

        // ========================================
        // 4. UPDATE_WEIGHT_MAP: Winner-Takes-All
        // Mỗi pixel chỉ thuộc về 1 ảnh có weight cao nhất
        // ========================================

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float maxWeight = 0;
                int winnerIdx = -1;

                // Tìm ảnh có weight cao nhất tại pixel này
                for (int idx = 0; idx < nodes.size(); idx++) {
                    FloatPointer wPtr = new FloatPointer(weightMaps[idx].data());
                    float w = wPtr.get(y * W + x);
                    if (w > maxWeight) {
                        maxWeight = w;
                        winnerIdx = idx;
                    }
                }

                // Set winner = 1, others = 0
                for (int idx = 0; idx < nodes.size(); idx++) {
                    FloatPointer wPtr = new FloatPointer(weightMaps[idx].data());
                    wPtr.put(y * W + x, (idx == winnerIdx) ? 1.0f : 0.0f);
                }
            }
        }

        // ========================================
        // 5. MULTIBAND BLENDING (Simplified version)
        // Với winner-takes-all, chỉ cần simple weighted sum
        // ========================================

        Mat result = new Mat(H, W, CV_32FC3, new Scalar(0.0, 0.0, 0.0, 0.0));

        for (int idx = 0; idx < nodes.size(); idx++) {
            Mat warpedImg = warpedImages[idx];
            Mat weightMap = weightMaps[idx];

            // Convert to float
            Mat img32F = new Mat();
            warpedImg.convertTo(img32F, CV_32FC3, 1.0/255.0, 0.0);

            // Multiply by weight
            for (int y = 0; y < H; y++) {
                FloatPointer imgPtr = new FloatPointer(img32F.data());
                FloatPointer wPtr = new FloatPointer(weightMap.data());
                FloatPointer resPtr = new FloatPointer(result.data());

                for (int x = 0; x < W; x++) {
                    float w = wPtr.get(y * W + x);
                    if (w > 0) {
                        int offset = (y * W + x) * 3;
                        resPtr.put(offset, resPtr.get(offset) + imgPtr.get(offset) * w);
                        resPtr.put(offset+1, resPtr.get(offset+1) + imgPtr.get(offset+1) * w);
                        resPtr.put(offset+2, resPtr.get(offset+2) + imgPtr.get(offset+2) * w);
                    }
                }
            }

            img32F.release();
        }

        // Convert back to 8-bit
        Mat finalRes = new Mat();
        result.convertTo(finalRes, CV_8UC3, 255.0, 0.0);

        // Cleanup
        T_offset.release();
        result.release();
        for (Mat img : warpedImages) if (img != null) img.release();
        for (Mat w : weightMaps) if (w != null) w.release();

        return finalRes;
    }
}