package com.stitching.imageStitching;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;

import java.util.List;

import static com.stitching.imageStitching.CylinderStitcher.ImageNode;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ImageBlenderFAST {

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
        // 3. Warp images + Create weight maps (FAST VERSION)
        // ========================================

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

            // ========================================
            // Tạo weight map NHANH bằng cách warp từ weight image gốc
            // ========================================

            int imgW = node.img.cols();
            int imgH = node.img.rows();

            // Tạo weight image từ ảnh gốc (weight cao ở giữa, thấp ở cạnh)
            Mat originalWeight = new Mat(imgH, imgW, CV_32F);
            FloatPointer owPtr = new FloatPointer(originalWeight.data());

            for (int y = 0; y < imgH; y++) {
                for (int x = 0; x < imgW; x++) {
                    // Normalize to [-0.5, 0.5]
                    double normX = (double)x / imgW - 0.5;
                    double normY = (double)y / imgH - 0.5;

                    // Weight: OpenPano formula
                    double weight = Math.max(0.0,
                            (0.5 - Math.abs(normX)) * (0.5 - Math.abs(normY))) + EPS;

                    owPtr.put(y * imgW + x, (float) weight);
                }
            }

            // Warp weight map (NHANH hơn nhiều vì OpenCV tối ưu)
            Mat warpedWeight = new Mat();
            warpPerspective(originalWeight, warpedWeight, H_final, new Size(W, H),
                    INTER_LINEAR, BORDER_CONSTANT, new Scalar(0));

            warpedImages[idx] = warpedImg;
            weightMaps[idx] = warpedWeight;

            H_final.release();
            originalWeight.release();
        }

        System.out.println("[Blend] Warping done, running winner-takes-all...");

        // ========================================
        // 4. UPDATE_WEIGHT_MAP: Winner-Takes-All
        // ========================================

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float maxWeight = 0;
                int winnerIdx = -1;

                // Tìm ảnh có weight cao nhất
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

        System.out.println("[Blend] Winner-takes-all done, feathering seams...");

        // ========================================
        // 5. Feather nhẹ đường nối (Optional)
        // ========================================

        int featherSize = 7;  // Rất nhỏ, chỉ làm mềm đường nối
        if (featherSize > 1) {
            for (int idx = 0; idx < nodes.size(); idx++) {
                Mat temp = new Mat();
                GaussianBlur(weightMaps[idx], temp,
                        new Size(featherSize, featherSize), 0);
                weightMaps[idx].release();
                weightMaps[idx] = temp;
            }
        }

        System.out.println("[Blend] Feathering done, blending images...");

        // ========================================
        // 6. Weighted sum blending
        // ========================================

        Mat result = new Mat(H, W, CV_32FC3, new Scalar(0.0, 0.0, 0.0, 0.0));

        for (int idx = 0; idx < nodes.size(); idx++) {
            Mat warpedImg = warpedImages[idx];
            Mat weightMap = weightMaps[idx];

            // Convert to float
            Mat img32F = new Mat();
            warpedImg.convertTo(img32F, CV_32FC3, 1.0/255.0, 0.0);

            // Multiply by weight và cộng vào result
            for (int c = 0; c < 3; c++) {
                Mat channel = new Mat();
                extractChannel(img32F, channel, c);

                Mat weighted = new Mat();
                multiply(channel, weightMap, weighted);

                Mat resultChannel = new Mat();
                extractChannel(result, resultChannel, c);

                add(resultChannel, weighted, resultChannel);

                insertChannel(resultChannel, result, c);

                channel.release();
                weighted.release();
                resultChannel.release();
            }

            img32F.release();
        }

        System.out.println("[Blend] Blending done, converting to 8-bit...");

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
