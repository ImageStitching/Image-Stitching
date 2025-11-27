/* package com.stitching.imageStitching.blender;

import com.stitching.imageStitching.ImageNode;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_core.MatVector;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ImageBlenderFAST {

    private static final double EPS = 1e-6;

    public static Mat blend(List<ImageNode> nodes) {
        System.out.println("[Blend] Calculating canvas size...");

        // 1. TÍNH CANVAS SIZE (GIỮ NGUYÊN)
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE;

        for (ImageNode node : nodes) {
            Size s = node.img.size();
            Mat corners = new Mat(4, 1, CV_32FC2);
            FloatPointer ptr = new FloatPointer(corners.data());
            ptr.put(0, 0); ptr.put(1, 0);
            ptr.put(2, s.width()); ptr.put(3, 0);
            ptr.put(4, s.width()); ptr.put(5, s.height());
            ptr.put(6, 0); ptr.put(7, s.height());

            Mat transCorners = new Mat();
            perspectiveTransform(corners, transCorners, node.globalTransform);

            FloatPointer tPtr = new FloatPointer(transCorners.data());
            for(int i=0; i<4; i++) {
                float x = tPtr.get(i*2);
                float y = tPtr.get(i*2+1);
                if(x < minX) minX = x;
                if(x > maxX) maxX = x;
                if(y < minY) minY = y;
                if(y > maxY) maxY = y;
            }
            corners.release(); transCorners.release();
        }

        int W = (int) Math.ceil(maxX - minX);
        int H = (int) Math.ceil(maxY - minY);

        if (W <= 0 || H <= 0 || W > 60000 || H > 60000) {
            System.err.println("Invalid canvas size: " + W + "x" + H);
            return null;
        }
        System.out.println("   -> Canvas: " + W + "x" + H);

        // 2. MA TRẬN DỊCH CHUYỂN (OFFSET)
        Mat T_offset = Mat.eye(3, 3, CV_64F).asMat();
        DoublePointer dp = new DoublePointer(T_offset.data());
        dp.put(2, -minX);
        dp.put(5, -minY);

        // 3. CHUẨN BỊ TÍCH LŨY
        Mat resultAccum = Mat.zeros(new Size(W, H), CV_32FC3).asMat();
        Mat weightAccum = Mat.zeros(new Size(W, H), CV_32F).asMat();

        System.out.println("[Blend] Warping & Compensating Exposure...");

        for (int idx = 0; idx < nodes.size(); idx++) {
            ImageNode node = nodes.get(idx);

            // T_final = T_offset * T_global
            Mat H_final = new Mat();
            gemm(T_offset, node.globalTransform, 1.0, new Mat(), 0.0, H_final);

            // A. Warp Ảnh Gốc
            Mat warpedImg = new Mat();
            warpPerspective(node.img, warpedImg, H_final, new Size(W, H), INTER_LINEAR, BORDER_CONSTANT, new Scalar(0,0,0,0));

            // Tạo Mask (Vùng có ảnh thực sự)
            Mat mask = new Mat();
            cvtColor(warpedImg, mask, COLOR_BGR2GRAY);
            threshold(mask, mask, 1, 255, THRESH_BINARY); // Pixel > 1 là có ảnh

            // B. [NÂNG CẤP 1] TẠO WEIGHT MAP BẰNG DISTANCE TRANSFORM
            // Giúp làm mờ biên cực mịn theo hình dáng thực tế của ảnh sau khi warp
            Mat distMap = new Mat();
            distanceTransform(mask, distMap, DIST_L2, 3);

            // Normalize weight về 0.0 - 1.0
            Mat weightMap = new Mat();
            normalize(distMap, weightMap, 0.0, 1.0, NORM_MINMAX, -1, null);

            // Convert Weight sang Float
            Mat weightFloat = new Mat();
            weightMap.convertTo(weightFloat, CV_32F);

            // Convert Ảnh sang Float
            Mat imgFloat = new Mat();
            warpedImg.convertTo(imgFloat, CV_32FC3);

            // C. [NÂNG CẤP 2] BÙ SÁNG (EXPOSURE COMPENSATION)
            // So sánh độ sáng vùng chồng lấn giữa ảnh hiện tại và phần đã tích lũy
            if (idx > 0) {
                // Tìm vùng chồng lấn: Nơi cả mask hiện tại và weightAccum cũ đều > 0
                Mat overlapMask = new Mat();
                Mat weightMask = new Mat();

                // Tạo mask từ weightAccum (những chỗ đã có ảnh)
                threshold(weightAccum, weightMask, EPS, 255, THRESH_BINARY);
                weightMask.convertTo(weightMask, CV_8U);

                // Giao của 2 vùng
                bitwise_and(mask, weightMask, overlapMask);

                // Nếu vùng chồng lấn đủ lớn, tính toán bù sáng
                if (countNonZero(overlapMask) > 1000) {
                    // Tính trung bình độ sáng của ảnh hiện tại trong vùng overlap
                    Scalar meanCurr = mean(imgFloat, overlapMask);
                    double brightnessCurr = (meanCurr.get(0) + meanCurr.get(1) + meanCurr.get(2)) / 3.0;

                    // Tính trung bình độ sáng của nền (đã tích lũy) trong vùng overlap
                    // Cần chia resultAccum cho weightAccum để lấy màu thực tế
                    // Tuy nhiên để nhanh, ta có thể xấp xỉ hoặc cắt ROI.
                    // Ở đây ta làm kỹ:

                    // Cắt vùng ROI overlap để tính cho nhanh (tránh tính cả canvas to)
                    Rect roi = boundingRect(overlapMask);
                    if (roi.width() > 0 && roi.height() > 0) {
                        Mat roiResult = resultAccum.apply(roi);
                        Mat roiWeight = weightAccum.apply(roi);
                        Mat roiMask = overlapMask.apply(roi);

                        // [FIX] Tách kênh để chia (3 kênh / 1 kênh)
                        Mat roiBackground = new Mat();
                        MatVector bgChannels = new MatVector();
                        split(roiResult, bgChannels); // Tách roiResult thành 3 kênh R, G, B

                        // Chia từng kênh cho roiWeight
                        for (long k = 0; k < bgChannels.size(); k++) {
                            Mat c = bgChannels.get(k);
                            divide(c, roiWeight, c);
                        }
                        merge(bgChannels, roiBackground); // Gộp lại thành ảnh màu
                        // Tạm thời chia để lấy màu trung bình nền
//                        Mat roiBackground = new Mat();
//                        divide(roiResult, roiWeight, roiBackground); // Chia cho weight để ra màu gốc

                        Scalar meanBg = mean(roiBackground, roiMask);
                        double brightnessBg = (meanBg.get(0) + meanBg.get(1) + meanBg.get(2)) / 3.0;

                        if (brightnessCurr > 1.0 && brightnessBg > 1.0) {
                            double gain = brightnessBg / brightnessCurr;

                            // Giới hạn gain để không bị đổi màu quá đà (0.7 - 1.4)
                            if (gain < 0.7) gain = 0.7;
                            if (gain > 1.4) gain = 1.4;

                            //System.out.println("   -> Compensating exposure for img " + idx + ": Gain = " + String.format("%.2f", gain));

                            // Áp dụng Gain vào ảnh hiện tại
                            multiply(imgFloat, new Mat(imgFloat.size(), CV_32FC3, new Scalar(gain, gain, gain, 0)), imgFloat);
                        }
                        roiResult.release(); roiWeight.release(); roiMask.release(); roiBackground.release();
                    }
                }
                overlapMask.release(); weightMask.release();
            }

            // D. CỘNG DỒN VÀO ACCUMULATOR
            MatVector channels = new MatVector();
            split(imgFloat, channels);

            // Nhân từng kênh màu với Weight: RGB * W
            for (long k = 0; k < channels.size(); k++) {
                Mat c = channels.get(k);
                multiply(c, weightFloat, c);
            }
            merge(channels, imgFloat);

            // Result += RGB*W
            add(resultAccum, imgFloat, resultAccum);
            // WeightSum += W
            add(weightAccum, weightFloat, weightAccum);

            // Giải phóng bộ nhớ
            H_final.release(); mask.release();
            distMap.release(); weightMap.release();
            warpedImg.release(); imgFloat.release(); weightFloat.release();
        }

        // 4. CHUẨN HÓA (NORMALIZE)
        System.out.println("[Blend] Normalizing & Converting...");

        // Cộng epsilon để tránh chia cho 0
        add(weightAccum, new Mat(new Size(W, H), CV_32F, new Scalar(EPS)), weightAccum);

        MatVector resChannels = new MatVector();
        split(resultAccum, resChannels);

        // Chia từng kênh: Pixel = (Tổng RGB*W) / (Tổng W)
        for (long k = 0; k < resChannels.size(); k++) {
            Mat c = resChannels.get(k);
            divide(c, weightAccum, c);
        }

        Mat finalFloat = new Mat();
        merge(resChannels, finalFloat);

        // 5. Convert về 8-bit
        Mat finalRes = new Mat();
        finalFloat.convertTo(finalRes, CV_8UC3);

        // Cleanup
        T_offset.release(); resultAccum.release();
        weightAccum.release(); finalFloat.release();

        return finalRes;
    }
}*/


package com.stitching.imageStitching.blender;

import com.stitching.imageStitching.ImageNode;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;

import java.util.List;

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
