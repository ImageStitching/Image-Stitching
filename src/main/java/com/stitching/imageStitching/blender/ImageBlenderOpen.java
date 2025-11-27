package com.stitching.imageStitching.blender;

import com.stitching.imageStitching.ImageNode;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_core.MatVector;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ImageBlenderOpen {
    private static final double EPS = 1e-6; // Epsilon để tránh chia cho 0

    public static Mat blend(List<ImageNode> nodes) {
        System.out.println("[Blend] Calculating canvas size...");

        // 1. TÍNH CANVAS SIZE (Khung tranh toàn cục)
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE;

        for (ImageNode node : nodes) {
            Size s = node.img.size();
            // 4 góc của ảnh gốc
            Mat corners = new Mat(4, 1, CV_32FC2);
            FloatPointer ptr = new FloatPointer(corners.data());
            ptr.put(0, 0); ptr.put(1, 0);
            ptr.put(2, s.width()); ptr.put(3, 0);
            ptr.put(4, s.width()); ptr.put(5, s.height());
            ptr.put(6, 0); ptr.put(7, s.height());

            // Biến đổi góc theo Global Transform để tìm biên
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

        // Kiểm tra an toàn kích thước
        if (W <= 0 || H <= 0 || W > 60000 || H > 60000) {
            System.err.println("Invalid canvas size: " + W + "x" + H + ". Check Transforms!");
            return null;
        }
        System.out.println("   -> Canvas: " + W + "x" + H);

        // 2. MA TRẬN DỊCH CHUYỂN (Offset về 0,0)
        Mat T_offset = Mat.eye(3, 3, CV_64F).asMat();
        DoublePointer dp = new DoublePointer(T_offset.data());
        dp.put(2, -minX);
        dp.put(5, -minY);

        // 3. CHUẨN BỊ TÍCH LŨY (ACCUMULATOR)
        // resultAccum: Lưu tử số (Tổng Màu * Trọng số) - Dùng Float 3 kênh
        Mat resultAccum = Mat.zeros(new Size(W, H), CV_32FC3).asMat();

        // weightAccum: Lưu mẫu số (Tổng Trọng số) - Dùng Float 1 kênh
        Mat weightAccum = Mat.zeros(new Size(W, H), CV_32F).asMat();

        System.out.println("[Blend] Warping & Accumulating (Linear)...");

        for (int idx = 0; idx < nodes.size(); idx++) {
            ImageNode node = nodes.get(idx);

            // T_final = T_offset * T_global
            Mat H_final = new Mat();
            gemm(T_offset, node.globalTransform, 1.0, new Mat(), 0.0, H_final);

            // A. Warp Ảnh Gốc
            Mat warpedImg = new Mat();
            warpPerspective(node.img, warpedImg, H_final, new Size(W, H), INTER_LINEAR, BORDER_CONSTANT, new Scalar(0,0,0,0));

            // B. Tạo Weight Map (Khoảng cách từ tâm - Giống OpenPano)
            int imgW = node.img.cols();
            int imgH = node.img.rows();
            Mat originalWeight = new Mat(imgH, imgW, CV_32F);
            FloatPointer owPtr = new FloatPointer(originalWeight.data());

            for (int y = 0; y < imgH; y++) {
                for (int x = 0; x < imgW; x++) {
                    // Chuẩn hóa tọa độ về [-0.5, 0.5]
                    double normX = (double)x / imgW - 0.5;
                    double normY = (double)y / imgH - 0.5;

                    // Weight giảm dần từ tâm (0.5) ra biên (0)
                    // Công thức: (0.5 - |x|) * (0.5 - |y|)
                    double weight = Math.max(0.0, (0.5 - Math.abs(normX)) * (0.5 - Math.abs(normY)));

                    // Mũ 1 (Tuyến tính) để blending đều
                    owPtr.put(y * imgW + x, (float) weight);
                }
            }

            // Warp Weight Map sang không gian chung
            Mat warpedWeight = new Mat();
            warpPerspective(originalWeight, warpedWeight, H_final, new Size(W, H), INTER_LINEAR, BORDER_CONSTANT, new Scalar(0));

            // C. Cộng dồn (Accumulate)
            Mat imgFloat = new Mat();
            warpedImg.convertTo(imgFloat, CV_32FC3);

            // [QUAN TRỌNG] Dùng MatVector thay vì List<Mat> để tránh lỗi biên dịch
            MatVector channels = new MatVector();
            split(imgFloat, channels);

            // Nhân từng kênh màu với Weight: RGB * W
            for (long k = 0; k < channels.size(); k++) {
                Mat c = channels.get(k);
                multiply(c, warpedWeight, c); // c = c * w
            }
            merge(channels, imgFloat);

            // Cộng vào tổng kết quả: Result += RGB*W
            add(resultAccum, imgFloat, resultAccum);

            // Cộng vào tổng trọng số: WeightSum += W
            add(weightAccum, warpedWeight, weightAccum);

            // Giải phóng bộ nhớ tạm
            H_final.release(); originalWeight.release();
            warpedImg.release(); warpedWeight.release();
            imgFloat.release();
            // channels không cần release thủ công, JavaGC sẽ lo, hoặc gọi channels.close() nếu muốn
        }

        // 4. CHUẨN HÓA (NORMALIZE)
        // Pixel = (Tổng RGB*W) / (Tổng W)
        System.out.println("[Blend] Normalizing & Converting...");

        // Tránh chia cho 0: Cộng epsilon vào mẫu số
        add(weightAccum, new Mat(new Size(W, H), CV_32F, new Scalar(EPS)), weightAccum);

        MatVector resChannels = new MatVector();
        split(resultAccum, resChannels);

        // Chia từng kênh
        for (long k = 0; k < resChannels.size(); k++) {
            Mat c = resChannels.get(k);
            divide(c, weightAccum, c);
        }

        Mat finalFloat = new Mat();
        merge(resChannels, finalFloat);

        // 5. Convert về 8-bit để hiển thị
        Mat finalRes = new Mat();
        finalFloat.convertTo(finalRes, CV_8UC3);

        // Dọn dẹp
        T_offset.release(); resultAccum.release();
        weightAccum.release(); finalFloat.release();

        return finalRes;
    }
}
