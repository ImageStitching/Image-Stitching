package com.stitching.imageStitching.warper;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class CylindricalWarper {
    
    /**
     * Uốn cong ảnh theo phép chiếu hình trụ.
     * @param image Ảnh gốc
     * @param f Tiêu cự (Focal Length)
     * @return Ảnh đã uốn cong và cắt bỏ viền đen
     */
    public static Mat warp(Mat image, double f) {
        int w = image.cols();
        int h = image.rows();

        Mat mapX = new Mat(h, w, CV_32F);
        Mat mapY = new Mat(h, w, CV_32F);
        
        FloatPointer pX = new FloatPointer(mapX.data());
        FloatPointer pY = new FloatPointer(mapY.data());
        
        float hW = w / 2.0f;
        float hH = h / 2.0f;
        float focal = (float) f;

        // Tạo bản đồ biến đổi ngược (Inverse Mapping)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // Tọa độ cực (cylindrical coordinates)
                float theta = (x - hW) / focal;
                float h_cyl = (y - hH) / focal;

                // Công thức chiếu ngược chuẩn từ OpenPano (warp.cc)
                float x_src = (float) (focal * Math.tan(theta)) + hW;
                float y_src = (float) (h_cyl * focal / Math.cos(theta)) + hH;
                // Hoặc dùng công thức tương đương: y_src = h_cyl * sqrt(f^2 + x_hat^2) + hH

                long idx = y * w + x;
                pX.put(idx, x_src);
                pY.put(idx, y_src);
            }
        }

        Mat result = new Mat();
        remap(image, result, mapX, mapY, INTER_LINEAR, BORDER_CONSTANT, new Scalar(0, 0, 0, 0));
        
        mapX.release(); 
        mapY.release();
        
        return cropBlackBorder(result);
    }

    private static Mat cropBlackBorder(Mat img) {
        Mat gray = new Mat();
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat points = new Mat();
        findNonZero(gray, points);
        if (points.empty()) {
            gray.release();
            points.release();
            return img;
        }
        Rect bb = boundingRect(points);
        Mat cropped = new Mat(img, bb);
        
        gray.release();
        points.release();
        return cropped;
    }
}