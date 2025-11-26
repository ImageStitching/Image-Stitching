package com.stitching.imageStitching.matchAndTransform;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_calib3d.*;
import static org.bytedeco.opencv.global.opencv_core.*;

public class TransformEstimator {

    /**
     * Tính ma trận Affine (2x3) rồi chuyển thành 3x3.
     * OpenPano dùng Affine cho Cylinder mode để tránh biến dạng phối cảnh.
     */
    public static Mat estimateAffine(Mat srcPoints, Mat dstPoints) {
        Mat mask = new Mat();
        // RANSAC threshold 4.0 pixel (OpenPano default)
        Mat aff2D = estimateAffine2D(srcPoints, dstPoints, mask, RANSAC, 4.0, 2000, 0.99, 0);
        mask.release();

        if (aff2D.empty()) return null;

        // Chuyển đổi 2x3 sang 3x3 Homography form để dễ nhân tích lũy
        Mat H = Mat.eye(3, 3, CV_64F).asMat();
        DoublePointer s = new DoublePointer(aff2D.data());
        DoublePointer d = new DoublePointer(H.data());
        
        // Copy 2 hàng đầu tiên: [a00 a01 tx]
        //                       [a10 a11 ty]
        for (int i = 0; i < 6; i++) d.put(i, s.get(i));
        
        return H;
    }

    /**
     * Sanity Check: Kiểm tra xem ma trận có hợp lý không.
     */
    public static boolean isTransformValid(Mat H) {
        if (H == null || H.empty()) return false;
        DoublePointer val = new DoublePointer(H.data());
        
        double sx = Math.sqrt(val.get(0)*val.get(0) + val.get(3)*val.get(3)); // Scale X approx
        double sy = Math.sqrt(val.get(1)*val.get(1) + val.get(4)*val.get(4)); // Scale Y approx
        
        // Scale không được thay đổi quá nhiều (0.8 - 1.2)
        if (sx < 0.8 || sx > 1.25 || sy < 0.8 || sy > 1.25) return false;
        
        return true;
    }
}