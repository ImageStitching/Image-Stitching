package com.stitching.openpanoStitcher;

import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_calib3d.*;

public class TransformEstimator {
    // Tính H từ ảnh Phải (src) -> ảnh Trái (dst)
    public Mat computeH(Mat srcPoints, Mat dstPoints) {
        Mat mask = new Mat();
        // return findHomography(srcPoints, dstPoints, RANSAC, 4.0, mask, 2000, 0.995);
        return findHomography(srcPoints, dstPoints, USAC_MAGSAC, 4.0, mask, 2000, 0.995);
    }
}