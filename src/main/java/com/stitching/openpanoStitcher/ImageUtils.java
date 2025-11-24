package com.stitching.openpanoStitcher;

import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.*;

public class ImageUtils {
    // Xoay 90 độ ngược chiều kim đồng hồ (CCW)
    // Dùng để biến: TOP -> LEFT, BOTTOM -> RIGHT
    public static Mat rotateLeft(Mat src) {
        Mat dst = new Mat();
        rotate(src, dst, ROTATE_90_COUNTERCLOCKWISE);
        return dst;
    }

    // Xoay 90 độ cùng chiều kim đồng hồ (CW)
    // Dùng để xoay kết quả về lại thẳng đứng
    public static Mat rotateRight(Mat src) {
        Mat dst = new Mat();
        rotate(src, dst, ROTATE_90_CLOCKWISE);
        return dst;
    }
}