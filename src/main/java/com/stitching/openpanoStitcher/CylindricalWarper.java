package com.stitching.openpanoStitcher;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class CylindricalWarper {

    public Mat warp(Mat image, double focalLength) {
        int width = image.cols();
        int height = image.rows();

        Mat mapX = new Mat(height, width, CV_32F);
        Mat mapY = new Mat(height, width, CV_32F);

        float f = (float) focalLength;
        float halfW = width / 2.0f;
        float halfH = height / 2.0f;

        float[] xData = new float[width * height];
        float[] yData = new float[width * height];

        // Tạo Map biến đổi ngược: Từ tọa độ đích (đã uốn) -> tìm tọa độ nguồn (ảnh phẳng)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Tọa độ tương đối so với tâm
                float x_dest = x - halfW;
                float y_dest = y - halfH;

                // Công thức Cylindrical Projection chuẩn
                // x_src = f * tan(x_dest / f)
                // y_src = y_dest / cos(x_dest / f)

                float x_src = (float) (f * Math.tan(x_dest / f));
                float y_src = (float) (y_dest / Math.cos(x_dest / f));

                // Đưa về hệ tọa độ ảnh gốc
                x_src = x_src + halfW;
                y_src = y_src + halfH;

                int index = y * width + x;
                xData[index] = x_src;
                yData[index] = y_src;
            }
        }

        new FloatPointer(mapX.data()).put(xData);
        new FloatPointer(mapY.data()).put(yData);

        Mat result = new Mat();
        // Cắt bỏ phần viền đen do uốn cong để ghép cho đẹp
        remap(image, result, mapX, mapY, INTER_LINEAR, BORDER_CONSTANT, new Scalar(0, 0, 0, 0));

        return result;
    }
}