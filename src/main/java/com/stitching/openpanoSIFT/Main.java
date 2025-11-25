package com.stitching.openpanoSIFT;

import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class Main {
    private static Path INPUT_PATH = Paths.get("src","main","resources","static","sift");

    public static void main(String[] args) {
        String imgName = "memeNgua6.jpg";
        String imagePath = INPUT_PATH.resolve(imgName).toString();
        System.out.println(" ---- > Loading image: " + imagePath);
        String filename = imagePath; // Đặt ảnh vào thư mục gốc dự án
        Mat src = imread(filename);
        if (src.empty()) {
            System.err.println("Cannot read image: " + filename);
            return;
        }

        // 1. Tiền xử lý: Grayscale + Float
        Mat gray = new Mat();
        cvtColor(src, gray, COLOR_BGR2GRAY);
        Mat floatGray = new Mat();
        gray.convertTo(floatGray, CV_32F, 1.0/255.0, 0.0);

        System.out.println("Processing " + floatGray.cols() + "x" + floatGray.rows() + "...");

        // 2. Scale Space
        ScaleSpace scaleSpace = new ScaleSpace();
        List<MatVector> gPyramid = scaleSpace.buildGaussianPyramid(floatGray);
        List<MatVector> dogPyramid = scaleSpace.buildDoGPyramid(gPyramid);

        // 3. Detect, Interpolate, Orient, Describe
        SiftDetector detector = new SiftDetector();
        List<SiftKeyPoint> keypoints = detector.run(gPyramid, dogPyramid);

        System.out.println("Found " + keypoints.size() + " SIFT keypoints.");

        // 4. Vẽ kết quả
        for (SiftKeyPoint kp : keypoints) {
            // Vẽ vòng tròn thể hiện scale
            int radius = (int) kp.scale; 
            Point center = new Point((int)kp.x, (int)kp.y);
            circle(src, center, radius, new Scalar(0, 255, 0, 0), 1, LINE_AA, 0);
            
            // Vẽ đường thẳng thể hiện hướng (Orientation)
            int endX = (int) (kp.x + radius * Math.cos(Math.toRadians(kp.angle)));
            int endY = (int) (kp.y + radius * Math.sin(Math.toRadians(kp.angle)));
            line(src, center, new Point(endX, endY), new Scalar(0, 0, 255, 0), 1, LINE_AA, 0);
        }

        String outputPath = INPUT_PATH.resolve("result_full_sift_"+imgName).toString();
        imwrite(outputPath, src);
        System.out.println("Ảnh kết quả lưu ở: " + outputPath);
    }
}