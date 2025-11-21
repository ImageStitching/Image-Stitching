package com.stitching.SIFT;

import com.stitching.imageOperator.Matrix_Image;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.SIFT;

import java.nio.file.Path;
import java.nio.file.Paths;

public class SIFTComparison {
    private static Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "sift");

    private static double[][] loadImageCustom(String path) {
        System.out.println("PHẦN 1: CUSTOM SIFT \n");
        double[][] image = Matrix_Image.create_DOUBLEgrayMatrix_from_color_image(path);
        if (image == null) throw new RuntimeException("Không thể load ảnh custom!");
        return image;
    }

    private static CustomResult runCustomSIFT(double[][] image) {
        long start = System.currentTimeMillis();
        SIFTFeatureDetector customDetector = new SIFTFeatureDetector(0, 3, 0.04, 10.0, 1.6, true, 5);
        ImageFeature f = customDetector.detectFeatures(image);
        long end = System.currentTimeMillis();

        System.out.printf(" Custom SIFT: %d keypoints\n", f.getNumKeypoints());
        System.out.printf("  Thời gian chạy: %.2f seconds\n\n", (end - start) / 1000.0);

        return new CustomResult(f, (end - start) / 1000.0);
    }

    private static Mat loadImageOpenCV(String path) {
        System.out.println(" PHẦN 2: OPENCV SIFT (ByteDeco)\n");

        Mat img = opencv_imgcodecs.imread(path, opencv_imgcodecs.IMREAD_GRAYSCALE);
        if (img.empty()) {
            throw new RuntimeException("Không thể load ảnh OpenCV!");
        }
        System.out.printf("  Ảnh OpenCV: %d x %d, channels=%d\n\n",
                img.rows(), img.cols(), img.channels());

        return img;
    }

    private static OpenCVResult runOpenCVSIFT(Mat img) {
        long start = System.currentTimeMillis();
        SIFT sift = SIFT.create();
        KeyPointVector keyPoints = new KeyPointVector();
        Mat descriptors = new Mat();
        sift.detectAndCompute(img, new Mat(), keyPoints, descriptors);
        long end = System.currentTimeMillis();
        System.out.printf("  OpenCV SIFT: %d keypoints\n", keyPoints.size());
        System.out.printf("  Kích thước Descriptor: [%d x %d]\n", descriptors.rows(), descriptors.cols());
        System.out.printf("  Thời gian chạy: %.2f seconds\n\n", (end - start) / 1000.0);

        return new OpenCVResult(keyPoints, descriptors, (end - start) / 1000.0);
    }

    private static void compareKeypoints(CustomResult custom, OpenCVResult openCV) {
        System.out.println("▶ PHẦN 3: CHI TIẾT SO SÁNH KEYPOINTS \n");

        int minKP = Math.min(custom.feature.getNumKeypoints(), (int) openCV.keyPoints.size());

        System.out.printf(" So sánh %d keypoint đầu tiên:\n\n", minKP);

        double sumDiffX = 0, sumDiffY = 0, sumDiffSize = 0, sumDiffAngle = 0;

        System.out.println("┌─────┬──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐");
        System.out.println("│ Idx │ Custom (x, y, size, angle)  │ OpenCV (x, y, size, angle)  │ Diff (x, y, size, angle)    │");
        System.out.println("├─────┼──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤");

        for (int i = 0; i < Math.min(10, minKP); i++) {
            ImageFeature.KeyPointInfo c = custom.feature.getKeyPoints().get(i);
            KeyPoint o = openCV.keyPoints.get(i);

            double diffX = Math.abs(c.pt_x - o.pt().x());
            double diffY = Math.abs(c.pt_y - o.pt().y());
            double diffSize = Math.abs(c.size - o.size());
            double diffAngle = Math.abs(c.angle - o.angle());

            sumDiffX += diffX;
            sumDiffY += diffY;
            sumDiffSize += diffSize;
            sumDiffAngle += diffAngle;

            System.out.printf("│ %3d │ (%7.1f,%7.1f,%5.2f,%6.1f°) │ (%7.1f,%7.1f,%5.2f,%6.1f°) │ (%6.1f,%6.1f,%5.2f,%6.1f°) │\n",
                    i, c.pt_x, c.pt_y, c.size, c.angle,
                    o.pt().x(), o.pt().y(), o.size(), o.angle(),
                    diffX, diffY, diffSize, diffAngle
            );
        }

        System.out.println("└─────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘");

        custom.avgDiffX = sumDiffX / minKP;
        custom.avgDiffY = sumDiffY / minKP;
        custom.avgDiffSize = sumDiffSize / minKP;
        custom.avgDiffAngle = sumDiffAngle / minKP;
    }

    private static void compareResponse(CustomResult custom, OpenCVResult openCV) {
        System.out.println("\n PHẦN 4: RESPONSE SCORE \n");
        System.out.println("┌─────┬──────────────────┬──────────────────┐");
        System.out.println("│ Idx │ Custom Response  │ OpenCV Response  │");
        System.out.println("├─────┼──────────────────┼──────────────────┤");

        int limit = Math.min(5, Math.min(custom.feature.getNumKeypoints(), (int) openCV.keyPoints.size()));
        for (int i = 0; i < limit; i++) {
            ImageFeature.KeyPointInfo c = custom.feature.getKeyPoints().get(i);
            KeyPoint o = openCV.keyPoints.get(i);
            System.out.printf("│ %3d │ %16.6f │ %16.6f │\n", i, c.response, o.response());
        }
        System.out.println("└─────┴──────────────────┴──────────────────┘");
    }

    private static void printConclusion(CustomResult custom, OpenCVResult openCV) {
        System.out.println("\n PHẦN 5: KẾT LUẬN \n");
        int customKP = custom.feature.getNumKeypoints();
        int opencvKP = (int) openCV.keyPoints.size();
        System.out.printf(" Custom SIFT: %4d keypoint (%.2f seconds)\n", customKP, custom.timeSec);
        System.out.printf(" OpenCV SIFT: %4d keypoint (%.2f seconds)\n", opencvKP, openCV.timeSec);

        double ratio = (double) customKP / opencvKP;
        System.out.printf(" Ratio (Custom/OpenCV): %.2f\n\n", ratio);
        if (Math.abs(ratio - 1.0) < 0.2) System.out.println(" Số lượng keypoint GẦN GIỐNG (chênh lệch < 20%)");
        else if (ratio > 1.2) System.out.println(" Custom phát hiện NHIỀU HƠN OpenCV");
        else System.out.println(" Custom phát hiện ÍT HƠN OpenCV");

        if (custom.avgDiffX < 2 && custom.avgDiffY < 2 && custom.avgDiffSize < 0.5) System.out.println(" Vị trí keypoints GẦN GIỐNG nhau");
        else System.out.println(" Vị trí keypoints CÓ KHÁC BIỆT");

        if (custom.avgDiffAngle < 5) System.out.println(" Hướng (Orientation) GẦN GIỐNG nhau");
        else System.out.println(" Hướng (Orientation) CÓ KHÁC BIỆT");
    }

    private static class CustomResult {
        ImageFeature feature;
        double timeSec;
        double avgDiffX, avgDiffY, avgDiffSize, avgDiffAngle;
        CustomResult(ImageFeature f, double t) {
            this.feature = f;
            this.timeSec = t;
        }
    }

    private static class OpenCVResult {
        KeyPointVector keyPoints;
        Mat descriptors;
        double timeSec;

        OpenCVResult(KeyPointVector kp, Mat des, double t) {
            this.keyPoints = kp;
            this.descriptors = des;
            this.timeSec = t;
        }
    }

    public static void main(String[] args) {

        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║     SIFT COMPARISON: Custom vs OpenCV (ByteDeco)                ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝\n");

        String imagePath = INPUT_PATH.resolve("org_img.png").toString();
        System.out.println("📷 Loading image: " + imagePath + "\n");

        // PHẦN 1: CUSTOM SIFT
        double[][] customImage = loadImageCustom(imagePath);
        CustomResult custom = runCustomSIFT(customImage);

        // PHẦN 2: OpenCV SIFT
        Mat opencvImage = loadImageOpenCV(imagePath);
        OpenCVResult openCV = runOpenCVSIFT(opencvImage);

        // PHẦN 3: So sánh Keypoints
        compareKeypoints(custom, openCV);

        // PHẦN 4: Response score
        compareResponse(custom, openCV);

        // PHẦN 5: Tổng kết
        printConclusion(custom, openCV);
    }
}
