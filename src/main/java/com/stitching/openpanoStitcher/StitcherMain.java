package com.stitching.openpanoStitcher;

import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftConfig;
import com.stitching.openpanoSIFT.SiftDetector;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_features2d.*;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class StitcherMain {
    // Đổi đường dẫn này về thư mục chứa ảnh của bạn
    private static Path INPUT_PATH = Paths.get("src","main","resources","static","sift_one_scene_vertical");

    public static void main(String[] args) {
        // QUAN TRỌNG: img1 là ảnh TRÁI, img2 là ảnh PHẢI
        // Theo ảnh bạn gửi: img2.jpg (bên trái), img1.jpg (bên phải)
        String pathLeft = INPUT_PATH.resolve("img2.jpg").toString();
        String pathRight = INPUT_PATH.resolve("img1.jpg").toString();

        Mat imgLeftRaw = imread(pathLeft);
        Mat imgRightRaw = imread(pathRight);

        if (imgLeftRaw.empty() || imgRightRaw.empty()) {
            System.err.println("Không tìm thấy ảnh!");
            return;
        }

        System.out.println("1. Đang uốn cong ảnh (Cylindrical Warping)...");
        CylindricalWarper warper = new CylindricalWarper();

        // Tiêu cự giả định. Với camera điện thoại, f ~ width là mức an toàn.
        // Bạn có thể thử chỉnh số này: 800, 1000, 1200.
        double f = imgLeftRaw.cols();

        Mat imgLeft = warper.warp(imgLeftRaw, f);
        Mat imgRight = warper.warp(imgRightRaw, f);

        // Lưu ảnh warp để debug xem nó có cong không
        imwrite(INPUT_PATH.resolve("debug_warp_left.jpg").toString(), imgLeft);
        imwrite(INPUT_PATH.resolve("debug_warp_right.jpg").toString(), imgRight);

        System.out.println("2. Đang tìm đặc trưng SIFT...");
        SiftData d1 = runSift(imgLeft);
        SiftData d2 = runSift(imgRight);

        System.out.println("3. Đang khớp điểm (Matching)...");
        FeatureMatcher matcher = new FeatureMatcher();
        Mat desc1 = matcher.convertDescriptorsToMat(d1.keypoints);
        Mat desc2 = matcher.convertDescriptorsToMat(d2.keypoints);

        // Match Left (query) vs Right (train)
        FeatureMatcher.MatchResult res = matcher.matchFeatures(d1.keypoints, desc1, d2.keypoints, desc2);

        if (res == null || res.inlierMatches.isEmpty()) {
            System.err.println("Không tìm thấy điểm trùng khớp! Hãy kiểm tra lại thứ tự ảnh.");
            return;
        }
        System.out.println("   Tìm thấy " + res.inlierMatches.size() + " cặp điểm trùng khớp.");

        // Vẽ đường nối để kiểm tra (Optional)
        Mat debugMatch = new Mat();
        drawMatches(imgLeft, convertToOpenCVKeyPoints(d1.keypoints),
                imgRight, convertToOpenCVKeyPoints(d2.keypoints),
                new DMatchVector(res.inlierMatches.toArray(new DMatch[0])), debugMatch);
        imwrite(INPUT_PATH.resolve("debug_matches.jpg").toString(), debugMatch);

        // 4. Tính ma trận dịch chuyển
        // Ta muốn ghép Right vào Left. Cần tìm H sao cho: H * Right = Left
        // Input của computeH là (srcPoints, dstPoints).
        // src = Right (img2), dst = Left (img1)
        TransformEstimator estimator = new TransformEstimator();
        Mat H = estimator.computeH(res.dstPoints, res.srcPoints);

        System.out.println("4. Đang ghép ảnh...");

        // Tạo Canvas đủ rộng
        Mat result = new Mat();
        Size canvasSize = new Size(imgLeft.cols() + imgRight.cols(), imgLeft.rows());

        // Warp ảnh phải sang vị trí mới
        warpPerspective(imgRight, result, H, canvasSize);

        // Dán ảnh trái vào vị trí gốc
        Mat roi = new Mat(result, new Rect(0, 0, imgLeft.cols(), imgLeft.rows()));

        // Tạo mask để cắt bỏ phần đen của ảnh trái khi dán đè lên
        Mat maskLeft = new Mat();
        cvtColor(imgLeft, maskLeft, COLOR_BGR2GRAY);
        threshold(maskLeft, maskLeft, 1, 255, THRESH_BINARY);

        imgLeft.copyTo(roi, maskLeft);

        // Cắt bỏ phần thừa màu đen bên phải canvas
        Mat finalResult = cropBlackBorder(result);

        String outPath = INPUT_PATH.resolve("result_panorama.jpg").toString();
        imwrite(outPath, finalResult);
        System.out.println("Xong! Kết quả tại: " + outPath);
    }

    // Hàm phụ trợ: Cắt bỏ viền đen thừa thãi
    private static Mat cropBlackBorder(Mat img) {
        Mat gray = new Mat();
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat points = new Mat();
        findNonZero(gray, points);
        Rect bb = boundingRect(points);
        return new Mat(img, bb);
    }

    // --- Các hàm Helper cũ (Giữ nguyên) ---
    static class SiftData { List<SiftKeyPoint> keypoints; }

    private static SiftData runSift(Mat img) {
        Mat gray = new Mat();
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat floatGray = new Mat();
        gray.convertTo(floatGray, CV_32F, 1.0/255.0, 0.0);
        ScaleSpace ss = new ScaleSpace();
        SiftDetector det = new SiftDetector();
        SiftData d = new SiftData();
        // Không cần upscale nữa vì ảnh warp đã đủ chi tiết và tránh nặng máy
        SiftConfig.DOUBLE_IMAGE_SIZE = false;
        d.keypoints = det.run(ss.buildGaussianPyramid(floatGray), ss.buildDoGPyramid(ss.buildGaussianPyramid(floatGray)));
        return d;
    }

    private static KeyPointVector convertToOpenCVKeyPoints(List<SiftKeyPoint> kps) {
        KeyPointVector vec = new KeyPointVector(kps.size());
        for (long i = 0; i < kps.size(); i++) {
            SiftKeyPoint kp = kps.get((int)i);
            KeyPoint cvKp = new KeyPoint(kp.x, kp.y, kp.scale, kp.angle, 0f, 0, -1);
            vec.put(i, cvKp);
        }
        return vec;
    }
}