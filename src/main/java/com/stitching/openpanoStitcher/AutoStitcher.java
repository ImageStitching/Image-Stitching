package com.stitching.openpanoStitcher;

import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftConfig;
import com.stitching.openpanoSIFT.SiftDetector;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.*;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_features2d.*;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class AutoStitcher {
    private static Path INPUT_PATH = Paths.get("src","main","resources","static","stitching_one_scene_crossline");

    // --- CẤU TRÚC DỮ LIỆU ĐỂ PHÂN TÍCH QUAN HỆ ẢNH ---
    enum StitchDirection { HORIZONTAL, VERTICAL }

    static class ImageRelation {
        StitchDirection direction;
        boolean needSwap; // True nếu thứ tự ảnh bị ngược (Phải->Trái hoặc Dưới->Trên)
    }
    // --------------------------------------------------

    public static void main(String[] args) {
        String path1 = INPUT_PATH.resolve("img1.jpg").toString();
        String path2 = INPUT_PATH.resolve("img2.jpg").toString();

        Mat img1 = imread(path1);
        Mat img2 = imread(path2);

        if (img1.empty() || img2.empty()) {
            System.err.println("Không đọc được ảnh!");
            return;
        }

        System.out.println("=== BẮT ĐẦU TỰ ĐỘNG GHÉP ẢNH (AUTO-ORDER & DIRECTION) ===");

        // BƯỚC 1: CHẠY SIFT SƠ BỘ TRÊN ẢNH GỐC
        // Chạy 1 lần ở đây để dùng cho việc phân tích hướng và thứ tự
        System.out.println("1. Phân tích đặc trưng ảnh gốc...");
        SiftConfig.DOUBLE_IMAGE_SIZE = false;
        SiftData d1 = runSift(img1);
        SiftData d2 = runSift(img2);

        // Matching sơ bộ
        FeatureMatcher matcher = new FeatureMatcher();
        Mat desc1 = matcher.convertDescriptorsToMat(d1.keypoints);
        Mat desc2 = matcher.convertDescriptorsToMat(d2.keypoints);
        FeatureMatcher.MatchResult res = matcher.matchFeatures(d1.keypoints, desc1, d2.keypoints, desc2);

        if (res == null || res.inlierMatches.isEmpty()) {
            System.err.println("Không tìm thấy điểm chung giữa 2 ảnh! Không thể ghép.");
            return;
        }

        // BƯỚC 2: PHÂN TÍCH HƯỚNG VÀ THỨ TỰ
        ImageRelation relation = analyzeRelation(res.inlierMatches, d1, d2);

        // BƯỚC 3: SẮP XẾP LẠI ẢNH (NẾU CẦN)
        Mat finalImg1, finalImg2;
        SiftData finalD1, finalD2;

        if (relation.needSwap) {
            System.out.println(">> PHÁT HIỆN NGƯỢC THỨ TỰ: Đang đảo vị trí ảnh...");
            finalImg1 = img2;
            finalImg2 = img1;
            finalD1 = d2;
            finalD2 = d1;
        } else {
            System.out.println(">> Thứ tự ảnh đã CHÍNH XÁC.");
            finalImg1 = img1;
            finalImg2 = img2;
            finalD1 = d1;
            finalD2 = d2;
        }

        Mat finalResult = null;

        // BƯỚC 4: XỬ LÝ GHÉP DỰA TRÊN HƯỚNG
        if (relation.direction == StitchDirection.HORIZONTAL) {
            System.out.println(">> CHẾ ĐỘ: GHÉP NGANG (Panorama 360)");
            // Ghép ngang -> Bật Warp hình trụ (true)
            // Truyền finalD1, finalD2 vào để nếu không warp thì dùng lại, đỡ tính toán
            finalResult = stitchRobust(finalImg1, finalImg2, finalD1, finalD2, true);
        } else {
            System.out.println(">> CHẾ ĐỘ: GHÉP DỌC (Planar)");
            System.out.println(">> Đang xoay ảnh để xử lý...");

            // Xoay ảnh sang ngang để tận dụng logic ghép
            Mat img1Rot = ImageUtils.rotateLeft(finalImg1);
            Mat img2Rot = ImageUtils.rotateLeft(finalImg2);

            // Ghép Dọc -> Tắt Warp (false) -> Giữ phẳng
            // Truyền null cho SiftData vì ảnh đã xoay, keypoint cũ không dùng được, cần tính lại bên trong hàm
            Mat stitchedRot = stitchRobust(img1Rot, img2Rot, null, null, false);

            if (stitchedRot != null) {
                // Xoay kết quả về lại chiều dọc
                finalResult = ImageUtils.rotateRight(stitchedRot);
            }
        }

        if (finalResult != null) {
            String outPath = INPUT_PATH.resolve("result_auto_panorama.jpg").toString();
            imwrite(outPath, finalResult);
            System.out.println("=== HOÀN TẤT! Kết quả: " + outPath + " ===");
        } else {
            System.err.println("Ghép thất bại.");
        }
    }

    // --- LOGIC PHÂN TÍCH MỚI ---
    private static ImageRelation analyzeRelation(List<DMatch> matches, SiftData d1, SiftData d2) {
        ImageRelation rel = new ImageRelation();

        double sumX1 = 0, sumX2 = 0;
        double sumY1 = 0, sumY2 = 0;
        double sumDeltaX = 0, sumDeltaY = 0;

        int count = Math.min(matches.size(), 100); // Lấy mẫu 100 điểm tốt nhất

        for(int i=0; i<count; i++) {
            DMatch m = matches.get(i);
            SiftKeyPoint p1 = d1.keypoints.get(m.queryIdx());
            SiftKeyPoint p2 = d2.keypoints.get(m.trainIdx());

            sumX1 += p1.x; sumX2 += p2.x;
            sumY1 += p1.y; sumY2 += p2.y;

            sumDeltaX += Math.abs(p1.x - p2.x);
            sumDeltaY += Math.abs(p1.y - p2.y);
        }

        double avgX1 = sumX1 / count;
        double avgX2 = sumX2 / count;
        double avgY1 = sumY1 / count;
        double avgY2 = sumY2 / count;

        double avgDeltaX = sumDeltaX / count;
        double avgDeltaY = sumDeltaY / count;

        // 1. Xác định Hướng: Lệch chiều nào nhiều hơn thì là hướng đó
        if (avgDeltaX > avgDeltaY) {
            rel.direction = StitchDirection.HORIZONTAL;
            System.out.println("   -> Detect: Horizontal (Ngang)");

            // 2. Xác định Thứ tự Ngang
            // Quy tắc: Ảnh 1 (Trái) -> Ảnh 2 (Phải).
            // Điểm khớp: Mép Phải ảnh 1 (x lớn) nối với Mép Trái ảnh 2 (x nhỏ).
            // Nên đúng ra avgX1 phải > avgX2 (trong hệ tọa độ từng ảnh riêng biệt 0..W).
            // *Chỉnh lại logic trước đó một chút cho chính xác với hệ tọa độ ảnh*:
            // Nếu ta đặt 2 ảnh lên bàn: [Img1][Img2].
            // Vùng chồng lấn là Phải Img1 và Trái Img2.
            // Tọa độ x trên Img1 sẽ lớn (gần Width). Tọa độ x trên Img2 sẽ nhỏ (gần 0).
            // => Nếu avgX1 < avgX2 => Img1 đang là bên Phải => NGƯỢC.
            if (avgX1 < avgX2) {
                rel.needSwap = true;
            }
        } else {
            rel.direction = StitchDirection.VERTICAL;
            System.out.println("   -> Detect: Vertical (Dọc)");

            // 2. Xác định Thứ tự Dọc
            // Quy tắc: Ảnh 1 (Trên) -> Ảnh 2 (Dưới).
            // Vùng chồng lấn: Dưới Img1 (y lớn) và Trên Img2 (y nhỏ).
            // => Nếu avgY1 < avgY2 => Img1 đang là ở Dưới => NGƯỢC.
            if (avgY1 < avgY2) {
                rel.needSwap = true;
            }
        }
        return rel;
    }

    // --- HÀM GHÉP ROBUST (Cập nhật nhận SiftData và useWarp) ---
    private static Mat stitchRobust(Mat imgLeftRaw, Mat imgRightRaw, SiftData d1Input, SiftData d2Input, boolean useWarp) {
        Mat imgLeft, imgRight;

        // 1. Xử lý Warp (Uốn cong hoặc Không)
        if (useWarp) {
            System.out.println("   -> 1. Uốn cong ảnh (Cylindrical)...");
            CylindricalWarper warper = new CylindricalWarper();
            double f = imgLeftRaw.cols();
            imgLeft = warper.warp(imgLeftRaw, f);
            imgRight = warper.warp(imgRightRaw, f);

            // Debug warp
            imwrite(INPUT_PATH.resolve("debug_auto_warp_left.jpg").toString(), imgLeft);
            imwrite(INPUT_PATH.resolve("debug_auto_warp_right.jpg").toString(), imgRight);
        } else {
            System.out.println("   -> Bỏ qua Uốn cong - Warp (Planar Mode)...");
            imgLeft = imgLeftRaw.clone();
            imgRight = imgRightRaw.clone();
        }

        // 2. Matching
        // Nếu dùng Warp, tọa độ thay đổi -> BẮT BUỘC chạy lại SIFT
        // Nếu không Warp và có data đầu vào -> Dùng lại data cũ
        SiftData d1, d2;
        if (useWarp || d1Input == null) {
            System.out.println("   -> Tính toán SIFT lại (do Warp hoặc Rotate)...");
            SiftConfig.DOUBLE_IMAGE_SIZE = false;
            d1 = runSift(imgLeft);
            d2 = runSift(imgRight);
        } else {
            System.out.println("   -> Tái sử dụng dữ liệu SIFT cũ...");
            d1 = d1Input;
            d2 = d2Input;
        }

        FeatureMatcher matcher = new FeatureMatcher();
        Mat desc1 = matcher.convertDescriptorsToMat(d1.keypoints);
        Mat desc2 = matcher.convertDescriptorsToMat(d2.keypoints);
        FeatureMatcher.MatchResult res = matcher.matchFeatures(d1.keypoints, desc1, d2.keypoints, desc2);

        if (res == null || res.inlierMatches.isEmpty()) return null;

        // Debug Matches
        Mat debugMatch = new Mat();
        drawMatches(imgLeft, convertToOpenCVKeyPoints(d1.keypoints),
                imgRight, convertToOpenCVKeyPoints(d2.keypoints),
                new DMatchVector(res.inlierMatches.toArray(new DMatch[0])), debugMatch);
        imwrite(INPUT_PATH.resolve("debug_auto_matches.jpg").toString(), debugMatch);

        // --- LOGIC TÍNH TOÁN BOUNDING BOX ---
        System.out.println("   -> Tính toán Canvas & Blending...");

        // 3. Tính Homography (imgRight -> imgLeft)
        TransformEstimator estimator = new TransformEstimator();
        Mat H = estimator.computeH(res.dstPoints, res.srcPoints);

        // 4. Tính toán 4 góc của ảnh phải sau biến đổi
        Mat corners2 = new Mat(4, 1, CV_32FC2);
        FloatPointer ptr = new FloatPointer(corners2.data());
        float w2 = imgRight.cols();
        float h2 = imgRight.rows();

        ptr.put(0, 0); ptr.put(1, 0);
        ptr.put(2, w2); ptr.put(3, 0);
        ptr.put(4, w2); ptr.put(5, h2);
        ptr.put(6, 0); ptr.put(7, h2);

        Mat transformedCorners2 = new Mat();
        perspectiveTransform(corners2, transformedCorners2, H);

        FloatPointer resPtr = new FloatPointer(transformedCorners2.data());
        float minX = 0, minY = 0, maxX = imgLeft.cols(), maxY = imgLeft.rows();

        for(int i=0; i<4; i++) {
            float x = resPtr.get(2*i);
            float y = resPtr.get(2*i+1);
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        int canvasW = (int) Math.round(maxX - minX);
        int canvasH = (int) Math.round(maxY - minY);

        // 5. Ma trận dịch chuyển T
        Mat T = Mat.eye(3, 3, CV_64F).asMat();
        if (minX < 0) new FloatPointer(T.ptr(0, 2)).put(-minX);
        if (minY < 0) new FloatPointer(T.ptr(1, 2)).put(-minY);

        // H_final = T * H
        Mat H_final = new Mat();
        gemm(T, H, 1.0, new Mat(), 0.0, H_final);

        // 6. Warp & Blending
        Mat result = new Mat();
        Size finalSize = new Size(canvasW, canvasH);

        warpPerspective(imgRight, result, H_final, finalSize);

        Mat imgLeftWarped = new Mat();
        warpPerspective(imgLeft, imgLeftWarped, T, finalSize);

        Mat mask1 = new Mat();
        cvtColor(imgLeftWarped, mask1, COLOR_BGR2GRAY);
        threshold(mask1, mask1, 1, 255, THRESH_BINARY);

        imgLeftWarped.copyTo(result, mask1);

        return cropBlackBorder(result);
    }

    // --- HELPERS (GIỮ NGUYÊN) ---
    static class SiftData { List<SiftKeyPoint> keypoints; }

    private static SiftData runSift(Mat img) {
        Mat gray = new Mat();
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat floatGray = new Mat();
        gray.convertTo(floatGray, CV_32F, 1.0/255.0, 0.0);
        ScaleSpace ss = new ScaleSpace();
        SiftDetector det = new SiftDetector();
        SiftData d = new SiftData();
        d.keypoints = det.run(ss.buildGaussianPyramid(floatGray), ss.buildDoGPyramid(ss.buildGaussianPyramid(floatGray)));
        return d;
    }

    private static Mat cropBlackBorder(Mat img) {
        Mat gray = new Mat();
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat points = new Mat();
        findNonZero(gray, points);
        if(points.empty()) return img;
        Rect bb = boundingRect(points);
        return new Mat(img, bb);
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