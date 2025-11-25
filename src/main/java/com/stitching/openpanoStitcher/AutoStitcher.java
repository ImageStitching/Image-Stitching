package com.stitching.openpanoStitcher;

import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftConfig;
import com.stitching.openpanoSIFT.SiftDetector;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_features2d.*;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class AutoStitcher {
    private static final Path INPUT_PATH = Paths.get("src","main","resources","static","one_scene_crossline_rotation");
    enum StitchDirection { HORIZONTAL, VERTICAL, DIAGONAL }
    enum CaptureMode {
        ROTATION,       // Đứng 1 chỗ xoay (Pano chuẩn) -> Cần Warp Trụ
        TRANSLATION,    // Trượt ngang/dọc/chéo (Scan phẳng) -> Cần Planar (Không Warp)
        ORBIT_INWARD    // Đi theo cung tròn chụp vào tâm -> Cần Planar (Không Warp)
    }

    static class ImageRelation {
        StitchDirection direction;
        boolean needSwap; // True nếu thứ tự ảnh bị ngược
        String debugMsg;  // Tin nhắn debug để biết code đang nghĩ gì
    }

    // Hàm này chỉ giữ lại tối đa 'maxPoints' điểm đầu tiên (thường là điểm mạnh nhất)
    private static void limitKeypoints(SiftData data, int maxPoints) {
        if (data == null || data.keypoints == null) return;

        int currentSize = data.keypoints.size();
        if (currentSize <= maxPoints) {
            System.out.println("   -> Số lượng điểm (" + currentSize + ") đã tối ưu, không cần cắt giảm.");
            return;
        }

        // Cắt danh sách, chỉ lấy sublist từ 0 đến maxPoint OpenCV SIFT thường trả về keypoint theo thứ tự từ mạnh đến yếu sẵn rồi, nên ta chỉ cần cắt đuôi là được.
        List<SiftKeyPoint> limitedList = new ArrayList<>(data.keypoints.subList(0, maxPoints));

        data.keypoints.clear();
        data.keypoints.addAll(limitedList);

        System.out.println("      -> Đã giảm số lượng điểm từ " + currentSize + " xuống " + maxPoints + " để giảm tải CPU.");
    }

    public static void main(String[] args) {
        // [CODE MỚI] THAM SỐ ĐẦU VÀO QUAN TRỌNG
        // Bạn đổi giá trị này tùy theo cách bạn chụp ảnh:
        // - CaptureMode.ROTATION: Nếu đứng yên xoay máy (Mặc định Pano).
        // - CaptureMode.TRANSLATION: Nếu di chuyển tịnh tiến (Quét mặt bàn, quét tường).
        // - CaptureMode.ORBIT_INWARD: Nếu đi vòng quanh vật thể.
        CaptureMode currentMode = CaptureMode.ROTATION;

        //String img_src_1 = "img13", img_src_2 = "img14";
        String img_src_1 = "result_img12_img13", img_src_2 = "result_img13_img14";
        String destination_img_name = "result_" + img_src_1 + "_" + img_src_2;

        String path1 = INPUT_PATH.resolve(img_src_1 + ".jpg").toString();
        String path2 = INPUT_PATH.resolve(img_src_2 + ".jpg").toString();

        Mat img1 = imread(path1);
        Mat img2 = imread(path2);

        if (img1.empty() || img2.empty()) {
            System.err.println("Không đọc được ảnh!");
            return;
        }

        System.out.println("=== BẮT ĐẦU TỰ ĐỘNG GHÉP ẢNH (AUTO-ORDER & OMNI-DIRECTION) ===");

        // BƯỚC 1: CHẠY SIFT SƠ BỘ TRÊN ẢNH GỐC
        System.out.println("1. Phân tích đặc trưng ảnh gốc...");
        SiftConfig.DOUBLE_IMAGE_SIZE = false;
        SiftData d1 = runSift(img1);
        SiftData d2 = runSift(img2);

        // GIOI HAN SO KEYPOINT DE GIAM CPU LUC MATCHING
        limitKeypoints(d1, 4000);
        limitKeypoints(d2, 4000);

        FeatureMatcher matcher = new FeatureMatcher();
        Mat desc1 = matcher.convertDescriptorsToMat(d1.keypoints);
        Mat desc2 = matcher.convertDescriptorsToMat(d2.keypoints);
        FeatureMatcher.MatchResult res = matcher.matchFeatures(d1.keypoints, desc1, d2.keypoints, desc2);

        if (res == null || res.inlierMatches.isEmpty()) {
            System.err.println("Không tìm thấy điểm chung giữa 2 ảnh! Không thể ghép.");
            return;
        }

        // BƯỚC 2: PHÂN TÍCH HƯỚNG VÀ THỨ TỰ
        // [CODE MỚI] Hàm này đã được nâng cấp để nhận diện chéo
        ImageRelation relation = analyzeRelation(res.inlierMatches, d1, d2);
        System.out.println(">> KẾT QUẢ PHÂN TÍCH: " + relation.debugMsg);

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

        // [CODE MỚI] QUYẾT ĐỊNH CHIẾN LƯỢC GHÉP DỰA TRÊN MODE VÀ DIRECTION
        // Logic:
        // - Nếu là ROTATION: Luôn ưu tiên Warp Trụ (cho Ngang) hoặc CenterProjection (cho Chéo).
        // - Nếu là TRANSLATION / ORBIT: Luôn ưu tiên Planar (Tắt Warp) để tránh méo vật thể phẳng.
        boolean useWarp = (currentMode == CaptureMode.ROTATION);

        // BƯỚC 4: XỬ LÝ GHÉP DỰA TRÊN HƯỚNG
        switch (relation.direction) {
            case HORIZONTAL:
                System.out.println(">> CHẾ ĐỘ: GHÉP NGANG (Warp Trụ)");
                // Nếu Rotation -> Warp Trụ Ngang.
                // Nếu Translation -> Planar Ngang.
                if (useWarp) {
                    // Với góc xoay ngang lớn, dùng Center để chia đều độ méo
                    finalResult = stitchCenter(finalImg1, finalImg2, finalD1, finalD2);
                } else {
                    // Trượt ngang hoặc vòng cung nhìn vào trong thì dùng Robust (Planar)
                    finalResult = stitchRobust(finalImg1, finalImg2, finalD1, finalD2, false);
                }
                break;

            case VERTICAL:
                System.out.println(">> CHẾ ĐỘ: GHÉP DỌC (Center Projection)");
                // Dọc dùng Center Stitcher để tránh img2 bị kéo dài quá mức
                // Xoay ảnh sang ngang để dễ tính toán SIFT/Match (tùy chọn, nhưng giữ nguyên logic cũ cho an toàn)
                Mat r1 = ImageUtils.rotateLeft(finalImg1);
                Mat r2 = ImageUtils.rotateLeft(finalImg2);

                Mat stitchedV;

                if (useWarp) {
                    // [CẬP NHẬT] Nếu là chụp Xoay/Nghiêng (Tilt) -> Dùng stitchCenter
                    // Để chia đều độ méo phối cảnh (Keystone effect)
                    stitchedV = stitchCenter(r1, r2, null, null);
                } else {
                    // Nếu là scan phẳng tịnh tiến hoặc vòng cung nhìn vào trong -> Dùng Robust tắt warp
                    stitchedV = stitchRobust(r1, r2, null, null, false);
                }
                if (stitchedV != null) {
                    finalResult = ImageUtils.rotateRight(stitchedV);
                }
                break;

            case DIAGONAL:
                System.out.println(">> CHẾ ĐỘ: GHÉP CHÉO (Center Projection)");
                // Dùng stitchCenter để chia đôi độ méo cho cả 2 ảnh
                if (useWarp) {
                    // Nếu xoay chéo (Tilt + Pan): Dùng Center Projection để giảm méo phối cảnh
                    finalResult = stitchCenter(finalImg1, finalImg2, finalD1, finalD2);
                } else {
                    // Nếu trượt chéo (Scan chéo) hoặc cunxng vòng cung chéo nhìn vào trong: Dùng Planar (Tắt Warp)
                    finalResult = stitchRobust(finalImg1, finalImg2, finalD1, finalD2, false);
                }
                break;
        }

        if (finalResult != null) {
            String outPath = INPUT_PATH.resolve(destination_img_name+".jpg").toString();
            imwrite(outPath, finalResult);
            System.out.println("=== HOÀN TẤT! Kết quả: " + outPath + " ===");
        } else {
            System.err.println("Ghép thất bại.");
        }
    }

    // --- [CODE MỚI] LOGIC PHÂN TÍCH QUAN HỆ (HỖ TRỢ NGANG, DỌC, CHÉO, AUTO-SWAP) ---
    private static ImageRelation analyzeRelation(List<DMatch> matches, SiftData d1, SiftData d2) {
        ImageRelation rel = new ImageRelation();

        // Tính vector dịch chuyển trung bình từ Ảnh 1 -> Ảnh 2
        // Vector V = P1 - P2
        double sumVx = 0, sumVy = 0;
        double sumAbsVx = 0, sumAbsVy = 0;

        int count = Math.min(matches.size(), 100); // Lấy mẫu 100 điểm

        for(int i=0; i<count; i++) {
            DMatch m = matches.get(i);
            SiftKeyPoint p1 = d1.keypoints.get(m.queryIdx());
            SiftKeyPoint p2 = d2.keypoints.get(m.trainIdx());

            double vx = p1.x - p2.x;
            double vy = p1.y - p2.y;

            sumVx += vx;
            sumVy += vy;
            sumAbsVx += Math.abs(vx);
            sumAbsVy += Math.abs(vy);
        }

        double avgVx = sumVx / count;       // Vector có hướng (quan trọng để xác định thứ tự)
        double avgVy = sumVy / count;
        double avgAbsVx = sumAbsVx / count; // Độ lớn tuyệt đối (quan trọng để xác định hướng)
        double avgAbsVy = sumAbsVy / count;

        // [CODE MỚI CHO CHÉO]
        // Tính tỷ lệ giữa chiều nhỏ và chiều lớn.
        // Nếu tỷ lệ > 0.35 (tức góc di chuyển khoảng > 19 độ so với trục chính), ta coi là CHÉO.
        double ratio = Math.min(avgAbsVx, avgAbsVy) / Math.max(avgAbsVx, avgAbsVy);
        // [FIX]: Tăng ngưỡng từ 0.35 lên 0.55
        // Ý nghĩa: Cạnh nhỏ phải bằng hơn một nửa cạnh lớn mới tính là Chéo.
        // Nếu nhỏ hơn, coi như là rung tay hoặc nhiễu -> ép về Ngang/Dọc.
        boolean isDiagonal = ratio > 0.5;

        // === LOGIC XÁC ĐỊNH HƯỚNG VÀ SWAP ===
        // Nguyên tắc Swap: Ta luôn muốn Img1 nằm ở vị trí "Gốc" (Top-Left).
        // Vector V = P1(trên Img1) - P2(trên Img2).
        // Vì điểm khớp nằm ở biên Phải/Dưới của Img1 và biên Trái/Trên của Img2:
        // -> Nếu Img1 đứng trước: x1 lớn, x2 nhỏ -> Vx > 0.
        // -> Nếu Img1 đứng trên:  y1 lớn, y2 nhỏ -> Vy > 0.
        // KẾT LUẬN: Nếu Vector DƯƠNG -> Đúng thứ tự. Nếu Vector ÂM -> Ngược -> Swap.

        if (isDiagonal) {
            rel.direction = StitchDirection.DIAGONAL;
            rel.debugMsg = String.format("Chéo (Diagonal). Tỷ lệ: %.2f", ratio);

            // Với ảnh chéo, ta kiểm tra tổng vector.
            // Nếu tổng < 0, nghĩa là Img1 đang nằm lệch về phía Phải-Dưới so với Img2 -> Swap
            if ((avgVx + avgVy) < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> Swap (Img1 đang ở góc Dưới-Phải)";
            }
        }
        else if (avgAbsVx > avgAbsVy) {
            rel.direction = StitchDirection.HORIZONTAL;
            rel.debugMsg = "Ngang (Horizontal)";
            // Nếu Vx < 0 -> Img1 đang nằm bên Phải -> Swap
            if (avgVx < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> Swap (Phải->Trái)";
            }
        }
        else {
            rel.direction = StitchDirection.VERTICAL;
            rel.debugMsg = "Dọc (Vertical)";
            // Nếu Vy < 0 -> Img1 đang nằm bên Dưới -> Swap
            if (avgVy < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> Swap (Dưới->Trên)";
            }
        }

        return rel;
    }

    // --- HÀM GHÉP ROBUST (GIỮ NGUYÊN CODE CŨ) ---
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
        try {
            Mat debugMatch = new Mat();
            drawMatches(imgLeft, convertToOpenCVKeyPoints(d1.keypoints),
                    imgRight, convertToOpenCVKeyPoints(d2.keypoints),
                    new DMatchVector(res.inlierMatches.toArray(new DMatch[0])), debugMatch);
            imwrite(INPUT_PATH.resolve("debug_auto_matches.jpg").toString(), debugMatch);
            debugMatch.release();
        } catch (Exception e) {
            System.err.println("      [WARNING] Lỗi vẽ ảnh debug (bỏ qua): " + e.getMessage());
        }

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

    // --- HÀM GHÉP TRUNG TÂM (Dùng cho DỌC và CHÉO) ---
    private static Mat stitchCenter(Mat img1, Mat img2, SiftData d1Input, SiftData d2Input) {
        System.out.println("   -> Chế độ Center Projection (Giảm méo)...");

        System.out.println("      -> SIFT lại");
        // 1. Matching (Nếu chưa có data thì chạy lại)
        SiftData d1 = (d1Input != null) ? d1Input : runSift(img1);
        SiftData d2 = (d2Input != null) ? d2Input : runSift(img2);

        limitKeypoints(d1, 2000);
        limitKeypoints(d2, 2000);

        System.out.println("      -> Matching lại");
        FeatureMatcher matcher = new FeatureMatcher();
        Mat desc1 = matcher.convertDescriptorsToMat(d1.keypoints);
        Mat desc2 = matcher.convertDescriptorsToMat(d2.keypoints);
        FeatureMatcher.MatchResult res = matcher.matchFeatures(d1.keypoints, desc1, d2.keypoints, desc2);

        if (res == null || res.inlierMatches.isEmpty()) return null;

        System.out.println("      -> Vẽ ảnh debug matching");
        // [FIX] Vẽ Debug an toàn (Bọc try-catch để không chết chương trình nếu lỗi vẽ)
        try {
            // Kiểm tra kích thước trước khi vẽ
            if (d1.keypoints.size() > 0 && d2.keypoints.size() > 0 && res.inlierMatches.size() > 0) {
                Mat debugMatch = new Mat();
                // Tạo vector cục bộ để đảm bảo vòng đời
                KeyPointVector kpv1 = convertToOpenCVKeyPoints(d1.keypoints);
                KeyPointVector kpv2 = convertToOpenCVKeyPoints(d2.keypoints);
                DMatchVector dmv = new DMatchVector(res.inlierMatches.toArray(new DMatch[0]));

                drawMatches(img1, kpv1, img2, kpv2, dmv, debugMatch);
                imwrite(INPUT_PATH.resolve("debug_center_matches.jpg").toString(), debugMatch);
            }
        } catch (Exception e) {
            System.err.println("   [WARNING] Không thể vẽ ảnh debug matches (Lỗi Index/Memory), nhưng vẫn tiếp tục ghép.");
            // e.printStackTrace(); // Bỏ qua lỗi để chạy tiếp
        }

        System.out.println("      -> Tính H, Homo");
        // 2. Tính H từ img2 -> img1
        TransformEstimator estimator = new TransformEstimator();
        Mat H = estimator.computeH(res.dstPoints, res.srcPoints);

        // 3. Tính Homography trung bình (H_half) để chia đôi độ méo
        Mat I = Mat.eye(3, 3, CV_64F).asMat();
        Mat H_diff = new Mat();
        subtract(H, I, H_diff);

        Mat H_half = new Mat();
        // Xấp xỉ: H_half = I + 0.5 * (H - I)
        addWeighted(I, 1.0, H_diff, 0.5, 0.0, H_half);

        // Sửa lỗi MatExpr: Dùng hàm invert() hoặc .asMat()
        Mat H_half_inv = new Mat();
        invert(H_half, H_half_inv); // Nghịch đảo ma trận

        System.out.println("      -> Tính Box");
        // 4. Tính Bounding Box chung
        Rect bb1 = computeBoundingBox(img1, H_half_inv);
        Rect bb2 = computeBoundingBox(img2, H_half);
        Rect finalBB = unionRect(bb1, bb2);

        // 5. Ma trận dịch chuyển về dương
        Mat T = Mat.eye(3, 3, CV_64F).asMat();
        new FloatPointer(T.ptr(0, 2)).put(-finalBB.x());
        new FloatPointer(T.ptr(1, 2)).put(-finalBB.y());

        Mat H1_final = new Mat(), H2_final = new Mat();
        gemm(T, H_half_inv, 1.0, new Mat(), 0.0, H1_final);
        gemm(T, H_half, 1.0, new Mat(), 0.0, H2_final);

        System.out.println("      -> Warp và Blend");
        // 6. Warp & Blend
        Mat result = new Mat();
        Size size = new Size(finalBB.width(), finalBB.height());

        Mat warped1 = new Mat();
        Mat warped2 = new Mat();

        // Warp cả 2 ảnh về không gian chung
        warpPerspective(img1, warped1, H1_final, size);
        warpPerspective(img2, warped2, H2_final, size);

        result = smartBlend(warped1, warped2);

        return cropBlackBorder(result);
    }

    // --- CÁC HÀM HELPER BẮT BUỘC CHO stitchCenter ---
    private static Rect computeBoundingBox(Mat img, Mat H) {
        Mat corners = new Mat(4, 1, CV_32FC2);
        FloatPointer ptr = new FloatPointer(corners.data());
        float w = img.cols(); float h = img.rows();
        ptr.put(0, 0); ptr.put(1, 0); ptr.put(2, w); ptr.put(3, 0);
        ptr.put(4, w); ptr.put(5, h); ptr.put(6, 0); ptr.put(7, h);

        Mat tCorners = new Mat();
        perspectiveTransform(corners, tCorners, H);
        FloatPointer res = new FloatPointer(tCorners.data());

        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;

        for(int i=0; i<4; i++) {
            float x = res.get(2*i); float y = res.get(2*i+1);
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }
        return new Rect((int)minX, (int)minY, (int)(maxX - minX), (int)(maxY - minY));
    }

    private static Rect unionRect(Rect r1, Rect r2) {
        int x = Math.min(r1.x(), r2.x());
        int y = Math.min(r1.y(), r2.y());
        int w = Math.max(r1.x() + r1.width(), r2.x() + r2.width()) - x;
        int h = Math.max(r1.y() + r1.height(), r2.y() + r2.height()) - y;
        return new Rect(x, y, w, h);
    }

    // --- HÀM PHA TRỘN MỚI: SEAM-CUTTING & FEATHERING ---
    private static Mat smartBlend(Mat img1, Mat img2) {
        // 1. Tạo Mask (8-bit)
        Mat gray1 = new Mat(), mask1 = new Mat();
        cvtColor(img1, gray1, COLOR_BGR2GRAY);
        threshold(gray1, mask1, 1, 255, THRESH_BINARY);
        gray1.release();

        Mat gray2 = new Mat(), mask2 = new Mat();
        cvtColor(img2, gray2, COLOR_BGR2GRAY);
        threshold(gray2, mask2, 1, 255, THRESH_BINARY);
        gray2.release();

        // 2. Distance Transform
        Mat dist1 = new Mat(), dist2 = new Mat();
        distanceTransform(mask1, dist1, DIST_L2, 3);
        distanceTransform(mask2, dist2, DIST_L2, 3);

        // 3. Tạo Seam Mask (8-bit)
        Mat seamMask = new Mat();
        compare(dist1, dist2, seamMask, CMP_GT);

        // 4. Tạo Alpha 3 kênh (Float)
        Mat seamMask3c = new Mat();
        cvtColor(seamMask, seamMask3c, COLOR_GRAY2BGR);

        Mat alpha3c = new Mat();
        seamMask3c.convertTo(alpha3c, CV_32F, 1.0/255.0, 0.0);

        GaussianBlur(alpha3c, alpha3c, new Size(45, 45), 0);
        //boxFilter(alpha3c, alpha3c, -1, new Size(45, 45));

        // 5. Chuẩn bị ảnh Float
        Mat img1F = new Mat(), img2F = new Mat();
        img1.convertTo(img1F, CV_32F);
        img2.convertTo(img2F, CV_32F);

        // --- [FIX LỖI MÀU Ở ĐÂY] ---
        // Tạo ma trận toàn số 1.0 trên cả 3 kênh
        Mat ones = new Mat(alpha3c.size(), CV_32FC3, new Scalar(1.0, 1.0, 1.0, 0.0));
        Mat beta3c = new Mat();
        subtract(ones, alpha3c, beta3c); // Beta = 1.0 - Alpha
        // ---------------------------

        // 6. Nhân chập
        Mat part1 = new Mat(), part2 = new Mat();
        multiply(img1F, alpha3c, part1);
        multiply(img2F, beta3c, part2);

        Mat resultF = new Mat();
        add(part1, part2, resultF);

        // 7. Convert về 8-bit
        Mat result = new Mat();
        resultF.convertTo(result, CV_8U);

        mask1.release(); mask2.release(); dist1.release(); dist2.release();
        return result;
    }

    // --- HELPERS (GIỮ NGUYÊN) ---
    static class SiftData { public List<SiftKeyPoint> keypoints; }

    static SiftData runSift(Mat img) {
        Mat gray = new Mat();
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat floatGray = new Mat();
        gray.convertTo(floatGray, CV_32F, 1.0/255.0, 0.0);
        gray.release();

        ScaleSpace ss = new ScaleSpace();
        SiftDetector det = new SiftDetector();
        SiftData d = new SiftData();
        List<MatVector> gaussianPyramid = ss.buildGaussianPyramid(floatGray);
        List<MatVector> dogPyramid = ss.buildDoGPyramid(gaussianPyramid);
        d.keypoints = det.run(gaussianPyramid, dogPyramid);
        floatGray.release();
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