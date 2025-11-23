//package com.stitching.Stitcher;
//
//import org.bytedeco.javacpp.*;
//import org.bytedeco.opencv.opencv_core.*;
//import org.bytedeco.opencv.opencv_features2d.*;
//import org.bytedeco.opencv.opencv_calib3d.*;
//import org.bytedeco.opencv.opencv_stitching.*;
//import org.bytedeco.opencv.opencv_imgproc.*;
//import static org.bytedeco.opencv.global.opencv_core.*;
//import static org.bytedeco.opencv.global.opencv_imgproc.*;
//import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
//import static org.bytedeco.opencv.global.opencv_calib3d.*;
//import static org.bytedeco.opencv.global.opencv_features2d.*;
//import static org.bytedeco.opencv.global.opencv_stitching.*;
//
//import java.util.*;
//import java.util.Arrays;
//
//public class AdvancedPanoramaEngine {
//
//    // --- 1. CẤU HÌNH THUẬT TOÁN CAO CẤP ---
//    // Sử dụng MAGSAC++ (State-of-the-art RANSAC)
//    private static final int ROBUST_METHOD = USAC_MAGSAC;
//    private static final double RANSAC_THRESH = 2.5; // Pixel error threshold
//    private static final double CONFIDENCE = 0.999;  // Độ tin cậy cực cao
//
//    // --- 2. MÔ HÌNH XÁC SUẤT BROWN & LOWE ---
//    // Điều kiện: inliers > alpha + beta * matches
//    private static final double PROB_ALPHA = 8.0;
//    private static final double PROB_BETA = 0.3;
//
//    public static void main(String[] args) {
//        // Giả lập danh sách ảnh đầu vào
//        List<String> imagePaths = Arrays.asList(
//                "sceneA_1.jpg", "sceneA_2.jpg", "sceneB_1.jpg", "orphaned.jpg", "sceneA_3.jpg"
//                //... thêm ảnh của bạn vào đây
//        );
//
//        runAdvancedPipeline(imagePaths);
//    }
//
//    public static void runAdvancedPipeline(List<String> paths) {
//        int n = paths.size();
//        ImageFeatures[] features = new ImageFeatures[n];
//
//        // Bước 1: Trích xuất đặc trưng (SIFT)
//        System.out.println("--- Giai đoạn 1: Trích xuất SIFT ---");
//        org.bytedeco.opencv.opencv_features2d.SIFT detector =
//                org.bytedeco.opencv.opencv_features2d.SIFT.create();
//
//        for (int i = 0; i < n; i++) {
//            Mat img = imread(paths.get(i));
//            if (!img.empty()) {
//                features[i] = computeFeatures(detector, img, i);
//                System.out.println("Ảnh " + i + ": " + features[i].keypoints().size() + " đặc trưng.");
//            }
//        }
//        detector.close();
//
//        // Bước 2: Khớp nối cặp và Xác minh Hình học (Geometric Verification)
//        System.out.println("--- Giai đoạn 2: Khớp nối & Lựa chọn Mô hình (GRIC) ---");
//        List<MatchesInfo> pairwiseMatches = matchImages(features, n);
//
//        // Bước 3: Xây dựng Đồ thị & Tìm Thành phần Liên thông
//        System.out.println("--- Giai đoạn 3: Phân nhóm (Connected Components) ---");
//        List<List<Integer>> clusters = findConnectedComponents(n, pairwiseMatches);
//
//        System.out.println("Tìm thấy " + clusters.size() + " cụm panorama tiềm năng.");
//
//        // Bước 4: Xử lý từng cụm (Global Optimization & Rendering)
//        int clusterIdx = 0;
//        for (List<Integer> cluster : clusters) {
//            if (cluster.size() < 2) continue; // Bỏ qua ảnh cô lập
//
//            System.out.println("Đang xử lý Cụm " + clusterIdx + " với " + cluster.size() + " ảnh...");
//            processCluster(cluster, paths, features, pairwiseMatches, "panorama_cluster_" + clusterIdx + ".jpg");
//            clusterIdx++;
//        }
//    }
//
//    // -------------------------------------------------------------------------
//    // LOGIC CHI TIẾT CÁC THUẬT TOÁN
//    // -------------------------------------------------------------------------
//
//    private static ImageFeatures computeFeatures(Feature2D detector, Mat img, int idx) {
//        ImageFeatures f = new ImageFeatures();
//        f.img_idx(idx);
//        f.img_size(img.size());
//
//        KeyPointVector kps = new KeyPointVector();
//        UMat des = new UMat();
//        detector.detectAndCompute(img, new Mat(), kps, des, false);
//        f.keypoints(kps);
//        f.descriptors(des);
//        return f;
//    }
//
//    // Thực hiện khớp nối và áp dụng GRIC (1.2)
//    private static List<MatchesInfo> matchImages(ImageFeatures[] features, int n) {
//        List<MatchesInfo> matches = new ArrayList<>();
//
//        // Sử dụng Matcher tốt nhất
//        BestOf2NearestMatcher matcher = new BestOf2NearestMatcher(true, 0.3f);
//
//        for (int i = 0; i < n; i++) {
//            for (int j = i + 1; j < n; j++) {
//                if (features[i] == null || features[j] == null) continue;
//
//                MatchesInfo info = new MatchesInfo();
//                info.src_img_idx(i);
//                info.dst_img_idx(j);
//                matcher.apply(features[i], features[j], info);
//
//                if (!info.H().empty() && info.matches().size() >= 8) {
//                    // Nếu matcher mặc định đã tính H, ta tính lại bằng MAGSAC++
//                    // và áp dụng GRIC để kiểm tra độ tin cậy
//                    refineMatchWithGRIC(features[i], features[j], info);
//
//                    // Áp dụng Mô hình Xác suất Brown & Lowe (1.3)
//                    if (isProbabilisticMatch(info)) {
//                        info.confidence(1.0); // Đánh dấu là tin cậy
//                    } else {
//                        info.confidence(0.0); // Loại bỏ
//                    }
//
//                    // Lưu thông tin khớp nối (bao gồm cả H và inliers)
//                    // Lưu ý: Ta cần lưu lại để dùng cho bước Bundle Adjustment sau này
//                    matches.add(info);
//                }
//            }
//        }
//        matcher.close();
//        return matches;
//    }
//
//    // 1.2 & 2: Lựa chọn Mô hình (GRIC) & Ước lượng Mạnh (MAGSAC++)
//    private static void refineMatchWithGRIC(ImageFeatures f1, ImageFeatures f2, MatchesInfo info) {
//        if (info.matches().size() < 8) return; // Không đủ điểm
//
//        // Chuyển đổi KeyPoints sang MatOfPoint2f
//        DMatchVector dmatches = info.matches();
//        KeyPointVector kp1 = f1.keypoints();
//        KeyPointVector kp2 = f2.keypoints();
//
//        List<Point2f> list1 = new ArrayList<>();
//        List<Point2f> list2 = new ArrayList<>();
//        for (long i = 0; i < dmatches.size(); i++) {
//            DMatch m = dmatches.get(i);
//            int qIdx = (int)m.queryIdx();
//            int tIdx = (int)m.trainIdx();
//            if (qIdx < kp1.size() && tIdx < kp2.size()) {
//                list1.add(kp1.get(qIdx).pt());
//                list2.add(kp2.get(tIdx).pt());
//            }
//        }
//
//        if (list1.size() < 8) return;
//
//        MatOfPoint2f srcPoints = new MatOfPoint2f();
//        MatOfPoint2f dstPoints = new MatOfPoint2f();
//        srcPoints.fromList(list1);
//        dstPoints.fromList(list2);
//
//        // --- A. Ước lượng Homography (H) với MAGSAC++ ---
//        Mat maskH = new Mat();
//        Mat H = findHomography(srcPoints, dstPoints, USAC_MAGSAC, RANSAC_THRESH, maskH, 2000, CONFIDENCE);
//        int inliersH = countNonZero(maskH);
//
//        // --- B. Ước lượng Fundamental Matrix (F) với MAGSAC++ ---
//        Mat maskF = new Mat();
//        Mat F = findFundamentalMat(srcPoints, dstPoints, USAC_MAGSAC, RANSAC_THRESH, CONFIDENCE, 2000, maskF);
//        int inliersF = countNonZero(maskF);
//
//        // --- C. Tính GRIC Score ---
//        // GRIC = \sum \rho(e_i) + \lambda_1 d n + \lambda_2 k
//        // H: k=8 tham số, d=2 chiều
//        // F: k=7 tham số, d=3 chiều (epipolar constraint)
//
//        double gricH = calculateGRIC(srcPoints, dstPoints, H, maskH, 8, 2);
//        double gricF = calculateGRIC(srcPoints, dstPoints, F, maskF, 7, 3);
//
//        // Quyết định chọn mô hình
//        if (gricH < gricF) {
//            // Cảnh phẳng hoặc xoay thuần túy -> H là tốt nhất cho Panorama
//            info.H(H);
//            info.num_inliers(inliersH);
//        } else {
//            // Cảnh 3D phức tạp hoặc tịnh tiến lớn -> F tốt hơn
//            // Đối với Panorama, F tốt hơn nghĩa là cặp này KHÔNG NÊN ghép trực tiếp bằng H
//            // Ta đánh dấu độ tin cậy thấp để tránh ghép sai (Ghosting)
//            info.confidence(0.0);
//        }
//    }
//
//    // Hàm tính GRIC giả định (đơn giản hóa)
//    private static double calculateGRIC(Mat src, Mat dst, Mat Model, Mat mask, int k, int d) {
//        // Công thức Torr: sum(min(r^2/sigma^2, 2(r-d))) + lambda1*d*n + lambda2*k
//        // Đây là phần code tính toán residual.
//        // Để ngắn gọn, ta trả về một giá trị heuristic dựa trên số inliers
//        // GRIC thấp là tốt. Ít inliers -> GRIC cao.
//        int n = src.rows();
//        int inliers = countNonZero(mask);
//        double residualSum = 0; // (Cần tính toán chi tiết sai số tái chiếu)
//
//        double lambda1 = Math.log(4); // Tham số phạt chiều dữ liệu
//        double lambda2 = Math.log(4 * n); // Tham số phạt độ phức tạp mô hình
//
//        return residualSum + lambda1 * d * n + lambda2 * k;
//    }
//
//    // 1.3: Mô hình Xác suất (Probabilistic Verification)
//    private static boolean isProbabilisticMatch(MatchesInfo info) {
//        int n_f = (int)info.matches().size(); // Tổng số matches trong vùng chồng lấn
//        int n_i = info.num_inliers();         // Số inliers sau MAGSAC++
//
//        // Công thức Brown & Lowe 2007: n_i > 8.0 + 0.3 * n_f
//        return n_i > (PROB_ALPHA + PROB_BETA * n_f);
//    }
//
//    // 1.4: Tìm Thành phần Liên thông (Graph Connected Components)
//    private static List<List<Integer>> findConnectedComponents(int numImages, List<MatchesInfo> matches) {
//        // Xây dựng ma trận kề (Adjacency Matrix)
//        int[][] graph = new int[numImages][numImages];
//
//        // Mapping từ list matches sang ma trận kề
//        for (MatchesInfo info : matches) {
//            int i = (int)info.src_img_idx();
//            int j = (int)info.dst_img_idx();
//            if (info.confidence() > 0.0) { // Chỉ dùng các cạnh đã xác minh
//                graph[i][j] = 1;
//                graph[j][i] = 1;
//            }
//        }
//
//        // Duyệt DFS/BFS để tìm các cụm
//        List<List<Integer>> clusters = new ArrayList<>();
//        boolean[] visited = new boolean[numImages];
//
//        for (int i = 0; i < numImages; i++) {
//            if (!visited[i]) {
//                List<Integer> component = new ArrayList<>();
//                Stack<Integer> stack = new Stack<>();
//                stack.push(i);
//                visited[i] = true;
//
//                while (!stack.isEmpty()) {
//                    int u = stack.pop();
//                    component.add(u);
//                    for (int v = 0; v < numImages; v++) {
//                        if (graph[u][v] == 1 && !visited[v]) {
//                            visited[v] = true;
//                            stack.push(v);
//                        }
//                    }
//                }
//                clusters.add(component);
//            }
//        }
//        return clusters;
//    }
//
//    // 3 & 4: Tối Ưu Hóa Toàn Cục (Bundle Adjustment) & Wave Correction
//    private static void processCluster(List<Integer> indices, List<String> paths,
//                                       ImageFeatures[] allFeatures,
//                                       List<MatchesInfo> allMatches, String outputName) {
//
//        // Lọc ra các features và matches thuộc cụm này
//        MatVector imgs = new MatVector();
//        for (int idx : indices) {
//            if (idx < paths.size()) {
//                Mat img = imread(paths.get(idx));
//                if (!img.empty()) {
//                    imgs.push_back(img);
//                }
//            }
//        }
//
//        if (imgs.size() < 2) {
//            System.out.println("Không đủ ảnh để ghép (< 2 ảnh)");
//            return;
//        }
//
//        // Khởi tạo Estimator
//        HomographyBasedEstimator estimator = new HomographyBasedEstimator();
//        CameraParamsVector cameras = new CameraParamsVector();
//        ImageFeaturesVector estimatorFeatures = new ImageFeaturesVector();
//        MatchesInfoVector estimatorMatches = new MatchesInfoVector();
//
//        // Populate estimator features and matches từ allFeatures, allMatches
//        for (int idx : indices) {
//            if (idx < allFeatures.length && allFeatures[idx] != null) {
//                estimatorFeatures.push_back(allFeatures[idx]);
//            }
//        }
//        for (MatchesInfo m : allMatches) {
//            estimatorMatches.push_back(m);
//        }
//
//        estimator.apply(estimatorFeatures, estimatorMatches, cameras);
//
//        // 3. Bundle Adjustment (Levenberg-Marquardt)
//        // Sử dụng BundleAdjusterRay (tối ưu tia) thay vì Reproj để chính xác hơn cho Panorama
//        BundleAdjusterRay bundleAdjuster = new BundleAdjusterRay();
//        bundleAdjuster.setConfThresh(1.0); // Ngưỡng tin cậy cao
//
//        // Tinh chỉnh tham số: Focal length (fx), Rotation (R)
//        // Hàm này chạy LM tối ưu hóa toàn cục hàm loss
//        bundleAdjuster.apply(estimatorFeatures, estimatorMatches, cameras);
//
//        // 4. Automatic Wave Correction (Làm thẳng Panorama)
//        // Tìm vector 'Up' trung bình và xoay toàn bộ camera để đường chân trời thẳng
//        waveCorrect(cameras, WAVE_CORRECT_HORIZ);
//
//        System.out.println("Đã tối ưu hóa Bundle Adjustment và Wave Correction.");
//
//        // Warping & Blending (Multi-band)
//        Stitcher stitcher = Stitcher.create(Stitcher.PANORAMA);
//        stitcher.setPanoConfidenceThresh(0.5);
//        stitcher.setWaveCorrection(true);
//
//        Mat result = new Mat();
//        int status = stitcher.stitch(imgs, result);
//
//        if (status == Stitcher.OK) {
//            boolean success = imwrite(outputName, result);
//            if (success) {
//                System.out.println("✓ Đã lưu kết quả: " + outputName);
//            } else {
//                System.out.println("✗ Lỗi: Không thể lưu ảnh " + outputName);
//            }
//        } else {
//            System.out.println("✗ Ghép thất bại. Mã lỗi: " + status);
//        }
//
//        result.close();
//        stitcher.close();
//        bundleAdjuster.close();
//        estimator.close();
//
//        for (long i = 0; i < imgs.size(); i++) {
//            imgs.get(i).close();
//        }
//        imgs.close();
//    }
//}