package com.stitching.imageStitching;

import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftDetector;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.opencv.opencv_core.*;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Arrays;
import java.util.stream.Collectors;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class CylinderStitcherEnhanced {
    private static final Path OUTPUT_PATH = Paths.get("src", "main", "resources", "static", "output");
    private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "example_data","myself");
    //private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "scene_crossline");

    // ===== [MỚI] ENUM CHO HƯỚNG GHÉP =====
    enum StitchDirection { HORIZONTAL, VERTICAL, DIAGONAL, UNKNOWN }

    // SỬ DỤNG ImageNode từ CylinderStitcher.ImageNode
    // Để tương thích với ImageBlenderFAST
    public static class ImageNode {
        public int id;
        public String filename;
        public Mat img;
        public List<SiftKeyPoint> keypoints;
        public Mat descriptors;
        public Mat globalTransform;

        public ImageNode(int id, String name, Mat img, List<SiftKeyPoint> kp, Mat desc) {
            this.id = id;
            this.filename = name;
            this.img = img;
            this.keypoints = kp;
            this.descriptors = desc;
            this.globalTransform = Mat.eye(3, 3, CV_64F).asMat();
        }

        public CylinderStitcher.ImageNode transform() {
            return new CylinderStitcher.ImageNode(this.id, this.filename, this.img, this.keypoints, this.descriptors);
        }
    }

    // [MỚI] Lưu thông tin hướng ghép riêng biệt
    private static Map<Integer, StitchDirection> nodeDirections = new HashMap<>();

    public static void main(String[] args) {
        System.out.println("=== OPENPANO JAVA AUTO-DIRECTION ===");

        File folder = INPUT_PATH.toFile();
        File[] files = folder.listFiles((d, n) ->
                n.toLowerCase().endsWith(".jpg") || n.toLowerCase().endsWith(".png"));
        if (files == null || files.length == 0) return;

        Arrays.sort(files, Comparator.comparing(File::getName));

        List<ImageNode> nodes = new ArrayList<>();
        FeatureMatcherWrapper matcher = new FeatureMatcherWrapper();

        // --- BƯỚC 1: LOAD & DETECT FEATURES (Chưa Warp) ---
        System.out.println("\n[Step 1] Loading & SIFT Detection...");
        for (int i = 0; i < files.length; i++) {
            System.out.println("-> " + files[i].getName());
            Mat raw = imread(files[i].getAbsolutePath());
            if (raw.empty()) continue;

            // [MỚI] CHƯA WARP - Giữ nguyên ảnh gốc để phân tích hướng
            Mat gray = new Mat();
            cvtColor(raw, gray, COLOR_BGR2GRAY);
            Mat fGray = new Mat();
            gray.convertTo(fGray, CV_32F, 1.0/255.0, 0.0);

            SiftDetector detector = new SiftDetector();
            ScaleSpace ss = new ScaleSpace();
            List<SiftKeyPoint> kps = detector.run(
                    ss.buildGaussianPyramid(fGray),
                    ss.buildDoGPyramid(ss.buildGaussianPyramid(fGray))
            );

            if(kps.size() > 4000) kps = new ArrayList<>(kps.subList(0, 4000));

            Mat desc = matcher.convertDescriptors(kps);
            nodes.add(new ImageNode(i, files[i].getName(), raw, kps, desc));

            gray.release();
            fGray.release();
        }

        if (nodes.size() < 2) return;

        // --- BƯỚC 2: PHÂN TÍCH HƯỚNG GHÉP ---
        System.out.println("\n[Step 2] Analyzing Stitch Directions...");
        analyzeStitchDirections(nodes, matcher);

        // --- BƯỚC 3: QUYẾT ĐỊNH CHIẾN LƯỢC WARP ---
        System.out.println("\n[Step 3] Applying Cylindrical Warp...");
        applyWarpStrategy(nodes);

        // --- BƯỚC 4: TÍNH TRANSFORMS ---
        System.out.println("\n[Step 4] Calculating Transforms...");
        computeTransforms(nodes, matcher);

        // --- BƯỚC 5: BLENDING ---
        System.out.println("\n[Step 5] Blending...");
        List<CylinderStitcher.ImageNode> nodes_new = nodes.stream()
                // 1. MAP: Tạo CylinderStitcher.ImageNode mới từ ImageNode cũ
                .map(n -> new CylinderStitcher.ImageNode(
                        n.id,           // Copy id
                        n.filename,     // Copy filename
                        n.img,          // Copy Mat img
                        n.keypoints,    // Copy keypoints
                        n.descriptors,   // Copy descriptors
                        n.globalTransform
                ))
                .collect(Collectors.toList());
        Mat result = ImageBlenderFAST.blend(nodes_new);

        if (result != null) {
            String name = "openpano_auto_direction";
            String outName = name + ".jpg";
            imwrite(OUTPUT_PATH.resolve(outName).toString(), result);
            System.out.println(">>> DONE: " + outName);
        }
    }

    // ===== [MỚI] HÀM PHÂN TÍCH HƯỚNG GHÉP =====
    private static void analyzeStitchDirections(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        for (int i = 0; i < nodes.size() - 1; i++) {
            ImageNode curr = nodes.get(i);
            ImageNode next = nodes.get(i + 1);

            // Match giữa 2 ảnh liên tiếp
            FeatureMatcherWrapper.MatchResult res = matcher.match(
                    curr.keypoints, curr.descriptors,
                    next.keypoints, next.descriptors
            );

            if (res == null || res.inlierMatches.size() < 20) {
                System.out.println("   -> " + curr.filename + " <-> " + next.filename +
                        ": NOT ENOUGH MATCHES");
                nodeDirections.put(curr.id, StitchDirection.UNKNOWN);
                continue;
            }

            // Phân tích vector dịch chuyển
            StitchDirection dir = analyzeMatchDirection(res.inlierMatches, curr, next);
            nodeDirections.put(curr.id, dir);

            System.out.println("   -> " + curr.filename + " <-> " + next.filename +
                    ": " + dir);
        }
    }

    // ===== [MỚI] LOGIC PHÂN TÍCH HƯỚNG (Port từ AutoStitcher) =====
    private static StitchDirection analyzeMatchDirection(
            List<DMatch> matches, ImageNode n1, ImageNode n2) {

        double sumVx = 0, sumVy = 0;
        double sumAbsVx = 0, sumAbsVy = 0;
        int count = Math.min(matches.size(), 100);

        for (int i = 0; i < count; i++) {
            DMatch m = matches.get(i);
            SiftKeyPoint p1 = n1.keypoints.get(m.queryIdx());
            SiftKeyPoint p2 = n2.keypoints.get(m.trainIdx());

            double vx = p1.x - p2.x;
            double vy = p1.y - p2.y;

            sumVx += vx;
            sumVy += vy;
            sumAbsVx += Math.abs(vx);
            sumAbsVy += Math.abs(vy);
        }

        double avgAbsVx = sumAbsVx / count;
        double avgAbsVy = sumAbsVy / count;

        // Tỷ lệ giữa chiều nhỏ và chiều lớn
        double ratio = Math.min(avgAbsVx, avgAbsVy) / Math.max(avgAbsVx, avgAbsVy);

        // Phát hiện CHÉO
        if (ratio > 0.5) {
            return StitchDirection.DIAGONAL;
        }
        // Ngang > Dọc
        else if (avgAbsVx > avgAbsVy) {
            return StitchDirection.HORIZONTAL;
        }
        // Dọc
        else {
            return StitchDirection.VERTICAL;
        }
    }

    // ===== [MỚI] ÁP DỤNG CHIẾN LƯỢC WARP =====
    private static void applyWarpStrategy(List<ImageNode> nodes) {
        // Kiểm tra xem có ảnh nào DỌC hoặc CHÉO không
        boolean hasVertical = false;
        boolean hasDiagonal = false;
        int horizontalCount = 0;

        for (ImageNode node : nodes) {
            StitchDirection dir = nodeDirections.getOrDefault(node.id, StitchDirection.UNKNOWN);
            if (dir == StitchDirection.VERTICAL) hasVertical = true;
            if (dir == StitchDirection.DIAGONAL) hasDiagonal = true;
            if (dir == StitchDirection.HORIZONTAL) horizontalCount++;
        }

        System.out.println("   -> Horizontal: " + horizontalCount +
                ", Vertical: " + hasVertical +
                ", Diagonal: " + hasDiagonal);

        // CHIẾN LƯỢC:
        // 1. Nếu TOÀN BỘ là NGANG → Cylindrical Warp (như cũ)
        // 2. Nếu có DỌC/CHÉO → KHÔNG WARP (Planar mode)

        boolean shouldWarp = !hasVertical && !hasDiagonal && horizontalCount > nodes.size() / 2;

        if (shouldWarp) {
            System.out.println("   -> Strategy: CYLINDRICAL WARP (Pure Panorama)");
            Mat tmp = nodes.get(0).img;
            double f = tmp.cols() * 1.0;

            for (ImageNode node : nodes) {
                Mat warped = CylindricalWarper.warp(node.img, f);

                // Tính lại SIFT trên ảnh warped
                Mat gray = new Mat();
                cvtColor(warped, gray, COLOR_BGR2GRAY);
                Mat fGray = new Mat();
                gray.convertTo(fGray, CV_32F, 1.0/255.0, 0.0);

                SiftDetector detector = new SiftDetector();
                ScaleSpace ss = new ScaleSpace();
                List<SiftKeyPoint> kps = detector.run(
                        ss.buildGaussianPyramid(fGray),
                        ss.buildDoGPyramid(ss.buildGaussianPyramid(fGray))
                );

                if(kps.size() > 4000) kps = new ArrayList<>(kps.subList(0, 4000));

                FeatureMatcherWrapper tempMatcher = new FeatureMatcherWrapper();
                Mat desc = tempMatcher.convertDescriptors(kps);

                node.img = warped;
                node.keypoints = kps;
                node.descriptors = desc;  // ← QUAN TRỌNG

                gray.release();
                fGray.release();
            }
        } else {
            System.out.println("   -> Strategy: PLANAR (Mixed Directions)");
            // Không warp - giữ nguyên ảnh gốc và keypoints
        }
    }

    // ===== [GIỮ NGUYÊN] HÀM COMPUTE TRANSFORMS =====
    private static void computeTransforms(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        int n = nodes.size();
        int mid = n / 2;
        ImageNode center = nodes.get(mid);
        System.out.println("   -> Anchor: " + center.filename);

        // Lan truyền sang Phải
        for (int i = mid; i < n - 1; i++) {
            ImageNode curr = nodes.get(i);
            ImageNode next = nodes.get(i + 1);

            FeatureMatcherWrapper.MatchResult res = matcher.match(
                    curr.keypoints, curr.descriptors,
                    next.keypoints, next.descriptors
            );

            if (res != null) {
                Mat T_curr_next = TransformEstimator.estimateAffine(res.srcPoints, res.dstPoints);
                if (T_curr_next != null && TransformEstimator.isTransformValid(T_curr_next)) {
                    Mat T_next_curr = new Mat();
                    invert(T_curr_next, T_next_curr, DECOMP_LU);

                    Mat g = new Mat();
                    gemm(curr.globalTransform, T_next_curr, 1.0, new Mat(), 0.0, g);
                    next.globalTransform = g;
                } else {
                    next.globalTransform = curr.globalTransform.clone();
                }
            }
        }

        // Lan truyền sang Trái
        for (int i = mid; i > 0; i--) {
            ImageNode curr = nodes.get(i);
            ImageNode prev = nodes.get(i - 1);

            FeatureMatcherWrapper.MatchResult res = matcher.match(
                    prev.keypoints, prev.descriptors,
                    curr.keypoints, curr.descriptors
            );

            if (res != null) {
                Mat T_prev_curr = TransformEstimator.estimateAffine(res.srcPoints, res.dstPoints);
                if (T_prev_curr != null && TransformEstimator.isTransformValid(T_prev_curr)) {
                    Mat g = new Mat();
                    gemm(curr.globalTransform, T_prev_curr, 1.0, new Mat(), 0.0, g);
                    prev.globalTransform = g;
                } else {
                    prev.globalTransform = curr.globalTransform.clone();
                }
            }
        }
    }
}