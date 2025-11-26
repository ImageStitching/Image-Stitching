package com.stitching.imageStitching;

import com.stitching.imageStitching.blender.ImageBlenderFAST;
import com.stitching.imageStitching.matchAndTransform.FeatureMatcherWrapper;
import com.stitching.imageStitching.matchAndTransform.TransformEstimator;
import com.stitching.imageStitching.warper.CylindricalWarper;
import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftDetector;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.opencv.opencv_core.*;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/***
 * Code này có thể xác định Ngang, chéo ,dọc nhwung ếu đảo thứ tự các ảnh thì sẽ ko xác định được.
 */
public class CylinderStitcherEnhanced {
    private static final Path OUTPUT_PATH = Paths.get("src", "main", "resources", "static", "output");
    //private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "example_data","myself");
    private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "scene_vertical_2");

    enum StitchDirection { HORIZONTAL, VERTICAL, DIAGONAL, UNKNOWN }
    private static Map<Integer, StitchDirection> nodeDirections = new HashMap<>();
    private static Map<Integer, Boolean> nodeSwapFlags = new HashMap<>();

    static class ImageRelation {
        StitchDirection direction;
        boolean needSwap;
        String debugMsg;
    }

    public static void main(String[] args) {
        System.out.println("=== OPENPANO JAVA AUTO-DIRECTION ===");
        File folder = INPUT_PATH.toFile();
        File[] files = folder.listFiles((d, n) -> n.toLowerCase().endsWith(".jpg") || n.toLowerCase().endsWith(".png"));
        if (files == null || files.length == 0) return;

        Arrays.sort(files, Comparator.comparing(File::getName));
        List<ImageNode> nodes = new ArrayList<>();
        FeatureMatcherWrapper matcher = new FeatureMatcherWrapper();

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
            List<MatVector> gaussianPyramid = ss.buildGaussianPyramid(fGray);
            List<MatVector> dogPyramid = ss.buildDoGPyramid(gaussianPyramid);
            List<SiftKeyPoint> kps = detector.run(gaussianPyramid, dogPyramid);

            if(kps.size() > 4000) kps = new ArrayList<>(kps.subList(0, 4000));

            Mat desc = matcher.convertDescriptors(kps);
            nodes.add(new ImageNode(i, files[i].getName(), raw, kps, desc));

            gray.release(); fGray.release(); gaussianPyramid.clear(); dogPyramid.clear();
        }

        if (nodes.size() < 2) return;

        System.out.println("\n[Step 2] Analyzing Stitch Directions...");
        analyzeStitchDirections(nodes, matcher);

        // ========== STEP 2.5: AUTO SORT ==========
        System.out.println("\n[Step 2.5] Auto-Sorting Images...");
        sortImagesByDirection(nodes, matcher);

        System.out.println("\n[Step 3] Applying Cylindrical Warp...");
        applyWarpStrategy(nodes);

        System.out.println("\n[Step 4] Calculating Transforms...");
        computeTransforms(nodes, matcher);

        System.out.println("\n[Step 5] Blending...");
        /* List<CylinderStitcher.ImageNode> nodes_new = nodes.stream()
                .map(n -> new CylinderStitcher.ImageNode(n.id, n.filename, n.img, n.keypoints, n.descriptors, n.globalTransform))
                .collect(Collectors.toList()); */
        Mat result = ImageBlenderFAST.blend(nodes);

        if (result != null) {
            String name = "openpano_auto_direction";
            String outName = name + ".jpg";
            imwrite(OUTPUT_PATH.resolve(outName).toString(), result);
            System.out.println(">>> DONE: " + outName);
        }
    }

    private static void analyzeStitchDirections(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        for (int i = 0; i < nodes.size() - 1; i++) {
            ImageNode curr = nodes.get(i);
            ImageNode next = nodes.get(i + 1);

            FeatureMatcherWrapper.MatchResult res = matcher.match(curr.keypoints, curr.descriptors, next.keypoints, next.descriptors);

            if (res == null || res.inlierMatches.size() < 20) {
                System.out.println("   -> " + curr.filename + " <-> " + next.filename + ": NOT ENOUGH MATCHES (" + (res != null ? res.inlierMatches.size() : 0) + ")");
                nodeDirections.put(curr.id, StitchDirection.UNKNOWN);
                continue;
            }

            ImageRelation rel = analyzeMatchDirection(res.inlierMatches, curr, next);
            nodeDirections.put(curr.id, rel.direction);
            System.out.println("   -> " + curr.filename + " <-> " + next.filename + ": " + rel.debugMsg);
        }
    }

    private static ImageRelation analyzeMatchDirection(List<DMatch> matches, ImageNode n1, ImageNode n2) {
        ImageRelation rel = new ImageRelation();
        double sumVx = 0, sumVy = 0; double sumAbsVx = 0, sumAbsVy = 0;
        int count = Math.min(matches.size(), 200);

        for (int i = 0; i < count; i++) {
            DMatch m = matches.get(i);
            SiftKeyPoint p1 = n1.keypoints.get(m.queryIdx());
            SiftKeyPoint p2 = n2.keypoints.get(m.trainIdx());
            double vx = p1.x - p2.x; double vy = p1.y - p2.y;
            sumVx += vx; sumVy += vy; sumAbsVx += Math.abs(vx); sumAbsVy += Math.abs(vy);
        }

        double avgVx = sumVx / count;
        double avgVy = sumVy / count;
        double avgAbsVx = sumAbsVx / count;
        double avgAbsVy = sumAbsVy / count;

        // === ĐIỀU KIỆN DIAGONAL (ĐÃ SỬA) ===
        double ratio = Math.min(avgAbsVx, avgAbsVy) / (Math.max(avgAbsVx, avgAbsVy) + 0.001);

        // FIX 1: Tăng ngưỡng ratio từ 0.5 lên 0.7
        boolean ratioCheck = ratio > 0.7;

        // FIX 2: Cả 2 chiều phải đủ lớn (> 30px) để loại bỏ nhiễu
        boolean magnitudeCheck = (avgAbsVx > 30) && (avgAbsVy > 30);

        // FIX 3: Độ chênh lệch không quá lớn (< 30%)
        double diffRatio = Math.abs(avgAbsVx - avgAbsVy) / Math.max(avgAbsVx, avgAbsVy);
        boolean balanceCheck = diffRatio < 0.3;

        boolean isDiagonal = ratioCheck && magnitudeCheck && balanceCheck;

        // Debug info
        String vectorInfo = String.format(
                "Vx=%.1f Vy=%.1f |Vx|=%.1f |Vy|=%.1f ratio=%.2f diff=%.2f",
                avgVx, avgVy, avgAbsVx, avgAbsVy, ratio, diffRatio
        );

        // === XÁC ĐỊNH HƯỚNG VÀ SWAP ===
        if (isDiagonal) {
            rel.direction = StitchDirection.DIAGONAL;
            rel.debugMsg = "Chéo (Diagonal). " + vectorInfo;
            // Kiểm tra swap cho diagonal (tổng vector < 0)
            if ((avgVx + avgVy) < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> SWAP (Img đàu ở góc Dưới-Phải)";
            }
        }
        else if (avgAbsVx > avgAbsVy) {
            rel.direction = StitchDirection.HORIZONTAL;
            rel.debugMsg = "Ngang (Horizontal). " + vectorInfo;
            // Kiểm tra swap cho ngang (Vx < 0 = Phải->Trái)
            if (avgVx < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> SWAP (Phải->Trái)";
            }
        }
        else {
            rel.direction = StitchDirection.VERTICAL;
            rel.debugMsg = "Dọc (Vertical). " + vectorInfo;
            // Kiểm tra swap cho dọc (Vy < 0 = Dưới->Trên)
            if (avgVy < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> SWAP (Dưới->Trên)";
            }
        }

        return rel;
    }

    private static void sortImagesByDirection(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        boolean swapped;
        int passCount = 0;
        int totalSwaps = 0;
        int maxPasses = nodes.size(); // Giới hạn để tránh vòng lặp vô hạn

        do {
            swapped = false;
            passCount++;

            if (passCount > maxPasses) {
                System.out.println("   [WARNING] Đã vượt quá " + maxPasses + " lần lặp, dừng sắp xếp.");
                break;
            }

            for (int i = 0; i < nodes.size() - 1; i++) {
                ImageNode curr = nodes.get(i);
                ImageNode next = nodes.get(i + 1);

                // Match lại để kiểm tra
                FeatureMatcherWrapper.MatchResult res = matcher.match(curr.keypoints, curr.descriptors, next.keypoints, next.descriptors);

                if (res == null || res.inlierMatches.size() < 20) {
                    continue; // Không đủ match
                }

                ImageRelation rel = analyzeMatchDirection(res.inlierMatches, curr, next);

                // Nếu phát hiện ngược thứ tự → Swap
                if (rel.needSwap) {
                    System.out.println("   -> Swapping: [" + i + "] " + curr.filename + " <-> [" + (i+1) + "] " + next.filename);
                    swapImageNodeContent(curr, next);
                    updateMapsAfterSwap(curr.id, next.id);
                    swapped = true; totalSwaps++;
                }
            }
        } while (swapped);

        System.out.println("   -> Hoàn thành sau " + passCount + " lần lặp, " + totalSwaps + " lần swap.");
        System.out.println("   -> Thứ tự cuối cùng:");
        for (int i = 0; i < nodes.size(); i++) {
            System.out.println("      [" + i + "] " + nodes.get(i).filename);
        }
    }

    private static void swapImageNodeContent(ImageNode a, ImageNode b) {
        // Lưu tạm nội dung của a
        String tempFilename = a.filename;
        Mat tempImg = a.img;
        List<SiftKeyPoint> tempKps = a.keypoints;
        Mat tempDesc = a.descriptors;
        Mat tempTransform = a.globalTransform;

        // Copy nội dung từ b sang a
        a.filename = b.filename;
        a.img = b.img;
        a.keypoints = b.keypoints;
        a.descriptors = b.descriptors;
        a.globalTransform = b.globalTransform;

        // Copy nội dung từ temp (a cũ) sang b
        b.filename = tempFilename;
        b.img = tempImg;
        b.keypoints = tempKps;
        b.descriptors = tempDesc;
        b.globalTransform = tempTransform;

        // LƯU Ý: a.id và b.id KHÔNG đổi!
    }

    private static void updateMapsAfterSwap(int idA, int idB) {
        // Swap trong nodeDirections
        StitchDirection dirA = nodeDirections.get(idA);
        StitchDirection dirB = nodeDirections.get(idB);
        if (dirA != null) nodeDirections.put(idB, dirA);
        if (dirB != null) nodeDirections.put(idA, dirB);

        // Swap trong nodeSwapFlags (nếu có dùng)
        Boolean flagA = nodeSwapFlags.get(idA);
        Boolean flagB = nodeSwapFlags.get(idB);
        if (flagA != null || flagB != null) {
            nodeSwapFlags.put(idA, flagB);
            nodeSwapFlags.put(idB, flagA);
        }
    }

    private static void applyWarpStrategy(List<ImageNode> nodes) {
        boolean hasVertical = false;
        boolean hasDiagonal = false;
        int horizontalCount = 0;

        for (ImageNode node : nodes) {
            StitchDirection dir = nodeDirections.getOrDefault(node.id, StitchDirection.UNKNOWN);
            if (dir == StitchDirection.VERTICAL) hasVertical = true;
            if (dir == StitchDirection.DIAGONAL) hasDiagonal = true;
            if (dir == StitchDirection.HORIZONTAL) horizontalCount++;
        }

        System.out.println("   -> Horizontal: " + horizontalCount + ", Vertical: " + hasVertical + ", Diagonal: " + hasDiagonal);
        // Chiến lược: Chỉ warp nếu TOÀN BỘ là ngang
        boolean shouldWarp = !hasVertical && !hasDiagonal && horizontalCount > nodes.size() / 2;
        // OPTION: Có thể force bật/tắt warp ở đây
        shouldWarp = false; // Force Planar
        // shouldWarp = true;  // Force Cylindrical

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
                List<MatVector> gaussianPyramid = ss.buildGaussianPyramid(fGray);
                List<MatVector> dogPyramid = ss.buildDoGPyramid(gaussianPyramid);
                List<SiftKeyPoint> kps = detector.run(gaussianPyramid, dogPyramid);

                if(kps.size() > 4000) kps = new ArrayList<>(kps.subList(0, 4000));

                FeatureMatcherWrapper tempMatcher = new FeatureMatcherWrapper();
                Mat desc = tempMatcher.convertDescriptors(kps);

                node.img = warped;
                node.keypoints = kps;
                node.descriptors = desc;  // ← QUAN TRỌNG

                gray.release(); fGray.release(); gaussianPyramid.clear(); dogPyramid.clear();
            }
        } else {
            System.out.println("   -> Strategy: PLANAR (Mixed Directions)");
        }
    }

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