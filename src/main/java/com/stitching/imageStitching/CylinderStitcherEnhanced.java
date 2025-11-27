package com.stitching.imageStitching;

import com.stitching.imageStitching.blender.ImageBlenderFAST;
import com.stitching.imageStitching.blender.ImageBlenderOpen;
import com.stitching.imageStitching.matchAndTransform.FeatureMatcherWrapper;
import com.stitching.imageStitching.matchAndTransform.TransformEstimator;
import com.stitching.imageStitching.warper.CylindricalWarper;
import com.stitching.imageStitching.warper.SphericalWarper;
import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftConfig;
import com.stitching.openpanoSIFT.SiftDetector;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import edu.princeton.cs.algorithms.MinPQ;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.opencv.opencv_core.*;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Arrays;

import static com.stitching.imageStitching.ImageNode.analyzeSequence;
import static com.stitching.imageStitching.ImageNode.getMatchScore;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/***
 * Code này có thể xác định Ngang, chéo ,dọc nhwung ếu đảo thứ tự các ảnh thì sẽ ko xác định được.
 */
public class CylinderStitcherEnhanced {
    private static final Path OUTPUT_PATH = Paths.get("src", "main", "resources", "static", "output");
    //private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "example_data", "CMU0");
    private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "scene_vertical_2");

    enum StitchDirection {HORIZONTAL, VERTICAL, DIAGONAL, UNKNOWN}

    private static Map<Integer, StitchDirection> nodeDirections = new HashMap<>();
    private static Map<Integer, Boolean> nodeSwapFlags = new HashMap<>();

    static class ImageRelation {
        StitchDirection direction;
        boolean needSwap;
        String debugMsg;
    }

    public static String run(boolean warp, Path INPUT_PATH, Path OUTPUT_PATH) {
        System.out.println("=== OPENPANO JAVA AUTO-DIRECTION ===");
        File folder = INPUT_PATH.toFile();
        File[] files = folder.listFiles((d, n) -> n.toLowerCase().endsWith(".jpg") || n.toLowerCase().endsWith(".png"));
        if (files == null || files.length == 0) return null;

        Arrays.sort(files, Comparator.comparing(File::getName));
        List<ImageNode> nodes = new ArrayList<>();
        FeatureMatcherWrapper matcher = new FeatureMatcherWrapper();

        SiftConfig.DOUBLE_IMAGE_SIZE = false;
        for (int i = 0; i < files.length; i++) {
            System.out.println("-> " + files[i].getName());
            Mat raw = imread(files[i].getAbsolutePath());
            if (raw.empty()) continue;

            // [MỚI] CHƯA WARP - Giữ nguyên ảnh gốc để phân tích hướng
            Mat gray = new Mat();
            cvtColor(raw, gray, COLOR_BGR2GRAY);
            Mat fGray = new Mat();
            gray.convertTo(fGray, CV_32F, 1.0 / 255.0, 0.0);

            SiftDetector detector = new SiftDetector();
            ScaleSpace ss = new ScaleSpace();
            List<MatVector> gaussianPyramid = ss.buildGaussianPyramid(fGray);
            List<MatVector> dogPyramid = ss.buildDoGPyramid(gaussianPyramid);
            List<SiftKeyPoint> kps = detector.run(gaussianPyramid, dogPyramid);

            // kps.sort((k1, k2) -> Float.compare(k2.response, k1.response));

            if (kps.size() > 4000) kps = new ArrayList<>(kps.subList(0, 4000));

            Mat desc = matcher.convertDescriptors(kps);
            nodes.add(new ImageNode(i, files[i].getName(), raw, kps, desc));

            gray.release();
            fGray.release();
            gaussianPyramid.clear();
            dogPyramid.clear();
        }

        if (nodes.size() < 2) return null;

        analyzeStitchDirections(nodes, matcher);

        sortImagesByDirection(nodes, matcher);

        applyWarpStrategy(nodes, warp);

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
            return outName;
        }
        return null;
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
        SiftConfig.DOUBLE_IMAGE_SIZE = false;
        for (int i = 0; i < files.length; i++) {
            System.out.println("-> " + files[i].getName());
            Mat raw = imread(files[i].getAbsolutePath());
            if (raw.empty()) continue;

            // [MỚI] CHƯA WARP - Giữ nguyên ảnh gốc để phân tích hướng
            Mat gray = new Mat();
            cvtColor(raw, gray, COLOR_BGR2GRAY);
            Mat fGray = new Mat();
            gray.convertTo(fGray, CV_32F, 1.0 / 255.0, 0.0);

            SiftDetector detector = new SiftDetector();
            ScaleSpace ss = new ScaleSpace();
            List<MatVector> gaussianPyramid = ss.buildGaussianPyramid(fGray);
            List<MatVector> dogPyramid = ss.buildDoGPyramid(gaussianPyramid);
            List<SiftKeyPoint> kps = detector.run(gaussianPyramid, dogPyramid);

            // kps.sort((k1, k2) -> Float.compare(k2.response, k1.response));

            if (kps.size() > 4000) kps = new ArrayList<>(kps.subList(0, 4000));

            Mat desc = matcher.convertDescriptors(kps);
            nodes.add(new ImageNode(i, files[i].getName(), raw, kps, desc));

            gray.release();
            fGray.release();
            gaussianPyramid.clear();
            dogPyramid.clear();
        }

        if (nodes.size() < 2) return;

        System.out.println("\n[Step 2] Analyzing Stitch Directions...");
        analyzeStitchDirections(nodes, matcher);

        // ========== STEP 2.5: AUTO SORT ==========
        System.out.println("\n[Step 2.5] Auto-Sorting Images...");
        sortImagesByDirection(nodes, matcher);

        System.out.println("\n[Step 3] Applying Cylindrical Warp...");
        applyWarpStrategy(nodes, true);

        System.out.println("\n[Step 4] Calculating Transforms...");
        computeTransforms(nodes, matcher);

        System.out.println("\n[Step 5] Blending...");
        /* List<CylinderStitcher.ImageNode> nodes_new = nodes.stream()
                .map(n -> new CylinderStitcher.ImageNode(n.id, n.filename, n.img, n.keypoints, n.descriptors, n.globalTransform))
                .collect(Collectors.toList()); */

        Mat result = ImageBlenderFAST.blend(nodes);
        // Mat result = ImageBlenderOpen.blend(nodes);

        if (result != null) {
            String name = "openpano_auto_direction";
            String outName = name + ".jpg";
            imwrite(OUTPUT_PATH.resolve(outName).toString(), result);
            System.out.println(">>> DONE: " + outName);
        }
    }

    public static void analyzeStitchDirections(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
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

    public static ImageRelation analyzeMatchDirection(List<DMatch> matches, ImageNode n1, ImageNode n2) {
        ImageRelation rel = new ImageRelation();
        double sumVx = 0, sumVy = 0;
        double sumAbsVx = 0, sumAbsVy = 0;

        int count = Math.min(matches.size(), 400);

        for (int i = 0; i < count; i++) {
            DMatch m = matches.get(i);
            SiftKeyPoint p1 = n1.keypoints.get(m.queryIdx()); // Điểm trên ảnh n1
            SiftKeyPoint p2 = n2.keypoints.get(m.trainIdx()); // Điểm trên ảnh n2

            // Vector V = P1 - P2
            double vx = p1.x - p2.x;
            double vy = p1.y - p2.y;

            sumVx += vx;
            sumVy += vy;
            sumAbsVx += Math.abs(vx);
            sumAbsVy += Math.abs(vy);
        }

        // 2. Tính trung bình
        double avgVx = sumVx / count;       // Vector có hướng (quan trọng để xác định thứ tự SWAP)
        double avgVy = sumVy / count;
        double avgAbsVx = sumAbsVx / count; // Độ lớn tuyệt đối (quan trọng để xác định HƯỚNG)
        double avgAbsVy = sumAbsVy / count;

        double ratio = (Math.min(avgAbsVx, avgAbsVy) + 0.001) / (Math.max(avgAbsVx, avgAbsVy) + 0.001);
        boolean isDiagonal = ratio > 0.5;

        String vectorInfo = String.format("Vx=%.1f Vy=%.1f |Vx|=%.1f |Vy|=%.1f ratio=%.2f", avgVx, avgVy, avgAbsVx, avgAbsVy, ratio);

        // 4. Phân loại hướng và kiểm tra Swap (dựa trên vector Âm/Dương)
        if (isDiagonal) {
            rel.direction = StitchDirection.DIAGONAL;
            rel.debugMsg = "Chéo (Diagonal). " + vectorInfo;
            if ((avgVx + avgVy) < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> SWAP (Do n1 ở góc Dưới-Phải)";
            }
        } else if (avgAbsVx > avgAbsVy) {
            rel.direction = StitchDirection.HORIZONTAL;
            rel.debugMsg = "Ngang (Horizontal). " + vectorInfo;
            if (avgVx < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> SWAP (Phải->Trái)";
            }
        } else {
            rel.direction = StitchDirection.VERTICAL;
            rel.debugMsg = "Dọc (Vertical). " + vectorInfo;
            if (avgVy < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> SWAP (Dưới->Trên)";
            }
        }
        return rel;
    }

    // Hàm này sẽ lỗi nếu gặp chụp ảnh xoay camera vòng tròn do không xác định rõ được cực phải và cực trái
    // Thay thế hoàn toàn hàm sortImagesByDirection cũ bằng hàm này
    /* private static void sortImagesByDirection(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        int n = nodes.size();
        System.out.println("   [Sort] Đang phân tích toàn cục " + n + " ảnh (Global Voting)...");

        // Bảng điểm để xếp hạng (Score càng NHỎ -> càng đứng ĐẦU/TRÊN/TRÁI)
        Map<Integer, Integer> scoreMap = new HashMap<>();
        for (ImageNode node : nodes) {
            scoreMap.put(node.id, 0);
        }

        // So sánh TẤT CẢ các cặp (All-pairs comparison)
        // Độ phức tạp N^2, nhưng với n < 20 thì chạy trong tíc tắc (vài ms)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                ImageNode nodeA = nodes.get(i);
                ImageNode nodeB = nodes.get(j);

                // Match A với B
                FeatureMatcherWrapper.MatchResult res = matcher.match(
                        nodeA.keypoints, nodeA.descriptors,
                        nodeB.keypoints, nodeB.descriptors
                );

                // Nếu không khớp hoặc ít điểm chung -> Bỏ qua, không vote
                if (res == null || res.inlierMatches.size() < 15) {
                    continue;
                }

                // Phân tích quan hệ A và B
                ImageRelation rel = analyzeMatchDirection(res.inlierMatches, nodeA, nodeB);

                // Logic Voting:
                // Nếu rel.needSwap = TRUE: Nghĩa là A đang đứng SAU B (theo thứ tự mong muốn).
                // => A bị cộng điểm (đẩy xuống dưới), B được trừ điểm (đẩy lên trên).
                // Ngược lại: A đứng trước B.

                if (rel.needSwap) {
                    // A đứng sau B
                    scoreMap.put(nodeA.id, scoreMap.get(nodeA.id) + 1);
                    scoreMap.put(nodeB.id, scoreMap.get(nodeB.id) - 1);
                    System.out.println("      Vote: " + nodeB.filename + " < " + nodeA.filename);
                } else {
                    // A đứng trước B
                    scoreMap.put(nodeA.id, scoreMap.get(nodeA.id) - 1);
                    scoreMap.put(nodeB.id, scoreMap.get(nodeB.id) + 1);
                    System.out.println("      Vote: " + nodeA.filename + " < " + nodeB.filename);
                }
            }
        }
        // Sắp xếp List dựa trên điểm số (Tăng dần)
        nodes.sort(Comparator.comparingInt(node -> scoreMap.get(node.id)));

        System.out.println("   -> Thứ tự sau sắp xếp (Global Sort):");
        for (int i = 0; i < nodes.size(); i++) {
            System.out.println("      [" + i + "] " + nodes.get(i).filename + " (Score: " + scoreMap.get(nodes.get(i).id) + ")");
        }
    } */

    // --- CHIẾN LƯỢC 2: CHAIN SORT (MẮT XÍCH) - Dùng khi tên file lộn xộn ---
    /* private static void sortImagesByDirection(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        System.out.println("   [Sort] Chiến lược: Seed & Grow (Tính toán trực tiếp - Không Cache)...");
        if (nodes.size() < 2) return;

        // --- BƯỚC 1: TÌM CẶP HẠT NHÂN (SEED) ---
        int bestA = -1, bestB = -1;
        int maxMatches = 0;

        // Duyệt tất cả các cặp để tìm cặp mạnh nhất
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = 0; j < nodes.size(); j++) {
                if (i == j) continue;

                ImageNode ni = nodes.get(i);
                ImageNode nj = nodes.get(j);

                // [FIX CRASH] Kiểm tra an toàn
                if (ni.descriptors == null || ni.descriptors.empty() ||
                        nj.descriptors == null || nj.descriptors.empty()) {
                    continue;
                }

                // Tính toán Match trực tiếp
                FeatureMatcherWrapper.MatchResult res = matcher.match(
                        ni.keypoints, ni.descriptors,
                        nj.keypoints, nj.descriptors
                );

                if (res != null && res.inlierMatches.size() > 30) {
                    // Chỉ cập nhật max, KHÔNG lưu cache
                    if (res.inlierMatches.size() > maxMatches) {
                        maxMatches = res.inlierMatches.size();
                        bestA = i;
                        bestB = j;
                    }
                }
            }
        }

        if (bestA == -1) {
            System.out.println("      [WARNING] Không tìm thấy liên kết mạnh > 30. Giữ nguyên.");
            return;
        }

        // --- BƯỚC 2: KHỞI TẠO CHUỖI ---
        LinkedList<ImageNode> chain = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();

        ImageNode nodeA = nodes.get(bestA);
        ImageNode nodeB = nodes.get(bestB);

        System.out.println("      -> Cặp hạt nhân: " + nodeA.filename + " & " + nodeB.filename + " (" + maxMatches + " matches)");

        // [RE-COMPUTE] Tính lại match cho cặp hạt nhân vì không dùng cache
        FeatureMatcherWrapper.MatchResult resAB = matcher.match(
                nodeA.keypoints, nodeA.descriptors,
                nodeB.keypoints, nodeB.descriptors
        );

        ImageRelation relAB = analyzeMatchDirection(resAB.inlierMatches, nodeA, nodeB);

        if (relAB.needSwap) {
            // A cần đứng sau B => [B, A]
            chain.add(nodeB); chain.add(nodeA);
            visited.add(nodeB.id); visited.add(nodeA.id);
            System.out.println("      Seed Order: " + nodeB.filename + " -> " + nodeA.filename);
        } else {
            // A đứng trước B => [A, B]
            chain.add(nodeA); chain.add(nodeB);
            visited.add(nodeA.id); visited.add(nodeB.id);
            System.out.println("      Seed Order: " + nodeA.filename + " -> " + nodeB.filename);
        }

        // --- BƯỚC 3: VÒNG LẶP MỞ RỘNG (GROWING) ---
        while (visited.size() < nodes.size()) {
            ImageNode head = chain.getFirst();
            ImageNode tail = chain.getLast();

            int bestCandidateId = -1;
            int bestScore = 0;
            boolean insertAtHead = false; // True: chèn đầu, False: chèn đuôi

            // Duyệt qua tất cả các node chưa được visit
            for (int i = 0; i < nodes.size(); i++) {
                ImageNode candidate = nodes.get(i);
                if (visited.contains(candidate.id)) continue;

                // [FIX CRASH] Kiểm tra lỗi descriptor
                if (candidate.descriptors == null || candidate.descriptors.empty()) {
                    visited.add(candidate.id); // Đánh dấu lỗi để không xét lại
                    System.out.println("      [SKIP] Ảnh lỗi descriptor: " + candidate.filename);
                    continue;
                }

                // 1. Thử khớp với TAIL (để nối vào đuôi)
                // [RE-COMPUTE] Tính trực tiếp Tail -> Candidate
                FeatureMatcherWrapper.MatchResult resTail = null;
                if (!tail.descriptors.empty()) {
                    resTail = matcher.match(
                            tail.keypoints, tail.descriptors,
                            candidate.keypoints, candidate.descriptors
                    );
                }

                if (resTail != null && resTail.inlierMatches.size() > 30) {
                    ImageRelation rel = analyzeMatchDirection(resTail.inlierMatches, tail, candidate);
                    // Nếu Tail -> Candidate mà KHÔNG SWAP => Candidate thực sự nằm sau Tail
                    if (!rel.needSwap) {
                        if (resTail.inlierMatches.size() > bestScore) {
                            bestScore = resTail.inlierMatches.size();
                            bestCandidateId = i;
                            insertAtHead = false; // Chèn đuôi
                        }
                    }
                }

                // 2. Thử khớp với HEAD (để nối vào đầu)
                // [RE-COMPUTE] Tính trực tiếp Candidate -> Head
                FeatureMatcherWrapper.MatchResult resHead = null;
                if (!head.descriptors.empty()) {
                    resHead = matcher.match(
                            candidate.keypoints, candidate.descriptors,
                            head.keypoints, head.descriptors
                    );
                }

                if (resHead != null && resHead.inlierMatches.size() > 30) {
                    ImageRelation rel = analyzeMatchDirection(resHead.inlierMatches, candidate, head);
                    // Nếu Candidate -> Head mà KHÔNG SWAP => Candidate thực sự nằm trước Head
                    if (!rel.needSwap) {
                        if (resHead.inlierMatches.size() > bestScore) {
                            bestScore = resHead.inlierMatches.size();
                            bestCandidateId = i;
                            insertAtHead = true; // Chèn đầu
                        }
                    }
                }
            } // End for candidate

            // Nếu tìm thấy ứng viên phù hợp
            if (bestCandidateId != -1) {
                ImageNode winner = nodes.get(bestCandidateId);
                if (insertAtHead) {
                    chain.addFirst(winner);
                    System.out.println("      + Chèn ĐẦU: " + winner.filename + " (Score: " + bestScore + ")");
                } else {
                    chain.addLast(winner);
                    System.out.println("      + Chèn ĐUÔI: " + winner.filename + " (Score: " + bestScore + ")");
                }
                visited.add(winner.id);
            } else {
                System.out.println("      [INFO] Không thể mở rộng chuỗi thêm nữa. Dừng tại đây.");
                break;
            }
        }

        // Cập nhật lại list nodes
        nodes.clear();
        nodes.addAll(chain);

        System.out.println("   -> Thứ tự tạm thời (Trước khi Reorient):");
        for(int i=0; i<nodes.size(); i++) {
            System.out.println("      [" + i + "] " + nodes.get(i).filename);
        }

        // --- BƯỚC 4: TÁI ĐỊNH HƯỚNG VẬT LÝ ---
        reorientCyclicList(nodes, matcher);

        System.out.println("   -> Thứ tự cuối cùng (Sau khi Reorient):");
        for(int i=0; i<nodes.size(); i++) {
            System.out.println("      [" + i + "] " + nodes.get(i).filename);
        }
    }
    */

    /**
     * Tái định hướng danh sách ảnh dựa trên Vectơ dịch chuyển (Vật lý).
     * Loại bỏ hoàn toàn sự phụ thuộc vào tên file.
     */
    /*private static void reorientCyclicList(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        // Cần ít nhất 2 ảnh để xác định vector hướng
        if (nodes.size() < 2) return;

        System.out.println("   [Reorient] Đang phân tích hướng vật lý (Physics-Based)...");

        // --- BƯỚC 1: XÁC ĐỊNH CẤU TRÚC (VÒNG TRÒN HAY ĐƯỜNG THẲNG) ---
        boolean isCyclic = false;
        // Chỉ check vòng tròn nếu có đủ số lượng ảnh (>= 4) để tránh false positive
        if (nodes.size() >= 4) {
            ImageNode first = nodes.get(0);
            ImageNode last = nodes.get(nodes.size() - 1);
            FeatureMatcherWrapper.MatchResult loopRes = matcher.match(
                    first.keypoints, first.descriptors,
                    last.keypoints, last.descriptors
            );
            // Ngưỡng 30 điểm match để xác nhận nối vòng
            if (loopRes != null && loopRes.inlierMatches.size() > 30) {
                isCyclic = true;
            }
        }
        System.out.println("      -> Cấu trúc: " + (isCyclic ? "VÒNG TRÒN (Circle)" : "ĐƯỜNG THẲNG (Strip)"));

        // --- BƯỚC 2: TÍNH VECTƠ HƯỚNG GIỮA 2 ẢNH ĐẦU TIÊN ---
        // Lấy mẫu cặp ảnh [0] -> [1] để xem dòng chảy đang đi về đâu
        ImageNode n0 = nodes.get(0);
        ImageNode n1 = nodes.get(1);

        FeatureMatcherWrapper.MatchResult res = matcher.match(
                n0.keypoints, n0.descriptors,
                n1.keypoints, n1.descriptors
        );

        // Nếu không đủ điểm match để tính vector -> Giữ nguyên (An toàn)
        if (res == null || res.inlierMatches.size() < 10) {
            System.out.println("      -> [WARN] Không đủ điểm match giữa n0-n1 để xác định hướng. Giữ nguyên.");
            return;
        }

        // Tính Vector trung bình: V = Coord(n0) - Coord(n1)
        // Lưu ý: n0 là Query, n1 là Train
        double sumVx = 0, sumVy = 0;
        for (DMatch m : res.inlierMatches) {
            SiftKeyPoint p1 = n0.keypoints.get(m.queryIdx()); // Điểm trên n0
            SiftKeyPoint p2 = n1.keypoints.get(m.trainIdx()); // Điểm trên n1

            // Công thức: Vector = P_truoc - P_sau
            sumVx += (p1.x - p2.x);
            sumVy += (p1.y - p2.y);
        }

        double avgVx = sumVx / res.inlierMatches.size();
        double avgVy = sumVy / res.inlierMatches.size();

        // --- BƯỚC 3: PHÂN TÍCH CHIỀU (DỰA TRÊN QUY CHUẨN PANORAMA) ---
        boolean isVerticalFlow = Math.abs(avgVy) > Math.abs(avgVx);
        boolean needReverse = false;
        String dirInfo = "";

        if (isVerticalFlow) {
            // == HƯỚNG DỌC (Vertical/Diagonal) ==
            // Quy chuẩn: TRÊN -> DƯỚI (Top to Bottom)
            // Logic: Ảnh n0 (Trên) đè lên n1 (Dưới).
            // Vùng chồng lấn là: Đáy n0 (Y lớn) trùng với Đỉnh n1 (Y nhỏ).
            // => Vy = Y_n0 - Y_n1 > 0 (DƯƠNG)
            if (avgVy > 0) {
                dirInfo = "Trên -> Dưới (Đúng chiều)";
                needReverse = false;
            } else {
                dirInfo = "Dưới -> Trên (NGƯỢC CHIỀU)";
                needReverse = true;
            }
        } else {
            // == HƯỚNG NGANG (Horizontal) ==
            // Quy chuẩn: TRÁI -> PHẢI (Left to Right)
            // Logic: Ảnh n0 (Trái) đè lên n1 (Phải).
            // Vùng chồng lấn là: Mép Phải n0 (X lớn) trùng với Mép Trái n1 (X nhỏ).
            // => Vx = X_n0 - X_n1 > 0 (DƯƠNG)
            if (avgVx > 0) {
                dirInfo = "Trái -> Phải (Đúng chiều)";
                needReverse = false;
            } else {
                dirInfo = "Phải -> Trái (NGƯỢC CHIỀU)";
                needReverse = true;
            }
        }

        System.out.println(String.format("      -> Vectơ n0->n1: Vx=%.1f, Vy=%.1f. Kết luận: %s", avgVx, avgVy, dirInfo));

        // --- BƯỚC 4: THỰC HIỆN HÀNH ĐỘNG ---
        if (needReverse) {
            System.out.println("      -> ĐANG ĐẢO NGƯỢC DANH SÁCH (REVERSE)...");
            Collections.reverse(nodes);

            // In lại thứ tự để kiểm tra
            System.out.println("      -> Thứ tự sau khi đảo:");
            for (int i = 0; i < nodes.size(); i++) {
                System.out.println("         [" + i + "] " + nodes.get(i).filename);
            }
        } else {
            System.out.println("      -> Giữ nguyên thứ tự.");
        }

        // GHI CHÚ VỚI VÒNG TRÒN (IS_CYCLIC = TRUE):
        // Vì không dùng tên file, ta không biết đâu là "ảnh số 1".
        // Ta chấp nhận điểm cắt do Chain Sort tìm ra (thường là nơi match yếu nhất hoặc điểm bắt đầu thuật toán).
        // Tuy nhiên, việc reverse ở trên đảm bảo vòng tròn quay ĐÚNG CHIỀU (thường là chiều kim đồng hồ hoặc trên xuống dưới).
    }
    */

    // Class phụ trợ để quản lý cạnh nối
    static class Edge implements Comparable<Edge> {
        int u, v;
        int score;
        public Edge(int u, int v, int score) {
            this.u = u; this.v = v; this.score = score;
        }
        @Override
        public int compareTo(Edge o) {
            return Integer.compare(o.score, this.score); // Giảm dần
        }
    }

    private static void sortImagesByDirection(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        System.out.println("   [Sort] Chiến lược: Maximum Spanning Chain (Graph Based)...");
        int n = nodes.size();
        if (n < 2) return;

        // BƯỚC 1: TÍNH TOÁN TẤT CẢ CÁC CẶP (ALL-PAIRS MATCHING)
        List<Edge> allEdges = new ArrayList<>();
        // Cache để dùng lại ở bước Reorient
        Map<String, FeatureMatcherWrapper.MatchResult> matchCache = new HashMap<>();

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) { // Chỉ tính i < j để không lặp
                ImageNode ni = nodes.get(i);
                ImageNode nj = nodes.get(j);

                if (ni.descriptors == null || ni.descriptors.empty() ||
                        nj.descriptors == null || nj.descriptors.empty()) continue;

                FeatureMatcherWrapper.MatchResult res = matcher.match(
                        ni.keypoints, ni.descriptors,
                        nj.keypoints, nj.descriptors
                );

                if (res != null && res.inlierMatches.size() > 30) {
                    allEdges.add(new Edge(i, j, res.inlierMatches.size()));
                    matchCache.put(ni.id + "_" + nj.id, res);
                    matchCache.put(nj.id + "_" + ni.id, res); // Lưu cả 2 chiều
                }
            }
        }

        // Sắp xếp các cạnh theo điểm số giảm dần (Mạnh nhất lên đầu)
        Collections.sort(allEdges);

        // BƯỚC 2: XÂY DỰNG ĐỒ THỊ (KRUSKAL-LIKE)
        // Mỗi node tối đa 2 hàng xóm (Degree <= 2)
        int[] degree = new int[n];
        // Danh sách kề: adj[i] chứa danh sách các node nối với i
        List<List<Integer>> adj = new ArrayList<>(n);
        for(int i=0; i<n; i++) adj.add(new ArrayList<>());

        // Union-Find đơn giản để check chu trình
        int[] parent = new int[n];
        for(int i=0; i<n; i++) parent[i] = i;
        int edgesCount = 0;

        for (Edge e : allEdges) {
            int u = e.u;
            int v = e.v;

            // Điều kiện 1: Mỗi node chỉ được nối tối đa 2 cạnh
            if (degree[u] >= 2 || degree[v] >= 2) continue;

            // Điều kiện 2: Không tạo chu trình sớm (trừ khi khép vòng cuối cùng)
            int rootU = findRoot(parent, u);
            int rootV = findRoot(parent, v);

            if (rootU != rootV) {
                // Nối 2 thành phần rời rạc
                adj.get(u).add(v);
                adj.get(v).add(u);
                degree[u]++;
                degree[v]++;
                parent[rootU] = rootV; // Union
                edgesCount++;
            } else {
                // Nếu cùng gốc -> Tạo chu trình (Vòng tròn)
                // Chỉ cho phép tạo vòng tròn nếu đã đi qua TẤT CẢ các điểm (edgesCount == n - 1)
                if (edgesCount == n - 1) {
                    adj.get(u).add(v);
                    adj.get(v).add(u);
                    degree[u]++;
                    degree[v]++;
                    edgesCount++;
                    System.out.println("      -> Đã khép vòng tròn (Circle Closed) tại cặp: "
                            + nodes.get(u).filename + " - " + nodes.get(v).filename);
                }
            }
        }

        // BƯỚC 3: DUYỆT ĐỒ THỊ ĐỂ TẠO CHUỖI
        // Tìm điểm bắt đầu: Node có bậc 1 (đầu mút). Nếu không có (toàn bậc 2) thì là vòng tròn -> chọn node 0.
        int startNode = -1;
        for (int i = 0; i < n; i++) {
            if (degree[i] == 1) {
                startNode = i;
                break;
            }
        }
        if (startNode == -1) startNode = 0; // Trường hợp vòng tròn kín

        List<ImageNode> sortedNodes = new ArrayList<>();
        boolean[] visited = new boolean[n];
        int curr = startNode;

        // DFS/BFS để traverse
        while (curr != -1) {
            visited[curr] = true;
            sortedNodes.add(nodes.get(curr));

            int nextNode = -1;
            for (int neighbor : adj.get(curr)) {
                if (!visited[neighbor]) {
                    nextNode = neighbor;
                    break;
                }
            }
            curr = nextNode;
        }

        // Nếu traverse thiếu (do đồ thị bị ngắt quãng), ném phần còn lại vào cuối (ít khi xảy ra với code này)
        if (sortedNodes.size() < n) {
            System.out.println("      [WARN] Đồ thị không liên thông hoàn toàn. Ghép phần dư...");
            for(int i=0; i<n; i++) if(!visited[i]) sortedNodes.add(nodes.get(i));
        }

        // Cập nhật lại list nodes
        nodes.clear();
        nodes.addAll(sortedNodes);

        System.out.println("   -> Thứ tự sau Graph Sort (Topo chuẩn):");
        for(ImageNode node : nodes) System.out.println("      " + node.filename);

        // BƯỚC 4: TÁI ĐỊNH HƯỚNG (REORIENT)
        // 1. Kiểm tra vòng tròn & Xoay về mỏ neo (để số 0 lên đầu)
        // 2. Kiểm tra chiều vật lý (để đảo ngược nếu cần)
        reorientCyclicList(nodes, matcher);
    }

    // Helper cho Union-Find
    private static int findRoot(int[] parent, int i) {
        if (parent[i] == i) return i;
        return parent[i] = findRoot(parent, parent[i]);
    }

    private static void reorientCyclicList(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        if (nodes.size() < 2) return;
        System.out.println("   [Reorient] Đang phân tích hướng và điểm bắt đầu...");

        // 1. Kiểm tra Vòng tròn
        boolean isCyclic = false;
        if (nodes.size() >= 4) {
            ImageNode first = nodes.get(0);
            ImageNode last = nodes.get(nodes.size() - 1);
            FeatureMatcherWrapper.MatchResult res = matcher.match(first.keypoints, first.descriptors, last.keypoints, last.descriptors);
            if (res != null && res.inlierMatches.size() > 30) isCyclic = true;
        }
        System.out.println("      -> Cấu trúc: " + (isCyclic ? "VÒNG TRÒN" : "ĐƯỜNG THẲNG"));

        // 2. TÌM ANCHOR (CHỈ ĐỂ XOAY VỊ TRÍ HIỂN THỊ, KHÔNG ẢNH HƯỞNG TOPO)
        // Mục đích: Đưa medium00 hoặc ảnh nhỏ nhất về đầu danh sách cho đẹp
        if (isCyclic) {
            int anchorIdx = findBestAnchorIndex(nodes); // Hàm tìm số nhỏ nhất
            if (anchorIdx > 0) {
                System.out.println("      -> Xoay vòng để đưa " + nodes.get(anchorIdx).filename + " về đầu.");
                Collections.rotate(nodes, -anchorIdx);
            }
        }

        // 3. KIỂM TRA CHIỀU VẬT LÝ (Physics Vector Check)
        ImageNode n0 = nodes.get(0);
        ImageNode n1 = nodes.get(1);
        FeatureMatcherWrapper.MatchResult res = matcher.match(n0.keypoints, n0.descriptors, n1.keypoints, n1.descriptors);

        if (res != null && res.inlierMatches.size() > 10) {
            double sumVx = 0, sumVy = 0;
            for (DMatch m : res.inlierMatches) {
                SiftKeyPoint p1 = n0.keypoints.get(m.queryIdx());
                SiftKeyPoint p2 = n1.keypoints.get(m.trainIdx());
                sumVx += (p1.x - p2.x);
                sumVy += (p1.y - p2.y);
            }
            double avgVx = sumVx / res.inlierMatches.size();
            double avgVy = sumVy / res.inlierMatches.size();

            boolean isVertical = Math.abs(avgVy) > Math.abs(avgVx);
            boolean needReverse = false;

            if (isVertical) {
                // Dọc: Nếu Vy < 0 (Dưới->Trên) => Ngược
                if (avgVy < 0) needReverse = true;
            } else {
                // Ngang: Nếu Vx < 0 (Phải->Trái) => Ngược
                if (avgVx < 0) needReverse = true;
            }

            if (needReverse) {
                System.out.println("      -> Phát hiện ngược chiều vật lý. Đảo ngược danh sách.");
                Collections.reverse(nodes);
            } else {
                System.out.println("      -> Chiều vật lý đã đúng.");
            }
        }

        System.out.println("   -> Thứ tự cuối cùng:");
        for(ImageNode node : nodes) System.out.println("      " + node.filename);
    }

    // Helper tìm ảnh có số nhỏ nhất để làm điểm bắt đầu hiển thị
    private static int findBestAnchorIndex(List<ImageNode> nodes) {
        int minNum = Integer.MAX_VALUE;
        int minIdx = 0;
        for(int i=0; i<nodes.size(); i++) {
            String name = nodes.get(i).filename;
            // Extract number logic
            try {
                String numStr = name.replaceAll("[^0-9]", "");
                if (!numStr.isEmpty()) {
                    int num = Integer.parseInt(numStr);
                    if (num < minNum) {
                        minNum = num;
                        minIdx = i;
                    }
                }
            } catch (Exception e) {}
        }
        return minIdx;
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

    private static void applyWarpStrategy(List<ImageNode> nodes, boolean warp) {
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
//        shouldWarp = false; // Force Planar
//        shouldWarp = true;  // Force Cylindrical
        shouldWarp =warp;

        if (shouldWarp) {
            System.out.println("   -> Strategy: CYLINDRICAL WARP (Pure Panorama)");
            Mat tmp = nodes.get(0).img;
            double f = tmp.cols() * 1.0;

            for (ImageNode node : nodes) {
                Mat warped = CylindricalWarper.warp(node.img, f);
                // Mat warped = SphericalWarper.warp(node.img, f);

                // Tính lại SIFT trên ảnh warped
                Mat gray = new Mat();
                cvtColor(warped, gray, COLOR_BGR2GRAY);
                Mat fGray = new Mat();
                gray.convertTo(fGray, CV_32F, 1.0 / 255.0, 0.0);

                SiftDetector detector = new SiftDetector();
                ScaleSpace ss = new ScaleSpace();
                List<MatVector> gaussianPyramid = ss.buildGaussianPyramid(fGray);
                List<MatVector> dogPyramid = ss.buildDoGPyramid(gaussianPyramid);
                List<SiftKeyPoint> kps = detector.run(gaussianPyramid, dogPyramid);

                if (kps.size() > 4000) kps = new ArrayList<>(kps.subList(0, 4000));

                FeatureMatcherWrapper tempMatcher = new FeatureMatcherWrapper();
                Mat desc = tempMatcher.convertDescriptors(kps);

                node.img = warped;
                node.keypoints = kps;
                node.descriptors = desc;  // ← QUAN TRỌNG

                gray.release();
                fGray.release();
                gaussianPyramid.clear();
                dogPyramid.clear();
            }
        } else {
            System.out.println("   -> Strategy: PLANAR (Mixed Directions)");
        }
    }

    // --- PHIÊN BẢN GLOBAL OPTIMIZATION (MÔ PHỎNG BUNDLE ADJUSTMENT) ---
    private static void computeTransforms(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        if (nodes.isEmpty()) return;
        int n = nodes.size();
        System.out.println("   -> [Step 4] Calculating Global Transforms (Center Anchor & Adaptive Straightening)...");

        // 1. CHỌN ANCHOR Ở GIỮA
        // Thay vì lấy nodes.get(0), ta lấy phần tử giữa để chia đều sai số sang 2 bên
        int mid = n / 2;
        nodes.get(mid).globalTransform = Mat.eye(3, 3, CV_64F).asMat();
        System.out.println("      -> Anchor Image: " + nodes.get(mid).filename + " (Index " + mid + ")");

        List<Mat> relativeTransforms = new ArrayList<>(Collections.nCopies(n - 1, null));

        // 2. Tính Pairwise Transform (Giữ nguyên - Tính hết các cặp liền kề)
        for (int i = 0; i < n - 1; i++) {
            ImageNode curr = nodes.get(i);
            ImageNode next = nodes.get(i + 1);
            FeatureMatcherWrapper.MatchResult res = matcher.match(curr.keypoints, curr.descriptors, next.keypoints, next.descriptors);

            Mat T = null;
            if (res != null && res.inlierMatches.size() > 10) {
                T = TransformEstimator.estimateAffine(res.srcPoints, res.dstPoints);
            }
            if (T == null || !TransformEstimator.isTransformValid(T)) {
                T = Mat.eye(3, 3, CV_64F).asMat();
            }
            relativeTransforms.set(i, T);
        }

        // 3. LAN TRUYỀN BIẾN ĐỔI TỪ GIỮA RA 2 ĐẦU (Center Propagation)

        // A. Lan truyền về phía SAU (Từ mid -> n-1)
        // Logic cũ: T_global_(i+1) = T_global_i * inv(T_i_i+1)
        for (int i = mid; i < n - 1; i++) {
            Mat T_rel = relativeTransforms.get(i);
            Mat T_inv = new Mat();
            invert(T_rel, T_inv, DECOMP_LU);

            Mat g = new Mat();
            gemm(nodes.get(i).globalTransform, T_inv, 1.0, new Mat(), 0.0, g);
            nodes.get(i + 1).globalTransform = g;
        }

        // B. Lan truyền về phía TRƯỚC (Từ mid -> 0)
        // Logic mới: T_global_i = T_global_(i+1) * T_i_i+1
        // (Đi lùi thì nhân với ma trận xuôi, không cần nghịch đảo)
        for (int i = mid - 1; i >= 0; i--) {
            Mat T_rel = relativeTransforms.get(i); // Transform từ i -> i+1

            Mat g = new Mat();
            // nodes[i+1] đã có tọa độ, nhân với T_rel để ra tọa độ của nodes[i]
            gemm(nodes.get(i + 1).globalTransform, T_rel, 1.0, new Mat(), 0.0, g);
            nodes.get(i).globalTransform = g;
        }

        // 4. STRAIGHTENING THÔNG MINH (ADAPTIVE)

        // A. Lấy tọa độ tâm và biên
        double[] centersX = new double[n];
        double[] centersY = new double[n];
        double minX = Double.MAX_VALUE, maxX = -Double.MAX_VALUE;
        double minY = Double.MAX_VALUE, maxY = -Double.MAX_VALUE;

        for(int i=0; i<n; i++) {
            DoubleIndexer idx = nodes.get(i).globalTransform.createIndexer();
            centersX[i] = idx.get(0, 2);
            centersY[i] = idx.get(1, 2);

            if(centersX[i] < minX) minX = centersX[i];
            if(centersX[i] > maxX) maxX = centersX[i];
            if(centersY[i] < minY) minY = centersY[i];
            if(centersY[i] > maxY) maxY = centersY[i];
        }

        double spanX = Math.abs(maxX - minX);
        double spanY = Math.abs(maxY - minY);

        // Quyết định chiều chính của Panorama
        boolean isVertical = spanY > spanX;

        System.out.println("      -> Detected Shape: " + (isVertical ? "VERTICAL (Dọc)" : "HORIZONTAL (Ngang)"));

        if (!isVertical) {
            // --- XỬ LÝ NGANG ---
            // Tính Slope Y theo X
            double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            for (int i = 0; i < n; i++) {
                sumX += centersX[i]; sumY += centersY[i];
                sumXY += centersX[i] * centersY[i]; sumXX += centersX[i] * centersX[i];
            }
            double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX + 1e-6);

            if (Math.abs(slope) > 0.2) {
                System.out.println("      -> Slope quá lớn (" + slope + ") -> Tắt Straightening.");
                slope = 0;
            } else {
                System.out.println("      -> Horizontal Slope: " + slope);
            }

            for (int i = 0; i < n; i++) {
                Mat G = nodes.get(i).globalTransform;
                DoubleIndexer idx = G.createIndexer();
                double currTy = idx.get(1, 2);

                // Xoay quanh trục (0,0) - Tâm của ảnh Anchor
                double correction = - (slope * centersX[i]);
                idx.put(1, 2, currTy + correction);
            }

        } else {
            // --- XỬ LÝ DỌC ---
            // Tính Slope X theo Y
            double sumY = 0, sumX = 0, sumYX = 0, sumYY = 0;
            for (int i = 0; i < n; i++) {
                sumY += centersY[i]; sumX += centersX[i];
                sumYX += centersY[i] * centersX[i]; sumYY += centersY[i] * centersY[i];
            }
            double slope = (n * sumYX - sumY * sumX) / (n * sumYY - sumY * sumY + 1e-6);

            if (Math.abs(slope) > 0.2) {
                System.out.println("      -> Slope quá lớn (" + slope + ") -> Tắt Straightening.");
                slope = 0;
            } else {
                System.out.println("      -> Vertical Slope: " + slope);
            }

            for (int i = 0; i < n; i++) {
                Mat G = nodes.get(i).globalTransform;
                DoubleIndexer idx = G.createIndexer();
                double currTx = idx.get(0, 2);

                // Xoay quanh trục (0,0) - Tâm của ảnh Anchor
                double correction = - (slope * centersY[i]);
                idx.put(0, 2, currTx + correction);
            }
        }
    }
    /* private static void computeTransforms(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
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
    }*/
}