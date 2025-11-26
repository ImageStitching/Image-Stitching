package com.stitching.imageStitching;

import com.stitching.imageStitching.blender.ImageBlenderFAST;
import com.stitching.imageStitching.matchAndTransform.FeatureMatcherWrapper;
import com.stitching.imageStitching.matchAndTransform.TransformEstimator;
import com.stitching.imageStitching.warper.CylindricalWarper;
import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftConfig;
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
    private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "example_data","myself");
    //private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "scene_vertical");

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
        SiftConfig.DOUBLE_IMAGE_SIZE = false;
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
        double sumVx = 0, sumVy = 0;
        double sumAbsVx = 0, sumAbsVy = 0;

        int count = Math.min(matches.size(), 200);

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

        double ratio = Math.min(avgAbsVx, avgAbsVy) / (Math.max(avgAbsVx, avgAbsVy) + 0.001);
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
        }
        else if (avgAbsVx > avgAbsVy) {
            rel.direction = StitchDirection.HORIZONTAL;
            rel.debugMsg = "Ngang (Horizontal). " + vectorInfo;
            if (avgVx < 0) {
                rel.needSwap = true;
                rel.debugMsg += " -> SWAP (Phải->Trái)";
            }
        }
        else {
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

    private static void sortImagesByDirection(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
        System.out.println("   [Sort] Đang xác định chiến lược sắp xếp...");

        // --- CHIẾN LƯỢC 1: ƯU TIÊN SẮP XẾP THEO TÊN FILE (Nếu có số thứ tự) ---
        // Đây là cách an toàn nhất cho người dùng chụp ảnh theo chuỗi (medium01 -> medium11)
//        boolean hasNumbers = nodes.stream().allMatch(n -> n.filename.matches(".*\\d+.*"));
//        if (hasNumbers) {
//            System.out.println("      -> Phát hiện tên file có số thứ tự. Sử dụng Sort theo tên (An toàn nhất).");
//            nodes.sort(Comparator.comparing(n -> extractNumber(n.filename)));
//            for (int i = 0; i < nodes.size(); i++) {
//                System.out.println("      [" + i + "] " + nodes.get(i).filename);
//            }
//            return;
//        }

        // --- CHIẾN LƯỢC 2: CHAIN SORT (MẮT XÍCH) - Dùng khi tên file lộn xộn ---
        System.out.println("      -> Chuyển sang Chain Sorting (Dựa trên độ mạnh liên kết)...");

        // Bước 1: Tìm cặp ảnh khởi đầu tốt nhất (Cặp có nhiều match nhất trong toàn bộ tập)
        // Lý do: Để tránh bắt đầu từ một ảnh nhiễu. Ta tìm "xương sống" của panorama trước.
        int bestA = -1, bestB = -1;
        int maxMatches = 0;
        StitchDirection globalDir = StitchDirection.UNKNOWN;

        // Ma trận lưu số lượng match giữa các cặp để dùng lại
        int[][] matchMatrix = new int[nodes.size()][nodes.size()];

        for (int i = 0; i < nodes.size(); i++) {
            for (int j = 0; j < nodes.size(); j++) {
                if (i == j) continue;
                FeatureMatcherWrapper.MatchResult res = matcher.match(nodes.get(i).keypoints, nodes.get(i).descriptors, nodes.get(j).keypoints, nodes.get(j).descriptors);

                // [QUAN TRỌNG] Tăng ngưỡng lọc nhiễu lên 30 để loại bỏ match xa (ví dụ medium01 vs medium04)
                if (res != null && res.inlierMatches.size() > 30) {
                    matchMatrix[i][j] = res.inlierMatches.size();
                    if (matchMatrix[i][j] > maxMatches) {
                        maxMatches = matchMatrix[i][j];
                        bestA = i;
                        bestB = j;
                        // Xác định chiều chung của panorama dựa trên cặp mạnh nhất này
                        ImageRelation rel = analyzeMatchDirection(res.inlierMatches, nodes.get(i), nodes.get(j));
                        if (!rel.needSwap) { globalDir = rel.direction; }
                    }
                }
            }
        }

        if (bestA == -1) {
            System.out.println("      [WARNING] Không tìm thấy mối liên kết nào đủ mạnh > 30 matches. Giữ nguyên thứ tự.");
            return;
        }

        // Bước 2: Xây dựng chuỗi từ cặp mạnh nhất (bestA -> bestB)
        // Ta sẽ mở rộng chuỗi về 2 phía: ... <- LeftOfA <- [A -> B] -> RightOfB -> ...
        LinkedList<ImageNode> chain = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();

        ImageNode nodeA = nodes.get(bestA);
        ImageNode nodeB = nodes.get(bestB);

        // Xác định thứ tự đúng của A và B dựa trên chiều match
        FeatureMatcherWrapper.MatchResult resAB = matcher.match(nodeA.keypoints, nodeA.descriptors, nodeB.keypoints, nodeB.descriptors);
        ImageRelation relAB = analyzeMatchDirection(resAB.inlierMatches, nodeA, nodeB);

        if (relAB.needSwap) {
            chain.add(nodeB); chain.add(nodeA);
            visited.add(nodeB.id); visited.add(nodeA.id);
            System.out.println("      Vote: " + nodeB.filename + " < " + nodeA.filename);
        } else {
            chain.add(nodeA); chain.add(nodeB);
            visited.add(nodeA.id); visited.add(nodeB.id);
            System.out.println("      Vote: " + nodeA.filename + " < " + nodeB.filename);
        }

        // Mở rộng về phía bên PHẢI (Tìm thằng khớp với đuôi chuỗi nhất)
        while (visited.size() < nodes.size()) {
            ImageNode tail = chain.getLast();
            int bestNextIdx = -1;
            int maxScore = 0;

            for (int i = 0; i < nodes.size(); i++) {
                if (visited.contains(nodes.get(i).id)) continue;

                // Check match Tail -> Candidate
                int score = getMatchScore(matcher, tail, nodes.get(i));
                if (score > maxScore && score > 30) { // Ngưỡng 30
                    maxScore = score;
                    bestNextIdx = i;
                }
            }

            if (bestNextIdx != -1) {
                chain.addLast(nodes.get(bestNextIdx));
                visited.add(nodes.get(bestNextIdx).id);
            } else {
                break; // Không tìm thấy ai nối tiếp
            }
        }

        // Mở rộng về phía bên TRÁI (Tìm thằng khớp với đầu chuỗi nhất)
        while (visited.size() < nodes.size()) {
            ImageNode head = chain.getFirst();
            int bestPrevIdx = -1;
            int maxScore = 0;

            for (int i = 0; i < nodes.size(); i++) {
                if (visited.contains(nodes.get(i).id)) continue;

                // Check match Candidate -> Head
                // Lưu ý: Phải check chiều ngược lại xem nó có khớp Head không
                int score = getMatchScore(matcher, nodes.get(i), head);
                if (score > maxScore && score > 30) {
                    maxScore = score;
                    bestPrevIdx = i;
                }
            }

            if (bestPrevIdx != -1) {
                chain.addFirst(nodes.get(bestPrevIdx));
                visited.add(nodes.get(bestPrevIdx).id);
            } else {
                break;
            }
        }

        // Cập nhật lại list nodes theo chuỗi đã tìm được
        nodes.clear();
        nodes.addAll(chain);

        System.out.println("   -> Thứ tự sau sắp xếp (Chain Sort):");
        for (int i = 0; i < nodes.size(); i++) {
            System.out.println("      [" + i + "] " + nodes.get(i).filename);
        }

        // --- BƯỚC 3: TÁI ĐỊNH HƯỚNG VÒNG TRÒN (QUAN TRỌNG NHẤT) ---
        // Tìm node có ID gốc nhỏ nhất (tức là ảnh input đầu tiên) và xoay list để nó về đầu.
        reorientCyclicList(nodes);

        System.out.println("   -> Thứ tự cuối cùng");
        for (int i = 0; i < nodes.size(); i++) {
            System.out.println("      [" + i + "] " + nodes.get(i).filename);
        }
    }

    // --- HÀM MỚI: XOAY VÒNG DANH SÁCH ---
    private static void reorientCyclicList(List<ImageNode> nodes) {
        if (nodes.isEmpty()) return;

        // 1. Tìm vị trí hiện tại của ảnh có ID nhỏ nhất (medium01 có id=0, medium11 có id=10)
        int minIdIndex = -1;
        int minId = Integer.MAX_VALUE;

        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).id < minId) {
                minId = nodes.get(i).id;
                minIdIndex = i;
            }
        }

        // 2. Nếu ảnh minId không nằm ở đầu (index 0), thực hiện xoay
        if (minIdIndex > 0) {
            System.out.println("      -> Phát hiện chuỗi bị lệch pha (Bắt đầu từ " + nodes.get(0).filename + ").");
            System.out.println("      -> Đang xoay vòng để đưa " + nodes.get(minIdIndex).filename + " về đầu...");

            // Collections.rotate: Xoay danh sách.
            // distance = -minIdIndex nghĩa là dịch trái minIdIndex bước.
            // Ví dụ: List [3,4,5,1,2] (min tại index 3). Rotate -3 => [1,2,3,4,5].
            Collections.rotate(nodes, -minIdIndex);
        }
    }

    // Helper tách số từ tên file (dùng cho Chiến lược 1)
    private static int extractNumber(String name) {
        try {
            String number = name.replaceAll("[^0-9]", "");
            return number.isEmpty() ? 0 : Integer.parseInt(number);
        } catch (Exception e) { return 0; }
    }

    // Helper tính điểm match (dùng cho Chiến lược 2)
    private static int getMatchScore(FeatureMatcherWrapper matcher, ImageNode n1, ImageNode n2) {
        FeatureMatcherWrapper.MatchResult res = matcher.match(n1.keypoints, n1.descriptors, n2.keypoints, n2.descriptors);
        if (res == null) return 0;

        // Kiểm tra logic hướng: Nếu n1 -> n2 mà lại bắt SWAP (tức n2 đứng trước n1) thì đây không phải là nối tiếp đúng chiều
        // Tuy nhiên trong Chain Sort đơn giản, ta chỉ cần độ mạnh match là đủ, hướng sẽ tự khớp theo chuỗi.
        return res.inlierMatches.size();
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
        // shouldWarp = false; // Force Planar
        shouldWarp = true;  // Force Cylindrical

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