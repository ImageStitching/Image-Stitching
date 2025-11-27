package com.stitching.imageStitching;

import com.stitching.imageStitching.blender.ImageBlenderFAST;
import com.stitching.imageStitching.matchAndTransform.FeatureMatcherWrapper;
import com.stitching.imageStitching.matchAndTransform.TransformEstimator;
import com.stitching.imageStitching.warper.CylindricalWarper;
import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftDetector;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
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
 * Bản này chỉ ghép ảnh ngang là chính và chưa thể xác định Hướng dọc, chéo, Sắp xếp lại các ảnh
 */
public class CylinderStitcher {
    private static final Path OUTPUT_PATH = Paths.get("src", "main", "resources", "static", "output");
    private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "example_data","CMU0");
    //private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "ptit");

    /*
    public static class ImageNode {
        public int id;
        public String filename;
        public Mat img;
        public List<SiftKeyPoint> keypoints;
        public Mat descriptors;
        public Mat globalTransform;

        public ImageNode(int id, String name, Mat img, List<SiftKeyPoint> kp, Mat desc) {
            this.id = id; this.filename = name; this.img = img;
            this.keypoints = kp; this.descriptors = desc;
            this.globalTransform = Mat.eye(3, 3, CV_64F).asMat();
        }
        public ImageNode(int id, String name, Mat img, List<SiftKeyPoint> kp, Mat desc, Mat globalTransform) {
            this.id = id; this.filename = name; this.img = img;
            this.keypoints = kp; this.descriptors = desc;
            this.globalTransform = globalTransform;
        }
    }
 */

    public static void main(String[] args) {
        System.out.println("=== OPENPANO JAVA MAIN ===");

        File folder = INPUT_PATH.toFile();
        File[] files = folder.listFiles((d, n) -> n.toLowerCase().endsWith(".jpg") || n.toLowerCase().endsWith(".png"));
        if (files == null || files.length == 0) return;

        // OpenPano Cylinder Mode yêu cầu ảnh có thứ tự
        Arrays.sort(files, Comparator.comparing(File::getName));

        List<ImageNode> nodes = new ArrayList<>();
        FeatureMatcherWrapper matcher = new FeatureMatcherWrapper();

        // --- BƯỚC 1: PRE-WARP ---
        Mat tmp = imread(files[0].getAbsolutePath());
        double f = tmp.cols() * 1.0; 
        tmp.release();
        
        System.out.println("\n[Step 1] Cylindrical Warping & SIFT...");
        for (int i = 0; i < files.length; i++) {
            System.out.println("-> " + files[i].getName());
            Mat raw = imread(files[i].getAbsolutePath());
            if (raw.empty()) continue;

            Mat warped = CylindricalWarper.warp(raw, f);
            
            // Tính SIFT (Code của bạn)
            Mat gray = new Mat(); cvtColor(warped, gray, COLOR_BGR2GRAY);
            Mat fGray = new Mat(); gray.convertTo(fGray, CV_32F, 1.0/255.0, 0.0);
            
            SiftDetector detector = new SiftDetector();
            ScaleSpace ss = new ScaleSpace();
            List<MatVector> gaussianPyramid = ss.buildGaussianPyramid(fGray);
            List<MatVector> dogPyramid = ss.buildDoGPyramid(gaussianPyramid);
            List<SiftKeyPoint> kps = detector.run(gaussianPyramid, dogPyramid);
            
            // Limit keypoints
            if(kps.size() > 4000) kps = new ArrayList<>(kps.subList(0, 4000));
            
            Mat desc = matcher.convertDescriptors(kps);
            nodes.add(new ImageNode(i, files[i].getName(), warped, kps, desc));
            
            gray.release(); fGray.release(); gaussianPyramid.clear(); dogPyramid.clear();
        }

        if (nodes.size() < 2) return;

        // --- BƯỚC 2: CENTER-OUT MATCHING (cylstitcher.cc) ---
        System.out.println("\n[Step 2] Calculating Transforms...");
        computeTransforms(nodes, matcher);

        // --- BƯỚC 3: BLENDING ---
        System.out.println("\n[Step 3] Blending...");
        Mat result = ImageBlenderFAST.blend(nodes);

        if (result != null) {
            String imgOut = "openpano_java_blender";
            String outName = imgOut + ".jpg";
            imwrite(OUTPUT_PATH.resolve(outName).toString(), result);
            System.out.println(">>> DONE: " + outName);
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
//    private static void computeTransforms(List<ImageNode> nodes, FeatureMatcherWrapper matcher) {
//        int n = nodes.size();
//        int mid = n / 2;
//        ImageNode center = nodes.get(mid);
//        System.out.println("   -> Anchor: " + center.filename);
//
//        // Lan truyền sang Phải
//        for (int i = mid; i < n - 1; i++) {
//            ImageNode curr = nodes.get(i);
//            ImageNode next = nodes.get(i + 1);
//
//            FeatureMatcherWrapper.MatchResult res = matcher.match(curr.keypoints, curr.descriptors, next.keypoints, next.descriptors);
//
//            if (res != null) {
//                // Tìm Affine từ Curr -> Next
//                Mat T_curr_next = TransformEstimator.estimateAffine(res.srcPoints, res.dstPoints);
//                if (T_curr_next != null && TransformEstimator.isTransformValid(T_curr_next)) {
//                    // Global_Next = Global_Curr * inv(T_curr_next)
//                    Mat T_next_curr = new Mat();
//                    invert(T_curr_next, T_next_curr, DECOMP_LU);
//
//                    Mat g = new Mat();
//                    gemm(curr.globalTransform, T_next_curr, 1.0, new Mat(), 0.0, g);
//                    next.globalTransform = g;
//                } else {
//                    next.globalTransform = curr.globalTransform.clone(); // Fallback
//                }
//            }
//        }
//
//        // Lan truyền sang Trái
//        for (int i = mid; i > 0; i--) {
//            ImageNode curr = nodes.get(i);
//            ImageNode prev = nodes.get(i - 1);
//
//            FeatureMatcherWrapper.MatchResult res = matcher.match(prev.keypoints, prev.descriptors, curr.keypoints, curr.descriptors);
//
//            if (res != null) {
//                // Tìm Affine từ Prev -> Curr
//                Mat T_prev_curr = TransformEstimator.estimateAffine(res.srcPoints, res.dstPoints);
//                if (T_prev_curr != null && TransformEstimator.isTransformValid(T_prev_curr)) {
//                    // Global_Prev = Global_Curr * T_Prev_Curr
//                    Mat g = new Mat();
//                    gemm(curr.globalTransform, T_prev_curr, 1.0, new Mat(), 0.0, g);
//                    prev.globalTransform = g;
//                } else {
//                    prev.globalTransform = curr.globalTransform.clone();
//                }
//            }
//        }
//    }
}