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

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class CylinderStitcher {
    private static final Path OUTPUT_PATH = Paths.get("src", "main", "resources", "static", "output");
    private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "example_data","myself");
    //private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "scene_vertical");

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
            List<SiftKeyPoint> kps = detector.run(
                ss.buildGaussianPyramid(fGray), 
                ss.buildDoGPyramid(ss.buildGaussianPyramid(fGray))
            );
            
            // Limit keypoints
            if(kps.size() > 4000) kps = new ArrayList<>(kps.subList(0, 4000));
            
            Mat desc = matcher.convertDescriptors(kps);
            nodes.add(new ImageNode(i, files[i].getName(), warped, kps, desc));
            
            gray.release(); fGray.release();
        }

        if (nodes.size() < 2) return;

        // --- BƯỚC 2: CENTER-OUT MATCHING (cylstitcher.cc) ---
        System.out.println("\n[Step 2] Calculating Transforms...");
        computeTransforms(nodes, matcher);

        // --- BƯỚC 3: BLENDING ---
        System.out.println("\n[Step 3] Blending...");
        Mat result = ImageBlenderFAST.blend(nodes);

        if (result != null) {
            String imgOut = "final_openpano_java_blender";
            String outName = imgOut + ".jpg";
            imwrite(OUTPUT_PATH.resolve(outName).toString(), result);
            System.out.println(">>> DONE: " + outName);
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
            
            FeatureMatcherWrapper.MatchResult res = matcher.match(curr.keypoints, curr.descriptors, next.keypoints, next.descriptors);
            
            if (res != null) {
                // Tìm Affine từ Curr -> Next
                Mat T_curr_next = TransformEstimator.estimateAffine(res.srcPoints, res.dstPoints);
                if (T_curr_next != null && TransformEstimator.isTransformValid(T_curr_next)) {
                    // Global_Next = Global_Curr * inv(T_curr_next)
                    Mat T_next_curr = new Mat();
                    invert(T_curr_next, T_next_curr, DECOMP_LU);
                    
                    Mat g = new Mat();
                    gemm(curr.globalTransform, T_next_curr, 1.0, new Mat(), 0.0, g);
                    next.globalTransform = g;
                } else {
                    next.globalTransform = curr.globalTransform.clone(); // Fallback
                }
            }
        }

        // Lan truyền sang Trái
        for (int i = mid; i > 0; i--) {
            ImageNode curr = nodes.get(i);
            ImageNode prev = nodes.get(i - 1);
            
            FeatureMatcherWrapper.MatchResult res = matcher.match(prev.keypoints, prev.descriptors, curr.keypoints, curr.descriptors);
            
            if (res != null) {
                // Tìm Affine từ Prev -> Curr
                Mat T_prev_curr = TransformEstimator.estimateAffine(res.srcPoints, res.dstPoints);
                if (T_prev_curr != null && TransformEstimator.isTransformValid(T_prev_curr)) {
                    // Global_Prev = Global_Curr * T_Prev_Curr
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