package com.stitching.openpanoStitcher;

import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftDetector;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import com.stitching.openpanoStitcher.FeatureMatcher;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.*;
import org.bytedeco.opencv.opencv_stitching.MultiBandBlender;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_calib3d.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_stitching.*;

public class OpenPanoStitcher {
    // ĐỔI ĐƯỜNG DẪN ẢNH CỦA BẠN TẠI ĐÂY
    private static final Path OUTPUT_PATH = Paths.get("src", "main", "resources", "static", "output");
    private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "example_data","myself");

    static class ImageNode {
        int id;
        String filename;
        Mat img;            // Ảnh đã warp trụ
        SiftData siftData;
        Mat descriptors;
        Mat globalTransform; // Vị trí toàn cục (3x3)

        public ImageNode(int id, String filename, Mat img, SiftData siftData, Mat descriptors) {
            this.id = id;
            this.filename = filename;
            this.img = img;
            this.siftData = siftData;
            this.descriptors = descriptors;
            this.globalTransform = Mat.eye(3, 3, CV_64F).asMat();
        }
    }

    static class SiftData { List<SiftKeyPoint> keypoints; }

    public static void main(String[] args) {
        System.out.println("=== OPENPANO JAVA (CYLINDRICAL + MULTIBAND BLEND) ===");

        File folder = INPUT_PATH.toFile();
        File[] fileList = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png"));

        if (fileList == null || fileList.length == 0) {
            System.err.println("Không tìm thấy ảnh."); return;
        }

        // 1. Sắp xếp file theo tên (img1 -> img2...)
        // OpenPano chế độ Cylinder yêu cầu ảnh phải có thứ tự quét
        Arrays.sort(fileList, Comparator.comparing(File::getName));

        List<ImageNode> nodes = new ArrayList<>();
        FeatureMatcher matcher = new FeatureMatcher();

        System.out.println("\n--- BƯỚC 1: PRE-WARP & SIFT ---");

        // Ước lượng tiêu cự f (Quan trọng: f càng lớn ảnh càng phẳng, f nhỏ ảnh càng cong)
        Mat tmp = imread(fileList[0].getAbsolutePath());
        double f = tmp.cols() * 1.0; // f = chiều rộng ảnh là mức trung bình tốt
        tmp.release();

        for (int i = 0; i < fileList.length; i++) {
            System.out.println("-> Processing: " + fileList[i].getName());
            Mat raw = imread(fileList[i].getAbsolutePath());
            if (raw.empty()) continue;

            // Uốn cong thành hình trụ (Giống OpenPano)
            Mat warped = warpCylindrical(raw, f);

            // Cắt bỏ viền đen để SIFT không bắt nhầm
            Mat cropped = cropBlackBorder(warped);

            // SIFT trên ảnh đã uốn
            SiftData sift = runSift(cropped);
            limitKeypoints(sift, 4000);
            Mat desc = matcher.convertDescriptorsToMat(sift.keypoints);

            nodes.add(new ImageNode(i, fileList[i].getName(), cropped, sift, desc));
        }

        if (nodes.size() < 2) return;

        // --- BƯỚC 2: TÍNH TOÁN VỊ TRÍ (CENTER-OUT) ---
        System.out.println("\n--- BƯỚC 2: TÍNH TOÁN CHUỖI BIẾN ĐỔI ---");
        computeGlobalTransforms(nodes, matcher);

        // --- BƯỚC 3: RENDER VỚI MULTIBAND BLENDER ---
        System.out.println("\n--- BƯỚC 3: PHA TRỘN ĐA TẦN SỐ (SHARP BLEND) ---");
        Mat result = renderWithMultiBand(nodes);

        if (result != null) {
            String outName = "openpano_final_sharp.jpg";
            imwrite(INPUT_PATH.resolve(outName).toString(), result);
            System.out.println(">>> DONE! Saved: " + outName);
        }
    }

    // =========================================================================
    // LOGIC: TÍNH TOÁN VỊ TRÍ TỪ TÂM RA (CENTER-OUT)
    // =========================================================================
    private static void computeGlobalTransforms(List<ImageNode> nodes, FeatureMatcher matcher) {
        int n = nodes.size();
        int mid = n / 2; // Chọn ảnh giữa làm gốc
        ImageNode center = nodes.get(mid);
        center.globalTransform = Mat.eye(3, 3, CV_64F).asMat();
        System.out.println("   -> Anchor Image: " + center.filename);

        // 1. Lan truyền sang PHẢI (Right Wing)
        for (int i = mid; i < n - 1; i++) {
            ImageNode curr = nodes.get(i);
            ImageNode next = nodes.get(i + 1);

            // Tìm biến đổi từ Curr -> Next
            Mat T_curr_to_next = getAffineTransform(curr, next, matcher);

            if (T_curr_to_next != null) {
                // Global_Next = Global_Curr * T_Next_To_Curr
                // T_curr_to_next là biến đổi xuôi. Ta cần nghịch đảo.
                Mat T_next_to_curr = new Mat();
                invert(T_curr_to_next, T_next_to_curr, DECOMP_LU);

                Mat global = new Mat();
                gemm(curr.globalTransform, T_next_to_curr, 1.0, new Mat(), 0.0, global);
                next.globalTransform = global;
            } else {
                // Mất dấu: Giả định dịch chuyển ngang bằng 90% chiều rộng ảnh
                System.out.println("      [WARN] Mất kết nối " + curr.filename + " -> " + next.filename);
                Mat T_shift = Mat.eye(3, 3, CV_64F).asMat();
                new org.bytedeco.javacpp.DoublePointer(T_shift.data()).put(2, -curr.img.cols() * 0.9); // Dịch trái (nghịch đảo của dịch phải)
                Mat global = new Mat();
                gemm(curr.globalTransform, T_shift, 1.0, new Mat(), 0.0, global);
                next.globalTransform = global;
            }
        }

        // 2. Lan truyền sang TRÁI (Left Wing)
        for (int i = mid; i > 0; i--) {
            ImageNode curr = nodes.get(i);
            ImageNode prev = nodes.get(i - 1);

            // Tìm biến đổi từ Prev -> Curr
            Mat T_prev_to_curr = getAffineTransform(prev, curr, matcher);

            if (T_prev_to_curr != null) {
                // Global_Prev = Global_Curr * T_Prev_To_Curr
                // T_prev_to_curr là biến đổi đưa Prev về Curr.
                Mat global = new Mat();
                gemm(curr.globalTransform, T_prev_to_curr, 1.0, new Mat(), 0.0, global);
                prev.globalTransform = global;
            } else {
                System.out.println("      [WARN] Mất kết nối " + prev.filename + " -> " + curr.filename);
                Mat T_shift = Mat.eye(3, 3, CV_64F).asMat();
                new org.bytedeco.javacpp.DoublePointer(T_shift.data()).put(2, curr.img.cols() * 0.9); // Dịch phải
                Mat global = new Mat();
                gemm(curr.globalTransform, T_shift, 1.0, new Mat(), 0.0, global);
                prev.globalTransform = global;
            }
        }
    }

    private static Mat getAffineTransform(ImageNode src, ImageNode dst, FeatureMatcher matcher) {
        FeatureMatcher.MatchResult res = matcher.matchFeatures(src.siftData.keypoints, src.descriptors, dst.siftData.keypoints, dst.descriptors);
        if (res != null && res.inlierMatches.size() > 10) {
            Mat mask = new Mat();
            // Dùng Affine (2x3) thay vì Homography để tránh nổ ảnh (vì đã warp trụ rồi)
            // RANSAC 4.0 pixel
            Mat aff = estimateAffine2D(res.srcPoints, res.dstPoints, mask, RANSAC, 4.0, 2000, 0.99, 0);
            mask.release();

            if (!aff.empty()) {
                // Chuyển Affine 2x3 thành 3x3
                return convertAffineTo3x3(aff);
            }
        }
        return null;
    }

    // =========================================================================
    // RENDER: SỬ DỤNG OPENCV MultiBandBlender
    // =========================================================================
    private static Mat renderWithMultiBand(List<ImageNode> nodes) {
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE;

        // 1. Tính kích thước Canvas
        for (ImageNode node : nodes) {
            float w = node.img.cols(); float h = node.img.rows();
            Mat corners = new Mat(4, 1, CV_32FC2);
            FloatPointer ptr = new FloatPointer(corners.data());
            ptr.put(0,0); ptr.put(1,0); ptr.put(2,w); ptr.put(3,0); ptr.put(4,w); ptr.put(5,h); ptr.put(6,0); ptr.put(7,h);
            Mat dst = new Mat(); perspectiveTransform(corners, dst, node.globalTransform);
            FloatPointer res = new FloatPointer(dst.data());
            for(int i=0; i<4; i++) {
                float x=res.get(2*i), y=res.get(2*i+1);
                minX=Math.min(minX,x); maxX=Math.max(maxX,x); minY=Math.min(minY,y); maxY=Math.max(maxY,y);
            }
            corners.release(); dst.release();
        }

        int W = (int)Math.ceil(maxX - minX); int H = (int)Math.ceil(maxY - minY);
        if (W > 30000 || H > 30000) { System.err.println("Canvas too big!"); return null; }
        System.out.println("   -> Canvas Size: " + W + "x" + H);

        // Ma trận dịch chuyển về gốc
        Mat T_offset = Mat.eye(3, 3, CV_64F).asMat();
        new org.bytedeco.javacpp.DoublePointer(T_offset.data()).put(2, -minX);
        new org.bytedeco.javacpp.DoublePointer(T_offset.data()).put(5, -minY);

        // 2. KHỞI TẠO MULTI-BAND BLENDER
        // type: false (CPU), num_bands: 5 (càng cao càng mượt nhưng chậm)
        // Sửa thành 3 tham số, tham số cuối là float (0.0f)
        MultiBandBlender blender = new MultiBandBlender(0, 5, 0);
        Rect roi = new Rect(0, 0, W, H);
        blender.prepare(roi);

        for (ImageNode node : nodes) {
            System.out.println("      + Feeding Blender: " + node.filename);

            Mat H_final = new Mat();
            gemm(T_offset, node.globalTransform, 1.0, new Mat(), 0.0, H_final);

            // Warp ảnh về vị trí trên Canvas
            Mat warpedImg = new Mat();
            warpPerspective(node.img, warpedImg, H_final, new Size(W, H));

            // Chuyển sang 16S (Short) vì Blender yêu cầu độ chính xác cao
            Mat img16S = new Mat();
            warpedImg.convertTo(img16S, CV_16SC3);

            // Tạo Mask cho vùng ảnh
            Mat mask = new Mat(node.img.size(), CV_8U, new Scalar(255));
            // Erode nhẹ để loại bỏ viền đen do nội suy
            rectangle(mask, new Point(0,0), new Point(mask.cols()-1, mask.rows()-1), new Scalar(0), 2, LINE_8, 0);

            Mat warpedMask = new Mat();
            warpPerspective(mask, warpedMask, H_final, new Size(W, H));

            // Đưa vào Blender
            blender.feed(img16S, warpedMask, new Point(0, 0));

            H_final.release(); warpedImg.release(); img16S.release(); mask.release(); warpedMask.release();
        }

        System.out.println("   -> Blending...");
        Mat result = new Mat();
        Mat resultMask = new Mat();

        // Trộn
        blender.blend(result, resultMask);

        // Chuyển về 8-bit để lưu
        Mat finalRes = new Mat();
        result.convertTo(finalRes, CV_8U);

        return cropBlackBorder(finalRes);
    }

    // --- UTILS ---
    private static Mat warpCylindrical(Mat image, double f) {
        int w = image.cols(); int h = image.rows();
        Mat mapX = new Mat(h, w, CV_32F), mapY = new Mat(h, w, CV_32F);
        FloatPointer pX = new FloatPointer(mapX.data()), pY = new FloatPointer(mapY.data());
        float hW = w/2f, hH = h/2f, focal = (float)f;
        for(int y=0; y<h; y++) {
            for(int x=0; x<w; x++) {
                float x_d = x - hW; float y_d = y - hH;
                float x_s = (float)(focal * Math.tan(x_d/focal)) + hW;
                float y_s = (float)(y_d / Math.cos(x_d/focal)) + hH;
                long idx = y*w + x; pX.put(idx, x_s); pY.put(idx, y_s);
            }
        }
        Mat res = new Mat(); remap(image, res, mapX, mapY, INTER_LINEAR, BORDER_CONSTANT, new Scalar(0,0,0,0));
        mapX.release(); mapY.release(); return cropBlackBorder(res);
    }

    private static Mat convertAffineTo3x3(Mat aff) {
        Mat H = Mat.eye(3,3,CV_64F).asMat();
        org.bytedeco.javacpp.DoublePointer s=new org.bytedeco.javacpp.DoublePointer(aff.data());
        org.bytedeco.javacpp.DoublePointer d=new org.bytedeco.javacpp.DoublePointer(H.data());
        for(int i=0; i<6; i++) d.put(i, s.get(i)); return H;
    }

    private static SiftData runSift(Mat img) {
        Mat g=new Mat(); if(img.channels()==3) cvtColor(img,g,COLOR_BGR2GRAY); else img.copyTo(g);
        Mat f=new Mat(); g.convertTo(f,CV_32F,1.0/255.0,0.0);
        SiftData d=new SiftData(); d.keypoints=new SiftDetector().run(new ScaleSpace().buildGaussianPyramid(f), new ScaleSpace().buildDoGPyramid(new ScaleSpace().buildGaussianPyramid(f)));
        return d;
    }
    private static void limitKeypoints(SiftData d, int m) { if(d!=null && d.keypoints.size()>m) d.keypoints=new ArrayList<>(d.keypoints.subList(0,m)); }
    private static Mat cropBlackBorder(Mat img) { Mat g=new Mat(); cvtColor(img,g,COLOR_BGR2GRAY); Mat p=new Mat(); findNonZero(g,p); if(p.empty()) return img; Rect bb=boundingRect(p); return new Mat(img,bb); }
}