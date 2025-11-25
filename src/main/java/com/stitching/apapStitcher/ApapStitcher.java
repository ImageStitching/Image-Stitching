package com.stitching.apapStitcher;

import com.stitching.openpanoSIFT.ScaleSpace;
import com.stitching.openpanoSIFT.SiftDetector;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import com.stitching.openpanoStitcher.FeatureMatcher;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.*;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_calib3d.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ApapStitcher {
    private static final Path INPUT_PATH = Paths.get("src", "main", "resources", "static", "one_scene_mytenancyhouse_rotation");
    private static final int GRID_SIZE = 40;
    private static final double GAMMA = 0.0025;
    private static final double SIGMA = 8.5;

    public static void main(String[] args) {
        System.out.println("=== APAP STITCHER (Robust Ver.) ===");

        File folder = INPUT_PATH.toFile();
        File[] files = folder.listFiles((d, n) -> n.toLowerCase().endsWith(".jpg") || n.toLowerCase().endsWith(".png"));
        if (files == null || files.length == 0) return;

        Arrays.sort(files, Comparator.comparing(File::getName));

        System.out.println("-> Base: " + files[0].getName());
        Mat canvas = imread(files[0].getAbsolutePath());
        if (canvas.empty()) { System.err.println("Lỗi đọc ảnh 1"); return; }

        // Canvas khởi tạo cần convert sang 3 kênh chuẩn để tránh lỗi
        if (canvas.channels() == 4) cvtColor(canvas, canvas, COLOR_BGRA2BGR);

        FeatureMatcher matcher = new FeatureMatcher();

        for (int i = 1; i < files.length; i++) {
            System.out.println("\n--- Stitching " + files[i].getName() + " ---");
            Mat nextImg = imread(files[i].getAbsolutePath());
            if (nextImg.empty()) continue;
            if (nextImg.channels() == 4) cvtColor(nextImg, nextImg, COLOR_BGRA2BGR);

            // 1. SIFT
            SiftData dCanvas = runSift(canvas);
            SiftData dNext = runSift(nextImg);

            // [FIX] Nếu canvas quá lớn hoặc không tìm thấy keypoint -> Dừng hoặc bỏ qua
            if (dCanvas.keypoints.size() < 10 || dNext.keypoints.size() < 10) {
                System.err.println("   [SKIP] Ít điểm đặc trưng SIFT. Giữ nguyên Canvas.");
                continue;
            }

            limitKeypoints(dCanvas, 5000);
            limitKeypoints(dNext, 5000);

            Mat descCanvas = matcher.convertDescriptorsToMat(dCanvas.keypoints);
            Mat descNext = matcher.convertDescriptorsToMat(dNext.keypoints);

            // 2. Match
            FeatureMatcher.MatchResult res = matcher.matchFeatures(dCanvas.keypoints, descCanvas, dNext.keypoints, descNext);
            if (res == null || res.inlierMatches.size() < 15) {
                System.err.println("   [SKIP] Không khớp được ảnh này.");
                continue;
            }

            // 3. Lọc RANSAC lấy điểm sạch
            Mat mask = new Mat();
            findHomography(res.dstPoints, res.srcPoints, RANSAC, 5.0, mask, 2000, 0.995);

            List<Point2f> srcPts = new ArrayList<>();
            List<Point2f> dstPts = new ArrayList<>();
            org.bytedeco.javacpp.BytePointer maskPtr = mask.data();
            for(int k=0; k<res.inlierMatches.size(); k++) {
                if(maskPtr.get(k) != 0) {
                    srcPts.add(getPoint(res.dstPoints, k)); // Next (Source)
                    dstPts.add(getPoint(res.srcPoints, k)); // Canvas (Dest)
                }
            }

            if (srcPts.size() < 8) {
                System.err.println("   [SKIP] RANSAC loại bỏ gần hết điểm.");
                continue;
            }

            // 4. APAP Warp & Merge
            // Hàm này sẽ trả về Canvas MỚI đã chứa cả ảnh cũ và ảnh mới
            Mat newCanvas = warpAndMergeApap(canvas, nextImg, srcPts, dstPts);

            if (newCanvas != null && !newCanvas.empty()) {
                canvas.release();
                canvas = newCanvas;
                System.out.println("   -> Ghép thành công. Canvas size: " + canvas.cols() + "x" + canvas.rows());
            } else {
                System.err.println("   [FAIL] Lỗi trong quá trình warp. Giữ nguyên Canvas cũ.");
            }

            // Dọn dẹp
            descCanvas.release(); descNext.release(); mask.release();
        }

        String outName = "apap_fixed_result.jpg";
        imwrite(INPUT_PATH.resolve(outName).toString(), canvas);
        System.out.println("\n>>> DONE: " + outName);
    }

    // =========================================================================
    // APAP WARP & MERGE (ĐÃ FIX LOGIC TỌA ĐỘ)
    // =========================================================================
    private static Mat warpAndMergeApap(Mat canvas, Mat nextImg, List<Point2f> srcPts, List<Point2f> dstPts) {
        // Lấy kích thước
        int width = nextImg.cols();
        int height = nextImg.rows();

        // 1. Tính Map biến dạng cho các đỉnh lưới của nextImg
        int gridCols = (int) Math.ceil((double) width / GRID_SIZE) + 1;
        int gridRows = (int) Math.ceil((double) height / GRID_SIZE) + 1;

        // Mảng lưu tọa độ đích của các đỉnh lưới
        float[] meshX = new float[gridRows * gridCols];
        float[] meshY = new float[gridRows * gridCols];

        // Biến lưu biên của ảnh mới sau khi warp
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;

        System.out.print("   + Computing Mesh: ");
        for (int gy = 0; gy < gridRows; gy++) {
            for (int gx = 0; gx < gridCols; gx++) {
                float px = Math.min(gx * GRID_SIZE, width - 1);
                float py = Math.min(gy * GRID_SIZE, height - 1);
                Point2f v = new Point2f(px, py);

                Mat H_local = computeLocalHomography(srcPts, dstPts, v);

                // p' = H * p
                double[] h = new double[9];
                new DoublePointer(H_local.data()).get(h);
                double w_new = h[6]*px + h[7]*py + h[8];
                double x_new = (h[0]*px + h[1]*py + h[2]) / w_new;
                double y_new = (h[3]*px + h[4]*py + h[5]) / w_new;

                int idx = gy * gridCols + gx;
                meshX[idx] = (float)x_new;
                meshY[idx] = (float)y_new;

                // Tìm biên của ảnh mới
                if (x_new < minX) minX = (float)x_new;
                if (x_new > maxX) maxX = (float)x_new;
                if (y_new < minY) minY = (float)y_new;
                if (y_new > maxY) maxY = (float)y_new;

                H_local.release();
            }
        }
        System.out.println("Done.");

        // 2. Tính kích thước Canvas TỔNG HỢP
        // Canvas cũ: (0,0) đến (canvas.cols, canvas.rows)
        // Ảnh mới: (minX, minY) đến (maxX, maxY)

        float finalMinX = Math.min(0, minX);
        float finalMinY = Math.min(0, minY);
        float finalMaxX = Math.max(canvas.cols(), maxX);
        float finalMaxY = Math.max(canvas.rows(), maxY);

        int W = (int)Math.ceil(finalMaxX - finalMinX);
        int H = (int)Math.ceil(finalMaxY - finalMinY);

        if (W > 30000 || H > 30000) {
            System.err.println("   [ERROR] Canvas nổ (" + W + "x" + H + "). Hủy bước này.");
            return null;
        }

        // 3. Tạo Canvas Mới
        Mat newCanvas = new Mat(H, W, canvas.type(), Scalar.all(0));

        // Dịch chuyển Canvas Cũ vào vị trí mới (trừ đi offset âm)
        int offX = (int)Math.abs(finalMinX);
        int offY = (int)Math.abs(finalMinY);

        Rect roiOld = new Rect(offX, offY, canvas.cols(), canvas.rows());
        Mat subCanvas = new Mat(newCanvas, roiOld);
        canvas.copyTo(subCanvas);

        // 4. Warp từng ô lưới của NextImg và dán đè vào NewCanvas
        for (int r = 0; r < gridRows - 1; r++) {
            for (int c = 0; c < gridCols - 1; c++) {
                // Src Coords
                float x1_s = c * GRID_SIZE;
                float y1_s = r * GRID_SIZE;
                float x2_s = Math.min((c+1) * GRID_SIZE, width);
                float y2_s = Math.min((r+1) * GRID_SIZE, height);

                // Dst Coords (từ mesh đã tính + offset toàn cục)
                int idx1 = r * gridCols + c;
                int idx2 = r * gridCols + (c+1);
                int idx3 = (r+1) * gridCols + (c+1);
                int idx4 = (r+1) * gridCols + c;

                Point2f p1 = new Point2f(meshX[idx1] + offX, meshY[idx1] + offY);
                Point2f p2 = new Point2f(meshX[idx2] + offX, meshY[idx2] + offY);
                Point2f p3 = new Point2f(meshX[idx3] + offX, meshY[idx3] + offY);
                Point2f p4 = new Point2f(meshX[idx4] + offX, meshY[idx4] + offY);

                // Cắt ô ảnh con từ NextImg
                Rect roiSrc = new Rect((int)x1_s, (int)y1_s, (int)(x2_s-x1_s), (int)(y2_s-y1_s));
                if (roiSrc.width() <= 0 || roiSrc.height() <= 0) continue;
                Mat cellSrc = new Mat(nextImg, roiSrc);

                // Tính Homography cục bộ cho ô này
                Mat srcQuad = new Mat(4, 1, CV_32FC2);
                FloatPointer sp = new FloatPointer(srcQuad.data());
                sp.put(0,0); sp.put(1,0);
                sp.put(2,roiSrc.width()); sp.put(3,0);
                sp.put(4,roiSrc.width()); sp.put(5,roiSrc.height());
                sp.put(6,0); sp.put(7,roiSrc.height());

                Mat dstQuad = new Mat(4, 1, CV_32FC2);
                FloatPointer dp = new FloatPointer(dstQuad.data());
                dp.put(0, p1.x()); dp.put(1, p1.y());
                dp.put(2, p2.x()); dp.put(3, p2.y());
                dp.put(4, p3.x()); dp.put(5, p3.y());
                dp.put(6, p4.x()); dp.put(7, p4.y());

                Mat H_cell = getPerspectiveTransform(srcQuad, dstQuad);

                // Warp ô
                // Tìm bounding box của ô đích
                float cx1 = Math.min(Math.min(p1.x(), p2.x()), Math.min(p3.x(), p4.x()));
                float cx2 = Math.max(Math.max(p1.x(), p2.x()), Math.max(p3.x(), p4.x()));
                float cy1 = Math.min(Math.min(p1.y(), p2.y()), Math.min(p3.y(), p4.y()));
                float cy2 = Math.max(Math.max(p1.y(), p2.y()), Math.max(p3.y(), p4.y()));

                int cellW = (int)Math.ceil(cx2 - cx1);
                int cellH = (int)Math.ceil(cy2 - cy1);

                // Dịch H về gốc (0,0) tạm thời để warp
                Mat T_cell = Mat.eye(3, 3, CV_64F).asMat();
                new DoublePointer(T_cell.data()).put(2, -cx1);
                new DoublePointer(T_cell.data()).put(5, -cy1);
                Mat H_warp = new Mat();
                gemm(T_cell, H_cell, 1.0, new Mat(), 0.0, H_warp);

                Mat cellWarped = new Mat();
                warpPerspective(cellSrc, cellWarped, H_warp, new Size(cellW, cellH));

                // Copy vào NewCanvas (Dùng mask để không đè đen)
                Mat mask = new Mat();
                cvtColor(cellWarped, mask, COLOR_BGR2GRAY);
                threshold(mask, mask, 1, 255, THRESH_BINARY);

                copyToWithMask(newCanvas, cellWarped, mask, (int)cx1, (int)cy1);

                cellSrc.release(); cellWarped.release(); mask.release();
                srcQuad.release(); dstQuad.release(); H_cell.release(); T_cell.release(); H_warp.release();
            }
        }
        return newCanvas;
    }

    private static void copyToWithMask(Mat dst, Mat src, Mat mask, int x, int y) {
        int w = Math.min(src.cols(), dst.cols() - x);
        int h = Math.min(src.rows(), dst.rows() - y);
        if (w <= 0 || h <= 0 || x < 0 || y < 0) return;

        Rect roiDst = new Rect(x, y, w, h);
        Rect roiSrc = new Rect(0, 0, w, h);

        Mat subDst = new Mat(dst, roiDst);
        Mat subSrc = new Mat(src, roiSrc);
        Mat subMask = new Mat(mask, roiSrc);

        subSrc.copyTo(subDst, subMask);
    }

    private static Mat computeLocalHomography(List<Point2f> src, List<Point2f> dst, Point2f gridPt) {
        int n = src.size();
        Mat A = new Mat(2 * n, 9, CV_64F);
        DoublePointer aPtr = new DoublePointer(A.data());
        double sigma2 = SIGMA * SIGMA;
        double gamma = GAMMA;

        for (int i = 0; i < n; i++) {
            Point2f s = src.get(i);
            Point2f d = dst.get(i);
            double dist2 = Math.pow(s.x() - gridPt.x(), 2) + Math.pow(s.y() - gridPt.y(), 2);
            double w = Math.exp(-dist2 / (sigma2 * Math.max(src.get(0).x(), 1000.0)));
            w = Math.max(w, gamma);

            double x=s.x(), y=s.y(), u=d.x(), v=d.y();
            aPtr.put(18*i + 0, 0);       aPtr.put(18*i + 1, 0);       aPtr.put(18*i + 2, 0);
            aPtr.put(18*i + 3, -w*x);    aPtr.put(18*i + 4, -w*y);    aPtr.put(18*i + 5, -w);
            aPtr.put(18*i + 6, w*v*x);   aPtr.put(18*i + 7, w*v*y);   aPtr.put(18*i + 8, w*v);
            aPtr.put(18*i + 9, w*x);     aPtr.put(18*i + 10, w*y);    aPtr.put(18*i + 11, w);
            aPtr.put(18*i + 12, 0);      aPtr.put(18*i + 13, 0);      aPtr.put(18*i + 14, 0);
            aPtr.put(18*i + 15, -w*u*x); aPtr.put(18*i + 16, -w*u*y); aPtr.put(18*i + 17, -w*u);
        }

        Mat wVal = new Mat(), uMat = new Mat(), vtMat = new Mat();
        SVDecomp(A, wVal, uMat, vtMat, SVD.FULL_UV);
        Mat hRow = vtMat.row(8);
        Mat H = new Mat(3, 3, CV_64F);
        DoublePointer hPtr = new DoublePointer(H.data());
        DoublePointer srcPtr = new DoublePointer(hRow.data());
        for(int k=0; k<9; k++) hPtr.put(k, srcPtr.get(k));

        // [FIX MULTIPLY ERROR] Dùng convertTo thay vì multiply
        double h33 = hPtr.get(8);
        if (Math.abs(h33) > 1e-8) {
            H.convertTo(H, -1, 1.0/h33, 0);
        }

        A.release(); wVal.release(); uMat.release(); vtMat.release();
        return H;
    }

    // Helpers
    private static Point2f getPoint(Mat ptMat, int idx) {
        FloatPointer ptr = new FloatPointer(ptMat.data());
        return new Point2f(ptr.get(2*idx), ptr.get(2*idx+1));
    }
    static class SiftData { List<SiftKeyPoint> keypoints; }
    private static SiftData runSift(Mat img) {
        Mat g=new Mat(); if(img.channels()==3) cvtColor(img,g,COLOR_BGR2GRAY); else img.copyTo(g);
        Mat f=new Mat(); g.convertTo(f,CV_32F,1.0/255.0,0.0);
        SiftData d=new SiftData(); d.keypoints=new SiftDetector().run(new ScaleSpace().buildGaussianPyramid(f), new ScaleSpace().buildDoGPyramid(new ScaleSpace().buildGaussianPyramid(f)));
        return d;
    }
    private static void limitKeypoints(SiftData d, int m) { if(d!=null && d.keypoints.size()>m) d.keypoints=new ArrayList<>(d.keypoints.subList(0,m)); }
    private static Mat cropBlackBorder(Mat img) { Mat g=new Mat(); cvtColor(img,g,COLOR_BGR2GRAY); Mat p=new Mat(); findNonZero(g,p); if(p.empty()) return img; Rect bb=boundingRect(p); return new Mat(img,bb); }
}