package com.stitching.openpanoStitcher;

import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_calib3d.*;

import java.util.ArrayList;
import java.util.List;

public class FeatureMatcher {
    private static final float RATIO_THRESHOLD = 0.75f;

    public static class MatchResult {
        public List<DMatch> inlierMatches = new ArrayList<>();
        public Mat srcPoints; // Mat chứa các điểm ảnh 1 (Left)
        public Mat dstPoints; // Mat chứa các điểm ảnh 2 (Right)
    }

    public MatchResult matchFeatures(List<SiftKeyPoint> kp1, Mat desc1, List<SiftKeyPoint> kp2, Mat desc2) {
        BFMatcher matcher = new BFMatcher(NORM_L2, false);
        //FlannBasedMatcher matcher = new FlannBasedMatcher();

        DMatchVectorVector knnMatches = new DMatchVectorVector();
        matcher.knnMatch(desc1, desc2, knnMatches, 2);

        // 1. Lọc thô (Lowe's Ratio Test)
        List<Point2f> pts1 = new ArrayList<>();
        List<Point2f> pts2 = new ArrayList<>();
        List<DMatch> goodMatches = new ArrayList<>();

        for (long i = 0; i < knnMatches.size(); i++) {
            DMatchVector matches = knnMatches.get(i);
            if (matches.size() < 2) continue;
            DMatch m = matches.get(0);
            DMatch n = matches.get(1);

            if (m.distance() < RATIO_THRESHOLD * n.distance()) {
                goodMatches.add(m);
                SiftKeyPoint p1 = kp1.get(m.queryIdx());
                SiftKeyPoint p2 = kp2.get(m.trainIdx());
                pts1.add(new Point2f(p1.x, p1.y));
                pts2.add(new Point2f(p2.x, p2.y));
            }
        }

        if (goodMatches.size() < 4) return null;

        // 2. Chuẩn bị Mat cho RANSAC (Dùng CV_32FC2 - 1 cột, 2 kênh cho chuẩn C++)
        int count = goodMatches.size();
        Mat m1 = new Mat(count, 1, CV_32FC2);
        Mat m2 = new Mat(count, 1, CV_32FC2);

        FloatPointer ptr1 = new FloatPointer(m1.data());
        FloatPointer ptr2 = new FloatPointer(m2.data());

        // Đổ dữ liệu vào mảng phẳng rồi copy 1 lần (Bulk Copy)
        float[] d1 = new float[count * 2];
        float[] d2 = new float[count * 2];

        for(int i=0; i<count; i++) {
            d1[2*i] = pts1.get(i).x();     d1[2*i+1] = pts1.get(i).y();
            d2[2*i] = pts2.get(i).x();     d2[2*i+1] = pts2.get(i).y();
        }
        ptr1.put(d1);
        ptr2.put(d2);

        // 3. Chạy RANSAC để lấy Mask lọc điểm nhiễu
        Mat mask = new Mat();
        findHomography(m1, m2, RANSAC, 4.0, mask, 2000, 0.995);

        // 4. Lọc lấy Inliers cuối cùng
        MatchResult res = new MatchResult();
        org.bytedeco.javacpp.BytePointer maskPtr = mask.data();

        List<Point2f> finalPts1 = new ArrayList<>();
        List<Point2f> finalPts2 = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            if (maskPtr.get(i) != 0) { // Nếu là Inlier
                res.inlierMatches.add(goodMatches.get(i));
                finalPts1.add(pts1.get(i));
                finalPts2.add(pts2.get(i));
            }
        }

        // Giảm ngưỡng tối thiểu xuống một chút (từ 10 xuống 8) cho các ca khó
        // if (res.inlierMatches.size() < 8) return null;

        // 5. Tạo Mat kết quả (cũng dùng CV_32FC2)
        int finalCount = finalPts1.size();
        res.srcPoints = new Mat(finalCount, 1, CV_32FC2);
        res.dstPoints = new Mat(finalCount, 1, CV_32FC2);

        float[] fd1 = new float[finalCount * 2];
        float[] fd2 = new float[finalCount * 2];

        for(int i=0; i<finalCount; i++) {
            fd1[2*i] = finalPts1.get(i).x();   fd1[2*i+1] = finalPts1.get(i).y();
            fd2[2*i] = finalPts2.get(i).x();   fd2[2*i+1] = finalPts2.get(i).y();
        }
        new FloatPointer(res.srcPoints.data()).put(fd1);
        new FloatPointer(res.dstPoints.data()).put(fd2);

        return res;
    }

    public Mat convertDescriptorsToMat(List<SiftKeyPoint> keypoints) {
        if (keypoints.isEmpty()) return new Mat();
        int rows = keypoints.size();
        Mat mat = new Mat(rows, 128, CV_32F);
        FloatPointer ptr = new FloatPointer(mat.data());
        float[] buf = new float[rows * 128];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(keypoints.get(i).descriptor, 0, buf, i * 128, 128);
        }
        ptr.put(buf);
        return mat;
    }
}