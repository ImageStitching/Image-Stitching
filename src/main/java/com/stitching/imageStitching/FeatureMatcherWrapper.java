package com.stitching.imageStitching;

import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.*;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;

public class FeatureMatcherWrapper {
    private FlannBasedMatcher matcher;

    public FeatureMatcherWrapper() {
        this.matcher = new FlannBasedMatcher();
    }

    public static class MatchResult {
        public List<DMatch> inlierMatches;
        public Mat srcPoints;
        public Mat dstPoints;

        public MatchResult(List<DMatch> matches, Mat src, Mat dst) {
            this.inlierMatches = matches;
            this.srcPoints = src;
            this.dstPoints = dst;
        }
    }

    public MatchResult match(List<SiftKeyPoint> kp1, Mat desc1, List<SiftKeyPoint> kp2, Mat desc2) {
        if (desc1.empty() || desc2.empty()) return null;

        DMatchVectorVector knnMatches = new DMatchVectorVector();
        matcher.knnMatch(desc1, desc2, knnMatches, 2);

        List<DMatch> goodMatches = new ArrayList<>();
        List<Point2f> pts1 = new ArrayList<>();
        List<Point2f> pts2 = new ArrayList<>();

        // Lowe's Ratio Test (Giống OpenPano)
        float ratioThresh = 0.75f;
        long size = knnMatches.size();

        for (long i = 0; i < size; i++) {
            DMatchVector matches = knnMatches.get(i);
            if (matches.size() < 2) continue;

            DMatch m = matches.get(0);
            DMatch n = matches.get(1);

            if (m.distance() < ratioThresh * n.distance()) {
                goodMatches.add(m);
                SiftKeyPoint p1 = kp1.get(m.queryIdx());
                SiftKeyPoint p2 = kp2.get(m.trainIdx());
                pts1.add(new Point2f(p1.x, p1.y));
                pts2.add(new Point2f(p2.x, p2.y));
            }
        }

        if (goodMatches.size() < 8) return null;

        return new MatchResult(goodMatches, listPointToMat(pts1), listPointToMat(pts2));
    }
    
    public Mat convertDescriptors(List<SiftKeyPoint> kps) {
        if (kps.isEmpty()) return new Mat();
        int rows = kps.size();
        int cols = 128;
        Mat mat = new Mat(rows, cols, CV_32F);
        FloatPointer ptr = new FloatPointer(mat.data());
        float[] buf = new float[rows * cols];
        for(int i=0; i<rows; i++) System.arraycopy(kps.get(i).descriptor, 0, buf, i*cols, cols);
        ptr.put(buf);
        return mat;
    }

    private Mat listPointToMat(List<Point2f> points) {
        Mat mat = new Mat(points.size(), 1, CV_32FC2);
        FloatPointer ptr = new FloatPointer(mat.data());
        for (int i = 0; i < points.size(); i++) {
            ptr.put(2 * i, points.get(i).x());
            ptr.put(2 * i + 1, points.get(i).y());
        }
        return mat;
    }
}