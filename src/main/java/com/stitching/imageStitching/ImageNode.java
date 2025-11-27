package com.stitching.imageStitching;

import com.stitching.imageStitching.matchAndTransform.FeatureMatcherWrapper;
import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.opencv.opencv_core.DMatch;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.Comparator;
import java.util.List;

import static com.stitching.imageStitching.CylinderStitcherEnhanced.analyzeMatchDirection;
import static org.bytedeco.opencv.global.opencv_core.CV_64F;

public class ImageNode {
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

    public ImageNode(int id, String name, Mat img, List<SiftKeyPoint> kp, Mat desc, Mat globalTransform) {
        this.id = id;
        this.filename = name;
        this.img = img;
        this.keypoints = kp;
        this.descriptors = desc;
        this.globalTransform = globalTransform;
    }

    public static class sequencePriority implements Comparator<ImageNode> {
        @Override
        public int compare(ImageNode o1, ImageNode o2) {
            FeatureMatcherWrapper matcher = new FeatureMatcherWrapper();
            CylinderStitcherEnhanced.ImageRelation relation = analyzeSequence(matcher, o1, o2);
            if(relation==null)
            if(relation.needSwap) return 1;
            return -1;
        }
    }

    public static int getMatchScore(FeatureMatcherWrapper matcher, ImageNode n1, ImageNode n2) {
        FeatureMatcherWrapper.MatchResult res = matcher.match(n1.keypoints, n1.descriptors, n2.keypoints, n2.descriptors);
        // Case 1: Không tìm thấy match nào hoặc match bị null
        if (res == null || res.inlierMatches == null) return 0;
        int score = res.inlierMatches.size();
        // Case 2: Có match nhưng quá ít (Nhiễu) -> Coi như bằng 0
        if (score < 15) return 0;
        return score;
    }

    public static int getMatchScore(FeatureMatcherWrapper.MatchResult res) {
        if (res == null || res.inlierMatches == null) return 0;
        int score = res.inlierMatches.size();
        // Case 2: Có match nhưng quá ít (Nhiễu) -> Coi như bằng 0
        if (score < 15) return 0;
        return score;
    }

    // helper tính dãy tương quan ImageRelation giữa 2 ảnh
    public static List<DMatch> getImageRelationDmatch(FeatureMatcherWrapper matcher, ImageNode n1, ImageNode n2) {
        FeatureMatcherWrapper.MatchResult res = matcher.match(n1.keypoints, n1.descriptors, n2.keypoints, n2.descriptors);
        if (res == null) return null;
        return res.inlierMatches;
    }

    // helper so sánh thứ tự giữa 2 ảnh
    public static CylinderStitcherEnhanced.ImageRelation analyzeSequence(FeatureMatcherWrapper matcher, ImageNode n1, ImageNode n2) {
        List<DMatch> dMatches = getImageRelationDmatch(matcher, n1, n2);
        if (dMatches == null) return null;
        return analyzeMatchDirection(dMatches, n1, n2);
    }
}