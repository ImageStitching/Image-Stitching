package com.stitching.RANSAC_matching;

import com.stitching.SIFT.SiftDescriptor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class CandidateMatchGenerator {

    /** Một cặp keypoint khớp giữa 2 ảnh A và B */
    public static class Candidate {
        public SiftDescriptor keyA;
        public SiftDescriptor keyB;
        public double distance;

        public Candidate(SiftDescriptor keyA, SiftDescriptor keyB, double distance) {
            this.keyA = keyA;
            this.keyB = keyB;
            this.distance = distance;
        }
    }

    /** Match giữa 2 ảnh bất kỳ trong danh sách nhiều ảnh */
    public static class MatchBetweenImages {
        public int indexA;                 // chỉ số ảnh A trong danh sách
        public int indexB;                 // chỉ số ảnh B trong danh sách
        public List<Candidate> matches;    // các cặp keypoint khớp giữa 2 ảnh

        public MatchBetweenImages(int indexA, int indexB, List<Candidate> matches) {
            this.indexA = indexA;
            this.indexB = indexB;
            this.matches = matches;
        }
    }

    /** Khoảng cách Euclid giữa 2 descriptor SIFT (double[]) */
    private static double euclideanDistance(double[] a, double[] b) {
        int len = Math.min(a.length, b.length);
        double sum = 0.0;
        for (int i = 0; i < len; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    /**
     * Sinh các cặp keypoint có khả năng khớp (nearest neighbor + Lowe ratio test).
     * So khớp giữa 2 ảnh: imageA ↔ imageB.
     */
    public static List<Candidate> generate(
            List<SiftDescriptor> imageA,
            List<SiftDescriptor> imageB,
            double ratioThreshold) {

        List<Candidate> candidates = new ArrayList<>();

        for (SiftDescriptor keyA : imageA) {
            double[] descA = keyA.descriptor;

            double bestDist = Double.MAX_VALUE;
            double secondBest = Double.MAX_VALUE;
            SiftDescriptor bestMatch = null;

            // So khớp với tất cả keypoint của ảnh B
            for (SiftDescriptor keyB : imageB) {
                double[] descB = keyB.descriptor;
                double dist = euclideanDistance(descA, descB);

                if (dist < bestDist) {
                    secondBest = bestDist;
                    bestDist = dist;
                    bestMatch = keyB;
                } else if (dist < secondBest) {
                    secondBest = dist;
                }
            }

            // Tránh chia cho ∞/0 khi không có second-best hợp lệ
            if (bestMatch == null || !Double.isFinite(secondBest) || secondBest <= 0) continue;

            // Lowe ratio test (thường đặt 0.8)
            if (bestDist < ratioThreshold * secondBest) {
                candidates.add(new Candidate(keyA, bestMatch, bestDist));
            }
        }

        return candidates;
    }

    /**
     * Mutual best match (lọc hai chiều A↔B) cho 2 ảnh.
     */
    public static List<Candidate> mutualMatches(
            List<SiftDescriptor> imageA,
            List<SiftDescriptor> imageB,
            double ratioThreshold) {

        List<Candidate> forward = generate(imageA, imageB, ratioThreshold);
        List<Candidate> backward = generate(imageB, imageA, ratioThreshold);

        Set<String> backwardSet = new HashSet<>();
        for (Candidate c : backward) {
            // lưu key theo dạng: (A.x,A.y)_(B.x,B.y)
            backwardSet.add(c.keyB.x + "," + c.keyB.y + "_" + c.keyA.x + "," + c.keyA.y);
        }

        List<Candidate> mutual = new ArrayList<>();
        for (Candidate c : forward) {
            String key = c.keyA.x + "," + c.keyA.y + "_" + c.keyB.x + "," + c.keyB.y;
            if (backwardSet.contains(key)) {
                mutual.add(c);
            }
        }
        return mutual;
    }

    /**
     * Tìm mutual matches cho N ảnh, chỉ xét cặp LIỀN KỀ:
     * (0,1), (1,2), (2,3), ...
     * Thích hợp cho panorama chụp theo chuỗi trái→phải.
     */
    public static List<MatchBetweenImages> mutualMatchesMultiAdjacent(
            List<List<SiftDescriptor>> allImages,
            double ratioThreshold) {

        List<MatchBetweenImages> result = new ArrayList<>();

        if (allImages == null || allImages.size() < 2) return result;

        for (int i = 0; i < allImages.size() - 1; i++) {
            List<SiftDescriptor> imgA = allImages.get(i);
            List<SiftDescriptor> imgB = allImages.get(i + 1);

            List<Candidate> matches = mutualMatches(imgA, imgB, ratioThreshold);
            if (!matches.isEmpty()) {
                result.add(new MatchBetweenImages(i, i + 1, matches));
            }
        }

        return result;
    }

    /**
     * Tìm mutual matches cho N ảnh, xét MỌI CẶP (i, j) với j > i.
     * Dùng khi muốn xây dựng full graph giữa các ảnh.
     */
    public static List<MatchBetweenImages> mutualMatchesMultiAllPairs(
            List<List<SiftDescriptor>> allImages,
            double ratioThreshold) {

        List<MatchBetweenImages> result = new ArrayList<>();

        if (allImages == null || allImages.size() < 2) return result;

        int n = allImages.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                List<SiftDescriptor> imgA = allImages.get(i);
                List<SiftDescriptor> imgB = allImages.get(j);

                List<Candidate> matches = mutualMatches(imgA, imgB, ratioThreshold);
                if (!matches.isEmpty()) {
                    result.add(new MatchBetweenImages(i, j, matches));
                }
            }
        }

        return result;
    }
}
