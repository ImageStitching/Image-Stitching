package com.stitching.RANSAC_matching;

import com.stitching.SIFT.SiftDescriptor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class CandidateMatchGenerator {

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
     * Mutual best match (lọc hai chiều A↔B)
     */
    public static List<Candidate> mutualMatches(
            List<SiftDescriptor> imageA,
            List<SiftDescriptor> imageB,
            double ratioThreshold) {

        List<Candidate> forward = generate(imageA, imageB, ratioThreshold);
        List<Candidate> backward = generate(imageB, imageA, ratioThreshold);

        Set<String> backwardSet = new HashSet<>();
        for (Candidate c : backward) {
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
}
