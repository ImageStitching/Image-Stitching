package com.stitching.selfcode.CandicateMatching_RANSAC_Homography;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class RANSAC {
    private final int numIterations;
    private final double threshold;

    public RANSAC(int numIterations, double threshold) {
        this.numIterations = numIterations;
        this.threshold = threshold;
    }

    public HomographyMatrix run(List<CandidateMatchGenerator.Candidate> matches) {
        if (matches.size() < 4) {
            System.err.println("Không đủ điểm khớp để chạy RANSAC (cần tối thiểu 4).");
            return null;
        }

        HomographyMatrix bestHomography = null;
        int maxInliers = 0;
        Random rand = new Random();

        for (int i = 0; i < numIterations; i++) {
            List<CandidateMatchGenerator.Candidate> randomSubset = new ArrayList<>();
            List<CandidateMatchGenerator.Candidate> tempMatches = new ArrayList<>(matches);
            Collections.shuffle(tempMatches, rand);
            
            for(int k=0; k<4; k++) randomSubset.add(tempMatches.get(k));

            HomographyMatrix H = computeHomographyDLT(randomSubset);
            if (H == null) continue;

            int currentInliers = 0;
            for (CandidateMatchGenerator.Candidate match : matches) {
                double[] projected = H.project(match.keyA.x, match.keyA.y);
                if (projected == null) continue;

                double distSq = Math.pow(projected[0] - match.keyB.x, 2) + Math.pow(projected[1] - match.keyB.y, 2);
                
                if (distSq < threshold * threshold) {
                    currentInliers++;
                }
            }

            if (currentInliers > maxInliers) {
                maxInliers = currentInliers;
                bestHomography = H;
            }
        }

        System.out.println("RANSAC hoàn tất. Số inliers tốt nhất: " + maxInliers + "/" + matches.size());
        return bestHomography;
    }

    private HomographyMatrix computeHomographyDLT(List<CandidateMatchGenerator.Candidate> fourMatches) {
        double[][] A = new double[8][8];
        double[] b = new double[8];

        for (int i = 0; i < 4; i++) {
            CandidateMatchGenerator.Candidate m = fourMatches.get(i);
            double x = m.keyA.x;
            double y = m.keyA.y;
            double u = m.keyB.x;
            double v = m.keyB.y;

            A[2*i][0] = x;
            A[2*i][1] = y;
            A[2*i][2] = 1;
            A[2*i][3] = 0;
            A[2*i][4] = 0;
            A[2*i][5] = 0;
            A[2*i][6] = -x * u;
            A[2*i][7] = -y * u;
            b[2*i] = u;

            A[2*i+1][0] = 0;
            A[2*i+1][1] = 0;
            A[2*i+1][2] = 0;
            A[2*i+1][3] = x;
            A[2*i+1][4] = y;
            A[2*i+1][5] = 1;
            A[2*i+1][6] = -x * v;
            A[2*i+1][7] = -y * v;
            b[2*i+1] = v;
        }

        try {
            double[] h = GaussianElimination.lsolve(A, b);
            if (h == null) return null;

            double[][] data = new double[3][3];
            data[0][0] = h[0]; data[0][1] = h[1]; data[0][2] = h[2];
            data[1][0] = h[3]; data[1][1] = h[4]; data[1][2] = h[5];
            data[2][0] = h[6]; data[2][1] = h[7]; data[2][2] = 1.0;

            return new HomographyMatrix(data);
        } catch (Exception e) {
            return null;
        }
    }
}