package com.stitching.SIFT;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SiftStage3 {

    private final int numOrientationBins = 36;
    private final double peakRatioThreshold = 0.8;

    public List<OrientedKeypoint> run(List<Keypoint> refinedKeypoints, List<List<SiftImage>> gaussianPyramid) {
        System.out.println("\n--- Bắt đầu Giai đoạn 3: Gán hướng cho điểm khóa ---");
        List<OrientedKeypoint> orientedKeypoints = new ArrayList<>();

        for (Keypoint kp : refinedKeypoints) {
            List<Double> orientations = calculateOrientations(kp, gaussianPyramid);
            for (double orientation : orientations) {
                orientedKeypoints.add(new OrientedKeypoint(kp, orientation));
            }
        }

        System.out.printf("Từ %d điểm khóa, đã tạo ra %d điểm khóa có hướng.\n", refinedKeypoints.size(), orientedKeypoints.size());
        System.out.println("--- Giai đoạn 3 Hoàn tất ---");
        return orientedKeypoints;
    }

    private List<Double> calculateOrientations(Keypoint kp, List<List<SiftImage>> gaussianPyramid) {
        SiftImage image = gaussianPyramid.get(kp.octave).get(kp.layer);
        double[][] imageData = image.data;
        int height = image.getHeight();
        int width = image.getWidth();

        double[] histogram = new double[numOrientationBins];
        int radius = (int) Math.round(3 * 1.5 * kp.sigma);
        double weightSigma = 1.5 * kp.sigma;

        int centerX = (int) Math.round(kp.x);
        int centerY = (int) Math.round(kp.y);

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int sampleY = centerY + i;
                int sampleX = centerX + j;
                if (sampleY <= 0 || sampleY >= height - 1 || sampleX <= 0 || sampleX >= width - 1) {
                    continue;
                }
                double dx = imageData[sampleY][sampleX + 1] - imageData[sampleY][sampleX - 1];
                double dy = imageData[sampleY+1][sampleX] - imageData[sampleY-1][sampleX];
                double magnitude = Math.sqrt(dx * dx + dy * dy);
                double orientation = Math.atan2(dy, dx);

                double weight = Math.exp(-(i * i + j * j) / (2 * weightSigma * weightSigma));

                // Chuyển hướng về khoảng [0, 2*PI ) và tìm bin tương ứng + Xử lý trường hợp 360 độ
                if (orientation < 0) {
                    orientation += 2 * Math.PI;
                }
//                int bin = (int) Math.round(orientation * numOrientationBins / (2 * Math.PI));
//                bin = (bin == numOrientationBins)? 0 : bin;
                int bin = (int) Math.floor(orientation * numOrientationBins / (2 * Math.PI));
                bin = Math.min(bin, numOrientationBins - 1);

                histogram[bin] += magnitude * weight;
            }
        }
        return findPeaks(histogram);
    }

    private List<Double> findPeaks(double[] histogram/*, int numOrientationBins, double peakRatioThreshold*/) {
        List<Double> orientations = new ArrayList<>();
        double maxPeakValue = Arrays.stream(histogram).max().orElse(0.0);

        for (int i = 0; i < numOrientationBins; i++) {
            double currentValue = histogram[i];
            double prevValue = histogram[(i - 1 + numOrientationBins) % numOrientationBins];
            double nextValue = histogram[(i + 1) % numOrientationBins];
            if (currentValue > prevValue && currentValue > nextValue && currentValue >= maxPeakValue * peakRatioThreshold) {
                double interp = 0.5 * (prevValue - nextValue) / (prevValue - 2 * currentValue + nextValue);
                interp = Math.max(-1.0, Math.min(1.0, interp));
                double peakBin = (i + interp + numOrientationBins) % numOrientationBins;
                double peakOrientation = peakBin * (2 * Math.PI / numOrientationBins);
                orientations.add(peakOrientation);
            }
        }
        return orientations;
    }

    public static void main(String[] args) {
    }
}
