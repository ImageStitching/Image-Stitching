package com.stitching.selfcode.SIFTlỏ;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SiftStage3 {
    private final int numOrientationBins = 36;
    private final double peakRatioThreshold = 0.8;

    private int imageHeight;
    private int imageWidth;
    private boolean enablePreciseUpscale;

    public SiftStage3() {}

    public void setImageDimensions(int height, int width, boolean enablePreciseUpscale) {
        this.imageHeight = height;
        this.imageWidth = width;
        this.enablePreciseUpscale = enablePreciseUpscale;
    }

    private boolean isWithinValidBounds(Keypoint kp, double minBorder) {
        double scaleFactor = enablePreciseUpscale ? Math.pow(2.0, kp.octave - 1.0) : Math.pow(2.0, kp.octave);
        double originalX = kp.x * scaleFactor;
        double originalY = kp.y * scaleFactor;
        boolean withinBounds = originalX >= minBorder && originalX < imageWidth - minBorder && originalY >= minBorder && originalY < imageHeight - minBorder;
        return withinBounds;
    }

    public List<OrientedKeypoint> run(List<Keypoint> refinedKeypoints, List<List<SiftImage>> gaussianPyramid) {
        setImageDimensions(gaussianPyramid.get(0).get(0).data.length,gaussianPyramid.get(0).get(0).data[0].length,refinedKeypoints.get(0).enable_precise_upscale);

        List<OrientedKeypoint> orientedKeypoints = new ArrayList<>();
        double MIN_BORDER = 1.0;
        for (Keypoint kp : refinedKeypoints) {
            if (!isWithinValidBounds(kp, MIN_BORDER)) continue;
            List<Double> orientations = calculateOrientations(kp, gaussianPyramid);
            if (orientations.isEmpty()) orientations.add(0.0);
            for (double orientation : orientations) orientedKeypoints.add(new OrientedKeypoint(kp, orientation));
        }
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
        int cx = (int) Math.round(kp.x);
        int cy = (int) Math.round(kp.y);

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int y = cy + dy;
                int x = cx + dx;
                if (y <= 0 || y >= height - 1 || x <= 0 || x >= width - 1)
                    continue;
                // Central difference
                double gx = imageData[y][x + 1] - imageData[y][x - 1];
                double gy = imageData[y + 1][x] - imageData[y - 1][x];
                double magnitude = Math.sqrt(gx * gx + gy * gy);
                double angle = Math.atan2(gy, gx);
                if (angle < 0) angle += 2.0 * Math.PI;
                double weight = Math.exp(-(dx * dx + dy * dy) / (2 * weightSigma * weightSigma));
                int bin = (int) (angle * numOrientationBins / (2 * Math.PI));
                bin = (bin + numOrientationBins) % numOrientationBins;
                histogram[bin] += weight * magnitude;
            }
        }
        for (int i = 0; i < 6; i++) {
            histogram = smoothHistogram(histogram);
        }
        return findPeaks(histogram);
    }

    private double[] smoothHistogram(double[] hist) {
        double[] out = new double[numOrientationBins];

        for (int i = 0; i < numOrientationBins; i++) {
            double prev = hist[(i - 1 + numOrientationBins) % numOrientationBins];
            double next = hist[(i + 1) % numOrientationBins];
            out[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * next;
        }
        return out;
    }

    private List<Double> findPeaks(double[] histogram) {
        List<Double> orientations = new ArrayList<>();
        double maxPeak = Arrays.stream(histogram).max().orElse(0.0);
        for (int i = 0; i < numOrientationBins; i++) {
            double prev = histogram[(i - 1 + numOrientationBins) % numOrientationBins];
            double curr = histogram[i];
            double next = histogram[(i + 1) % numOrientationBins];
            if (curr > prev && curr > next && curr >= maxPeak * peakRatioThreshold) {
                // Quadratic interpolation
                double interp = 0.5 * (prev - next) / (prev - 2 * curr + next);
                interp = Math.max(-1.0, Math.min(1.0, interp));
                double peakBin = (i + interp + numOrientationBins) % numOrientationBins;
                double angle = peakBin * (2 * Math.PI / numOrientationBins);
                orientations.add(angle);
            }
        }
        return orientations;
    }
}
