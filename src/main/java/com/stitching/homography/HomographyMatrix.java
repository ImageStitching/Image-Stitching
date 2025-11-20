package com.stitching.homography;

import com.stitching.SIFT.Keypoint;

public class HomographyMatrix {
    private double[][] data;

    public HomographyMatrix(double[][] data) {
        this.data = data;
    }

    public double[][] getData() {
        return data;
    }

    public double[] project(double x, double y) {
        double z_prime = data[2][0] * x + data[2][1] * y + data[2][2];
        if (Math.abs(z_prime) < 1e-10) return null;

        double x_prime = (data[0][0] * x + data[0][1] * y + data[0][2]) / z_prime;
        double y_prime = (data[1][0] * x + data[1][1] * y + data[1][2]) / z_prime;

        return new double[]{x_prime, y_prime};
    }

    public HomographyMatrix inverse() {
        return null; 
    }
}