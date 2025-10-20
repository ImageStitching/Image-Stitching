package com.stitching.SIFT;

public class SiftImage {
    public final double[][] data;
    public final double sigma;

    public SiftImage(double[][] data, double sigma) {
        this.data = data;
        this.sigma = sigma;
    }

    public int getWidth() {
        return data.length;
    }

    public int getHeight() {
        return data.length;
    }
}



