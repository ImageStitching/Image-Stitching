package com.stitching.SIFTlỏ;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class SiftImage {
    public final double[][] data;
    public final double sigma;

    public SiftImage(double[][] data, double sigma) {
        this.data = data;
        this.sigma = sigma;
    }

    public int getWidth() {
        return data[0].length;
    }

    public int getHeight() {
        return data.length;
    }

    public double div(SiftImage that) {
        return (this.sigma > that.sigma) ? (this.sigma / that.sigma) : (that.sigma / this.sigma);
    }
}