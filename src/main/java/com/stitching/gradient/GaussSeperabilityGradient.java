package com.stitching.gradient;

import java.util.Arrays;

public class GaussSeperabilityGradient {
    private static double[] GAUSS_KERNEL;
    private static double[] DERIVATIVE_GAUSS_KERNEL;

    public int kernel_size;
    public double sigma;

    public GaussSeperabilityGradient(int size, double sigma) {
        this.kernel_size = size; this.sigma = sigma;
        GAUSS_KERNEL = createGaussianKernel1D(size, sigma);
        DERIVATIVE_GAUSS_KERNEL = createDerivativeGaussianKernel1D(size, sigma);
    }
    /**
     * Tạo một nhân lọc Gaussian 1D.
     */
    private double[] createGaussianKernel1D(int size, double sigma) {
        if (size % 2 == 0) throw new IllegalArgumentException("Kích thước kernel phải là số lẻ.");
        double[] kernel = new double[size];
        int center = size / 2;
        double sum = 0;
        for (int i = 0; i < size; i++) {
            int d = i - center;
            kernel[i] = Math.exp(-(d * d) / (2 * sigma * sigma));
            sum += kernel[i];
        }
        // Chuẩn hóa để tổng bằng 1
        for (int i = 0; i < size; i++) {
            kernel[i] /= sum;
        }
        return kernel;
    }

    /**
     * Tạo nhân lọc đạo hàm của Gaussian theo 1 hướng x hoặc y.
     * Công thức: dG/dx(x, y) = -x / (2 * pi * sigma^4) * exp(-(x^2 + y^2) / (2 * sigma^2))
     * Để đơn giản, ta có thể tính: -x * G(x, y) / (sigma^2) rồi chuẩn hóa.
     * Kernel này có tính đối xứng lẻ và tổng các phần tử bằng 0.
     */
    private double[] createDerivativeGaussianKernel1D(int size, double sigma) {
        if (size % 2 == 0) throw new IllegalArgumentException("Kích thước kernel phải là số lẻ.");
        double[] kernel = new double[size];
        int center = size / 2;
        double sum = 0;

        for (int i = 0; i < size; i++) {
            int d = i - center;
            kernel[i] = -d * Math.exp(-(d * d) / (2 * sigma * sigma));
            sum += i < center ? -kernel[i] : kernel[i]; // Chuẩn hóa để tổng các giá trị dương bằng 1
        }

        if (sum != 0) {
            for (int i = 0; i < size; i++) {
                kernel[i] /= sum;
            }
        }
        return kernel;
    }

    /**
     * Thực hiện tích chập 1D trên ảnh.
     * @param image Ảnh đầu vào.
     * @param kernel Kernel 1D.
     * @param isHorizontal true nếu tích chập theo chiều ngang, false nếu theo chiều dọc.
     * @return Ảnh mới sau khi tích chập.
     */
    private double[][] convolve1D(double[][] image, double[] kernel, boolean isHorizontal) {
        int height = image.length;
        int width = image[0].length;
        int kernelSize = kernel.length;
        int center = kernelSize / 2;
        double[][] output = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0;
                for (int k = 0; k < kernelSize; k++) {
                    if (isHorizontal) {
                        int imageX = x + k - center;
                        imageX = Math.max(0, Math.min(imageX, width - 1));
                        sum += image[y][imageX] * kernel[k];
                    } else { // Vertical
                        int imageY = y + k - center;
                        imageY = Math.max(0, Math.min(imageY, height - 1));
                        sum += image[imageY][x] * kernel[k];
                    }
                }
                output[y][x] = sum;
            }
        }
        return output;
    }

    public double[][] gradientX(int[][] grayImage) {
        // 2. Tính toán Gx (Gradient theo hướng x) Gx = (Image * dG_1D_horizontal) * G_1D_vertical
        double[][] grayImg = Arrays.stream(grayImage)
                                .map(row -> Arrays.stream(row).asDoubleStream().toArray())
                                .toArray(double[][]::new);
        double[][] tempGx = convolve1D( grayImg , DERIVATIVE_GAUSS_KERNEL, true); // true = horizontal
        double[][] gx = convolve1D(tempGx, GAUSS_KERNEL, false); // false = vertical
        return gx;
    }

    public double[][] gradientY(int[][] grayImage) {
        //Tính toán Gy (Gradient theo hướng y): Gy = (Image * G_1D_horizontal) * dG_1D_vertical
        double[][] grayImg = Arrays.stream(grayImage)
                .map(row -> Arrays.stream(row).asDoubleStream().toArray())
                .toArray(double[][]::new);
        double[][] tempGy = convolve1D( grayImg , GAUSS_KERNEL, true); // true = horizontal
        double[][] gy = convolve1D(tempGy, DERIVATIVE_GAUSS_KERNEL, false); // false = vertical
        return gy;
    }

    // 2. Tính toán Gx (Gradient theo hướng x)
    // Gx = (Image * dG_1D_horizontal) * G_1D_vertical
    //double[][] tempGx = convolve1D(grayImage, derivativeKernel, true); // true = horizontal
    //double[][] gx = convolve1D(tempGx, gaussianKernel, false);       // false = vertical

    // 3. Tính toán Gy (Gradient theo hướng y)
    // Gy = (Image * G_1D_horizontal) * dG_1D_vertical
    //double[][] tempGy = convolve1D(grayImage, gaussianKernel, true);   // true = horizontal
    //double[][] gy = convolve1D(tempGy, derivativeKernel, false);     // false = vertical
}
