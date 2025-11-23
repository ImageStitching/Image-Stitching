package com.stitching.filter_convolution_gauss;

import edu.princeton.cs.introcs.Picture;

import java.awt.*;
import java.nio.file.Path;
import java.nio.file.Paths;

public class SeparabilityGauss {
    private static final Path OUTPUT_PATH = Paths.get("src","main","resources","static","image");

    /**
     * Tạo một hạt nhân Gaussian 1D.
     * @param sigma Độ lệch chuẩn.
     * @return Hạt nhân Gaussian 1D đã được chuẩn hóa.
     */
    private static float[] create1DGaussianKernel(int size, float sigma) {
        int radius = size / 2;
        float[] kernel = new float[size];
        float sum = 0;
        for (int i = 0; i < size; i++) {
            int d = i - radius;
            float value = (float) (Math.exp(-(d * d) / (2 * sigma * sigma)) / (Math.sqrt(2 * Math.PI) * sigma));
            kernel[i] = value;
            sum += value;
        }
        for (int i = 0; i < size; i++) kernel[i] /= sum;
        return kernel;
    }

    /**
     * Áp dụng bộ lọc Gaussian khả tách (hiệu quả hơn) lên ảnh.
     *
     * @param image Ảnh đầu vào.
     * @param sigma Độ lệch chuẩn.
     * @return Ảnh đã được làm mờ.
     */
    public Picture separabilityGaussianFilter(Picture image, int size,float sigma) {
        int width = image.width();
        int height = image.height();

        // Tạo kernel Gaussian 1D
        float[] kernel = create1DGaussianKernel(size, sigma);
        int radius = kernel.length / 2;

        Picture outputImage = new Picture(width, height);

        float[][][] tempPixels = new float[height][width][3];

        // --- Lượt 1: Lọc theo chiều ngang ---
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sumR = 0, sumG = 0, sumB = 0;

                for (int k = 0; k < kernel.length; k++) {
                    int pixelX = x + k - radius;
                    if (pixelX < 0) pixelX = 0;
                    if (pixelX >= width) pixelX = width - 1;

                    Color c = image.get(pixelX, y);
                    sumR += c.getRed() * kernel[k];
                    sumG += c.getGreen() * kernel[k];
                    sumB += c.getBlue() * kernel[k];
                }

                tempPixels[y][x][0] = sumR;
                tempPixels[y][x][1] = sumG;
                tempPixels[y][x][2] = sumB;
            }
        }

        // --- Lượt 2: Lọc theo chiều dọc ---
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sumR = 0, sumG = 0, sumB = 0;

                for (int k = 0; k < kernel.length; k++) {
                    int pixelY = y + k - radius;
                    if (pixelY < 0) pixelY = 0;
                    if (pixelY >= height) pixelY = height - 1;

                    sumR += tempPixels[pixelY][x][0] * kernel[k];
                    sumG += tempPixels[pixelY][x][1] * kernel[k];
                    sumB += tempPixels[pixelY][x][2] * kernel[k];
                }

                int outR = Math.min(Math.max(Math.round(sumR), 0), 255);
                int outG = Math.min(Math.max(Math.round(sumG), 0), 255);
                int outB = Math.min(Math.max(Math.round(sumB), 0), 255);
                outputImage.set(x, y, new Color(outR, outG, outB));
            }
        }
        return outputImage;
    }

    public static double[][] seperabilityGauss(double[][] img, double sigma) {
        int radius = (int) Math.ceil(3 * sigma);
        int size = 2 * radius + 1;
        int height = img.length;
        int width = img[0].length;

        float[] kernel = create1DGaussianKernel(size, (float) sigma);
        double[][] tempImage = new double[height][width];
        double[][] outputImage = new double[height][width];

        // --- Lượt 1: Lọc theo chiều ngang ---
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0;
                for (int k = 0; k < kernel.length; k++) {
                    int pixelX = x + k - radius;
                    if (pixelX < 0) pixelX = 0;
                    if (pixelX >= width) pixelX = width - 1;

                    sum += img[y][pixelX] * kernel[k];
                }
                tempImage[y][x] = sum;
            }
        }

        // Lượt 2: Lọc theo chiều dọc
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0;
                for (int k = 0; k < kernel.length; k++) {
                    int pixelY = y + k - radius;
                    if (pixelY < 0) pixelY = 0;
                    if (pixelY >= height) pixelY = height - 1;

                    sum += tempImage[pixelY][x] * kernel[k];
                }
                outputImage[y][x] = sum;
            }
        }
        return outputImage;
    }
}
