package com.stitching.filter_convolution_gauss;

import com.stitching.imageOperator.ColourImageToGray;
import edu.princeton.cs.introcs.Picture;

import java.nio.file.Path;
import java.nio.file.Paths;

public class Gauss {
    private static final Path OUTPUT_PATH = Paths.get("src","main","resources","static","image");
    /**
     * Tạo một hạt nhân Gaussian 2D.
     *
     * @param sigma Độ lệch chuẩn (mức độ mờ).
     * @return Hạt nhân Gaussian 2D đã được chuẩn hóa.
     */
    public static float[][] createGaussianKernel(float sigma) {
        int radius = (int) Math.ceil(3 * sigma);
        int size = 2 * radius + 1;
        float[][] kernel = new float[size][size];
        float sum = 0;

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int dx = x - radius;
                int dy = y - radius;
                float value = (float) (Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)) / (2 * Math.PI * sigma * sigma));
                kernel[y][x] = value;
                sum += value;
            }
        }

        // Chuẩn hóa hạt nhân để tổng các phần tử bằng 1
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                kernel[y][x] /= sum;
            }
        }
        return kernel;
    }

    /**
     * Áp dụng bộ lọc Gaussian lên ảnh.
     *
     * @param link_image link Ảnh đầu vào.
     * @param sigma Độ lệch chuẩn của bộ lọc Gaussian.
     * @return Ảnh đã được làm mờ bằng bộ lọc Gaussian.
     */
    public static Picture gaussianFilter(String link_image, float sigma) {
        float[][] kernel = createGaussianKernel(sigma);
        return LinearFiltering.linearFilter(
                ColourImageToGray.createGrayPictureFromLink(OUTPUT_PATH.resolve("imgColor.png").toString()),
                kernel
        );
    }

}
