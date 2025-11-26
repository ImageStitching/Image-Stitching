package com.stitching.selfcode.filter_convolution_gauss;

import com.stitching.selfcode.imageOperator.ColourImageToGray;
import edu.princeton.cs.introcs.Picture;

import java.awt.*;
import java.nio.file.Path;
import java.nio.file.Paths;

public class LinearFiltering {
    private static final Path OUTPUT_PATH = Paths.get("src","main","resources","static","image");
    // Làm mờ ảnh
    public static final int KERNEL_SIZE = 3;
    public static final float SHARP_AMOUNT = 2;
    public static final float[][] BOX_BLUR = new float[KERNEL_SIZE][KERNEL_SIZE];
    static {
        float val = 1.0f / (KERNEL_SIZE * KERNEL_SIZE);
        for (int i = 0; i < KERNEL_SIZE; i++)
            for (int j = 0; j < KERNEL_SIZE; j++)
                BOX_BLUR[i][j] = val;
    }

    // Làm nét ảnh
    public static final float[][] SHARPEN = new float[KERNEL_SIZE][KERNEL_SIZE];
    static {
        int size = KERNEL_SIZE;
        float amount = SHARP_AMOUNT;
        if (size % 2 == 0 || size < 3) {
            throw new IllegalArgumentException("Kernel size phải là số lẻ >= 3");
        }

        float blurValue = 1.0f / (size * size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                SHARPEN[i][j] = -amount * blurValue;
            }
        }
        int center = size / 2;
        SHARPEN[center][center] += (1.0f + amount);
    }
    //{
//            { 0, -1,  0},
//            {-1,  5, -1},
//            { 0, -1,  0}
//    };

    public static Picture linearFilter(Picture image, float[][] kernel) {
        int width = image.width();
        int height = image.height();
        int kernelWidth = kernel.length;
        int kernelHeight = kernel.length;
        int kernelCenterX = kernelWidth / 2;
        int kernelCenterY = kernelHeight / 2;

        Picture outputImage = new Picture(width, height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sumR = 0, sumG = 0, sumB = 0;

                for (int ky = 0; ky < kernelHeight; ky++) {
                    for (int kx = 0; kx < kernelWidth; kx++) {
                        int pixelX = x + (kx - kernelCenterX);
                        int pixelY = y + (ky - kernelCenterY);

                        // Xử lý các pixel ở biên bằng cách lặp lại pixel cạnh (clamping)
                        if (pixelX < 0) pixelX = 0;
                        if (pixelX >= width) pixelX = width - 1;
                        if (pixelY < 0) pixelY = 0;
                        if (pixelY >= height) pixelY = height - 1;

                        Color pixelColor = image.get(pixelX,pixelY);
                        float kernelVal = kernel[ky][kx];

                        sumR += pixelColor.getRed() * kernelVal;
                        sumG += pixelColor.getGreen() * kernelVal;
                        sumB += pixelColor.getBlue() * kernelVal;
                    }
                }

                int outR = Math.min(Math.max((int) sumR, 0), 255);
                int outG = Math.min(Math.max((int) sumG, 0), 255);
                int outB = Math.min(Math.max((int) sumB, 0), 255);

                outputImage.set(x, y, new Color(outR, outG, outB));
            }
        }
        return outputImage;
    }

    public static void main(String[] args) {
        linearFilter(ColourImageToGray.createGrayPictureFromLink(OUTPUT_PATH.resolve("imgColor.png").toString()), SHARPEN).show();
    }
}
