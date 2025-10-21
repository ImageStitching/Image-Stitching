package com.stitching.SIFT;

/****
 * Trong các giai đoạn giảm kích thước ảnh để phục vụ xây dựng Octave Pymarid , thì ta ần thu nhor ảnh và phóng to ảnh
 * Lúc ảnh ban đầu cần dùng phép Nội suy để phóng to gấp đôi sau đó sẽ làm mịn để xác định được nhiều Keypoints hơn
 */
public class Up_DownSample {

    /**
     * Tăng gấp đôi kích thước ảnh bằng phép nội suy song tuyến tính (Bilinear Interpolation).
     * @param originalImage ma trận ảnh gốc 2 chiều.
     * @param enable_precise_upscale boolean = True thì bật Nội suy, = False thì không nội suy
     */
    public static double[][] upsampleWithLinearInterpolation(double[][] originalImage, boolean enable_precise_upscale) {
        if(! enable_precise_upscale)
            return originalImage;

        int originalHeight = originalImage.length;
        int originalWidth = originalImage[0].length;
        int newHeight = originalHeight * 2;
        int newWidth = originalWidth * 2;
        double[][] upsampledImage = new double[newHeight][newWidth];

        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                double origX = (double) x / 2.0;
                double origY = (double) y / 2.0;

                int x1 = (int) Math.floor(origX);
                int y1 = (int) Math.floor(origY);
                int x2 = Math.min(x1 + 1, originalWidth - 1);
                int y2 = Math.min(y1 + 1, originalHeight - 1);

                double val_y1x1 = originalImage[y1][x1];
                double val_y1x2 = originalImage[y1][x2];
                double val_y2x1 = originalImage[y2][x1];
                double val_y2x2 = originalImage[y2][x2];

                double dx = origX - x1;
                double dy = origY - y1;

                double topInterpolation = val_y1x1 * (1 - dx) + val_y1x2 * dx;
                double bottomInterpolation = val_y2x1 * (1 - dx) + val_y2x2 * dx;

                upsampledImage[y][x] = topInterpolation * (1 - dy) + bottomInterpolation * dy;
            }
        }
        return upsampledImage;
    }


    public static double[][] downsample(double[][] image) {
        int newHeight = image.length / 2;
        int newWidth = image[0].length / 2;
        double[][] newImage = new double[newHeight][newWidth];
        for (int r = 0; r < newHeight; r++) {
            for (int c = 0; c < newWidth; c++) {
                newImage[r][c] = image[r * 2][c * 2];
            }
        }
        return newImage;
    }

}
