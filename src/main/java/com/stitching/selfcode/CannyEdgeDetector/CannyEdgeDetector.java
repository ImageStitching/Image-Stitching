package com.stitching.selfcode.CannyEdgeDetector;

import com.stitching.selfcode.gradient.GaussSeperabilityGradient;
import com.stitching.selfcode.imageOperator.Matrix_Image;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Lớp này thực hiện Giai đoạn 1 của thuật toán Canny Edge Detector.
 * Nó bao gồm việc giảm nhiễu và tính toán gradient của ảnh bằng cách
 * sử dụng phương pháp tách biệt (separable) của bộ lọc đạo hàm Gaussian,
 * giúp tăng hiệu quả tính toán.
 */
public class CannyEdgeDetector {
    private static final Path OUTPUT_PATH = Paths.get("src","main","resources","static","image");

    private static GaussSeperabilityGradient gauss;

    public CannyEdgeDetector(int kernelSize, double sigma) {
        gauss = new GaussSeperabilityGradient(kernelSize,sigma);
    }

    public static class CannyStage1Result {
        public final double[][] gx;
        public final double[][] gy;
        public final double[][] magnitude;
        public final double[][] orientation; // In radians

        public CannyStage1Result(double[][] gx, double[][] gy, double[][] magnitude, double[][] orientation) {
            this.gx = gx;
            this.gy = gy;
            this.magnitude = magnitude;
            this.orientation = orientation;
        }

        @Override
        public String toString() {
//            return "CannyStage1Result{" +
//                    "gx=" + Arrays.toString(gx) +
//                    ", gy=" + Arrays.toString(gy) +
//                    ", magnitude=" + Arrays.toString(magnitude) +
//                    ", orientation=" + Arrays.toString(orientation) +
//                    '}';
            String result = "*********** CannyStage1Result ***********";
            
//            result += "\n\nĐạo hàm theo hướng x là Gx:\n\n";
//            result += Arrays.stream(gx)
//                    .map(row -> Arrays.stream(row)
//                            .mapToObj(v -> String.format("%.2f", v))
//                            .collect(Collectors.joining(" ")))
//                    .collect(Collectors.joining("\n"));
//
//            result += "\n\nĐạo hàm theo hướng y là Gy:\n\n";
//            result += Arrays.stream(gy)
//                    .map(row -> Arrays.stream(row)
//                            .mapToObj(v -> String.format("%.2f", v))
//                            .collect(Collectors.joining(" ")))
//                    .collect(Collectors.joining("\n"));
//
            result += "\n\n Độ dốc magitude và hướng orientation (theo độ thang góc 360 độ) của từng pixel\n";
            result += "Ví dụ hình chữ nhật pixel[150][50] -> pixel[180][80]\n";
            for(int y=150; y<181; y++) {
                for (int x = 50; x < 81; x++)
                    result += "<" + String.format("%.2f", magnitude[y][x]) + " - " + String.format("%.2f", orientation[y][x] * 180 / Math.PI) + "> ";
                result += "\n";
            }
            return result;
        }
    }

    /**
     * Thực hiện giai đoạn 1 của thuật toán Canny.
     * @param grayImage Ảnh thang độ xám đầu vào (dạng mảng 2D).
     * @param kernelSize Kích thước của kernel (nên là số lẻ, ví dụ 5, 7).
     * @param sigma     Độ lệch chuẩn (sigma) của hàm Gaussian.
     * @return Một đối tượng chứa ảnh gradient Gx, Gy, độ lớn và hướng của gradient.
     */
    public CannyStage1Result reduceNoiseAndCalcGradient(int[][] grayImage) {
        int height = grayImage.length;
        int width = grayImage[0].length;

        // Làm mịn và đạo hàm với Derective Gauss
        double[][] gx = gauss.gradientX(grayImage);
        double[][] gy = gauss.gradientY(grayImage);

        // Tính toán độ lớn (magnitude) và hướng (orientation) của gradient
        double[][] magnitude = new double[height][width];
        double[][] orientation = new double[height][width]; // Lưu dưới dạng radian

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                magnitude[y][x] = Math.sqrt(gx[y][x] * gx[y][x] + gy[y][x] * gy[y][x]);
                orientation[y][x] = Math.atan2(gy[y][x], gx[y][x]);
            }
        }

        return new CannyStage1Result(gx, gy, magnitude, orientation);
    }

    /**
     * Giai đoạn 2: Non-Maximum Suppression (Loại bỏ điểm không phải cực đại).
     * Mục đích: Làm mỏng các đường biên dày, chỉ giữ lại các pixel là cực đại cục bộ.
     *
     * @param stage1Result Kết quả từ Giai đoạn 1, chứa độ lớn và hướng gradient.
     * @return Một ảnh thang độ xám (dạng mảng 2D) với các đường biên đã được làm mỏng.
     */
    public double[][] nonMaximumSuppression(CannyStage1Result stage1Result) {
        int height = stage1Result.magnitude.length;
        int width = stage1Result.magnitude[0].length;
        double[][] suppressedMag = new double[height][width];

        // Duyệt qua các pixel bên trong ảnh (bỏ qua đường viền 1 pixel)
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double angle = stage1Result.orientation[y][x]; // Hướng gradient tính bằng radian
                double mag = stage1Result.magnitude[y][x];

                double angleDegrees = angle * 180.0 / Math.PI;
                if (angleDegrees < 0) {
                    angleDegrees += 180;
                }

                double neighbor1 = 0, neighbor2 = 0;

                if ((0 <= angleDegrees && angleDegrees < 22.5) || (157.5 <= angleDegrees && angleDegrees <= 180)) {
                    neighbor1 = stage1Result.magnitude[y][x + 1];
                    neighbor2 = stage1Result.magnitude[y][x - 1];
                }
                else if (22.5 <= angleDegrees && angleDegrees < 67.5) {
                    neighbor1 = stage1Result.magnitude[y - 1][x + 1];
                    neighbor2 = stage1Result.magnitude[y + 1][x - 1];
                }
                else if (67.5 <= angleDegrees && angleDegrees < 112.5) {
                    neighbor1 = stage1Result.magnitude[y - 1][x];
                    neighbor2 = stage1Result.magnitude[y + 1][x];
                }
                else if (112.5 <= angleDegrees && angleDegrees < 157.5) {
                    neighbor1 = stage1Result.magnitude[y - 1][x - 1];
                    neighbor2 = stage1Result.magnitude[y + 1][x + 1];
                }

                // Nếu pixel hiện tại là cực đại cục bộ, giữ lại giá trị độ lớn
                if (mag >= neighbor1 && mag >= neighbor2) {
                    suppressedMag[y][x] = mag;
                } else {
                    suppressedMag[y][x] = 0;
                }
            }
        }
        return suppressedMag;
    }

    public static void main(String[] args) {
        CannyEdgeDetector cannyEdgeDetector = new CannyEdgeDetector(3,1.6);
        CannyStage1Result cannyStage1Result = cannyEdgeDetector.reduceNoiseAndCalcGradient(
                Matrix_Image.create_INTgrayMatrix_from_color_image(OUTPUT_PATH.resolve("imgColor.png").toString())
        );
        System.out.println(cannyStage1Result);
//        System.out.println(Arrays.toString(cannyEdgeDetector.nonMaximumSuppression(cannyStage1Result)));
    }
}
