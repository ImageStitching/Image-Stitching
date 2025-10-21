package com.stitching.SIFT;

import java.util.ArrayList;
import java.util.List;

/**
 * Giai đoạn 4: Tạo bộ mô tả SIFT (Descriptor)
 * Mỗi keypoint được biểu diễn bằng vector 128 chiều (4x4x8)
 */
public class SiftStage4 {

    private final int DESCRIPTOR_WIDTH = 4; // Lưới 4x4 subregions
    private final int DESCRIPTOR_BINS = 8; // 8 bins cho mỗi histogram hướng
    private final int DESCRIPTOR_SIZE = DESCRIPTOR_WIDTH * DESCRIPTOR_WIDTH * DESCRIPTOR_BINS; // 128
    private final double DESCRIPTOR_MAG_THRESHOLD = 0.2; // Ngưỡng cắt để giảm ảnh hưởng của gradient lớn

    /**
     * Hàm chính để chạy Giai đoạn 4.
     * @param orientedKeypoints Danh sách các điểm khóa có hướng từ Giai đoạn 3
     * @param gaussianPyramid Kim tự tháp Gaussian
     * @return Danh sách các bộ mô tả SIFT
     */
    public List<SiftDescriptor> run(List<OrientedKeypoint> orientedKeypoints, List<List<SiftImage>> gaussianPyramid) {
        System.out.println("\n--- Bắt đầu Giai đoạn 4: Tạo bộ mô tả điểm khóa ---");
        List<SiftDescriptor> descriptors = new ArrayList<>();

        for (OrientedKeypoint okp : orientedKeypoints) {
            double[] descriptorVector = createDescriptor(okp, gaussianPyramid);
            if (descriptorVector != null) {
                descriptors.add(new SiftDescriptor(okp, descriptorVector));
            }
        }

        System.out.printf("Đã tạo ra %d bộ mô tả SIFT (mỗi descriptor có %d chiều).\n",
                descriptors.size(), DESCRIPTOR_SIZE);
        System.out.println("--- Giai đoạn 4 Hoàn tất ---");
        return descriptors;
    }

    /**
     * Tạo vector mô tả 128 chiều cho một điểm khóa có hướng.
     * Vector được tính từ lưới 4x4 subregions, mỗi subregion có histogram 8 bins.
     */
    private double[] createDescriptor(OrientedKeypoint okp, List<List<SiftImage>> gaussianPyramid) {
        SiftImage image = gaussianPyramid.get(okp.octave).get(okp.layer);
        double[][] imageData = image.data;
        int height = image.getHeight();
        int width = image.getWidth();

        // Bán kính vùng mô tả: 3 * 1.5 * sigma (theo paper gốc của Lowe)
        // Nhân với sqrt(2) để bao phủ vùng sau khi xoay
        double radius = 3.0 * 1.5 * okp.sigma * Math.sqrt(2);
        int radiusInt = (int) Math.round(radius);

        // Khởi tạo histogram 3D: [y_bin][x_bin][orientation_bin]
        double[][][] hist = new double[DESCRIPTOR_WIDTH][DESCRIPTOR_WIDTH][DESCRIPTOR_BINS];

        double cos_t = Math.cos(okp.orientation);
        double sin_t = Math.sin(okp.orientation);

        // Duyệt qua tất cả các pixel trong vùng bán kính
        for (int i = -radiusInt; i <= radiusInt; i++) {
            for (int j = -radiusInt; j <= radiusInt; j++) {
                int sampleY = (int) Math.round(okp.y) + i;
                int sampleX = (int) Math.round(okp.x) + j;

                // Kiểm tra biên để tính gradient
                if (sampleY <= 0 || sampleY >= height - 1 ||
                        sampleX <= 0 || sampleX >= width - 1) {
                    continue;
                }

                // Xoay tọa độ về hệ tọa độ chuẩn hóa của keypoint
                // (j, i) là offset từ keypoint, xoay về hướng chuẩn
                double rotatedX = (j * cos_t + i * sin_t);
                double rotatedY = (-j * sin_t + i * cos_t);

                // Chuẩn hóa theo scale của keypoint và chuyển về tọa độ lưới 4x4
                // Mỗi bin có width = 3 * sigma (theo paper Lowe)
                double binWidth = 3.0 * okp.sigma;
                double histX = rotatedX / binWidth + DESCRIPTOR_WIDTH / 2.0 - 0.5;
                double histY = rotatedY / binWidth + DESCRIPTOR_WIDTH / 2.0 - 0.5;

                // Bỏ qua các pixel nằm ngoài vùng descriptor 4x4
                if (histX < -1.0 || histX >= DESCRIPTOR_WIDTH ||
                        histY < -1.0 || histY >= DESCRIPTOR_WIDTH) {
                    continue;
                }

                // Tính gradient tại điểm mẫu
                double dx = imageData[sampleY][sampleX + 1] - imageData[sampleY][sampleX - 1];
                double dy = imageData[sampleY + 1][sampleX] - imageData[sampleY - 1][sampleX];
                double magnitude = Math.sqrt(dx * dx + dy * dy);
                double orientation = Math.atan2(dy, dx);

                // Xoay hướng gradient về hệ tọa độ của keypoint
                double rotatedOrientation = orientation - okp.orientation;
                while (rotatedOrientation < 0) rotatedOrientation += 2 * Math.PI;
                while (rotatedOrientation >= 2 * Math.PI) rotatedOrientation -= 2 * Math.PI;

                // Tính bin hướng (0-7)
                double histO = rotatedOrientation * DESCRIPTOR_BINS / (2.0 * Math.PI);

                // Áp dụng trọng số Gaussian để giảm ảnh hưởng của các pixel xa
                // Sigma của Gaussian window = DESCRIPTOR_WIDTH / 2 (theo Lowe)
                double gaussianSigma = DESCRIPTOR_WIDTH / 2.0;
                double weight = Math.exp(-(histX * histX + histY * histY) /
                        (2.0 * gaussianSigma * gaussianSigma));
                double weightedMagnitude = magnitude * weight;

                // Nội suy tuyến tính 3 chiều (trilinear interpolation)
                // Phân bố weighted magnitude vào 8 bins lân cận
                trilinearInterpolation(hist, histY, histX, histO, weightedMagnitude);
            }
        }

        // Chuyển histogram 3D thành vector 1D và chuẩn hóa
        return flattenAndNormalize(hist);
    }

    /**
     * Nội suy tuyến tính 3 chiều để phân bố giá trị vào các bins lân cận.
     * Giúp descriptor ổn định hơn với các thay đổi nhỏ về vị trí và hướng.
     */
    private void trilinearInterpolation(double[][][] hist, double y, double x, double o, double magnitude) {
        int y0 = (int) Math.floor(y);
        int x0 = (int) Math.floor(x);
        int o0 = (int) Math.floor(o);

        double dy = y - y0;
        double dx = x - x0;
        double dor = o - o0;

        // Duyệt qua 8 bins lân cận (2x2x2)
        for (int dy_bin = 0; dy_bin <= 1; dy_bin++) {
            int y_bin = y0 + dy_bin;
            if (y_bin < 0 || y_bin >= DESCRIPTOR_WIDTH) continue;

            double wy = (dy_bin == 0) ? (1.0 - dy) : dy;

            for (int dx_bin = 0; dx_bin <= 1; dx_bin++) {
                int x_bin = x0 + dx_bin;
                if (x_bin < 0 || x_bin >= DESCRIPTOR_WIDTH) continue;

                double wx = (dx_bin == 0) ? (1.0 - dx) : dx;

                for (int do_bin = 0; do_bin <= 1; do_bin++) {
                    int o_bin = (o0 + do_bin) % DESCRIPTOR_BINS; // Xử lý wrap-around cho orientation

                    double wo = (do_bin == 0) ? (1.0 - dor) : dor;

                    // Phân bố magnitude theo trọng số nội suy
                    hist[y_bin][x_bin][o_bin] += magnitude * wy * wx * wo;
                }
            }
        }
    }

    /**
     * Chuyển histogram 3D thành vector 1D (128 chiều) và thực hiện chuẩn hóa.
     * Chuẩn hóa giúp descriptor bất biến với thay đổi độ sáng và độ tương phản.
     */
    private double[] flattenAndNormalize(double[][][] hist) {
        double[] descriptor = new double[DESCRIPTOR_SIZE];
        int index = 0;

        // Flatten theo thứ tự: y -> x -> orientation
        for (int i = 0; i < DESCRIPTOR_WIDTH; i++)
            for (int j = 0; j < DESCRIPTOR_WIDTH; j++)
                for (int k = 0; k < DESCRIPTOR_BINS; k++)
                    descriptor[index++] = hist[i][j][k];

        // Bước 1: Chuẩn hóa L2 (Euclidean normalization)
        double norm = 0.0;
        for (double val : descriptor) {
            norm += val * val;
        }
        norm = Math.sqrt(norm);

        if (norm > 1e-7) {
            for (int i = 0; i < descriptor.length; i++) {
                descriptor[i] /= norm;
            }
        } else {
            // Nếu vector rỗng, trả về null
            return null;
        }

        // Bước 2: Cắt ngưỡng các giá trị lớn (threshold clipping)
        // Giảm ảnh hưởng của các gradient cực mạnh (non-linear illumination)
        for (int i = 0; i < descriptor.length; i++) {
            if (descriptor[i] > DESCRIPTOR_MAG_THRESHOLD) {
                descriptor[i] = DESCRIPTOR_MAG_THRESHOLD;
            }
        }

        // Bước 3: Chuẩn hóa L2 lần nữa
        norm = 0.0;
        for (double val : descriptor) {
            norm += val * val;
        }
        norm = Math.sqrt(norm);

        if (norm > 1e-7) {
            for (int i = 0; i < descriptor.length; i++) {
                descriptor[i] /= norm;
            }
        }

        return descriptor;
    }
}