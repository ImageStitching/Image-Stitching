package com.stitching.SIFT;

import java.util.ArrayList;
import java.util.List;

/**
 * Lớp này đại diện cho một bộ mô tả SIFT hoàn chỉnh.
 * Nó chứa tất cả thông tin của OrientedKeypoint và vector 128 chiều.
 */
//class SiftDescriptor extends OrientedKeypoint {
//    public final double[] descriptor;
//
//    public SiftDescriptor(OrientedKeypoint okp, double[] descriptor) {
//        super(okp, okp.orientation);
//        this.descriptor = descriptor;
//    }
//
//    @Override
//    public String toString() {
//        return super.toString() + String.format(" | Descriptor[0..2]=%.3f, %.3f,...", descriptor[0], descriptor[1]);
//    }
//}
public class SiftStage4 {

    private final int DESCRIPTOR_WIDTH = 4; // Lưới 4x4
    private final int DESCRIPTOR_BINS = 8; // 8 bin hướng
    private final int WINDOW_GRID_SIZE = 16; // Cửa sổ 16x16
    private final double DESCRIPTOR_MAG_THRESHOLD = 0.2;

    /**
     * Hàm chính để chạy Giai đoạn 4.
     * @param orientedKeypoints Danh sách các điểm khóa có hướng từ Giai đoạn 3.
     * @param gaussianPyramid Kim tự tháp Gaussian.
     * @return Danh sách các bộ mô tả SIFT.
     */
    public List<SiftDescriptor> run(List<OrientedKeypoint> orientedKeypoints, List<List<SiftImage>> gaussianPyramid) {
        System.out.println("\n--- Bắt đầu Giai đoạn 4: Tạo bộ mô tả điểm khóa ---");
        List<SiftDescriptor> descriptors = new ArrayList<>();

        for (OrientedKeypoint okp : orientedKeypoints) {
            double[] descriptorVector = createDescriptor(okp, gaussianPyramid);
            if (descriptorVector!= null) {
                descriptors.add(new SiftDescriptor(okp, descriptorVector));
            }
        }

        System.out.printf("Đã tạo ra %d bộ mô tả SIFT.\n", descriptors.size());
        System.out.println("--- Giai đoạn 4 Hoàn tất ---");
        return descriptors;
    }

    /**
     * Tạo vector mô tả 128 chiều cho một điểm khóa có hướng.
     */
    private double[] createDescriptor(OrientedKeypoint okp, List<List<SiftImage>> gaussianPyramid) {
        SiftImage image = gaussianPyramid.get(okp.octave).get(okp.layer);
        double[][] imageData = image.data;
        int height = image.getHeight();
        int width = image.getWidth();

        // Kích thước thực của cửa sổ mô tả, phụ thuộc vào sigma của điểm khóa
        double windowSize = WINDOW_GRID_SIZE * okp.sigma;
        int radius = (int) Math.round(windowSize * Math.sqrt(2) * (DESCRIPTOR_WIDTH + 1) * 0.5);

        // Khởi tạo histogram 3D (4x4x8)
        double[] hist = new double[];

        double cos_t = Math.cos(okp.orientation);
        double sin_t = Math.sin(okp.orientation);

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int sampleY = (int) Math.round(okp.y) + i;
                int sampleX = (int) Math.round(okp.x) + j;
                if (sampleY <= 0 || sampleY >= height - 1 || sampleX <= 0 || sampleX >= width - 1) continue;

                // Xoay tọa độ mẫu về hệ tọa độ của điểm khóa
                double rotatedX = (j * cos_t + i * sin_t) / okp.sigma;
                double rotatedY = (-j * sin_t + i * cos_t) / okp.sigma;

                // Kiểm tra xem điểm mẫu có nằm trong lưới 4x4 không
                if (Math.abs(rotatedX) >= DESCRIPTOR_WIDTH / 2.0 |

                        | Math.abs(rotatedY) >= DESCRIPTOR_WIDTH / 2.0) {
                    continue;
                }

                // Tính gradient tại điểm mẫu
                double dx = imageData[sampleX + 1] - imageData[sampleX - 1];
                double dy = imageData[sampleX] - imageData[sampleX];
                double magnitude = Math.sqrt(dx * dx + dy * dy);
                double orientation = Math.atan2(dy, dx);

                // Xoay hướng gradient tương đối so với hướng của điểm khóa
                double rotatedOrientation = orientation - okp.orientation;
                while (rotatedOrientation < 0) rotatedOrientation += 2 * Math.PI;
                while (rotatedOrientation >= 2 * Math.PI) rotatedOrientation -= 2 * Math.PI;

                // Áp dụng trọng số Gaussian
                double weight = Math.exp(-(rotatedX * rotatedX + rotatedY * rotatedY) / (2.0 * (0.5 * DESCRIPTOR_WIDTH) * (0.5 * DESCRIPTOR_WIDTH)));
                double weightedMagnitude = magnitude * weight;

                // Nội suy ba chiều tuyến tính (phiên bản đơn giản hóa: gán vào bin gần nhất)
                // Tọa độ trong lưới 4x4
                double histX = rotatedX + DESCRIPTOR_WIDTH / 2.0 - 0.5;
                double histY = rotatedY + DESCRIPTOR_WIDTH / 2.0 - 0.5;
                // Bin hướng
                double histO = rotatedOrientation * DESCRIPTOR_BINS / (2 * Math.PI);

                int x_bin = (int) Math.floor(histX);
                int y_bin = (int) Math.floor(histY);
                int o_bin = (int) Math.floor(histO);

                if (x_bin >= 0 && x_bin < DESCRIPTOR_WIDTH && y_bin >= 0 && y_bin < DESCRIPTOR_WIDTH && o_bin >= 0 && o_bin < DESCRIPTOR_BINS) {
                    hist[o_bin] += weightedMagnitude;
                }
            }
        }

        // Chuyển histogram 3D thành vector 128 chiều và chuẩn hóa
        return flattenAndNormalize(hist);
    }

    /**
     * Chuyển histogram 3D thành vector 1D và thực hiện chuẩn hóa.
     */
    private double flattenAndNormalize(double hist) {
        double[] descriptor = new double[];
        int index = 0;
        for (int i = 0; i < DESCRIPTOR_WIDTH; i++) {
            for (int j = 0; j < DESCRIPTOR_WIDTH; j++) {
                for (int k = 0; k < DESCRIPTOR_BINS; k++) {
                    descriptor[index++] = hist[k];
                }
            }
        }

        // Bước 1: Chuẩn hóa L2
        double norm = 0.0;
        for (double val : descriptor) {
            norm += val * val;
        }
        norm = Math.sqrt(norm);
        if (norm > 1e-7) {
            for (int i = 0; i < descriptor.length; i++) {
                descriptor[i] /= norm;
            }
        }

        // Bước 2: Cắt ngưỡng
        for (int i = 0; i < descriptor.length; i++) {
            if (descriptor[i] > DESCRIPTOR_MAG_THRESHOLD) {
                descriptor[i] = DESCRIPTOR_MAG_THRESHOLD;
            }
        }

        // Bước 3: Chuẩn hóa L2 lại
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