package com.stitching.SIFT;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Lớp này đại diện cho một điểm khóa đã được gán hướng.
 * Nó kế thừa thông tin từ Keypoint và thêm thuộc tính orientation.
 */
//class OrientedKeypoint extends Keypoint {
//    public final double orientation; // Hướng của điểm khóa, tính bằng radian
//
//    public OrientedKeypoint(Keypoint keypoint, double orientation) {
//        super(keypoint.x, keypoint.y, keypoint.octave, keypoint.layer, keypoint.sigma);
//        this.orientation = orientation;
//    }
//
//    @Override
//    public String toString() {
//        return super.toString() + String.format(" | Orientation=%.2f°", Math.toDegrees(orientation));
//    }
//}

public class SiftStage3 {

    private final int numOrientationBins = 36;
    private final double peakRatioThreshold = 0.8;

    /**
     * Hàm chính để chạy Giai đoạn 3.
     * @param refinedKeypoints Danh sách các điểm khóa từ Giai đoạn 2.
     * @param gaussianPyramid Kim tự tháp Gaussian được xây dựng ở Giai đoạn 1.
     * @return Danh sách các điểm khóa đã được gán hướng.
     */
    public List<OrientedKeypoint> run(List<Keypoint> refinedKeypoints, List<List<SiftImage>> gaussianPyramid) {
        System.out.println("\n--- Bắt đầu Giai đoạn 3: Gán hướng cho điểm khóa ---");
        List<OrientedKeypoint> orientedKeypoints = new ArrayList<>();

        for (Keypoint kp : refinedKeypoints) {
            List<Double> orientations = calculateOrientations(kp, gaussianPyramid);
            for (double orientation : orientations) {
                orientedKeypoints.add(new OrientedKeypoint(kp, orientation));
            }
        }

        System.out.printf("Từ %d điểm khóa, đã tạo ra %d điểm khóa có hướng.\n", refinedKeypoints.size(), orientedKeypoints.size());
        System.out.println("--- Giai đoạn 3 Hoàn tất ---");
        return orientedKeypoints;
    }

    /**
     * Tính toán một hoặc nhiều hướng cho một điểm khóa duy nhất.
     */
    private List<Double> calculateOrientations(Keypoint kp, List<List<SiftImage>> gaussianPyramid) {
        SiftImage image = gaussianPyramid.get(kp.octave).get(kp.layer);
        double[][] imageData = image.data;
        int height = image.getHeight();
        int width = image.getWidth();

        double[] histogram = new double[9];
        int radius = (int) Math.round(3 * 1.5 * kp.sigma);
        double weightSigma = 1.5 * kp.sigma;

        int centerX = (int) Math.round(kp.x);
        int centerY = (int) Math.round(kp.y);

        // Lặp qua các pixel trong vùng lân cận
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int sampleY = centerY + i;
                int sampleX = centerX + j;
                if (sampleY <= 0 || sampleY >= height - 1 || sampleX <= 0 || sampleX >= width - 1) {
                    continue;
                }
                double dx = imageData[sampleY][sampleX + 1] - imageData[sampleY][sampleX - 1];
                double dy = imageData[sampleY+1][sampleX] - imageData[sampleY-1][sampleX];
                double magnitude = Math.sqrt(dx * dx + dy * dy);
                double orientation = Math.atan2(dy, dx);

                // Tính trọng số Gaussian
                double weight = Math.exp(-(i * i + j * j) / (2 * weightSigma * weightSigma));

                // Chuyển hướng về khoảng [0, 2*PI] và tìm bin tương ứng
                if (orientation < 0) {
                    orientation += 2 * Math.PI;
                }
                int bin = (int) Math.round(orientation * numOrientationBins / (2 * Math.PI));
                bin = (bin == numOrientationBins)? 0 : bin; // Xử lý trường hợp 360 độ

                histogram[bin] += magnitude * weight;
            }
        }

        // Tìm các đỉnh trong histogram
        return findPeaks(histogram);
    }

    /**
     * Tìm đỉnh chính và các đỉnh phụ trong biểu đồ hướng (orientation histogram)
     * theo thuật toán SIFT.
     *
     * @param histogram          mảng chứa histogram hướng (thường có 36 phần tử)
     * @param //numOrientationBins số bin (thường là 36)
     * @param //peakRatioThreshold tỉ lệ để giữ đỉnh phụ (thường 0.8)
     * @return danh sách các hướng (orientation) tính bằng radian
     */
    private List<Double> findPeaks(double[] histogram/*, int numOrientationBins, double peakRatioThreshold*/) {
        List<Double> orientations = new ArrayList<>();
        double maxPeakValue = Arrays.stream(histogram).max().orElse(0.0);

        for (int i = 0; i < numOrientationBins; i++) {
            double currentValue = histogram[i];
            double prevValue = histogram[(i - 1 + numOrientationBins) % numOrientationBins];
            double nextValue = histogram[(i + 1) % numOrientationBins];
            if (currentValue > prevValue && currentValue > nextValue && currentValue >= maxPeakValue * peakRatioThreshold) {
                double interp = 0.5 * (prevValue - nextValue) / (prevValue - 2 * currentValue + nextValue);
                interp = Math.max(-1.0, Math.min(1.0, interp));
                double peakBin = (i + interp + numOrientationBins) % numOrientationBins;
                double peakOrientation = peakBin * (2 * Math.PI / numOrientationBins);
                orientations.add(peakOrientation);
            }
        }
        return orientations;
    }


    // Ví dụ cách sử dụng
    public static void main(String args) {
        // Giả lập đầu vào từ Giai đoạn 2
        List<Keypoint> refinedKeypoints = new ArrayList<>();
        refinedKeypoints.add(new Keypoint(150.2, 100.8, 0, 2, 2.54));
        refinedKeypoints.add(new Keypoint(200.5, 250.1, 1, 3, 4.03));

        // Giả lập kim tự tháp Gaussian
        List<List<com.stitching.SIFT.SiftImage>> gaussianPyramid = new ArrayList<>();
        //... (Trong thực tế, kim tự tháp này sẽ chứa dữ liệu ảnh thật)

        // Chạy Giai đoạn 3
        SiftStage3 stage3 = new SiftStage3();
        List<OrientedKeypoint> orientedKeypoints = stage3.run(refinedKeypoints, gaussianPyramid);

        // In kết quả
        for (OrientedKeypoint okp : orientedKeypoints) {
            System.out.println(okp);
        }
    }

}
