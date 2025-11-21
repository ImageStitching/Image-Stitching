package com.stitching.SIFT;

import lombok.Getter;
import java.util.ArrayList;
import java.util.List;

/**
 * ImageFeature: Định dạng OUTPUT giống như OpenCV detectFeatures
 * Chứa danh sách keypoints và descriptor matrix
 */
@Getter
public class ImageFeature {
    private List<KeyPointInfo> keyPoints;
    private byte[][] descriptors;  // [numKeypoints x 128]
    private int numKeypoints;

    public ImageFeature(List<SiftDescriptor> siftDescriptors) {
        this.numKeypoints = siftDescriptors.size();
        this.keyPoints = new ArrayList<>();
        this.descriptors = new byte[numKeypoints][128];

        // ⭐ CONVERT SiftDescriptor -> ImageFeature
        for (int i = 0; i < siftDescriptors.size(); i++) {
            SiftDescriptor sd = siftDescriptors.get(i);

            // Tính tọa độ gốc từ octave coordinates
            int scaleFactor = 1 << sd.octave;  // 2^octave
            if (sd.enable_precise_upscale) scaleFactor = scaleFactor >> 1;

            double originalX = sd.x * scaleFactor;
            double originalY = sd.y * scaleFactor;
            double size = sd.sigma * 2.0;  // Size = 2 * sigma
            double angle = Math.toDegrees(sd.orientation);  // Radian -> Degree

            // Tạo KeyPointInfo
            KeyPointInfo kp = new KeyPointInfo(
                    originalX, originalY, size, angle,
                    sd.response, sd.octave, -1  // class_id = -1 (default)
            );
            this.keyPoints.add(kp);

            // Copy descriptor
            System.arraycopy(sd.descriptor, 0, this.descriptors[i], 0, 128);
        }
    }

    /**
     * KeyPointInfo: Tương ứng với cv::KeyPoint của OpenCV
     */
    @Getter
    public static class KeyPointInfo {
        public double pt_x;         // x coordinate
        public double pt_y;         // y coordinate
        public double size;         // keypoint neighborhood size
        public double angle;        // orientation in degrees [0, 360)
        public double response;     // keypoint strength
        public int octave;          // pyramid octave
        public int class_id;        // class identifier

        public KeyPointInfo(double pt_x, double pt_y, double size, double angle,
                            double response, int octave, int class_id) {
            this.pt_x = pt_x;
            this.pt_y = pt_y;
            this.size = size;
            this.angle = angle;
            this.response = response;
            this.octave = octave;
            this.class_id = class_id;
        }

        @Override
        public String toString() {
            return String.format(
                    "KeyPoint(%.2f, %.2f) size=%.2f angle=%.1f° response=%.4f octave=%d",
                    pt_x, pt_y, size, angle, response, octave
            );
        }
    }

    @Override
    public String toString() {
        return String.format("ImageFeature[keypoints=%d, descriptor_dims=128]\n" +
                        "First 3 keypoints:\n%s",
                numKeypoints,
                keyPoints.stream()
                        .limit(3)
                        .map(KeyPointInfo::toString)
                        .reduce((a, b) -> a + "\n" + b)
                        .orElse("No keypoints"));
    }

    // ⭐ HỮU DỤNG: Lấy descriptor của keypoint thứ i dưới dạng float [0, 1]
    public double[] getDescriptorAsFloat(int index) {
        double[] result = new double[128];
        for (int i = 0; i < 128; i++) {
//            result[i] = (descriptors[index][i] & 0xFF) / 512.0;
            result[i] = (descriptors[index][i] & 0xFF) / 255.0;
        }
        return result;
    }

    // ⭐ HỮU DỤNG: Lấy descriptor của keypoint thứ i dưới dạng byte [0, 255]
    public byte[] getDescriptorAsByte(int index) {
        return descriptors[index];
    }
}
