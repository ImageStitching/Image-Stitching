package com.stitching.SIFTlỏ;

import lombok.Getter;
import java.util.ArrayList;
import java.util.List;

/**
 * ImageFeature: Định dạng OUTPUT giống như OpenCV detectFeatures
 * Chứa danh sách keypoints và descriptor matrix
 */
@Getter
public class ImageFeature {
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

    private List<KeyPointInfo> keyPoints;
    private byte[][] descriptors;  // [numKeypoints x 128]
    private int numKeypoints;

    public ImageFeature(List<SiftDescriptor> siftDescriptors,
                        int originalImageHeight, int originalImageWidth) {
        this.keyPoints = new ArrayList<>();
        List<byte[]> validDescriptors = new ArrayList<>();

        final double MIN_BORDER = 1.0;  // Ít nhất 1 pixel từ cạnh

        for (SiftDescriptor sd : siftDescriptors) {
            double scaleFactor = sd.enable_precise_upscale ? Math.pow(2.0, sd.octave - 1.0) : Math.pow(2.0, sd.octave);

            double originalX = sd.x * scaleFactor;
            double originalY = sd.y * scaleFactor;

            // Kiểm tra ranh giới
//            if (originalX < MIN_BORDER || originalX >= originalImageWidth - MIN_BORDER || originalY < MIN_BORDER || originalY >= originalImageHeight - MIN_BORDER) {
//                continue;
//            }

            KeyPointInfo kp = new KeyPointInfo(
                    originalX, originalY,
                    sd.sigma * 2.0,
                    Math.toDegrees(sd.orientation),
                    sd.response, sd.octave, -1
            );
            this.keyPoints.add(kp);
            validDescriptors.add(sd.descriptor);
        }

        this.numKeypoints = this.keyPoints.size();
        this.descriptors = new byte[this.numKeypoints][128];
        for (int i = 0; i < validDescriptors.size(); i++) {
            System.arraycopy(validDescriptors.get(i), 0, this.descriptors[i], 0, 128);
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
    public double[] getDescriptorAsFloat(int index) {
        double[] result = new double[128];
        for (int i = 0; i < 128; i++) {
            result[i] = (descriptors[index][i] & 0xFF) / 255.0;
        }
        return result;
    }

    public byte[] getDescriptorAsByte(int index) {
        return descriptors[index];
    }
}
