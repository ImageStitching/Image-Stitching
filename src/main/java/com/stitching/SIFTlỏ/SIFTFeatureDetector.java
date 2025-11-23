package com.stitching.SIFTlỏ;

import com.stitching.imageOperator.Matrix_Image;
import edu.princeton.cs.introcs.Picture;
import lombok.Getter;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@Getter
public class SIFTFeatureDetector {
    private static Path INPUT_PATH = Paths.get("src","main","resources","static","sift");

    // Tham số SIFT int nfeatures, int nOctaveLayers, double contrastThreshold, double edgeThreshold, double sigma, boolean enable_precise_upscale

    private final int nfeatures;
    private final int nOctaveLayers;
    private final double contrastThreshold;
    private final double edgeThreshold;
    private final double sigma;
    private final boolean enable_precise_upscale;

    private final int maxInterpolationSteps = 5;
    private final int numOctaves;

    private SiftStage1 siftStage1;
    private SiftStage2 siftStage2;
    private SiftStage3 siftStage3;
    private SiftStage4 siftStage4;

    public SIFTFeatureDetector(int nfeatures, int nOctaveLayers, double contrastThreshold, double edgeThreshold, double sigma, boolean enable_precise_upscale, int numOctaves) {
        this.nfeatures = nfeatures;
        this.nOctaveLayers = nOctaveLayers;
        this.contrastThreshold = contrastThreshold;
        this.edgeThreshold = edgeThreshold;
        this.sigma = sigma;
        this.enable_precise_upscale = enable_precise_upscale;
        this.numOctaves = numOctaves;

        this.siftStage1 = new SiftStage1(nOctaveLayers,sigma, numOctaves, enable_precise_upscale);
        this.siftStage2 = new SiftStage2(contrastThreshold, edgeThreshold, nOctaveLayers, enable_precise_upscale);
        this.siftStage3 = new SiftStage3();
        this.siftStage4 = new SiftStage4();
    }

    public SIFTFeatureDetector() {
        this(0, 3, 0.1, 10.0, 1.6, true, 4);
    }

    // Thêm method này vào SIFTFeatureDetector.java để debug chi tiết

    private void debugCoordinateTransformation(ImageFeature imageFeature, double[][] originalImage) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("DEBUG: COORDINATE TRANSFORMATION VERIFICATION");
        System.out.println("=".repeat(80));

        int imgHeight = originalImage.length;
        int imgWidth = originalImage[0].length;

        System.out.printf("Original Image Size: %d x %d\n", imgHeight, imgWidth);
        System.out.printf("Enable Precise Upscale: %s\n", enable_precise_upscale);
        System.out.printf("Total Keypoints: %d\n\n", imageFeature.getNumKeypoints());

        // Thống kê tọa độ
        double minX = Double.MAX_VALUE, maxX = Double.MIN_VALUE;
        double minY = Double.MAX_VALUE, maxY = Double.MIN_VALUE;
        int outOfBounds = 0;
        int atZero = 0;

        for (int i = 0; i < Math.min(20, imageFeature.getNumKeypoints()); i++) {
            ImageFeature.KeyPointInfo kp = imageFeature.getKeyPoints().get(i);

            boolean isOutOfBounds = kp.pt_x < 0 || kp.pt_x >= imgWidth || kp.pt_y < 0 || kp.pt_y >= imgHeight;
            boolean isAtZero = (kp.pt_x < 1 && kp.pt_y < 1);

            if (isOutOfBounds) outOfBounds++;
            if (isAtZero) atZero++;

            minX = Math.min(minX, kp.pt_x);
            maxX = Math.max(maxX, kp.pt_x);
            minY = Math.min(minY, kp.pt_y);
            maxY = Math.max(maxY, kp.pt_y);

            String status = isOutOfBounds ? " ⚠️ OUT OF BOUNDS" : (isAtZero ? " ⚠️ AT ZERO" : " ✅ OK");

            System.out.printf("[%3d] octave=%d | (%.2f, %.2f) | size=%.2f | angle=%.1f°%s\n",
                    i, kp.octave, kp.pt_x, kp.pt_y, kp.size, kp.angle, status);
        }

        System.out.println("\n" + "-".repeat(80));
        System.out.println("COORDINATE STATISTICS:");
        System.out.println("-".repeat(80));
        System.out.printf("X Range: [%.2f, %.2f] (Expected: [0, %d])\n", minX, maxX, imgWidth);
        System.out.printf("Y Range: [%.2f, %.2f] (Expected: [0, %d])\n", minY, maxY, imgHeight);
        System.out.printf("Keypoints at (0,0): %d\n", atZero);
        System.out.printf("Keypoints out of bounds: %d\n", outOfBounds);

        // Phân tích theo octave
        System.out.println("\n" + "-".repeat(80));
        System.out.println("KEYPOINT DISTRIBUTION BY OCTAVE:");
        System.out.println("-".repeat(80));

        java.util.Map<Integer, Integer> octaveCount = new java.util.HashMap<>();
        for (ImageFeature.KeyPointInfo kp : imageFeature.getKeyPoints()) {
            octaveCount.put(kp.octave, octaveCount.getOrDefault(kp.octave, 0) + 1);
        }

        octaveCount.forEach((octave, count) ->
                System.out.printf("Octave %d: %3d keypoints (%.1f%%)\n",
                        octave, count, (count * 100.0) / imageFeature.getNumKeypoints())
        );

        // Final verdict
        System.out.println("\n" + "=".repeat(80));
        System.out.println("VERDICT:");
        System.out.println("=".repeat(80));

        boolean isGood = outOfBounds == 0 && atZero < imageFeature.getNumKeypoints() * 0.05;

        if (isGood) {
            System.out.println("✅ COORDINATES LOOK GOOD!");
            System.out.println("   - Keypoints are within image bounds");
            System.out.println("   - Scale factor calculation appears correct");
        } else {
            System.out.println("❌ COORDINATES HAVE ISSUES!");
            if (outOfBounds > 0) {
                System.out.printf("   - %d keypoints are out of bounds\n", outOfBounds);
            }
            if (atZero > imageFeature.getNumKeypoints() * 0.05) {
                System.out.printf("   - %d keypoints are clustered at origin\n", atZero);
            }
        }
        System.out.println("=".repeat(80) + "\n");
    }

    public ImageFeature detectFeatures(double[][] image) {
        System.out.println("========== PHÁT HIỆN SIFT FEATURES ==========");
        System.out.printf("Ảnh kích thước: %d dọc x %d ngang\n", image.length, image[0].length);
        System.out.printf("Tham số: nOctaveLayers=%d, contrastThreshold=%.3f, edgeThreshold=%.1f, sigma=%.1f\n",
                nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

        long startTime = System.currentTimeMillis();

        List<KeypointCandidate> candidates = siftStage1.run(image);
        System.out.printf("\n Giai đoạn 1: %d ứng viên keypoint\n", candidates.size());

        List<List<SiftImage>> gaussianPyramid = siftStage1.getGaussianPyramid();
        List<List<SiftImage>> dogPyramid = siftStage1.getDogPyramid();

        List<Keypoint> refinedKeypoints = siftStage2.run(candidates, dogPyramid, gaussianPyramid);
        System.out.printf(" Giai đoạn 2: %d keypoint sau lọc\n", refinedKeypoints.size());

        List<OrientedKeypoint> orientedKeypoints = siftStage3.run(refinedKeypoints, gaussianPyramid);
        System.out.printf(" Giai đoạn 3: %d keypoint có hướng\n", orientedKeypoints.size());

        List<SiftDescriptor> descriptors = siftStage4.run(orientedKeypoints, gaussianPyramid);
        System.out.printf(" Giai đoạn 4: %d descriptor tính toán\n", descriptors.size());

        ImageFeature imageFeature = new ImageFeature(descriptors,image[0].length, image.length);
        debugCoordinateTransformation(imageFeature, image);

        if (nfeatures > 0 && imageFeature.getNumKeypoints() > nfeatures) {
            imageFeature = filterByResponseScore(imageFeature, nfeatures);
            System.out.printf("   Lọc theo nfeatures: giữ lại %d features\n", imageFeature.getNumKeypoints());
        }

        long endTime = System.currentTimeMillis();
        System.out.printf("\n      KẾT QUẢ     \n");
        System.out.printf("Tổng keypoints: %d\n", imageFeature.getNumKeypoints());
        System.out.printf("Descriptor dimensions: [%d x 128]\n", imageFeature.getNumKeypoints());
        System.out.printf("Thời gian xử lý: %.2f giây\n", (endTime - startTime) / 1000.0);
        System.out.println();

        return imageFeature;
    }

    private ImageFeature filterByResponseScore(ImageFeature features, int topN) {
        return features;
    }

    private void printKeypoints(ImageFeature imageFeature) {
        System.out.println("┌─────┬──────────────────────────────┬──────────────────────────────┬───────────────── ────┐");
        System.out.println("│ Idx │  row    │    col    |    size_radius  |   angle   │    response    |    octave     |");
        System.out.println("├─────┼──────────────────────────────┼──────────────────────────────┼──────────────────────┤");
        List<ImageFeature.KeyPointInfo> kp = imageFeature.getKeyPoints();
        int n = kp.size();
        for (int i = 0; i < n; i++) {
            ImageFeature.KeyPointInfo c = kp.get(i);
            System.out.printf("│ %3d │  %7.1f   │   %7.1f   |     %5.1f      |   %6.1f độ   │    %5.1f    |    %d   |\n",
            i, c.pt_x, c.pt_y, c.size, c.angle, c.response, c.octave);
        }
    }

    public static void main(String[] args) {
        try {
            String imagePath = INPUT_PATH.resolve("memeNgua6.jpg").toString();
            System.out.println(" ---- > Loading image: " + imagePath);
            double[][] image = Matrix_Image.create_DOUBLEgrayMatrix_from_color_image(imagePath);
            if (image == null) {
                System.err.println(" Lỗi: Không thể load ảnh!");
                return;
            }

            System.out.println("\n OPTION 1: Tham số mặc định OpenCV");
            SIFTFeatureDetector sift1 = new SIFTFeatureDetector(0, 3, 0.1, 10.0, 1.6, true, 4);
            ImageFeature features1 = sift1.detectFeatures(image);

            sift1.printKeypoints(features1);

            Picture input = new Picture(imagePath);
            KeypointVisualizer.drawKeypoints(input,features1.getKeyPoints());
        } catch (Exception e) {
            System.err.println(" Lỗi: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
