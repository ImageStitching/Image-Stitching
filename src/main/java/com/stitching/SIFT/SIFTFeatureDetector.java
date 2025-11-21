package com.stitching.SIFT;

import com.stitching.imageOperator.Matrix_Image;
import lombok.Getter;
import org.springframework.beans.factory.annotation.Autowired;

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
        this(0, 3, 0.04, 10.0, 1.6, true, 5);
    }

    public ImageFeature detectFeatures(double[][] image) {
        System.out.println("========== PHÁT HIỆN SIFT FEATURES ==========");
        System.out.printf("Ảnh kích thước: %d x %d\n", image.length, image[0].length);
        System.out.printf("Tham số: nOctaveLayers=%d, contrastThreshold=%.3f, edgeThreshold=%.1f, sigma=%.1f\n",
                nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

        long startTime = System.currentTimeMillis();

        List<KeypointCandidate> candidates = siftStage1.run(image);
        System.out.printf("\n✓ Giai đoạn 1: %d ứng viên keypoint\n", candidates.size());

        List<List<SiftImage>> gaussianPyramid = siftStage1.getGaussianPyramid();
        List<List<SiftImage>> dogPyramid = siftStage1.getDogPyramid();

        List<Keypoint> refinedKeypoints = siftStage2.run(candidates, dogPyramid, gaussianPyramid);
        System.out.printf("✓ Giai đoạn 2: %d keypoint sau lọc\n", refinedKeypoints.size());

        List<OrientedKeypoint> orientedKeypoints = siftStage3.run(refinedKeypoints, gaussianPyramid);
        System.out.printf("✓ Giai đoạn 3: %d keypoint có hướng\n", orientedKeypoints.size());

        List<SiftDescriptor> descriptors = siftStage4.run(orientedKeypoints, gaussianPyramid);
        System.out.printf("✓ Giai đoạn 4: %d descriptor tính toán\n", descriptors.size());

        ImageFeature imageFeature = new ImageFeature(descriptors);

        // LỌC theo nfeatures nếu cần
        if (nfeatures > 0 && imageFeature.getNumKeypoints() > nfeatures) {
            imageFeature = filterByResponseScore(imageFeature, nfeatures);
            System.out.printf("✓ Lọc theo nfeatures: giữ lại %d features\n", imageFeature.getNumKeypoints());
        }

        long endTime = System.currentTimeMillis();
        System.out.printf("\n========== KẾT QUẢ ==========\n");
        System.out.printf("Tổng keypoints: %d\n", imageFeature.getNumKeypoints());
        System.out.printf("Descriptor dimensions: [%d x 128]\n", imageFeature.getNumKeypoints());
        System.out.printf("Thời gian xử lý: %.2f giây\n", (endTime - startTime) / 1000.0);
        System.out.println("=============================\n");

        return imageFeature;
    }

    /**
     * Lọc features theo response score (giữ top N features)
     */
    private ImageFeature filterByResponseScore(ImageFeature features, int topN) {
        return features;
    }

    public static void main(String[] args) {
        try {
            // ========== OPTION 1: Tham số mặc định (giống OpenCV) ==========
            System.out.println("╔════════════════════════════════════════════════════════╗");
            System.out.println("║         SIFT FEATURE DETECTION - OpenCV Style          ║");
            System.out.println("╚════════════════════════════════════════════════════════╝\n");

            // Load ảnh
            String imagePath = INPUT_PATH.resolve("org_img.png").toString();
            System.out.println("📷 Loading image: " + imagePath);
            double[][] image = Matrix_Image.create_DOUBLEgrayMatrix_from_color_image(imagePath);

            if (image == null) {
                System.err.println(" Lỗi: Không thể load ảnh!");
                return;
            }

            // ========== OPTION 1: Tham số mặc định ==========
            System.out.println("\n OPTION 1: Tham số mặc định OpenCV");
            SIFTFeatureDetector sift1 = new SIFTFeatureDetector();
            ImageFeature features1 = sift1.detectFeatures(image);

            if (features1.getNumKeypoints() > 0) {
                System.out.println("\n Top 5 Keypoints:");
                for (int i = 0; i < Math.min(5, features1.getNumKeypoints()); i++) {
                    ImageFeature.KeyPointInfo kp = features1.getKeyPoints().get(i);
                    System.out.printf("  [%d] %s\n", i + 1, kp);
                }
            }

            // ========== OPTION 2: Tham số tùy chỉnh ==========
            System.out.println("\n OPTION 2: Tham số tùy chỉnh");
            // int nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, enable_precise_upscale, numOctaves
            SIFTFeatureDetector sift2 = new SIFTFeatureDetector(
                    0,          // nfeatures: 0 = detect all
                    3,          // nOctaveLayers: 3 layers per octave
                    0.02,       // contrastThreshold: lower = detect more features
                    10.0,       // edgeThreshold: higher = keep more edge features
                    1.6,        // sigma: Gaussian blur parameter
                    true,       // enable_precise_upscale: upscale ảnh trước khi tạo pyramid
                    5           // numOctaves: number of pyramid levels
            );
            ImageFeature features2 = sift2.detectFeatures(image);

            if (features2.getNumKeypoints() > 0) {
                System.out.println("\n Top 5 Keypoints:");
                for (int i = 0; i < Math.min(5, features2.getNumKeypoints()); i++) {
                    ImageFeature.KeyPointInfo kp = features2.getKeyPoints().get(i);
                    System.out.printf("  [%d] %s\n", i + 1, kp);
                }
            }

            // ========== OPTION 3: So sánh ==========
            System.out.println("\n OPTION 3: So sánh kết quả");
            System.out.printf("Option 1 keypoints: %d\n", features1.getNumKeypoints());
            System.out.printf("Option 2 keypoints: %d\n", features2.getNumKeypoints());

        } catch (Exception e) {
            System.err.println(" Lỗi: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
