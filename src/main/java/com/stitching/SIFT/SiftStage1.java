package com.stitching.SIFT;

import com.stitching.filter_convolution_gauss.SeparabilityGauss;
import com.stitching.imageOperator.ColourImageToGray;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class SiftStage1 {
    private final int nOctaveLayers;
    private final double sigma;
    private final int numOctaves; // numOctaves = floor(log2(min(image_width, image_height))) - 3
    private final boolean enable_precise_upscale; // tham số để Nội suy cho ảnh size nhân đôi trước khi tạo Pymarids

    public SiftStage1(int nOctaveLayers, double sigma, int numOctaves, boolean enable_precise_upscale) {
        this.nOctaveLayers = nOctaveLayers;
        this.sigma = sigma;
        this.numOctaves = numOctaves;
        this.enable_precise_upscale = enable_precise_upscale;
    }

    public List<KeypointCandidate> run(double[][] initialImage) {
        System.out.println("--- Bắt đầu Giai đoạn 1: Phát hiện cực trị trong không gian tỷ lệ ---");

        System.out.printf("Ảnh gốc có kích thước: %d x %d\n", initialImage.length, initialImage.length);
        double[][] upsampledImage = Up_DownSample.upsampleWithLinearInterpolation(initialImage, this.enable_precise_upscale);
        System.out.printf("Ảnh sau khi nội suy có kích thước: %d x %d\n", upsampledImage.length, upsampledImage.length);

        // Sau khi có nội suy hoặc không có nội suy thì vẫn dùng Ma trận kết quả đó
        List<List<SiftImage>> gaussianPyramid = buildGaussianPyramid(upsampledImage);
        List<List<SiftImage>> dogPyramid = buildDogPyramid(gaussianPyramid);
        List<KeypointCandidate> candidates = findScaleSpaceExtrema(dogPyramid);

        System.out.println("--- Giai đoạn 1 Hoàn tất ---");
        return candidates;
    }

    // Các hàm buildGaussianPyramid, buildDogPyramid, findScaleSpaceExtrema,
    /*****
     sigma là hệ số sigma ban đầu cho ảnh ở octave 0, tức là sigma =1.6 (thường chọn thế) ở octave thứ 0 .
     Thì ảnh gốc ở octave sau có độ sigma = 2x lần ảnh ở octave trước đó.
     Với tứng layer trong octave s+3 layers thì mỗi lần sigma mới = k^l * sigma ở layer 0 của octave này
     currentSigma = sigma * Math.pow(2.0,o) * Math.pow(2.0, (double) l / nOctaveLayers );
     với:
     sigma*Math.pow(2.0,o) là sigma ở layer 0 trong octave thứ o.
     theo từng layer l thì currentSigma = sigma_layer_gốc_của_octave  *  k^l
     với: (k = 2^(1/s))
     với: s = nOctaveLayers
     nên sau l layer thì currentSigma của layer hiện tại = sigma_layer_gốc_của_octave * Math.pow(2, l / nOctaveLayers)
     *****/
    private List<List<SiftImage>> buildGaussianPyramid(double[][] baseImage) {
        System.out.println("1. Xây dựng Kim tự tháp Gaussian...");
        List<List<SiftImage>> pyramid = new ArrayList<>();
        double k = Math.pow(2.0, 1.0 / nOctaveLayers);
        double[][] currentImage = baseImage;

        for (int o = 0; o < numOctaves; o++) {
            List<SiftImage> octave = new ArrayList<>();
            System.out.println("  Đang xử lý Octave " + o);

            for (int l = 0; l < nOctaveLayers + 3; l++) {
                double currentSigma = sigma * Math.pow(2.0,o) * Math.pow(2.0, (double) l / nOctaveLayers );
                double[][] blurredImage = convolveWithSeparableGaussian(currentImage, currentSigma);
                octave.add(new SiftImage(blurredImage, currentSigma));
            }
            pyramid.add(octave);
            if (o < numOctaves - 1) {
                SiftImage nextBaseImage = octave.get(nOctaveLayers);
                currentImage = Up_DownSample.downsample(nextBaseImage.data);
            }
        }
        return pyramid;
    }

    private List<List<SiftImage>> buildDogPyramid(List<List<SiftImage>> gaussianPyramid) {
        System.out.println("2. Xây dựng Kim tự tháp DoG...");
        List<List<SiftImage>> dogPyramid = new ArrayList<>();

        for (int o = 0; o < gaussianPyramid.size(); o++) {
            List<SiftImage> gaussianOctave = gaussianPyramid.get(o);
            List<SiftImage> dogOctave = new ArrayList<>();
            System.out.println("  Tính toán DoG cho Octave " + o);

            for (int l = 0; l < gaussianOctave.size() - 1; l++) {
                SiftImage img1 = gaussianOctave.get(l);
                SiftImage img2 = gaussianOctave.get(l + 1);

                double[][] dogData = new double[img1.getHeight()][img1.getWidth()];
                for (int r = 0; r < img1.getHeight(); r++) {
                    for (int c = 0; c < img1.getWidth(); c++) {
                        dogData[r][c] = img2.data[r][c] - img1.data[r][c];
                    }
                }
                dogOctave.add(new SiftImage(dogData, img1.sigma));
            }
            dogPyramid.add(dogOctave);
        }
        return dogPyramid;
    }

    private List<KeypointCandidate> findScaleSpaceExtrema(List<List<SiftImage>> dogPyramid) {
        System.out.println("3. Tìm kiếm các điểm cực trị cục bộ...");
        List<KeypointCandidate> candidates = new ArrayList<>();

        for (int o = 0; o < dogPyramid.size(); o++) {
            List<SiftImage> dogOctave = dogPyramid.get(o);
            System.out.println("  Quét cực trị trên Octave " + o);

            for (int l = 1; l < dogOctave.size() - 1; l++) {
                SiftImage currentLayer = dogOctave.get(l);
                SiftImage prevLayer = dogOctave.get(l - 1);
                SiftImage nextLayer = dogOctave.get(l + 1);

                int height = currentLayer.getHeight();
                int width = currentLayer.getWidth();

                for (int r = 1; r < height - 1; r++) {
                    for (int c = 1; c < width - 1; c++) {
                        double pixelValue = currentLayer.data[r][c];
                        if (isLocalExtremum(pixelValue, prevLayer, currentLayer, nextLayer, r, c)) {
                            candidates.add(new KeypointCandidate(c, r, o, l, currentLayer.sigma, enable_precise_upscale));
                        }
                    }
                }
            }
        }
        return candidates;
    }

    private boolean isLocalExtremum(double pixelValue, SiftImage prev, SiftImage current, SiftImage next, int r, int c) {
        boolean isMax = true;
        boolean isMin = true;

        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (pixelValue < prev.data[r + dr][c + dc]) isMax = false;
                if (pixelValue > prev.data[r + dr][c + dc]) isMin = false;
                if (pixelValue < next.data[r + dr][c + dc]) isMax = false;
                if (pixelValue > next.data[r + dr][c + dc]) isMin = false;
                if (dr != 0 || dc != 0) {
                    if (pixelValue < current.data[r + dr][c + dc]) isMax = false;
                    if (pixelValue > current.data[r + dr][c + dc]) isMin = false;
                }
            }
        }
        return isMax || isMin;
    }

    private double[][] convolveWithSeparableGaussian(double[][] image, double sigma) {
        return SeparabilityGauss.seperabilityGauss(image, sigma);
    }

    public static void main(String[] args) {
        int nOctaveLayers = 3;
        double sigma = 1.6;
        int numOctaves = 4;
        boolean enable_precise_upscale = true;

        Runtime runtime = Runtime.getRuntime();
//        System.gc();
//        long before = runtime.totalMemory() - runtime.freeMemory();
//
//        double[][] dummyImage = Matrix_Image.create_DOUBLEgrayMatrix_from_color_image("src/main/resources/static/image/img.png");
//
//        System.gc();
//        long after = runtime.totalMemory() - runtime.freeMemory();
//        System.out.printf("Tăng bộ nhớ: %.2f MB%n", (after - before) / (1024.0 * 1024.0));
//
//        System.gc();
//        before = runtime.totalMemory() - runtime.freeMemory();
//
//        // --- Chạy Giai đoạn 1 ---
//        SiftStage1 stage1 = new SiftStage1(nOctaveLayers, sigma, numOctaves, true);
//        double[][] upsampled = Up_DownSample.upsampleWithLinearInterpolation(dummyImage, enable_precise_upscale);
//
//
//        System.gc();
//        after = runtime.totalMemory() - runtime.freeMemory();
//        System.out.printf("Tăng bộ nhớ: %.2f MB%n", (after - before) / (1024.0 * 1024.0));
//
//        dummyImage = null;
//        upsampled = null;

//        System.out.println("\nẢnh sau khi nội suy:");
//        Picture pic = new Picture(Matrix_Image.create_grayImage_from_gray_matrix(upsampled));
//        pic.show();

        // Chạy toàn bộ pipeline với ảnh lớn hơn (giả lập)
        System.out.println("\n--- Chạy pipeline đầy đủ với ảnh lớn hơn ---");

        Path OUTPUT_PATH = Paths.get("src", "main", "resources", "static", "sift");
        String linkIMG = OUTPUT_PATH.resolve("org_img.png").toString();

        System.gc();
        long before = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("Bộ nhớ ban đầu : %.2f MB%n", before / (1024.0 * 1024.0));

        double[][] largerDummyImage = ColourImageToGray.grayMatrix(linkIMG);

//        for (int i =0; i<10; i++) {
//            for (int j = 0; j < 10; j++) {
//                System.out.print(largerDummyImage[i][j] + " ");
//            }
//            System.out.println();
//        }

        System.gc();
        long after = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("Bộ nhớ sau khi chạy tạo ma trận gốc : %.2f MB%n", after / (1024.0 * 1024.0));
        System.out.printf("Tăng bộ nhớ: %.2f MB%n", (after - before) / (1024.0 * 1024.0));

//        for (int i =0; i<10; i++) {
//            for (int j = 0; j < 10; j++) {
//                System.out.print(largerDummyImage[i][j] + " ");
//            }
//            System.out.println();
//        }

//        before = after;

        SiftStage1 stage1 = new SiftStage1(nOctaveLayers, sigma, numOctaves, true);
        List<KeypointCandidate> siftImages = stage1.run(largerDummyImage);

        System.gc();
        after = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("\nBộ nhớ sau khi chạy tạo xong SiftStage1 : %.2f MB%n", after / (1024.0 * 1024.0));
        System.out.printf("Tăng bộ nhớ: %.2f MB%n", (after - before) / (1024.0 * 1024.0));
    }
}