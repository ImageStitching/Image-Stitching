package com.stitching.SIFT;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class SiftStage2 {

    // --- CÁC THAM SỐ CỦA SIFT (Tương ứng với OpenCV) ---
    private final double contrastThreshold;
    private final double edgeThreshold;
    private final int nOctaveLayers;
    private final int maxInterpolationSteps = 5;
    private final boolean enable_precise_upscale;

    public SiftStage2(double contrastThreshold, double edgeThreshold, int nOctaveLayers, boolean enable_precise_upscale) {
        this.contrastThreshold = contrastThreshold;
        this.edgeThreshold = edgeThreshold;
        this.nOctaveLayers = nOctaveLayers;
        this.enable_precise_upscale = enable_precise_upscale;
    }

    public List<Keypoint> run(List<KeypointCandidate> candidates, List<List<SiftImage>> dogPyramid, List<List<SiftImage>> gaussianPyramid) {
//        System.out.println("\n--- Bắt đầu Giai đoạn 2: Định vị và Lọc điểm khóa ---");
        List<Keypoint> refinedKeypoints = new ArrayList<>();

        for (KeypointCandidate candidate : candidates) {
            // Bước 1: Định vị chính xác vị trí điểm cực trị xem co Không hội tụ hoặc nằm ngoài biên ?
            double[] localizationResult = locateExtremumViaQuadraticFit(candidate, dogPyramid);
            if (localizationResult == null) {
                continue;
            }

            double[] offset = {localizationResult[0], localizationResult[1], localizationResult[2]};
            int o = (int) localizationResult[3];
            int l = (int) localizationResult[4];
            int r = (int) localizationResult[5];
            int c = (int) localizationResult[6];

            // Bước 2: Loại bỏ các điểm có độ tương phản thấp
            if (isLowContrast(offset, dogPyramid, o, l, r, c)) {
                continue;
            }

            // Bước 3: Loại bỏ các phản hồi tại cạnh
            if (isEdgeResponse(dogPyramid, o, l, r, c)) {
                continue;
            }

            // ⭐ TÍNH TOÁN RESPONSE (Contrast value)
            double[] gradient = computeGradient(dogPyramid, o, l, r, c);
            double response = dogPyramid.get(o).get(l).data[r][c] +
                    0.5 * (gradient[0] * offset[0] + gradient[1] * offset[1] + gradient[2] * offset[2]);

            // Nếu vượt qua tất cả, tạo một Keypoint đã tinh chỉnh
            double refinedX = c + offset[0];
            double refinedY = r + offset[1];
            double refinedLayer = l + offset[2];

            // Sigma được tính từ octave 0 của kim tự tháp Gaussian
            double initialOctaveSigma = dogPyramid.get(o).get(0).sigma;
            double refinedSigma = initialOctaveSigma * Math.pow(2.0, (refinedLayer) / nOctaveLayers);

            refinedKeypoints.add(new Keypoint(refinedX, refinedY, o, (int) Math.round(refinedLayer), refinedSigma, enable_precise_upscale, Math.abs(response)));
        }

//        System.out.printf("Từ %d ứng viên, còn lại %d điểm khóa sau khi lọc.\n", candidates.size(), refinedKeypoints.size());
//        System.out.println("--- Giai đoạn 2 Hoàn tất ---");
        return refinedKeypoints;
    }

    // BƯỚC 1: ĐỊNH VỊ VỚI ĐỘ CHÍNH XÁC DƯỚI MỨC PIXEL
    /**
     * Tinh chỉnh vị trí của một điểm ứng viên bằng cách khớp một hàm bậc hai 3D.
     * @return Một mảng double chứa [dx, dy, d_layer, o, l, r, c] đã được cập nhật, hoặc null nếu thất bại.
     */
    private double[] locateExtremumViaQuadraticFit(KeypointCandidate candidate, List<List<SiftImage>> dogPyramid) {
        int o = candidate.octave;
        int l = candidate.layer;
        int r = candidate.y;
        int c = candidate.x;

        double[] offset = new double[3];
        int step = 0;

        while (step < maxInterpolationSteps) {
            int height = dogPyramid.get(o).get(l).getHeight();
            int width = dogPyramid.get(o).get(l).getWidth();
            if (l < 1 || l > nOctaveLayers - 2 || r < 1 || r > height - 2 || c < 1 || c > width - 2) {
                return null;
            }
            double[] gradient = computeGradient(dogPyramid, o, l, r, c);
            double[][] hessian = computeHessian(dogPyramid, o, l, r, c);
            double[] d = solveLinearSystem3x3(hessian, new double[]{-gradient[0], -gradient[1], -gradient[2]});
            if (d == null) return null;
            offset = d;
            // Nếu hội tụ (dịch nhỏ hơn 0.5 pixel)
            if (Math.abs(offset[0]) < 0.5 && Math.abs(offset[1]) < 0.5 && Math.abs(offset[2]) < 0.5) break;
            c += (int) Math.round(offset[0]);
            r += (int) Math.round(offset[1]);
            l += (int) Math.round(offset[2]);

            if (l < 1 || l > nOctaveLayers - 2 || r < 1 || r > height - 2 || c < 1 || c > width - 2) {
                return null;
            }
            step++;
        }
        if (step >= maxInterpolationSteps) return null; // Không hội tụ
        return new double[]{offset[0], offset[1], offset[2], o, l, r, c};
    }

    private double[] computeGradient(List<List<SiftImage>> dogPyramid, int o, int l, int r, int c) {
        double dx = (dogPyramid.get(o).get(l).data[r][c + 1] - dogPyramid.get(o).get(l).data[r][c - 1]) / 2.0;
        double dy = (dogPyramid.get(o).get(l).data[r + 1][c] - dogPyramid.get(o).get(l).data[r - 1][c]) / 2.0;
        double ds = (dogPyramid.get(o).get(l + 1).data[r][c] - dogPyramid.get(o).get(l - 1).data[r][c]) / 2.0;
        return new double[]{dx, dy, ds};
    }

    private double[][] computeHessian(List<List<SiftImage>> dogPyramid, int o, int l, int r, int c) {
        double[][] H = new double[3][3];
        double v = dogPyramid.get(o).get(l).data[r][c];

        double dxx = dogPyramid.get(o).get(l).data[r][c + 1] + dogPyramid.get(o).get(l).data[r][c - 1] - 2 * v;
        double dyy = dogPyramid.get(o).get(l).data[r + 1][c] + dogPyramid.get(o).get(l).data[r - 1][c] - 2 * v;
        double dss = dogPyramid.get(o).get(l + 1).data[r][c] + dogPyramid.get(o).get(l - 1).data[r][c] - 2 * v;
        double dxy = (dogPyramid.get(o).get(l).data[r + 1][c + 1] - dogPyramid.get(o).get(l).data[r + 1][c - 1]
                - dogPyramid.get(o).get(l).data[r - 1][c + 1] + dogPyramid.get(o).get(l).data[r - 1][c - 1]) / 4.0;
        double dxs = (dogPyramid.get(o).get(l + 1).data[r][c + 1] - dogPyramid.get(o).get(l + 1).data[r][c - 1]
                - dogPyramid.get(o).get(l - 1).data[r][c + 1] + dogPyramid.get(o).get(l - 1).data[r][c - 1]) / 4.0;
        double dys = (dogPyramid.get(o).get(l + 1).data[r + 1][c] - dogPyramid.get(o).get(l + 1).data[r - 1][c]
                - dogPyramid.get(o).get(l - 1).data[r + 1][c] + dogPyramid.get(o).get(l - 1).data[r - 1][c]) / 4.0;

        H[0][0] = dxx;
        H[1][1] = dyy;
        H[2][2] = dss;
        H[0][1] = H[1][0] = dxy;
        H[0][2] = H[2][0] = dxs;
        H[1][2] = H[2][1] = dys;

        return H;
    }

    private double[] solveLinearSystem3x3(double[][] A, double[] b) {
        double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

        if (Math.abs(det) < 1e-10) return null; // ma trận suy biến

        double[][] inv = new double[3][3];

        inv[0][0] =  (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det;
        inv[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) / det;
        inv[0][2] =  (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det;

        inv[1][0] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]) / det;
        inv[1][1] =  (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det;
        inv[1][2] = -(A[0][0] * A[1][2] - A[0][2] * A[1][0]) / det;

        inv[2][0] =  (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det;
        inv[2][1] = -(A[0][0] * A[2][1] - A[0][1] * A[2][0]) / det;
        inv[2][2] =  (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det;

        double[] x = new double[3];
        for (int i = 0; i < 3; i++) {
            x[i] = inv[i][0] * b[0] + inv[i][1] * b[1] + inv[i][2] * b[2];
        }
        return x;
    }

    // BƯỚC 2: LOẠI BỎ CÁC ĐIỂM CÓ ĐỘ TƯƠNG PHẢN THẤP
    /** Kiểm tra xem điểm đã tinh chỉnh có độ tương phản quá thấp hay không.  true nếu điểm có độ tương phản thấp và nên bị loại bỏ.
     * D(xˆ) = D + 1/2 * (∂D-T/∂x) * x^
     */
    private boolean isLowContrast(double[] offset, List<List<SiftImage>> dogPyramid, int o, int l, int r, int c) {
        double[] gradient = computeGradient(dogPyramid, o, l, r, c);
        double contrast = dogPyramid.get(o).get(l).data[r][c] + 0.5 * (gradient[0] * offset[0] + gradient[1] * offset[1] + gradient[2] * offset[2]);
        // Theo bài báo của Lowe, ngưỡng này được chia cho số lớp
        return Math.abs(contrast) < (contrastThreshold / nOctaveLayers);
    }

    // BƯỚC 3: LOẠI BỎ CÁC PHẢN HỒI TẠI CẠNH
    /** Kiểm tra xem điểm có nằm trên một cạnh hay không bằng cách sử dụng ma trận Hessian 2D . true nếu điểm nằm trên cạnh và nên bị loại bỏ.
     */
    private boolean isEdgeResponse(List<List<SiftImage>> dogPyramid, int o, int l, int r, int c) {
        double[][] hessian2D = computeHessian2D(dogPyramid, o, l, r, c);
        double D_xx = hessian2D[0][0];
        double D_yy = hessian2D[1][1];
        double D_xy = hessian2D[1][0];
        double trace = D_xx + D_yy;
        double det = D_xx * D_yy - D_xy * D_xy;

        if (det <= 0) return true;
        double threshold_r = edgeThreshold;
        return (trace * trace) / det > (threshold_r + 1) * (threshold_r + 1) / threshold_r;
    }

    private double[][] computeHessian2D(List<List<SiftImage>> dogPyramid, int o, int l, int r, int c) {
        double[][] H = new double[2][2];
        double v = dogPyramid.get(o).get(l).data[r][c];
        H[0][0] = dogPyramid.get(o).get(l).data[r][c + 1] + dogPyramid.get(o).get(l).data[r][c - 1] - 2 * v;
        H[1][1] = dogPyramid.get(o).get(l).data[r + 1][c] + dogPyramid.get(o).get(l).data[r - 1][c] - 2 * v;
        H[1][0] = H[0][1] = (dogPyramid.get(o).get(l).data[r + 1][c + 1] - dogPyramid.get(o).get(l).data[r + 1][c - 1]
                - dogPyramid.get(o).get(l).data[r - 1][c + 1] + dogPyramid.get(o).get(l).data[r - 1][c - 1]) / 4.0;
        return H;
    }

    public static void main(String[] args) {
        int nOctaveLayers = 3;
        double sigma = 1.6;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10.0;
        int numOctaves = 4;
        boolean enable_precise_upscale = true;

        double[][] dummyImage = new double[2][2];

        SiftStage1 stage1 = new SiftStage1(nOctaveLayers, sigma, numOctaves, true);
        List<KeypointCandidate> candidates = new ArrayList<>();
        List<List<SiftImage>> dogPyramid = new ArrayList<>();
        List<List<SiftImage>> gaussianPyramid = new ArrayList<>();

        SiftStage2 stage2 = new SiftStage2(contrastThreshold, edgeThreshold,nOctaveLayers, enable_precise_upscale);
        List<Keypoint> refinedKeypoints = stage2.run(candidates, dogPyramid, gaussianPyramid);

        System.out.printf("Sau Giai đoạn 1, có %d ứng viên.\n", candidates.size());
        System.out.printf("Sau Giai đoạn 2, còn lại %d điểm khóa đã được tinh chỉnh.\n", refinedKeypoints.size());
        System.out.println("Đây là một vài ví dụ (nếu có):");

        for (int i = 0; i < Math.min(5, refinedKeypoints.size()); i++) {
            System.out.println(refinedKeypoints.get(i));
        }
    }
}