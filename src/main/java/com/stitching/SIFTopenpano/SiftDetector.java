package com.stitching.SIFTopenpano;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.javacpp.indexer.*;
import static org.bytedeco.opencv.global.opencv_core.*;

import java.util.ArrayList;
import java.util.List;

public class SiftDetector {

    // --- MAIN PIPELINE ---
    public List<SiftKeyPoint> run(List<MatVector> gaussianPyramid, List<MatVector> dogPyramid) {
        List<SiftKeyPoint> keypoints = new ArrayList<>();
        double threshold = SiftConfig.getContrastThreshold();

        for (int o = 0; o < dogPyramid.size(); o++) {
            MatVector dogOctave = dogPyramid.get(o);
            MatVector gaussOctave = gaussianPyramid.get(o);
            long numLayers = dogOctave.size();

            for (int s = 1; s < numLayers - 1; s++) {
                Mat imgBelow = dogOctave.get(s - 1);
                Mat imgCurr  = dogOctave.get(s);
                Mat imgAbove = dogOctave.get(s + 1);

                FloatIndexer idxBelow = imgBelow.createIndexer();
                FloatIndexer idxCurr  = imgCurr.createIndexer();
                FloatIndexer idxAbove = imgAbove.createIndexer();

                int rows = imgCurr.rows();
                int cols = imgCurr.cols();

                // 1. Find Extrema
                for (int y = 5; y < rows - 5; y++) {
                    for (int x = 5; x < cols - 5; x++) {
                        float val = idxCurr.get(y, x);
                        if (Math.abs(val) < threshold) continue;

                        if (isExtremum(val, x, y, idxBelow, idxCurr, idxAbove)) {
                            // 2. Interpolation & Edge Rejection
                            SiftKeyPoint kp = interpolate(x, y, o, s, idxBelow, idxCurr, idxAbove);
                            if (kp != null) {
                                // 3. Orientation Assignment
                                assignOrientation(kp, gaussOctave.get(s)); // Lấy ảnh Gaussian tương ứng
                                
                                // 4. Descriptor Generation
                                computeDescriptor(kp, gaussOctave.get(s));
                                
                                keypoints.add(kp);
                            }
                        }
                    }
                }
                idxBelow.release(); idxCurr.release(); idxAbove.release();
            }
        }
        return keypoints;
    }

    // --- STEP 1: EXTREMA CHECK ---
    private boolean isExtremum(float val, int x, int y, FloatIndexer below, FloatIndexer curr, FloatIndexer above) {
        boolean isMax = val > 0; 
        
        // Check 8 neighbors in current
        if (val <= curr.get(y-1, x-1) || val <= curr.get(y-1, x) || val <= curr.get(y-1, x+1) ||
            val <= curr.get(y,   x-1) ||                        val <= curr.get(y,   x+1) ||
            val <= curr.get(y+1, x-1) || val <= curr.get(y+1, x) || val <= curr.get(y+1, x+1)) isMax = false;
            
        if (!isMax) {
            // Check below
            for (int dy = -1; dy <= 1; dy++) 
                for (int dx = -1; dx <= 1; dx++) 
                    if (val <= below.get(y+dy, x+dx)) return false;
            // Check above
            for (int dy = -1; dy <= 1; dy++) 
                for (int dx = -1; dx <= 1; dx++) 
                    if (val <= above.get(y+dy, x+dx)) return false;
        }
        return true; 
        // Lưu ý: Logic chuẩn cần check (isMax || isMin). Đây là bản rút gọn check Max.
    }

    // --- STEP 2: INTERPOLATION (OpenPano sift.cc logic) ---
    private SiftKeyPoint interpolate(int c, int r, int octave, int layer, FloatIndexer below, FloatIndexer curr, FloatIndexer above) {
        // Gradient (dx, dy, ds)
        float dx = (curr.get(r, c+1) - curr.get(r, c-1)) * 0.5f;
        float dy = (curr.get(r+1, c) - curr.get(r-1, c)) * 0.5f;
        float ds = (above.get(r, c) - below.get(r, c)) * 0.5f;

        // Hessian (3x3)
        float v2 = 2.0f * curr.get(r, c);
        float dxx = curr.get(r, c+1) + curr.get(r, c-1) - v2;
        float dyy = curr.get(r+1, c) + curr.get(r-1, c) - v2;
        float dss = above.get(r, c) + below.get(r, c) - v2;
        float dxy = (curr.get(r+1, c+1) - curr.get(r+1, c-1) - curr.get(r-1, c+1) + curr.get(r-1, c-1)) * 0.25f;
        float dxs = (above.get(r, c+1) - above.get(r, c-1) - below.get(r, c+1) + below.get(r, c-1)) * 0.25f;
        float dys = (above.get(r+1, c) - above.get(r-1, c) - below.get(r+1, c) - below.get(r-1, c)) * 0.25f;

        // Solve Hx = -D
        Mat H = new Mat(3, 3, CV_32F);
        FloatIndexer hIdx = H.createIndexer();
        hIdx.put(0,0, dxx); hIdx.put(0,1, dxy); hIdx.put(0,2, dxs);
        hIdx.put(1,0, dxy); hIdx.put(1,1, dyy); hIdx.put(1,2, dys);
        hIdx.put(2,0, dxs); hIdx.put(2,1, dys); hIdx.put(2,2, dss);
        
        Mat D = new Mat(3, 1, CV_32F);
        FloatIndexer dIdx = D.createIndexer();
        dIdx.put(0,0, -dx); dIdx.put(1,0, -dy); dIdx.put(2,0, -ds);

        Mat X = new Mat();
        solve(H, D, X, DECOMP_LU); // Giải hệ phương trình
        
        FloatIndexer xIdx = X.createIndexer();
        float ox = xIdx.get(0, 0); // Offset x
        float oy = xIdx.get(1, 0); // Offset y
        float os = xIdx.get(2, 0); // Offset scale

        // Check sự hội tụ
        if (Math.abs(ox) > 0.5 || Math.abs(oy) > 0.5 || Math.abs(os) > 0.5) return null;

        // Edge check (Hessian determinant/trace) - Chỉ dùng dxx, dyy, dxy
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;
        if (det <= 0) return null;
        float edgeThresh = (float) SiftConfig.EDGE_THRESHOLD;
        if ((tr * tr) / det >= ((edgeThresh + 1) * (edgeThresh + 1) / edgeThresh)) return null;

        // Tính tọa độ và scale cuối cùng
        float scaleInOctave = (float) (SiftConfig.SIGMA_INIT * Math.pow(2.0, (layer + os) / SiftConfig.SCALES_PER_OCTAVE));
        float realScale = scaleInOctave * (float) Math.pow(2.0, octave);
        float realX = (c + ox) * (float) Math.pow(2.0, octave);
        float realY = (r + oy) * (float) Math.pow(2.0, octave);

        return new SiftKeyPoint(realX, realY, octave, layer, realScale);
    }

    // --- STEP 3: ORIENTATION (orientation.cc logic) ---
    private void assignOrientation(SiftKeyPoint kp, Mat img) {
        // Ảnh này là ảnh Gaussian đã làm mờ tại layer tương ứng
        FloatIndexer idx = img.createIndexer();
        int rows = img.rows();
        int cols = img.cols();

        // Tọa độ trên layer hiện tại
        float scl = kp.scale / (float)Math.pow(2, kp.octave);
        int r = (int) (kp.y / Math.pow(2, kp.octave));
        int c = (int) (kp.x / Math.pow(2, kp.octave));
        int radius = (int) (3 * 1.5 * scl);

        float[] hist = new float[36];

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int y = r + i;
                int x = c + j;
                if (y <= 0 || y >= rows - 1 || x <= 0 || x >= cols - 1) continue;

                float dx = idx.get(y, x+1) - idx.get(y, x-1);
                float dy = idx.get(y+1, x) - idx.get(y-1, x);
                float mag = (float) Math.sqrt(dx*dx + dy*dy);
                float angle = (float) Math.toDegrees(Math.atan2(dy, dx)); 
                if (angle < 0) angle += 360;

                float weight = (float) Math.exp(-(i*i + j*j) / (2 * (1.5 * scl) * (1.5 * scl)));
                int bin = (int) (angle / 10);
                if (bin >= 36) bin = 0;
                
                hist[bin] += mag * weight;
            }
        }
        
        // Tìm hướng chính (Max Peak)
        float maxVal = 0;
        int maxBin = 0;
        for(int k=0; k<36; k++) {
            if (hist[k] > maxVal) { maxVal = hist[k]; maxBin = k; }
        }
        kp.angle = maxBin * 10 + 5; // Lấy tâm của bin
    }

    // --- STEP 4: DESCRIPTOR (descriptor logic) ---
    private void computeDescriptor(SiftKeyPoint kp, Mat img) {
        // Chuẩn bị
        FloatIndexer idx = img.createIndexer();
        int rows = img.rows();
        int cols = img.cols();
        
        float scale = kp.scale / (float)Math.pow(2, kp.octave);
        float angleRad = (float) Math.toRadians(kp.angle);
        float cos_t = (float) Math.cos(angleRad);
        float sin_t = (float) Math.sin(angleRad);

        float hist_width = SiftConfig.DESCRIPTOR_HIST_WIDTH; // 4
        float bins = SiftConfig.DESCRIPTOR_HIST_BINS; // 8
        
        // Vùng lấy mẫu thực tế
        int radius = (int) ((hist_width + 1) * Math.sqrt(2) * (scale * SiftConfig.DESCRIPTOR_HIST_WIDTH + 0.5) / 2.0);
        if (radius < 1) radius = 1;

        int r_kp = (int) (kp.y / Math.pow(2, kp.octave));
        int c_kp = (int) (kp.x / Math.pow(2, kp.octave));

        // Duyệt qua vùng pixel xung quanh keypoint
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                // Xoay tọa độ (i,j) về hệ trục của Keypoint
                float j_rot = j * cos_t - i * sin_t;
                float i_rot = j * sin_t + i * cos_t;
                
                // Chuẩn hóa về tọa độ bin (-2 đến 2 cho lưới 4x4)
                float r_bin = i_rot / scale + hist_width / 2f - 0.5f;
                float c_bin = j_rot / scale + hist_width / 2f - 0.5f;

                if (r_bin > -1 && r_bin < hist_width && c_bin > -1 && c_bin < hist_width) {
                    int y = r_kp + i; 
                    int x = c_kp + j;
                    if (y <= 0 || y >= rows - 1 || x <= 0 || x >= cols - 1) continue;

                    float dx = idx.get(y, x+1) - idx.get(y, x-1);
                    float dy = idx.get(y+1, x) - idx.get(y-1, x);
                    float mod = (float) Math.sqrt(dx*dx + dy*dy);
                    float ori = (float) Math.atan2(dy, dx);
                    
                    // Xoay hướng gradient theo hướng chính
                    float ori_rot = ori - angleRad;
                    while (ori_rot < 0) ori_rot += 2 * Math.PI;
                    while (ori_rot >= 2 * Math.PI) ori_rot -= 2 * Math.PI;
                    
                    float o_bin = (float) (ori_rot * bins / (2 * Math.PI));
                    float weight = (float) Math.exp(-(r_bin*r_bin + c_bin*c_bin) / (0.5 * hist_width * hist_width)); // Gaussian weighting window

                    // Nội suy 3 chiều (Trilinear interpolation) vào descriptor
                    distributeToHistogram(kp.descriptor, r_bin, c_bin, o_bin, mod * weight);
                }
            }
        }
        
        // Chuẩn hóa và clamp
        normalizeAndClamp(kp.descriptor);
    }

    private void distributeToHistogram(float[] desc, float r, float c, float o, float mag) {
        int r0 = (int) Math.floor(r);
        int c0 = (int) Math.floor(c);
        int o0 = (int) Math.floor(o);
        
        float dr = r - r0;
        float dc = c - c0;
        float do_ = o - o0;

        for (int ir = 0; ir <= 1; ir++) {
            int r_idx = r0 + ir;
            if (r_idx >= 0 && r_idx < 4) {
                for (int ic = 0; ic <= 1; ic++) {
                    int c_idx = c0 + ic;
                    if (c_idx >= 0 && c_idx < 4) {
                        for (int io = 0; io <= 1; io++) {
                            int o_idx = (o0 + io) % 8; // Wrap around orientation
                            float val = mag * (ir == 0 ? 1 - dr : dr) 
                                            * (ic == 0 ? 1 - dc : dc) 
                                            * (io == 0 ? 1 - do_ : do_);
                            
                            // Index trong mảng 128 (4x4x8)
                            int idx = (r_idx * 4 + c_idx) * 8 + o_idx;
                            desc[idx] += val;
                        }
                    }
                }
            }
        }
    }

    private void normalizeAndClamp(float[] vec) {
        float sum = 0;
        for (float v : vec) sum += v * v;
        sum = (float) Math.sqrt(sum);
        if (sum == 0) return;
        
        // Normalize
        for (int i = 0; i < vec.length; i++) {
            vec[i] /= sum;
            if (vec[i] > 0.2f) vec[i] = 0.2f; // Clamp at 0.2
        }
        
        // Re-normalize
        sum = 0;
        for (float v : vec) sum += v * v;
        sum = (float) Math.sqrt(sum);
        if (sum == 0) return;
        for (int i = 0; i < vec.length; i++) vec[i] /= sum;
    }
}