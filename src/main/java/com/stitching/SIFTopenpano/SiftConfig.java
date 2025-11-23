package com.stitching.SIFTopenpano;

public class SiftConfig {
    // Cấu hình cơ bản
    public static final int NUM_OCTAVES = 4;
    public static final int SCALES_PER_OCTAVE = 3;
    public static final double SIGMA_INIT = 1.6;
    public static final double CONTRAST_THRESHOLD = 0.04;
    public static final double EDGE_THRESHOLD = 10.0;
    public static final boolean DOUBLE_IMAGE_SIZE = true;
    
    // Cấu hình Descriptor & Orientation
    public static final int DESCRIPTOR_HIST_WIDTH = 4; // Lưới 4x4
    public static final int DESCRIPTOR_HIST_BINS = 8;  // 8 hướng
    public static final double MAG_N_SIGMA = 1.5;      // Hệ số bán kính lấy mẫu
    public static final double SIFT_DESCR_WIDTH = 3.0; // Độ rộng descriptor

    
    // Ngưỡng contrast thực tế
    public static double getContrastThreshold() {
        return CONTRAST_THRESHOLD / SCALES_PER_OCTAVE;
    }
}