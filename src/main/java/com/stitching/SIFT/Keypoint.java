package com.stitching.SIFT;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

/**
 * Lớp này đại diện cho một điểm khóa (Keypoint) đã được tinh chỉnh và lọc sau Giai đoạn 2.
 * Nó chứa vị trí chính xác ở mức dưới pixel.
 */
@AllArgsConstructor
@Getter
@Setter
public class Keypoint {
    public final double x, y; // Tọa độ dưới pixel và sigma đã tinh chỉnh
    public final int octave, layer;
    public final double sigma;

//    public Keypoint(double x, double y, int octave, int layer, double sigma) {
//        this.x = x; this.y = y; this.octave = octave; this.layer = layer; this.sigma = sigma;
//    }

    @Override
    public String toString() {
        // Octave 0 tương ứng với ảnh đã nhân đôi, nên ta phải chia 2 để về tọa độ gốc
        int scaleFactor = 1 << (octave - 1);
        double originalX = x * scaleFactor;
        double originalY = y * scaleFactor;
        return String.format(
                "Refined Keypoint[Octave=%d, Layer=%d] at (%.2f, %.2f) -> Original Coords (~%.2f, ~%.2f) with sigma=%.2f",
                octave, layer, x, y, originalX, originalY, sigma
        );
    }
    public boolean equal(Keypoint that) {
        if(that==null) return false;
        if(this.getClass() != that.getClass()) return false;
        return ((this.x == that.x) &&(this.y == that.y) && (this.octave == that.octave) && (this.layer == that.layer) && (this.sigma == that.sigma));
    }
}
