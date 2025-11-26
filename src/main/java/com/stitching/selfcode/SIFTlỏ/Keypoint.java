package com.stitching.selfcode.SIFTlỏ;

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
    public final double x, y; // Tọa độ dưới pixel và sigma đã tinh chỉnh , x là chiều width, y là height
    public final int octave, layer;
    public final double sigma;
    public final boolean enable_precise_upscale;
    public final double response;       // Độ mạnh của keypoint (contrast)

    public boolean equal(Keypoint that) {
        if(that==null) return false;
        if(this.getClass() != that.getClass()) return false;
        return ((this.x == that.x) &&(this.y == that.y) && (this.octave == that.octave) && (this.layer == that.layer) && (this.sigma == that.sigma));
    }
}
