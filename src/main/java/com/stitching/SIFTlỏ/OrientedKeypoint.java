package com.stitching.SIFTlỏ;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class OrientedKeypoint extends Keypoint {
    public final double orientation; // Hướng của điểm khóa, tính bằng radian [0;2r)

    public OrientedKeypoint(Keypoint keypoint, double orientation) {
        super(keypoint.x, keypoint.y, keypoint.octave, keypoint.layer, keypoint.sigma, keypoint.enable_precise_upscale, keypoint.response);
        double norm_orientation = orientation;
        while (norm_orientation < 0) norm_orientation += 2 * Math.PI;
        while (norm_orientation >= 2 * Math.PI) norm_orientation -= 2 * Math.PI;
        this.orientation = norm_orientation;
    }

    @Override
    public String toString() {
        return super.toString() + String.format(" | Orientation=%.2f°", Math.toDegrees(orientation));
    }
}
