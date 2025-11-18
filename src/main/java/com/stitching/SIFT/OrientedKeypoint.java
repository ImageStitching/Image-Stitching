package com.stitching.SIFT;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class OrientedKeypoint extends Keypoint {
    public final double orientation; // Hướng của điểm khóa, tính bằng radian

    public OrientedKeypoint(Keypoint keypoint, double orientation) {
        super(keypoint.x, keypoint.y, keypoint.octave, keypoint.layer, keypoint.sigma, keypoint.enable_precise_upscale);
        this.orientation = orientation;
    }

    @Override
    public String toString() {
        return super.toString() + String.format(" | Orientation=%.2f°", Math.toDegrees(orientation));
    }
}
