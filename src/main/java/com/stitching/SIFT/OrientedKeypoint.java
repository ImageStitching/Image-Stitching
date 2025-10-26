package com.stitching.SIFT;

import lombok.Getter;
import lombok.Setter;

/**
 * Lớp này đại diện cho một điểm khóa đã được gán hướng.
 * Nó kế thừa thông tin từ Keypoint và thêm thuộc tính orientation.
 */
@Getter
@Setter
public class OrientedKeypoint extends Keypoint {
    public final double orientation; // Hướng của điểm khóa, tính bằng radian

    public OrientedKeypoint(Keypoint keypoint, double orientation) {
        super(keypoint.x, keypoint.y, keypoint.octave, keypoint.layer, keypoint.sigma);
        this.orientation = orientation;
    }

    @Override
    public String toString() {
        return super.toString() + String.format(" | Orientation=%.2f°", Math.toDegrees(orientation));
    }
}
