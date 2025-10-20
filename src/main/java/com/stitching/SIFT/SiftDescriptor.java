package com.stitching.SIFT;

/**
 * Lớp này đại diện cho một bộ mô tả SIFT hoàn chỉnh.
 * Nó chứa tất cả thông tin của OrientedKeypoint và vector 128 chiều.
 */
public class SiftDescriptor extends OrientedKeypoint {
    public final double[] descriptor;

    public SiftDescriptor(OrientedKeypoint okp, double[] descriptor) {
        super(okp, okp.orientation);
        this.descriptor = descriptor;
    }

    @Override
    public String toString() {
        return super.toString() + String.format(" | Descriptor[0..2]=%.3f, %.3f,...", descriptor[0], descriptor[1]);
    }
}
