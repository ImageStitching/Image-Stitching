package com.stitching.SIFT;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
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
