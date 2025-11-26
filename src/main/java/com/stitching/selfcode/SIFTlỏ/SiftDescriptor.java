package com.stitching.selfcode.SIFTlỏ;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class SiftDescriptor extends OrientedKeypoint {
    public final byte[] descriptor;

    public SiftDescriptor(OrientedKeypoint okp, double[] descriptorFloat) {
        super(okp, okp.orientation);
        // CONVERT: float [0, 1] -> byte [0, 511]
        this.descriptor = new byte[128];
        for (int i = 0; i < 128; i++) {
            int value = (int) (descriptorFloat[i] * 512.0);
            value = Math.max(0, Math.min(511, value));  // OpenCV dùng 512, clamp 0-511
            this.descriptor[i] = (byte) value;
        }
    }

    @Override
    public String toString() {
        return super.toString() + String.format(" | Descriptor[0..2]=%d, %d,...", descriptor[0] & 0xFF, descriptor[1] & 0xFF);
    }

    public double[] doubleDescriptor() {
        double[] descript = new double[128];
        for (int i = 0; i < 128; i++) {
            double value = ((descriptor[i] & 0xFF) / 512.0f);
            descript[i] = Math.max(0, Math.min(511, value));
        }
        return descript;
    }
}
