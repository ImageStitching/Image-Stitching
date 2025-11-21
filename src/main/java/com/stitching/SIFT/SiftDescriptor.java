package com.stitching.SIFT;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class SiftDescriptor extends OrientedKeypoint {
    public final byte[] descriptor;

    public SiftDescriptor(OrientedKeypoint okp, double[] descriptorFloat) {
        super(okp, okp.orientation);
        // CONVERT: float [0, 1] -> byte [0, 255]
        this.descriptor = new byte[128];
        for (int i = 0; i < 128; i++) {
//            int value = (int) Math.round(descriptorFloat[i] * 512.0);  // OpenCV quantization
//            this.descriptor[i] = (byte) Math.max(0, Math.min(255, value));
            int value = Math.min(255, Math.max(0, (int) Math.round(descriptorFloat[i] * 255.0)));
            this.descriptor[i] = (byte) (value & 0xFF);
        }
    }

    @Override
    public String toString() {
        return super.toString() + String.format(" | Descriptor[0..2]=%d, %d,...", descriptor[0] & 0xFF, descriptor[1] & 0xFF);
    }

    public double[] doubleDescriptor() {
        double[] descript = new double[128];
        for (int i = 0; i < 128; i++) {
//            double value = (double) Math.round(descriptor[i] / 512.0);  // OpenCV quantization
            double value = ((descriptor[i] & 0xFF) / 255.0);
            descript[i] = Math.max(0, Math.min(255, value));
        }
        return descript;
    }
}
