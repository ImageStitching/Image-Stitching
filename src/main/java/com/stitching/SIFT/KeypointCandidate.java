package com.stitching.SIFT;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Comparator;

@AllArgsConstructor
@Getter
public class KeypointCandidate {
    public final int x, y, octave, layer;
    public final double sigma;
    public final boolean enable_precise_upscale;
    public static final Comparator<KeypointCandidate> BY_ALL = new ByAll();

//    public KeypointCandidate(int x, int y, int octave, int layer, double sigma, boolean enablePreciseUpscale) {
//        this.x = x;
//        this.y = y;
//        this.octave = octave;
//        this.layer = layer;
//        this.sigma = sigma;
//        this.enable_precise_upscale = enablePreciseUpscale;
//    }

    @Override
    public String toString() {
        // Octave 0 tương ứng với ảnh đã nhân đôi, nên ta phải chia 2 để về tọa độ gốc
        int scaleFactor;
        if(this.enable_precise_upscale) scaleFactor = 1 << (octave - 1); // 2^(octave-1)
        else scaleFactor = 1 << (octave);  // 2^(octave)
        int originalX = x * scaleFactor;
        int originalY = y * scaleFactor;
        return String.format(
                "Candidate[Octave=%d, Layer=%d] at (%d, %d) -> Original Coords (~%d, ~%d) with sigma=%.2f",
                octave, layer, x, y, originalX, originalY, sigma
        );
    }

    public static class ByAll implements Comparator<KeypointCandidate> {
        @Override
        public int compare(KeypointCandidate o1, KeypointCandidate o2) {
            return 0;
        }
    }

    public boolean equal(KeypointCandidate that) {
        if(that==null) return false;
        if(this.getClass() != that.getClass()) return false;
        return ((this.x == that.x) &&(this.y == that.y) && (this.octave == that.octave) && (this.layer == that.layer) && (this.sigma == that.sigma));
    }
}

