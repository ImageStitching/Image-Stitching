package com.stitching.openpanoSIFT;

public class SiftKeyPoint {
    public float x, y;          // Tọa độ trên ảnh gốc
    public int octave;          // Thuộc octave nào
    public int layer;           // Layer trong octave
    public float scale;         // Scale thực tế (sigma)
    public float angle;         // Hướng (degree 0-360)
    public float[] descriptor;  // Vector 128 chiều

    public SiftKeyPoint(float x, float y, int octave, int layer, float scale) {
        this.x = x;
        this.y = y;
        this.octave = octave;
        this.layer = layer;
        this.scale = scale;
        this.descriptor = new float[128];
    }
}