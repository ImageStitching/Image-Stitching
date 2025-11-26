package com.stitching.imageStitching.warper;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;


public class ExteriorCylindricalWarper {
    /**
     * Approximate exterior cylindrical warp: map image onto the OUTER surface of a vertical cylinder.
     * This is an approximate inverse-mapping suitable when the camera or viewpoints should see the image
     * from the outside of the cylinder (i.e., the printed photos face outward).
     * <p>
     * NOTE: Exterior cylinder mathematically is similar to interior but with reversed sign of radius/visibility.
     * This implementation flips the interior mapping across the Z axis to simulate an exterior surface.
     */
    public static Mat warp(Mat src, double f, int outW, int outH) {
        int w = src.cols();
        int h = src.rows();
        double cx = w / 2.0, cy = h / 2.0;


        Mat mapX = new Mat(outH, outW, CV_32F);
        Mat mapY = new Mat(outH, outW, CV_32F);
        FloatPointer pX = new FloatPointer(mapX.data());
        FloatPointer pY = new FloatPointer(mapY.data());


// We'll treat longitude lon = (u - outW/2)/f; latitude lat = (v - outH/2)/f;
// For an exterior cylinder, the direction vector of a surface point is:
// X = sin(lon), Y = lat, Z = cos(lon)
// To find which source pixel maps there, we simulate a camera at Z = -f looking toward +Z (flip) and intersect ray with plane


        for (int v = 0; v < outH; v++) {
            for (int u = 0; u < outW; u++) {
                double lon = ((double) u - outW / 2.0) / f; // radians approx
                double lat = ((double) v - outH / 2.0) / f;


// surface point on cylinder of radius R=1 (direction)
                double sx = Math.sin(lon);
                double sy = lat; // small angle approx
                double sz = Math.cos(lon);


// Now simulate projecting that surface point back to source image camera at origin looking along +Z.
// Ray from camera (0,0,0) through (sx,sy,sz) intersects camera image plane z=1 → image point (sx/sz, sy/sz)
                double imgX = f * (sx / sz) + cx;
                double imgY = f * (sy / sz) + cy;


                int idx = v * outW + u;
                pX.put(idx, (float) imgX);
                pY.put(idx, (float) imgY);
            }
        }


        Mat dst = new Mat();
        remap(src, dst, mapX, mapY, INTER_LINEAR, BORDER_CONSTANT, new Scalar(0, 0, 0, 0));
        mapX.release();
        mapY.release();
        return dst;
    }


    public static void main(String[] args) {
        Mat src = imread("img_1.jpg");
        if (src.empty()) {
            System.err.println("read error");
            return;
        }
        int outW = src.cols() * 2;
        int outH = src.rows();
        double f = src.cols() / (2.0 * Math.tan(Math.toRadians(50.0) / 2.0));
        Mat ex = warp(src, f, outW, outH);
        imwrite("exterior_cyl.jpg", ex);
        System.out.println("Exterior saved");
    }
}