package com.stitching.imageStitching.warper;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.remap;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;

public class SphericalWarper {
    /**
     * Warp an image to spherical (equirectangular-style) panorama.
     *
     * @param src  input image
     * @param f    focal length in pixels (typical: width/ (2*tan(fov/2)) )
     * @param outW output panorama width (choose larger for long panoramas)
     * @param outH output panorama height
     * @return warped panorama Mat
     */
    public static Mat warp(Mat src, double f/*, int outW, int outH*/) {
        int w = src.cols();
        int h = src.rows();
        double cx = w / 2.0, cy = h / 2.0;

        int outH = h, outW = w;


        Mat mapX = new Mat(outH, outW, CV_32F);
        Mat mapY = new Mat(outH, outW, CV_32F);


        FloatPointer pX = new FloatPointer(mapX.data());
        FloatPointer pY = new FloatPointer(mapY.data());


// For each pixel in panorama (u,v) compute spherical coordinates lon/lat then project back to source image plane
        for (int v = 0; v < outH; v++) {
            for (int u = 0; u < outW; u++) {
// longitude: map u in [0..outW) -> [-pi..+pi)
                double lon = ((double) u - outW / 2.0) / f; // approx in radians
                double lat = ((double) v - outH / 2.0) / f; // approx


// 3D direction vector (unit sphere)
                double x = Math.cos(lat) * Math.sin(lon);
                double y = Math.sin(lat);
                double z = Math.cos(lat) * Math.cos(lon);


// project to camera plane of source image (pinhole): (X/Z, Y/Z) scaled by f and shifted by cx,cy
                double srcX = (f * (x / z)) + cx;
                double srcY = (f * (y / z)) + cy;


                int idx = v * outW + u;
                pX.put(idx, (float) srcX);
                pY.put(idx, (float) srcY);
            }
        }


        Mat dst = new Mat();
        remap(src, dst, mapX, mapY, INTER_LINEAR, BORDER_CONSTANT, new Scalar(0, 0, 0, 0));
        mapX.release();
        mapY.release();
        return dst;
    }


    // Example usage in main (for testing)
    public static void main(String[] args) {
        Mat img = imread("img_1.jpg");
        if (img.empty()) {
            System.err.println("Cannot read");
            return;
        }
        int outW = img.cols() * 3; // example: panorama wider than source
        int outH = img.rows();
        double f = img.cols() / (2.0 * Math.tan(Math.toRadians(50.0) / 2.0)); // guess focal from FOV 50 deg
        Mat pano = warp(img, f/*, outW, outH*/);
        imwrite("spherical_pano.jpg", pano);
        System.out.println("Spherical saved");
    }
}