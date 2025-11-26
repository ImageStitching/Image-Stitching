package com.stitching.imageStitching;

import com.stitching.openpanoSIFT.SiftKeyPoint;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_64F;

public class ImageNode {
    public int id;
    public String filename;
    public Mat img;
    public List<SiftKeyPoint> keypoints;
    public Mat descriptors;
    public Mat globalTransform;

    public ImageNode(int id, String name, Mat img, List<SiftKeyPoint> kp, Mat desc) {
        this.id = id;
        this.filename = name;
        this.img = img;
        this.keypoints = kp;
        this.descriptors = desc;
        this.globalTransform = Mat.eye(3, 3, CV_64F).asMat();
    }

    public ImageNode(int id, String name, Mat img, List<SiftKeyPoint> kp, Mat desc, Mat globalTransform) {
        this.id = id;
        this.filename = name;
        this.img = img;
        this.keypoints = kp;
        this.descriptors = desc;
        this.globalTransform = globalTransform;
    }
}