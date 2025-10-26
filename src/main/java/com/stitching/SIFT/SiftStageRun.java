package com.stitching.SIFT;

import com.stitching.imageOperator.Matrix_Image;

import java.nio.file.Path;
import java.nio.file.Paths;

public class SiftStageRun {
    private static Path INPUT_PATH = Paths.get("src","main","resource","static","sift");

    public static void main(String[] args) {
        double[][] org_img = Matrix_Image.create_DOUBLEgrayMatrix_from_color_image(INPUT_PATH.resolve("org_img.png").toString());
        SiftStage1 siftStage1 = new SiftStage1(3,1.6,5,true);
    }
}
