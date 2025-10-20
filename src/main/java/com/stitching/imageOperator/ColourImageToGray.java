package com.stitching.imageOperator;

import edu.princeton.cs.introcs.Picture;

import java.awt.*;

public class ColourImageToGray {
    private static double[][] ORG_MATRIX;

    private static void takeMatrix(String linkImage) {
        Picture picture = new Picture(linkImage);
        ORG_MATRIX = new double[picture.width()][picture.height()];
        for(int w = 0; w< picture.width(); w++)
            for(int h =0; h< picture.height(); h++) {
                ORG_MATRIX[w][h] = 0.299 * picture.get(w,h).getRed() + 0.587 * picture.get(w,h).getGreen()
                        + 0.114 * picture.get(w,h).getBlue();
            }
    }

    public static double[][] grayMatrix(String linkImage) {
        return ORG_MATRIX;
    }

    public static Picture createGrayPictureFromLink(String linkImage) {
        takeMatrix(linkImage);
        double[][] m = ORG_MATRIX;
        Picture picture = new Picture(m.length, m[0].length);
        for(int w = 0; w< picture.width(); w++)
            for(int h =0; h< picture.height(); h++) {
                int gray = (int) Math.round(m[w][h]);
                // clamp về [0,255]
                gray = Math.max(0, Math.min(255, gray));
                Color c = new Color(gray, gray, gray);
                picture.set(w, h, c);
            }
        return picture;
    }

    public static void main(String[] args) {
        Picture a = createGrayPictureFromLink("src/main/java/org/example/operations/imgColor.png");
        a.save("src/main/java/org/example/operations/imgGray.png");
        a.show();
    }
}
