package com.stitching.selfcode.imageOperator;

import edu.princeton.cs.introcs.Picture;

import java.awt.*;
import java.util.ArrayList;

public class Matrix_Image {
    // Luôn alf format [height][width]

    public static Picture create_grayImage_from_gray_matrix(double[][] matrix) {
        Picture pic = new Picture(matrix[0].length, matrix.length);
        for(int y =0; y< matrix.length; y++)
            for(int x = 0; x< matrix[0].length; x++) {
                int grayValue = (int) Math.round(matrix[y][x]);
                grayValue = Math.max(0, Math.min(255, grayValue));
                pic.set( x , y ,
                        new Color(grayValue, grayValue, grayValue) // <-- Sửa tại đây
                );
            }
        return pic;
    }

    public static ArrayList<Integer>[][] create_color_matrix_from_color_image(String linkImage) {
        Picture picture = new Picture(linkImage);
        ArrayList<Integer>[][] matrix = new ArrayList[picture.height()][picture.width()];
        for(int w = 0; w< picture.width(); w++)
            for(int h =0; h< picture.height(); h++) {
                Color c = picture.get(w,h);
                matrix[h][w] = new ArrayList<>();
                matrix[h][w].add(picture.get(w, h).getRed());
                matrix[h][w].add(picture.get(w, h).getGreen());
                matrix[h][w].add(picture.get(w, h).getBlue());
            }
        return matrix;
    }

    public static int[][] create_INTgrayMatrix_from_color_image(String linkImage) {
        Picture picture = new Picture(linkImage);
        int[][] matrix = new int[picture.height()][picture.width()];
        for(int w = 0; w< picture.width(); w++)
            for(int h =0; h< picture.height(); h++) {
                matrix[h][w] = (int) Math.round(0.299 * picture.get(w,h).getRed() + 0.587 * picture.get(w,h).getGreen()
                        + 0.114 * picture.get(w,h).getBlue());
            }
        return matrix;
    }
    public static double[][] create_DOUBLEgrayMatrix_from_color_image(String linkImage) {
        Picture picture = new Picture(linkImage);
        double[][] matrix = new double[picture.height()][picture.width()];
        for(int w = 0; w< picture.width(); w++)
            for(int h =0; h< picture.height(); h++) {
                matrix[h][w] = Math.round(0.299 * picture.get(w,h).getRed() + 0.587 * picture.get(w,h).getGreen()
                        + 0.114 * picture.get(w,h).getBlue());
            }
        return matrix;
    }

    public static void main(String[] args) {
        ArrayList<Integer>[][] matrix = create_color_matrix_from_color_image("static/image/imgColor.png");
    }

}
