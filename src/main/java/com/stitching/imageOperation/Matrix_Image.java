package com.stitching.imageOperation;

import edu.princeton.cs.introcs.Picture;

import java.awt.*;
import java.util.ArrayList;

public class Matrix_Image {
    public static Picture create_grayImage_from_gray_matrix(double[][] matrix) {
        Picture pic = new Picture(matrix.length, matrix[0].length);
        for(int i = 0; i< matrix.length; i++)
            for(int j =0; j< matrix[0].length; j++) {
                int grayValue = (int) Math.round(matrix[i][j]);
                grayValue = Math.max(0, Math.min(255, grayValue));

                pic.set( i , j ,
                        new Color(grayValue, grayValue, grayValue) // <-- Sửa tại đây
                );
            }
        return pic;
    }

    public static ArrayList<Integer>[][] create_color_matrix_from_color_image(String linkImage) {
        Picture picture = new Picture(linkImage);
        ArrayList<Integer>[][] matrix = new ArrayList[picture.width()][picture.height()];
        for(int w = 0; w< picture.width(); w++)
            for(int h =0; h< picture.height(); h++) {
                Color c = picture.get(w,h);
                matrix[w][h] = new ArrayList<>();
                matrix[w][h].add(picture.get(w, h).getRed());
                matrix[w][h].add(picture.get(w, h).getGreen());
                matrix[w][h].add(picture.get(w, h).getBlue());
            }
        return matrix;
    }

    public static int[][] create_INTgrayMatrix_from_color_image(String linkImage) {
        Picture picture = new Picture(linkImage);
        int[][] matrix = new int[picture.width()][picture.height()];
        for(int w = 0; w< picture.width(); w++)
            for(int h =0; h< picture.height(); h++) {
                matrix[w][h] = (int) Math.round(0.299 * picture.get(w,h).getRed() + 0.587 * picture.get(w,h).getGreen()
                        + 0.114 * picture.get(w,h).getBlue());
            }
        return matrix;
    }
    public static double[][] create_DOUBLEgrayMatrix_from_color_image(String linkImage) {
        Picture picture = new Picture(linkImage);
        double[][] matrix = new double[picture.width()][picture.height()];
        for(int w = 0; w< picture.width(); w++)
            for(int h =0; h< picture.height(); h++) {
                matrix[w][h] = Math.round(0.299 * picture.get(w,h).getRed() + 0.587 * picture.get(w,h).getGreen()
                        + 0.114 * picture.get(w,h).getBlue());
            }
        return matrix;
    }

    public static void main(String[] args) {
        ArrayList<Integer>[][] matrix = create_color_matrix_from_color_image("static/image/imgColor.png");
    }

}
