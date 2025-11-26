package com.stitching.selfcode.SIFTlỏ;

import edu.princeton.cs.introcs.Picture;
import edu.princeton.cs.introcs.StdDraw;

import java.awt.*;
import java.util.List;

public class KeypointVisualizer {
    public static void drawKeypoints(Picture input, List<ImageFeature.KeyPointInfo> keypoints) {
        input.show();
        int width = input.width();
        int height = input.height();

        StdDraw.setCanvasSize(width, height);
        StdDraw.setXscale(0, width);
        StdDraw.setYscale(0, height);

//        for (int r = 0; r < height; r++) {
//            for (int c = 0; c < width; c++) {
//                Color color = input.get(c,r);
//                StdDraw.setPenColor(color);
//                StdDraw.point(c, height - 1 - r); // lưu ý đảo trục y
//            }
//        }

        for (ImageFeature.KeyPointInfo kp : keypoints) {
            int x = (int) Math.round(kp.pt_x);
            int y = (int) Math.round(kp.pt_y);
            Color color = switch (kp.octave) {
                case 0 -> Color.RED;
                case 1 -> Color.GREEN;
                case 2 -> Color.BLUE;
                case 3 -> Color.YELLOW;
                case 4 -> Color.CYAN;
                default -> Color.WHITE;
            };
            StdDraw.setPenColor(color);
            StdDraw.filledCircle(x, height - 1 - y, 3); // radius = 3 pixels
        }
        StdDraw.show();
    }
}
