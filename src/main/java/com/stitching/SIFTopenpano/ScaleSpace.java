package com.stitching.SIFTopenpano;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import java.util.ArrayList;
import java.util.List;

public class ScaleSpace {

    public List<MatVector> buildGaussianPyramid(Mat baseImage) {
        List<MatVector> pyramid = new ArrayList<>();
        Mat currentImg = baseImage.clone();

        if (SiftConfig.DOUBLE_IMAGE_SIZE) {
            Mat upscaled = new Mat();
            // Upscale x2 dùng nội suy Linear
            resize(currentImg, upscaled, new Size(), 2.0, 2.0, INTER_LINEAR);

            // Theo Lowe: Ảnh gốc coi như có sigma = 0.5.
            // Khi upscale x2, sigma thực tế cũng tăng -> cần tính toán lại sigma blur ban đầu nếu muốn chuẩn xác tuyệt đối.
            // Tuy nhiên, để đơn giản và giống OpenPano, ta chỉ cần upscale và gán lại.
            currentImg = upscaled;
        }
        
        double k = Math.pow(2, 1.0 / SiftConfig.SCALES_PER_OCTAVE);
        double[] sigmas = new double[SiftConfig.SCALES_PER_OCTAVE + 3];
        
        sigmas[0] = SiftConfig.SIGMA_INIT;
        for (int i = 1; i < sigmas.length; i++) {
            double prevSigma = Math.pow(k, i - 1) * SiftConfig.SIGMA_INIT;
            double totalSigma = Math.pow(k, i) * SiftConfig.SIGMA_INIT;
            sigmas[i] = Math.sqrt(totalSigma * totalSigma - prevSigma * prevSigma);
        }

        for (int o = 0; o < SiftConfig.NUM_OCTAVES; o++) {
            MatVector octave = new MatVector(sigmas.length);
            octave.put(0, currentImg);

            for (int i = 1; i < sigmas.length; i++) {
                Mat prev = octave.get(i - 1);
                Mat next = new Mat();
                GaussianBlur(prev, next, new Size(0, 0), sigmas[i], sigmas[i], BORDER_DEFAULT);
                octave.put(i, next);
            }
            pyramid.add(octave);

            if (o < SiftConfig.NUM_OCTAVES - 1) {
                Mat baseNext = octave.get(SiftConfig.SCALES_PER_OCTAVE);
                Mat downsampled = new Mat();
                resize(baseNext, downsampled, new Size(), 0.5, 0.5, INTER_NEAREST);
                currentImg = downsampled;
            }
        }
        return pyramid;
    }

    public List<MatVector> buildDoGPyramid(List<MatVector> gPyramid) {
        List<MatVector> dogPyramid = new ArrayList<>();
        for (MatVector octave : gPyramid) {
            long size = octave.size();
            MatVector dogOctave = new MatVector(size - 1);
            for (long i = 0; i < size - 1; i++) {
                Mat diff = new Mat();
                subtract(octave.get(i + 1), octave.get(i), diff);
                dogOctave.put(i, diff);
            }
            dogPyramid.add(dogOctave);
        }
        return dogPyramid;
    }
}