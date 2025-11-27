package com.stitching.API;

import com.stitching.imageStitching.CylinderStitcherEnhanced;
import org.bytedeco.opencv.presets.opencv_core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Service
public class StitchingService {

    private static final Path INPUTPATH = Paths.get("src", "main", "resources", "uploads");

    // Thư mục lưu ảnh kết quả: stitch/
    private static final Path OUTPUTPATH = Paths.get("src", "main", "resources", "stitch");

    public String stitchImages(boolean warp) throws IOException {
        String filename = CylinderStitcherEnhanced.run(warp, INPUTPATH, OUTPUTPATH);
        return "http://localhost:8080/stitch/" + filename;
    }
}