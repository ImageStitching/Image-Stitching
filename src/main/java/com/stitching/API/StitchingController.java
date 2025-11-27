package com.stitching.API;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*") // Cho phép gọi từ file HTML bất kỳ
public class StitchingController {

    @Autowired
    private StitchingService stitchingService;

    @Autowired
    private ImageStorageService imageStorageService;

    @PostMapping(value = "/stitch", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> stitchImages(
            @RequestParam("images") List<MultipartFile> images,
            @RequestParam(value = "mode", defaultValue = "0") int mode) {
        try {
            if (images == null || images.isEmpty()) {
                Map<String, String> error = new HashMap<>();
                error.put("error", "Vui lòng chọn ảnh.");
                return ResponseEntity.badRequest().body(error);
            }

            // Kiểm tra mode hợp lệ (chỉ nhận 0 hoặc 1)
            if (mode != 0 && mode != 1) {
                Map<String, String> error = new HashMap<>();
                error.put("error", "Tham số mode chỉ nhận giá trị 0 hoặc 1.");
                return ResponseEntity.badRequest().body(error);
            }

            // Kiểm tra tất cả file có phải là ảnh hợp lệ
            for (MultipartFile image : images) {
                if (!imageStorageService.isValidImageFile(image)) {
                    Map<String, String> error = new HashMap<>();
                    error.put("error", "File không hợp lệ: " + image.getOriginalFilename());
                    return ResponseEntity.badRequest().body(error);
                }
            }

            // Lưu ảnh vào uploads/ (tự động xóa ảnh cũ)
            imageStorageService.storeMultiple(images);

            // Gọi service xử lý - trả về URL ảnh đã lưu
            // mode: 0 = algorithm A, 1 = algorithm B (hoặc tùy logic của bạn)
            boolean warp;
            if(mode == 1) warp=true;
            else warp = false;
            String imageUrl = stitchingService.stitchImages(warp);

            // Trả về URL để frontend hiển thị
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("imageUrl", imageUrl);
            response.put("mode", mode);
            response.put("message", "Ghép ảnh thành công với mode " + mode + "!");

            return ResponseEntity.ok().body(response);

        } catch (Exception e) {
            e.printStackTrace();
            Map<String, String> error = new HashMap<>();
            error.put("error", "Lỗi: " + e.getMessage());
            return ResponseEntity.internalServerError().body(error);
        }
    }
}