package com.stitching.API;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

@Service
public class ImageStorageService {

	// Thư mục lưu ảnh input: uploads/
	private final Path uploadsPath = Paths.get("src", "main", "resources", "uploads");

	/**
	 * Xóa tất cả ảnh cũ trong thư mục uploads
	 */
	public void clearUploadsDirectory() throws IOException {
		if (!Files.exists(uploadsPath)) {
			return; // Nếu thư mục chưa tồn tại thì không cần xóa
		}

		try (Stream<Path> files = Files.list(uploadsPath)) {
			files.filter(Files::isRegularFile)
					.forEach(file -> {
						try {
							Files.delete(file);
							System.out.println("Đã xóa file: " + file.getFileName());
						} catch (IOException e) {
							System.err.println("Không thể xóa file: " + file.getFileName());
						}
					});
		}
		System.out.println("=== ĐÃ XÓA TOÀN BỘ ẢNH CŨ TRONG UPLOADS ===");
	}

	/**
	 * Lưu danh sách file ảnh được upload từ người dùng
	 * Trả về danh sách đường dẫn file đã lưu
	 */
	public List<Path> storeMultiple(List<MultipartFile> files) throws IOException {
		clearUploadsDirectory();
		createDirectoryIfNotExists(uploadsPath);

		List<Path> savedPaths = new ArrayList<>();

		for (MultipartFile file : files) {
			String filename = file.getOriginalFilename();
			if (filename == null || filename.isEmpty()) {
				filename = "image_" + System.currentTimeMillis() + ".jpg";
			}

			Path targetPath = uploadsPath.resolve(filename);
			Files.copy(file.getInputStream(), targetPath, StandardCopyOption.REPLACE_EXISTING);
			savedPaths.add(targetPath);
		}

		return savedPaths;
	}

	/**
	 * Kiểm tra xem file upload có phải ảnh không
	 */
	public boolean isValidImageFile(MultipartFile file) {
		if (file == null) return false;

		String contentType = file.getContentType();
		if (contentType == null || !contentType.startsWith("image/")) {
			return false;
		}

		String originalFilename = file.getOriginalFilename();
		return originalFilename != null &&
				originalFilename.matches("(?i).+\\.(jpg|jpeg|png|gif|bmp|webp)$");
	}

	/**
	 * Tạo thư mục nếu chưa tồn tại
	 */
	private void createDirectoryIfNotExists(Path path) throws IOException {
		if (!Files.exists(path)) {
			Files.createDirectories(path);
		}
	}

	/**
	 * Lấy đường dẫn thư mục uploads
	 */
	public Path getUploadsPath() {
		return uploadsPath;
	}
}