import cv2
import numpy as np
import time


def extract_sift_features(image_path_left, image_path_right):
    """
    Thực hiện Giai đoạn 1: Trích xuất Keypoints và Descriptors bằng SIFT.
    """
    # --- 1. Tải ảnh ---
    img1 = cv2.imread(image_path_left, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image_path_right, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print("Lỗi: Không thể tải ảnh. Vui lòng kiểm tra đường dẫn.")
        return None, None, None, None, None, None

    # --- 2. Khởi tạo SIFT Detector ---
    try:
        detector = cv2.SIFT_create()
    except AttributeError:
        # Nếu SIFT không khả dụng sử dụng ORB
        print("Cảnh báo: SIFT không khả dụng. Sử dụng ORB thay thế.")
        detector = cv2.ORB_create(nfeatures=5000)

    start_time = time.time()
    print(f"Bắt đầu trích xuất đặc trưng với thuật toán: {detector.__class__.__name__}...")

    # --- 3. Thực hiện Trích xuất (Detect & Compute) ---
    # detectAndCompute thực hiện toàn bộ 4 bước trích xuất

    # Ảnh 1
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)

    # Ảnh 2
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    end_time = time.time()

    # --- 4. Tổng kết và Chuẩn bị đầu ra ---
    print(f"Hoàn thành Giai đoạn 1 sau {end_time - start_time:.4f} giây.")
    print(f"Ảnh 1: Tìm thấy {len(keypoints1)} Keypoints.")
    print(f"Ảnh 2: Tìm thấy {len(keypoints2)} Keypoints.")
    print(
        f"Kích thước Descriptor: {descriptors1.shape if descriptors1 is not None else 'N/A'} (sẵn sàng cho Giai đoạn 2).")

    return img1, img2, keypoints1, descriptors1, keypoints2, descriptors2


# ----------------------------------------------------
# KHỞI CHẠY VÍ DỤ VÀ TRỰC QUAN HÓA
# ----------------------------------------------------

# Thay thế bằng đường dẫn ảnh thực tế
img_left_path = "image_left.jpg"
img_right_path = "image_right.jpg"

# Chạy Giai đoạn 1
img1, img2, kp1, des1, kp2, des2 = extract_sift_features(img_left_path, img_right_path)

if des1 is not None:
    # Trực quan hóa kết quả để kiểm tra chất lượng
    print("\nTrực quan hóa Keypoints...")

    # Vẽ Keypoints lên Ảnh 1
    img_kp1 = cv2.drawKeypoints(img1, kp1, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                color=(0, 255, 0))  # Màu xanh lá

    # Vẽ Keypoints lên Ảnh 2
    img_kp2 = cv2.drawKeypoints(img2, kp2, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                color=(0, 255, 0))

    cv2.imshow("Keypoints tren Anh 1", img_kp1)
    cv2.imshow("Keypoints tren Anh 2", img_kp2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()