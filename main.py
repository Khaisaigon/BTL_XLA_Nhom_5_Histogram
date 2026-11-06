
import os
import cv2
import argparse

# Import các module
from src.preprocessing import preprocess_image

from src.equalization import (
    manual_equalization,
    opencv_equalization
)

from src.metrics import (
    calculate_metrics,
    calculate_exact_match
)

from src.visualization import display_results


def main():

    # 1. Khởi tạo bộ phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(
        description="Histogram Equalization - Manual vs OpenCV"
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image"
    )

    # 2. Đọc tham số từ terminal
    args = parser.parse_args()

    image_path = args.image_path

    # 3. Kiểm tra đường dẫn ảnh
    if not os.path.isfile(image_path):
        parser.error(f"Image not found: {image_path}")

    # 4. Tạo thư mục lưu kết quả
    os.makedirs("outputs", exist_ok=True)

    # 5. Đọc và tiền xử lý ảnh
    original, image = preprocess_image(image_path)

    print("===== IMAGE INFORMATION =====")
    print("Input path:", image_path)
    print("Original shape:", original.shape)
    print("Processed shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image min:", image.min())
    print("Image max:", image.max())

    # 6. Manual Histogram Equalization
    manual, hist, pdf, cdf, lut = manual_equalization(
        image
    )

    # 7. OpenCV Histogram Equalization
    opencv = opencv_equalization(image)

    # 8. Đánh giá sai khác
    metrics = calculate_metrics(
        manual, opencv
    )

    exact = calculate_exact_match(
        manual, opencv
    )

    # 9. In kết quả đánh giá
    print("\n===== METRICS COMPARISON =====")

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print("\n===== EXACT MATCH =====")

    for name, value in exact.items():
        print(f"{name}: {value}")

    # 10. Lưu ảnh kết quả
    cv2.imwrite(
        "outputs/manual_result.png", manual
    )

    cv2.imwrite(
        "outputs/opencv_result.png", opencv
    )

    # 11. Hiển thị ảnh và biểu đồ
    display_results(
        image,
        manual,
        opencv,
        "outputs/comparison.png"
    )

    print("\n===== COMPLETED =====")


if __name__ == "__main__":
    main()
