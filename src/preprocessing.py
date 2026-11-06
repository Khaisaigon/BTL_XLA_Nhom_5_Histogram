
import cv2
import numpy as np


def load_image(image_path):
    """
    Đọc ảnh đầu vào bất kỳ.

    IMREAD_UNCHANGED:
        Giữ nguyên độ sâu bit và số kênh ảnh.
    """

    image = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise FileNotFoundError(
            f"Cannot load image: {image_path}"
        )

    return image


def convert_to_grayscale(image):
    """
    Chuyển ảnh màu thành ảnh Grayscale.

    Hỗ trợ:
        - Grayscale
        - BGR
        - BGRA
    """

    if image.ndim == 2:
        gray = image

    elif image.ndim == 3:

        channels = image.shape[2]

        if channels == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        elif channels == 4:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY
            )

        else:
            raise ValueError(
                "Unsupported image channels."
            )

    else:
        raise ValueError(
            "Unsupported image dimensions."
        )

    return gray


def convert_to_8bit(image):
    """
    Chuyển ảnh Grayscale về 8-bit.

    Nếu ảnh đã là uint8:
        Giữ nguyên dữ liệu.

    Nếu ảnh là uint16 hoặc float:
        Chuẩn hóa min-max về 0-255.
    """

    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float64)

    min_value = np.min(image)
    max_value = np.max(image)

    if max_value == min_value:
        return np.zeros(
            image.shape,
            dtype=np.uint8
        )

    normalized = (
        (image - min_value) /
        (max_value - min_value)
    ) * 255

    result = np.clip(
        np.rint(normalized),
        0,
        255
    ).astype(np.uint8)

    return result


def preprocess_image(image_path):
    """
    Hàm tiền xử lý tổng hợp.

    Quy trình:
        1. Đọc ảnh.
        2. Chuyển Grayscale.
        3. Chuyển về 8-bit.

    Không thay đổi kích thước ảnh.

    Returns:
        original: Ảnh ban đầu.
        processed: Ảnh Grayscale 8-bit.
    """

    original = load_image(image_path)

    gray = convert_to_grayscale(original)

    processed = convert_to_8bit(gray)

    return original, processed
