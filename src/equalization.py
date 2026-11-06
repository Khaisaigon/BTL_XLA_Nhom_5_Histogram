
import numpy as np
import cv2

from .histogram import (
    calculate_histogram,
    normalize_histogram,
    calculate_cdf
)


def create_equalization_lut(cdf):
    """
    Tạo bảng ánh xạ LUT gồm 256 mức xám.

    Sử dụng CDF và CDF minimum.

    Returns:
        lut: Bảng ánh xạ uint8.
    """

    lut = np.zeros(256, dtype=np.uint8)

    # Tìm CDF nhỏ nhất khác 0
    non_zero = np.flatnonzero(cdf)

    if len(non_zero) == 0:
        return lut

    cdf_min = cdf[non_zero[0]]

    # Trường hợp ảnh chỉ có một mức xám
    if np.isclose(cdf_min, 1.0):
        return np.arange(256, dtype=np.uint8)

    # Chuẩn hóa CDF và ánh xạ về 0-255
    mapped_values = (
        (cdf - cdf_min) / (1.0 - cdf_min)
    ) * 255.0

    # Làm tròn về số nguyên gần nhất
    lut = np.clip(
        np.rint(mapped_values), 0, 255
    ).astype(np.uint8)

    return lut


def manual_equalization(image):
    """
    Cân bằng Histogram bằng phương pháp thủ công.

    Các bước:
        1. Histogram
        2. PDF
        3. CDF
        4. LUT
        5. Ánh xạ pixel
    """

    histogram = calculate_histogram(image)

    pdf = normalize_histogram(histogram)

    cdf = calculate_cdf(pdf)

    lut = create_equalization_lut(cdf)

    # Áp dụng LUT lên ảnh gốc
    result = lut[image]

    return result, histogram, pdf, cdf, lut


def opencv_equalization(image):
    """
    Cân bằng Histogram sử dụng OpenCV.
    """

    result = cv2.equalizeHist(image)

    return result
