
import numpy as np


def calculate_histogram(image):
    """
    Tính Histogram thủ công cho ảnh xám 8-bit.

    Returns:
        histogram: Mảng 256 phần tử.
    """

    histogram = np.zeros(256, dtype=np.int64)

    height, width = image.shape

    # Duyệt từng pixel trong ảnh
    for y in range(height):
        for x in range(width):

            gray_value = image[y, x]

            histogram[gray_value] += 1

    return histogram


def normalize_histogram(histogram):
    """
    Chuẩn hóa Histogram thành PDF.

    PDF[k] = histogram[k] / tổng số pixel
    """

    total_pixels = np.sum(histogram)

    pdf = histogram.astype(np.float64) / total_pixels

    return pdf


def calculate_cdf(pdf):
    """
    Tính CDF bằng phép cộng dồn PDF.

    Returns:
        cdf: Mảng 256 phần tử.
    """

    cdf = np.zeros(256, dtype=np.float64)

    cumulative = 0.0

    for i in range(256):

        cumulative += pdf[i]

        cdf[i] = cumulative

    return cdf
