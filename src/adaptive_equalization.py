
import numpy as np
import cv2


def manual_ahe(image, window_size=31):
    """
    Adaptive Histogram Equalization tự xây dựng.

    Parameters:
        image: Ảnh Grayscale uint8.
        window_size: Kích thước cửa sổ cục bộ.
                     Phải là số lẻ và >= 3.

    Returns:
        result: Ảnh sau Adaptive Histogram Equalization.
    """

    # ==========================================
    # 1. VALIDATE INPUT
    # ==========================================

    if image.ndim != 2:
        raise ValueError("Input must be a grayscale image.")

    if image.dtype != np.uint8:
        raise ValueError("Input image must have dtype uint8.")

    if not isinstance(window_size, (int, np.integer)):
        raise ValueError("Window size must be an integer.")

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("Window size must be odd and >= 3.")

    height, width = image.shape

    # ==========================================
    # 2. PADDING
    # ==========================================

    padding = window_size // 2

    padded_image = np.pad(
        image,
        pad_width=padding,
        mode="reflect"
    )

    # ==========================================
    # 3. INITIALIZE RESULT
    # ==========================================

    result = np.zeros_like(image)

    total_pixels = window_size * window_size

    # ==========================================
    # 4. SLIDING WINDOW
    # ==========================================

    for y in range(height):

        for x in range(width):

            # Lấy vùng lân cận của pixel hiện tại
            window = padded_image[
                y:y + window_size,
                x:x + window_size
            ]

            # Giá trị grayscale của pixel trung tâm
            center_value = int(image[y, x])

            # ==================================
            # 5. CALCULATE LOCAL HISTOGRAM
            # ==================================

            histogram = np.bincount(
                window.ravel(),
                minlength=256
            )

            # ==================================
            # 6. CALCULATE LOCAL CDF
            # ==================================

            cdf = np.cumsum(histogram)

            # Tìm mức xám đầu tiên xuất hiện
            non_zero = np.flatnonzero(histogram)

            cdf_min = cdf[non_zero[0]]

            # ==================================
            # 7. CREATE LOCAL MAPPING
            # ==================================

            denominator = total_pixels - cdf_min

            if denominator == 0:
                # Trường hợp cửa sổ chỉ có một mức xám
                new_value = center_value

            else:
                mapped_value = (
                    (cdf[center_value] - cdf_min)
                    / denominator
                ) * 255

                new_value = int(
                    np.clip(np.rint(mapped_value), 0, 255)
                )

            # ==================================
            # 8. UPDATE PIXEL
            # ==================================

            result[y, x] = new_value

    return result


def opencv_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Adaptive Histogram Equalization có giới hạn
    tương phản bằng OpenCV (CLAHE).
    """

    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Input must be a uint8 grayscale image.")

    if clip_limit <= 0:
        raise ValueError("clip_limit must be greater than 0.")

    if (
        len(tile_grid_size) != 2
        or any(not isinstance(v, (int, np.integer)) or v < 1
               for v in tile_grid_size)
    ):
        raise ValueError("tile_grid_size must contain two positive integers.")

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    result = clahe.apply(image)

    return result
