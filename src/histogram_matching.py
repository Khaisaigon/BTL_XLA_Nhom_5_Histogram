
import numpy as np

from skimage.exposure import match_histograms


def validate_images(source, target):
    """
    Kiểm tra hai ảnh đầu vào.

    Yêu cầu:
        - Ảnh Grayscale 2 chiều.
        - Kiểu dữ liệu uint8.
        - Ảnh không được rỗng.
    """

    for name, image in [
        ("Source", source),
        ("Target", target)
    ]:

        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"{name} phải là mảng NumPy."
            )

        if image.ndim != 2:
            raise ValueError(
                f"{name} phải là ảnh Grayscale."
            )

        if image.dtype != np.uint8:
            raise ValueError(
                f"{name} phải có kiểu dữ liệu uint8."
            )

        if image.size == 0:
            raise ValueError(
                f"{name} không được rỗng."
            )


def manual_histogram_matching(
    source,
    target,
    return_lut=False
):
    """
    Thuật toán Histogram Matching tự xây dựng.

    Các bước:
        1. Tìm các mức xám duy nhất.
        2. Đếm tần suất xuất hiện.
        3. Tính CDF của Source và Target.
        4. Nội suy tuyến tính giữa các quantile.
        5. Xây dựng LUT ánh xạ mức xám.
        6. Biến đổi ảnh Source.

    Trả về:
        - Ảnh sau Matching ở dạng float64.
        - LUT nếu return_lut=True.
    """

    validate_images(source, target)

    # ==========================================
    # 1. TÌM MỨC XÁM VÀ TẦN SUẤT
    # ==========================================

    source_values, source_counts = np.unique(
        source,
        return_counts=True
    )

    target_values, target_counts = np.unique(
        target,
        return_counts=True
    )

    # ==========================================
    # 2. TÍNH CDF CHO SOURCE
    # ==========================================

    source_cdf = np.cumsum(
        source_counts,
        dtype=np.int64
    ) / source.size

    # ==========================================
    # 3. TÍNH CDF CHO TARGET
    # ==========================================

    target_cdf = np.cumsum(
        target_counts,
        dtype=np.int64
    ) / target.size

    # ==========================================
    # 4. NỘI SUY TUYẾN TÍNH
    # ==========================================

    # Với mỗi quantile của Source,
    # tìm mức xám tương ứng trên CDF của Target.

    matched_values = np.interp(
        source_cdf,
        target_cdf,
        target_values
    )

    # ==========================================
    # 5. XÂY DỰNG BẢNG ÁNH XẠ LUT
    # ==========================================

    # Khởi tạo 256 mức xám.
    # Những mức xám không xuất hiện trong Source
    # không ảnh hưởng đến ảnh kết quả.

    lut = np.arange(
        256,
        dtype=np.float64
    )

    lut[source_values] = matched_values

    # ==========================================
    # 6. BIẾN ĐỔI ẢNH SOURCE
    # ==========================================

    matched_image = lut[source]

    if return_lut:
        return matched_image, lut

    return matched_image


def skimage_histogram_matching(source, target):
    """
    Histogram Matching sử dụng scikit-image.

    Phương pháp này được dùng làm đối chứng
    với thuật toán Manual.
    """

    validate_images(source, target)

    result = match_histograms(
        source,
        target,
        channel_axis=None
    )

    return result


def convert_matching_to_8bit(image):
    """
    Chuyển ảnh sau nội suy thành Grayscale uint8.

    Các giá trị được:
        - Làm tròn.
        - Giới hạn trong khoảng [0, 255].
        - Chuyển về uint8.
    """

    result = np.clip(
        np.rint(image),
        0,
        255
    ).astype(np.uint8)

    return result


def calculate_histogram_distance(image, target):
    """
    Đánh giá sự khác biệt phân bố mức xám
    giữa ảnh cần kiểm tra và ảnh Target.

    Các chỉ số:
        1. Histogram L1 Distance.
        2. Wasserstein-1 Distance.

    Giá trị càng nhỏ thể hiện phân bố
    mức xám càng gần ảnh Target.
    """

    validate_images(image, target)

    # ==========================================
    # 1. TÍNH HISTOGRAM CHUẨN HÓA
    # ==========================================

    hist_image = np.bincount(
        image.ravel(),
        minlength=256
    ).astype(np.float64)

    hist_target = np.bincount(
        target.ravel(),
        minlength=256
    ).astype(np.float64)

    hist_image /= image.size
    hist_target /= target.size

    # ==========================================
    # 2. HISTOGRAM L1 DISTANCE
    # ==========================================

    l1_distance = np.sum(
        np.abs(hist_image - hist_target)
    )

    # ==========================================
    # 3. TÍNH HAI CDF
    # ==========================================

    cdf_image = np.cumsum(hist_image)
    cdf_target = np.cumsum(hist_target)

    # ==========================================
    # 4. WASSERSTEIN-1 DISTANCE
    # ==========================================

    wasserstein_distance = np.sum(
        np.abs(
            cdf_image[:-1] - cdf_target[:-1]
        )
    )

    return {
        "Histogram L1": float(l1_distance),
        "Wasserstein-1": float(wasserstein_distance)
    }
