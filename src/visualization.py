
import matplotlib.pyplot as plt
import numpy as np

from .histogram import (
    calculate_histogram,
    normalize_histogram,
    calculate_cdf
)


def display_results(
    original,
    manual,
    opencv,
    save_path=None,
    show=True,
    titles=None,
    additional_image=None
):
    """
    Hiển thị ảnh Grayscale, Histogram và CDF.

    Bố cục:
        - Cột trái: Ảnh Grayscale.
        - Cột phải: Normalized Histogram và CDF.
        - Mỗi hàng tương ứng với một ảnh.

    Function 01:
        Original, Manual Equalization, OpenCV Equalization.

    Function 02:
        Original, Manual AHE, OpenCV CLAHE.

    Function 03:
        Source, Target, Manual Matching,
        Scikit-image Matching.

    Tham số:
        original: Ảnh đầu vào thứ nhất.
        manual: Ảnh đầu vào thứ hai.
        opencv: Ảnh đầu vào thứ ba.
        save_path: Đường dẫn lưu biểu đồ (nếu có).
        show: Có hiển thị bằng plt.show() hay không.
        titles: Danh sách tiêu đề tùy chỉnh.
        additional_image: Ảnh thứ tư (nếu có).

    Kết quả:
        Trả về đối tượng Figure của Matplotlib.
    """

    # ==========================================
    # 1. THIẾT LẬP DANH SÁCH ẢNH
    # ==========================================

    images = [original, manual, opencv]

    if additional_image is not None:
        images.append(additional_image)

    # Tiêu đề mặc định cho Function 01
    if titles is None:

        titles = [
            "Original Grayscale",
            "Manual Equalization",
            "OpenCV Equalization"
        ]

        if additional_image is not None:
            titles.append("Additional Image")

    # Kiểm tra số lượng tiêu đề và số lượng ảnh
    if len(titles) != len(images):
        raise ValueError(
            "Số lượng tiêu đề phải bằng số lượng ảnh."
        )

    # ==========================================
    # 2. TẠO BỐ CỤC BIỂU ĐỒ
    # ==========================================

    number_of_images = len(images)

    fig, axes = plt.subplots(
        number_of_images,
        2,
        figsize=(14, number_of_images * 4),
        gridspec_kw={
            "width_ratios": [1, 1]
        },
        squeeze=False
    )

    intensity = np.arange(256)

    # ==========================================
    # 3. DUYỆT QUA TỪNG ẢNH
    # ==========================================

    for i, (title, image) in enumerate(zip(titles, images)):

        image_ax = axes[i, 0]
        graph_ax = axes[i, 1]

        # ======================================
        # 3.1. HIỂN THỊ ẢNH GRAYSCALE
        # ======================================

        image_ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        image_ax.set_title(title)
        image_ax.axis("off")

        # ======================================
        # 3.2. TÍNH HISTOGRAM
        # ======================================

        histogram = calculate_histogram(image)

        # ======================================
        # 3.3. CHUẨN HÓA HISTOGRAM
        # ======================================

        normalized_hist = normalize_histogram(
            histogram
        )

        # ======================================
        # 3.4. TÍNH CDF
        # ======================================

        cdf = calculate_cdf(normalized_hist)

        # ======================================
        # 3.5. VẼ HISTOGRAM VÀ CDF
        # ======================================

        graph_ax.plot(
            intensity,
            normalized_hist,
            "r",
            label="histogram"
        )

        graph_ax.plot(
            intensity,
            cdf,
            "b--",
            label="cdf"
        )

        graph_ax.set_title(
            f"{title} - Histogram and CDF"
        )

        graph_ax.set_xlim(0, 255)
        graph_ax.set_ylim(0, 1.1)

        graph_ax.set_yticks(
            np.arange(0, 1.1, 0.1)
        )

        graph_ax.set_xlabel("Intensity Level")
        graph_ax.set_ylabel("Probability")

        graph_ax.legend(loc="upper left")
        graph_ax.grid(alpha=0.2)

    # ==========================================
    # 4. HOÀN THIỆN BIỂU ĐỒ
    # ==========================================

    plt.tight_layout()

    # Lưu biểu đồ nếu có đường dẫn
    if save_path:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    # Hiển thị nếu được yêu cầu
    if show:
        plt.show()

    return fig


def display_cdf_comparison(
    source,
    target,
    manual,
    library
):
    """
    So sánh CDF giữa bốn ảnh của Function 03.

    Mục đích:
        Kiểm tra mức độ tương đồng về phân bố
        mức xám giữa ảnh Target và ảnh Matching.
    """

    # ==========================================
    # 1. DANH SÁCH ẢNH
    # ==========================================

    images = [
        ("Source Image", source),
        ("Target Image", target),
        ("Manual Matching", manual),
        ("Scikit-image Matching", library)
    ]

    # ==========================================
    # 2. TẠO BIỂU ĐỒ
    # ==========================================

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    intensity = np.arange(256)

    # ==========================================
    # 3. TÍNH VÀ VẼ CDF
    # ==========================================

    for title, image in images:

        histogram = calculate_histogram(image)

        normalized_hist = normalize_histogram(
            histogram
        )

        cdf = calculate_cdf(normalized_hist)

        ax.plot(
            intensity,
            cdf,
            linewidth=1.8,
            label=title
        )

    # ==========================================
    # 4. THIẾT LẬP BIỂU ĐỒ
    # ==========================================

    ax.set_title(
        "CDF Comparison - Histogram Matching"
    )

    ax.set_xlabel("Intensity Level")
    ax.set_ylabel("Cumulative Probability")

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1.1)

    ax.set_yticks(
        np.arange(0, 1.1, 0.1)
    )

    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    return fig
