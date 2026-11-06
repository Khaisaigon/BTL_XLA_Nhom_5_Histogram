
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.preprocessing import (
    convert_to_grayscale,
    convert_to_8bit
)

from src.histogram_matching import (
    manual_histogram_matching,
    skimage_histogram_matching,
    convert_matching_to_8bit,
    calculate_histogram_distance
)

from src.metrics import (
    calculate_metrics,
    calculate_exact_match
)

from src.visualization import (
    display_results,
    display_cdf_comparison
)


def read_uploaded_image(uploaded_file):
    """
    Đọc ảnh được tải lên từ giao diện Streamlit.

    Quy trình:
        1. Đọc dữ liệu ảnh.
        2. Giải mã bằng OpenCV.
        3. Chuyển sang Grayscale.
        4. Chuẩn hóa về ảnh 8-bit.
    """

    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    original = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_UNCHANGED
    )

    if original is None:
        raise ValueError(
            "Không thể đọc ảnh được tải lên."
        )

    gray = convert_to_grayscale(original)

    processed = convert_to_8bit(gray)

    return processed


@st.cache_data(show_spinner=False)
def run_histogram_matching(source, target):
    """
    Thực hiện hai phương pháp Histogram Matching.

    Phương pháp 1:
        Manual Histogram Matching bằng NumPy.

    Phương pháp 2:
        Histogram Matching bằng scikit-image.

    Trả về ảnh kết quả và bảng LUT.
    """

    # ==========================================
    # 1. MANUAL HISTOGRAM MATCHING
    # ==========================================

    manual_float, lut = manual_histogram_matching(
        source,
        target,
        return_lut=True
    )

    # ==========================================
    # 2. SCIKIT-IMAGE HISTOGRAM MATCHING
    # ==========================================

    library_float = skimage_histogram_matching(
        source,
        target
    )

    # ==========================================
    # 3. CHUYỂN ẢNH VỀ UINT8
    # ==========================================

    manual = convert_matching_to_8bit(
        manual_float
    )

    library = convert_matching_to_8bit(
        library_float
    )

    return (
        manual,
        library,
        lut,
        manual_float,
        library_float
    )


def encode_image_png(image):
    """
    Mã hóa ảnh thành PNG để người dùng tải về.
    """

    success, encoded = cv2.imencode(
        ".png",
        image
    )

    if not success:
        raise RuntimeError(
            "Không thể mã hóa ảnh PNG."
        )

    return encoded.tobytes()


def render():
    """
    Giao diện chính của Function 03.
    """

    st.header("Function 03 - Histogram Matching")

    st.write(
        "Biến đổi phân bố mức xám của ảnh nguồn "
        "để gần với phân bố mức xám của ảnh đích."
    )

    # ==========================================
    # 1. UPLOAD ẢNH
    # ==========================================

    st.subheader("1. Upload Images")

    col_source, col_target = st.columns(2)

    with col_source:

        uploaded_source = st.file_uploader(
            "Source Image",
            type=[
                "png", "jpg", "jpeg",
                "bmp", "tif", "tiff"
            ],
            key="matching_source"
        )

    with col_target:

        uploaded_target = st.file_uploader(
            "Target Image",
            type=[
                "png", "jpg", "jpeg",
                "bmp", "tif", "tiff"
            ],
            key="matching_target"
        )

    # Chỉ xử lý khi đã tải đủ hai ảnh
    if uploaded_source is None or uploaded_target is None:

        st.info(
            "Vui lòng tải lên cả Source Image và Target Image."
        )

        return

    try:

        source = read_uploaded_image(
            uploaded_source
        )

        target = read_uploaded_image(
            uploaded_target
        )

    except Exception as error:

        st.error(
            f"Lỗi đọc ảnh: {error}"
        )

        return

    # ==========================================
    # 2. THÔNG TIN ẢNH
    # ==========================================

    st.subheader("2. Image Information")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            source,
            caption="Source Image",
            clamp=True,
            use_container_width=True
        )

        st.write(
            f"Resolution: {source.shape[1]} x {source.shape[0]}"
        )

        st.write(
            f"Data Type: {source.dtype}"
        )

    with col2:

        st.image(
            target,
            caption="Target Image",
            clamp=True,
            use_container_width=True
        )

        st.write(
            f"Resolution: {target.shape[1]} x {target.shape[0]}"
        )

        st.write(
            f"Data Type: {target.dtype}"
        )

    st.caption(
        "Hai ảnh đều được chuyển về Grayscale 8-bit. "
        "Source và Target không cần có cùng kích thước."
    )

    # ==========================================
    # 3. THUẬT TOÁN
    # ==========================================

    st.subheader("3. Algorithm")

    st.markdown(
        """
        **Manual Histogram Matching**

        - Tính histogram của Source và Target.
        - Tính phân bố tích lũy CDF.
        - Xác định các quantile tương ứng.
        - Nội suy tuyến tính để tìm mức xám mới.
        - Áp dụng bảng ánh xạ LUT lên Source.

        **Phương pháp đối chứng**

        Sử dụng `skimage.exposure.match_histograms()`.
        """
    )

    try:

        (
            manual,
            library,
            lut,
            manual_float,
            library_float
        ) = run_histogram_matching(
            source,
            target
        )

    except Exception as error:

        st.error(
            f"Lỗi Histogram Matching: {error}"
        )

        return

    # ==========================================
    # 4. HIỂN THỊ ẢNH VÀ BIỂU ĐỒ
    # ==========================================

    st.subheader("4. Image and Graph Visualization")

    # Sử dụng chung visualization.py
    # với Function 01 và Function 02.

    fig = display_results(
        original=source,
        manual=target,
        opencv=manual,
        additional_image=library,
        save_path=None,
        show=False,
        titles=[
            "Source Image",
            "Target Image",
            "Manual Histogram Matching",
            "Scikit-image Histogram Matching"
        ]
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # ==========================================
    # 4.1. SO SÁNH CDF
    # ==========================================

    st.markdown("#### CDF Comparison")

    fig_cdf = display_cdf_comparison(
        source=source,
        target=target,
        manual=manual,
        library=library
    )

    st.pyplot(
        fig_cdf,
        use_container_width=True
    )

    plt.close(fig_cdf)

    # ==========================================
    # 5. ĐÁNH GIÁ CHẤT LƯỢNG
    # ==========================================

    st.subheader("5. Image Quality Analysis")

    # ==========================================
    # 5.1. SO SÁNH MANUAL VỚI THƯ VIỆN
    # ==========================================

    st.markdown(
        "#### A. Manual vs Scikit-image"
    )

    try:

        metrics = calculate_metrics(
            manual,
            library
        )

        exact_metrics = calculate_exact_match(
            manual,
            library
        )

        metrics.update(exact_metrics)

        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })

        st.dataframe(
            metrics_df,
            hide_index=True,
            use_container_width=True
        )

    except Exception as error:

        st.warning(
            f"Không thể tính các chỉ số: {error}"
        )

    # Kiểm tra sai khác trước khi chuyển uint8
    max_float_error = np.max(
        np.abs(manual_float - library_float)
    )

    st.write(
        "Maximum Absolute Error (float64):",
        float(max_float_error)
    )

    # ==========================================
    # 5.2. ĐÁNH GIÁ HISTOGRAM MATCHING
    # ==========================================

    st.markdown(
        "#### B. Histogram Distance vs Target"
    )

    comparison_images = [
        ("Source Image", source),
        ("Manual Matching", manual),
        ("Scikit-image Matching", library)
    ]

    distance_results = []

    for name, image in comparison_images:

        distances = calculate_histogram_distance(
            image,
            target
        )

        distance_results.append({
            "Image": name,
            **distances
        })

    distance_df = pd.DataFrame(
        distance_results
    )

    st.dataframe(
        distance_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Histogram L1 và Wasserstein-1 càng nhỏ "
        "thì phân bố mức xám của ảnh càng gần Target."
    )

    # ==========================================
    # 6. TẢI KẾT QUẢ
    # ==========================================

    st.subheader("6. Download Results")

    col_download1, col_download2 = st.columns(2)

    with col_download1:

        st.download_button(
            label="Download Manual Matching",
            data=encode_image_png(manual),
            file_name="manual_histogram_matching.png",
            mime="image/png",
            key="download_manual_matching"
        )

    with col_download2:

        st.download_button(
            label="Download Scikit-image Matching",
            data=encode_image_png(library),
            file_name="skimage_histogram_matching.png",
            mime="image/png",
            key="download_library_matching"
        )

    # ==========================================
    # 7. HIỂN THỊ BẢNG LUT
    # ==========================================

    with st.expander(
        "View Lookup Table (LUT)"
    ):

        lut_df = pd.DataFrame({
            "Source Gray Level": np.arange(256),
            "Mapped Gray Level": lut
        })

        st.dataframe(
            lut_df,
            hide_index=True,
            use_container_width=True
        )
