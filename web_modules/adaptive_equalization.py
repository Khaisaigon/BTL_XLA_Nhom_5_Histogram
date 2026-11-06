
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import (
    convert_to_grayscale,
    convert_to_8bit
)

from src.adaptive_equalization import (
    manual_ahe,
    opencv_clahe
)

from src.metrics import calculate_metrics

from src.visualization import display_results


# ==================================================
# 1. BỘ NHỚ ĐỆM XỬ LÝ
# ==================================================

@st.cache_data(show_spinner=False)
def run_manual_ahe(image, window_size):
    """
    Chạy thuật toán Manual AHE.

    Lưu kết quả vào cache để tránh tính toán lại
    khi ảnh và tham số đầu vào không thay đổi.
    """
    return manual_ahe(image, window_size)


@st.cache_data(show_spinner=False)
def run_opencv_clahe(image, clip_limit, tile_grid_size):
    """
    Chạy thuật toán OpenCV CLAHE.

    Lưu kết quả theo ảnh và các tham số đầu vào.
    """
    return opencv_clahe(
        image,
        clip_limit,
        tile_grid_size
    )


# ==================================================
# 2. GIAO DIỆN CHÍNH
# ==================================================

def render():

    st.title("Adaptive Histogram Equalization")

    st.write(
        "Xây dựng và so sánh thuật toán Adaptive Histogram "
        "Equalization (AHE) tự triển khai với Contrast Limited "
        "Adaptive Histogram Equalization (CLAHE) của OpenCV."
    )

    # ==================================================
    # 1. UPLOAD IMAGE
    # ==================================================

    st.subheader("1. Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg", "jpeg", "png",
            "bmp", "tif", "tiff"
        ],
        key="ahe_upload"
    )

    if uploaded_file is None:
        st.info("Vui lòng tải ảnh lên để bắt đầu xử lý.")
        return

    # ==================================================
    # ĐỌC ẢNH
    # ==================================================

    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    original = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_UNCHANGED
    )

    if original is None:
        st.error(
            "Không thể đọc ảnh. Vui lòng kiểm tra file ảnh."
        )
        return

    # ==================================================
    # TIỀN XỬ LÝ ẢNH
    # ==================================================

    try:

        # Chuyển ảnh về Grayscale
        gray = convert_to_grayscale(original)

        # Chuyển về ảnh 8-bit
        image = convert_to_8bit(gray)

    except (ValueError, TypeError) as error:

        st.error(f"Preprocessing error: {error}")
        return

    height, width = image.shape

    # ==================================================
    # 2. IMAGE INFORMATION
    # ==================================================

    st.subheader("2. Image Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Width", f"{width} px")
        st.metric("Height", f"{height} px")

    with col2:
        st.metric("Bit Depth", "8-bit")
        st.metric("Data Type", str(image.dtype))

    with col3:
        st.metric("Min Intensity", int(image.min()))
        st.metric("Max Intensity", int(image.max()))

    st.write("**File name:**", uploaded_file.name)
    st.write("**Original shape:**", original.shape)

    file_size = len(uploaded_file.getvalue()) / 1024

    st.write(
        "**File size:**",
        f"{file_size:.2f} KB"
    )

    # ==================================================
    # 3. ALGORITHM PARAMETERS
    # ==================================================

    st.subheader("3. Algorithm Parameters")

    # --------------------------------------------------
    # THAM SỐ MANUAL AHE
    # --------------------------------------------------

    st.markdown("#### Manual AHE")

    window_size = st.number_input(
        "Window Size",
        min_value=3,
        max_value=101,
        value=31,
        step=2,
        help=(
            "Nhập kích thước cửa sổ cục bộ. "
            "Phải là số lẻ, ví dụ 15, 31, 51."
        )
    )

    if window_size % 2 == 0:
        st.error("Window Size phải là số lẻ.")
        return

    st.caption(
        f"Manual AHE sử dụng cửa sổ "
        f"{window_size} × {window_size} pixel."
    )

    st.divider()

    # --------------------------------------------------
    # THAM SỐ OPENCV CLAHE
    # --------------------------------------------------

    st.markdown("#### OpenCV CLAHE")

    col1, col2 = st.columns(2)

    with col1:

        clip_limit = st.number_input(
            "Clip Limit",
            min_value=0.1,
            max_value=100.0,
            value=2.0,
            step=0.5,
            help=(
                "Giới hạn tương phản của CLAHE. "
                "Giá trị lớn hơn thường cho phép "
                "tăng cường tương phản mạnh hơn."
            )
        )

    with col2:

        tile_grid_input = st.text_input(
            "Tile Grid Size",
            value="8x8",
            help=(
                "Nhập số ô theo định dạng ngang x dọc. "
                "Ví dụ: 8x8, 16x16, 8x16."
            )
        )

    # --------------------------------------------------
    # KIỂM TRA TILE GRID SIZE
    # --------------------------------------------------

    try:

        tile_values = (
            tile_grid_input
            .lower()
            .replace(" ", "")
            .split("x")
        )

        if len(tile_values) != 2:
            raise ValueError

        tile_width = int(tile_values[0])
        tile_height = int(tile_values[1])

        if tile_width < 1 or tile_height < 1:
            raise ValueError

        tile_grid_size = (
            tile_width,
            tile_height
        )

    except ValueError:

        st.error(
            "Tile Grid Size không hợp lệ. "
            "Vui lòng nhập theo định dạng: 8x8 hoặc 16x16."
        )
        return

    st.caption(
        f"CLAHE sử dụng clipLimit={clip_limit}, "
        f"tileGridSize={tile_grid_size}."
    )

    st.info(
        "Lưu ý: Tile Grid Size là số ô chia ảnh theo "
        "chiều ngang và chiều dọc, không phải kích thước "
        "pixel của mỗi ô."
    )

    # ==================================================
    # 4. ALGORITHM OVERVIEW
    # ==================================================

    st.subheader("4. Algorithm Overview")

    st.markdown(
        """
        **Manual Adaptive Histogram Equalization (AHE)**

        - Nhận ảnh Grayscale 8-bit đầu vào.
        - Xác định kích thước cửa sổ cục bộ (`window_size`).
        - Duyệt lần lượt qua từng pixel của ảnh.
        - Trích xuất vùng lân cận bằng Sliding Window.
        - Tính histogram và phân bố tích lũy CDF của vùng cục bộ.
        - Xác định `CDF_min` và ánh xạ mức xám của pixel trung tâm.
        - Lặp lại cho toàn bộ pixel để thu được ảnh Manual AHE.

        **Phương pháp đối chứng – OpenCV CLAHE**

        Sử dụng `cv2.createCLAHE()` của thư viện OpenCV.

        - Chia ảnh thành các vùng nhỏ theo `tileGridSize`.
        - Tính histogram cho từng vùng.
        - Giới hạn histogram dựa trên `clipLimit`.
        - Phân phối lại phần histogram bị cắt.
        - Thực hiện cân bằng histogram trong từng vùng.
        - Nội suy giữa các vùng lân cận để giảm ranh giới giữa các ô.

        **Lưu ý:** AHE và CLAHE là hai thuật toán khác nhau.
        CLAHE bổ sung cơ chế giới hạn tương phản nhằm hạn chế
        hiện tượng khuếch đại nhiễu của AHE thông thường.
        """
    )

    # ==================================================
    # 5. PROCESSING
    # ==================================================

    st.subheader("5. Processing")

    try:

        with st.spinner("Đang chạy Manual AHE..."):

            manual = run_manual_ahe(
                image,
                window_size
            )

        with st.spinner("Đang chạy OpenCV CLAHE..."):

            clahe = run_opencv_clahe(
                image,
                clip_limit,
                tile_grid_size
            )

    except Exception as error:

        st.error(f"Processing error: {error}")
        return

    st.success("Processing completed!")

    # ==================================================
    # 6. IMAGE AND GRAPH VISUALIZATION
    # ==================================================

    st.subheader("6. Image and Graph Visualization")

    try:

        fig = display_results(
            image,
            manual,
            clahe,
            save_path=None,
            show=False,
            titles=[
                "Original Grayscale",
                "Manual AHE",
                "OpenCV CLAHE"
            ]
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    except Exception as error:

        st.error(f"Visualization error: {error}")
        return

    # ==================================================
    # 7. IMAGE QUALITY ANALYSIS
    # ==================================================

    st.subheader("7. Image Quality Analysis")

    st.write(
        "Đánh giá sự thay đổi giữa ảnh Grayscale ban đầu "
        "và ảnh sau khi áp dụng từng thuật toán."
    )

    try:

        manual_metrics = calculate_metrics(
            image,
            manual
        )

        clahe_metrics = calculate_metrics(
            image,
            clahe
        )

    except Exception as error:

        st.error(f"Metrics calculation error: {error}")
        return

    metric_names = [
        "MAE",
        "MSE",
        "RMSE",
        "PSNR",
        "SSIM",
        "NCC"
    ]

    # Hàm định dạng giá trị của các chỉ số
    def format_value(value):

        if np.isinf(value):
            return "∞"

        if np.isnan(value):
            return "N/A"

        return f"{value:.6f}"

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Original vs Manual AHE")

        for name in metric_names:

            st.metric(
                label=name,
                value=format_value(manual_metrics[name])
            )

    with col2:

        st.markdown("#### Original vs OpenCV CLAHE")

        for name in metric_names:

            st.metric(
                label=name,
                value=format_value(clahe_metrics[name])
            )

    st.caption(
        "Các chỉ số này thể hiện mức độ khác biệt giữa "
        "ảnh gốc và ảnh sau xử lý. Không nên hiểu trực tiếp "
        "rằng MAE thấp hoặc SSIM cao luôn đồng nghĩa với "
        "kết quả Equalization tốt hơn."
    )

    # ==================================================
    # 8. DOWNLOAD RESULTS
    # ==================================================

    st.subheader("8. Download Results")

    col1, col2 = st.columns(2)

    # Mã hóa hai ảnh kết quả thành PNG
    success_manual, manual_encoded = cv2.imencode(
        ".png",
        manual
    )

    success_clahe, clahe_encoded = cv2.imencode(
        ".png",
        clahe
    )

    with col1:

        if success_manual:

            st.download_button(
                label="Download Manual AHE",
                data=manual_encoded.tobytes(),
                file_name="manual_ahe.png",
                mime="image/png"
            )

    with col2:

        if success_clahe:

            st.download_button(
                label="Download OpenCV CLAHE",
                data=clahe_encoded.tobytes(),
                file_name="opencv_clahe.png",
                mime="image/png"
            )
