
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import (
    convert_to_grayscale,
    convert_to_8bit
)

from src.equalization import (
    manual_equalization,
    opencv_equalization
)

from src.metrics import (
    calculate_metrics,
    calculate_exact_match
)

from src.visualization import display_results


def render():

    st.title("Histogram Equalization")

    st.write(
        "So sánh thuật toán Histogram Equalization "
        "tự xây dựng với thư viện OpenCV."
    )

    # ==========================================
    # 1. UPLOAD IMAGE
    # ==========================================

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    )

    if uploaded_file is None:
        st.info("Vui lòng tải ảnh lên để bắt đầu.")
        return

    # ==========================================
    # 2. READ IMAGE
    # ==========================================

    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    original = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_UNCHANGED
    )

    if original is None:
        st.error("Không thể đọc ảnh.")
        return

    # ==========================================
    # 3. PREPROCESSING
    # ==========================================

    gray = convert_to_grayscale(original)
    image = convert_to_8bit(gray)

    height, width = image.shape

    # ==========================================
    # 4. IMAGE INFORMATION
    # ==========================================

    st.subheader("1. Image Information")

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
    st.write(
        "**File size:**",
        f"{len(uploaded_file.getvalue()) / 1024:.2f} KB"
    )

    # ==========================================
    # 5. HISTOGRAM EQUALIZATION
    # ==========================================

    st.subheader("2. Histogram Equalization")

    manual, hist, pdf, cdf, lut = manual_equalization(image)

    opencv = opencv_equalization(image)

    st.markdown(
    """
    **Manual Histogram Equalization**

    - Tính histogram của ảnh Grayscale đầu vào.
    - Chuẩn hóa histogram để thu được phân bố xác suất PDF.
    - Tính hàm phân phối tích lũy CDF từ PDF.
    - Xây dựng bảng ánh xạ LUT theo công thức cân bằng histogram.
    - Áp dụng LUT để biến đổi mức xám của từng pixel.
    - Thu được ảnh sau khi cân bằng histogram.

   

    **Phương pháp đối chứng**

    Sử dụng `cv2.equalizeHist()` của thư viện OpenCV.
    """
)

    # ==========================================
    # 6. IMAGE AND GRAPH VISUALIZATION
    # ==========================================

    st.subheader("3. Image and Graph Visualization")

    # Gọi trực tiếp module visualization.py
    # Không tự xây dựng lại biểu đồ trên web.

    fig = display_results(
        image,
        manual,
        opencv,
        save_path=None,
        show=False
    )

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # ==========================================
    # 7. METRICS COMPARISON
    # ==========================================

    st.subheader("4. Metrics Comparison")

    metrics = calculate_metrics(manual, opencv)
    exact = calculate_exact_match(manual, opencv)

    # Hàm định dạng kết quả
    def format_metric(value):
        if np.isinf(value):
            return "∞"
        return f"{value:.6f}"

    # Hàng 1: MAE, MSE, RMSE
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", format_metric(metrics["MAE"]))

    with col2:
        st.metric("MSE", format_metric(metrics["MSE"]))

    with col3:
        st.metric("RMSE", format_metric(metrics["RMSE"]))

    # Hàng 2: PSNR, SSIM, NCC
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("PSNR (dB)", format_metric(metrics["PSNR"]))

    with col2:
        st.metric("SSIM", format_metric(metrics["SSIM"]))

    with col3:
        st.metric("NCC", format_metric(metrics["NCC"]))

    # Hàng 3: Exact Match và Max Absolute Error
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Exact Match (%)",
            f'{exact["Exact Match (%)"]:.4f}%'
        )

    with col2:
        st.metric(
            "Max Absolute Error",
            exact["Max Absolute Error"]
        )

    # ==========================================
    # 8. DOWNLOAD RESULTS
    # ==========================================

    st.subheader("5. Download Results")

    col1, col2 = st.columns(2)

    success, manual_encoded = cv2.imencode(
        ".png", manual
    )

    success2, opencv_encoded = cv2.imencode(
        ".png", opencv
    )

    with col1:
        if success:
            st.download_button(
                label="Download Manual Result",
                data=manual_encoded.tobytes(),
                file_name="manual_result.png",
                mime="image/png"
            )

    with col2:
        if success2:
            st.download_button(
                label="Download OpenCV Result",
                data=opencv_encoded.tobytes(),
                file_name="opencv_result.png",
                mime="image/png"
            )


    # ==========================================
    # 9. LOOKUP TABLE (LUT)
    # ==========================================

    st.subheader("6. Lookup Table (LUT)")

    with st.expander("View Lookup Table (LUT)"):

        st.write(
            "Bảng ánh xạ 256 mức xám từ ảnh gốc "
            "sang ảnh sau Histogram Equalization."
        )

        # Tạo bảng LUT gồm 256 mức xám
        lut_df = pd.DataFrame({
            "Original Gray Level": np.arange(256),
            "Mapped Gray Level": lut
        })

        # Hiển thị bảng LUT
        st.dataframe(
            lut_df,
            hide_index=True,
            use_container_width=True
        )
