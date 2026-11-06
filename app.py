
import streamlit as st

from web_modules.histogram_equalization import (
    render as histogram_equalization
)

from web_modules.adaptive_equalization import (
    render as adaptive_equalization
)

from web_modules.histogram_matching import (
    render as histogram_matching
)


# ==========================================
# 1. CẤU HÌNH GIAO DIỆN WEB
# ==========================================

st.set_page_config(
    page_title="BTL_XLA_NHOM_5 ",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. THÔNG TIN ĐỀ TÀI
# ==========================================

PROJECT_NAME = (
    "Histogram và Histogram Equalization"
)

GROUP_NAME = "Nhóm 5"

# Thay thông tin bên dưới bằng tên và MSSV thực tế
GROUP_MEMBERS = [
    {
        "STT": 1,
        "Họ và tên": "Trần Vĩnh Huy",
        "MSSV": "2311256"
    },
    {
        "STT": 2,
        "Họ và tên": "Phạm Hoàng Đăng",
        "MSSV": "2310732"
    },
    {
        "STT": 3,
        "Họ và tên": "Đàm Lê Khải",
        "MSSV": "2311545"
    }
]


# ==========================================
# 3. TRANG CHỦ - THÔNG TIN DỰ ÁN
# ==========================================

def render_home():
    """
    Hiển thị thông tin tổng quan về đề tài,
    thành viên nhóm và các chức năng đã xây dựng.
    """

    st.title("BTL Xử Lý Ảnh - Nhóm 5")

    st.subheader(PROJECT_NAME)

    st.markdown(f"**Nhóm thực hiện:** {GROUP_NAME}")

    st.divider()

    # --------------------------------------
    # GIỚI THIỆU ĐỀ TÀI
    # --------------------------------------

    st.header("1. Giới thiệu đề tài")

    st.write(
        "Đề tài nghiên cứu và xây dựng các thuật toán "
        "xử lý histogram trên ảnh Grayscale 8-bit. "
        "Các thuật toán được triển khai thủ công bằng Python "
        ", sau đó so sánh với các phương pháp "
        "có sẵn trong OpenCV và scikit-image."
    )

    # --------------------------------------
    # CÁC CHỨC NĂNG CỦA DỰ ÁN
    # --------------------------------------

    st.header("2. Các chức năng")

    st.markdown(
        """
        **Function 01 - Histogram Equalization**

        - Tính Histogram, PDF và CDF.
        - Xây dựng thuật toán cân bằng Histogram.
        - So sánh với `cv2.equalizeHist()`.

        **Function 02 - Adaptive Histogram Equalization**

        - Xây dựng thuật toán AHE bằng Sliding Window.
        - Tính histogram và CDF theo từng vùng cục bộ.
        - So sánh với CLAHE của OpenCV.

        **Function 03 - Histogram Matching**

        - Tính CDF của ảnh Source và Target.
        - Ánh xạ các mức xám bằng Linear Interpolation.
        - So sánh với `skimage.exposure.match_histograms()`.
        """
    )

    # --------------------------------------
    # THÔNG TIN THÀNH VIÊN
    # --------------------------------------

    st.header("3. Thành viên thực hiện")

    st.dataframe(
        GROUP_MEMBERS,
        hide_index=True,
        use_container_width=True
    )

   

# ==========================================
# 4. ĐĂNG KÝ CÁC MODULE
# ==========================================

modules = {
    "Home - Project Information": render_home,
    "Histogram Equalization": histogram_equalization,
    "Adaptive Histogram Equalization": adaptive_equalization,
    "Histogram Matching": histogram_matching
}


# ==========================================
# 5. SIDEBAR
# ==========================================

st.sidebar.title("Xử Lý Ảnh")

st.sidebar.caption("BTL Xử Lý Ảnh - Nhóm 5")

st.sidebar.markdown(f"**{GROUP_NAME}**")

st.sidebar.divider()

selected_module = st.sidebar.selectbox(
    "Select Algorithm",
    list(modules.keys())
)

st.sidebar.divider()

# Thông tin rút gọn về dự án
st.sidebar.markdown("### Project Information")

st.sidebar.write(PROJECT_NAME)

st.sidebar.caption(
    "Python | NumPy | OpenCV | Streamlit"
)


# ==========================================
# 6. CHẠY MODULE ĐƯỢC CHỌN
# ==========================================

modules[selected_module]()
