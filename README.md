
# Histogram & Histogram Equalization

**Báo cáo BTL: Xử lý ảnh – Nhóm 5**

Dự án nghiên cứu và xây dựng các thuật toán xử lý Histogram trên ảnh Grayscale 8-bit bằng Python. 

Hệ thống cung cấp giao diện Web bằng Streamlit và Jupyter Notebook phục vụ thực nghiệm, trực quan hóa và đánh giá kết quả.

---

## 1. Giới thiệu 

### Mục tiêu

- Xây dựng các thuật toán biến đổi Histogram bằng Python.
- So sánh thuật toán tự xây dựng với các thư viện xử lý ảnh.
- Đánh giá kết quả bằng các chỉ số định lượng.
- Xây dựng giao diện Web hỗ trợ thực nghiệm trực quan.



---

## 2. Các chức năng chính

### Function 01 – Histogram Equalization

Xây dựng thuật toán cân bằng Histogram nhằm cải thiện phân bố mức xám và tăng cường độ tương phản của ảnh.

**Manual Histogram Equalization:**

1. Tính Histogram của ảnh Grayscale.
2. Chuẩn hóa Histogram để thu được PDF.
3. Tính Cumulative Distribution Function (CDF).
4. Xây dựng Lookup Table (LUT) từ CDF.
5. Ánh xạ mức xám để tạo ảnh kết quả.

**Phương pháp đối chứng:**

```python
cv2.equalizeHist()
```

**Kết quả:**
- Ảnh Grayscale ban đầu.
- Ảnh Manual Histogram Equalization.
- Ảnh OpenCV Histogram Equalization.
- Biểu đồ Histogram và CDF.
- Các chỉ số đánh giá và so sánh hai phương pháp.

### Function 02 – Adaptive Histogram Equalization

Xây dựng thuật toán Adaptive Histogram Equalization (AHE) để tăng cường độ tương phản cục bộ của ảnh.

**Manual AHE:**

1. Xác định kích thước cửa sổ cục bộ (Window Size).
2. Duyệt qua từng pixel bằng Sliding Window.
3. Tính Histogram và CDF trong vùng lân cận.
4. Ánh xạ mức xám của pixel trung tâm.
5. Lặp lại cho toàn bộ ảnh để tạo ảnh AHE.

**Phương pháp đối chứng:**

```python
cv2.createCLAHE(
    clipLimit=clipLimit,
    tileGridSize=tileGridSize()
)
```

**Tham số hỗ trợ:**
- `window_size`: Kích thước cửa sổ của Manual AHE.
- `clipLimit`: Giới hạn tương phản của CLAHE.
- `tileGridSize`: Số vùng chia ảnh của CLAHE.

**Lưu ý:** Manual AHE và OpenCV CLAHE không phải hai cách triển khai hoàn toàn giống nhau. CLAHE bổ sung cơ chế giới hạn Histogram nhằm hạn chế hiện tượng khuếch đại nhiễu.

### Function 03 – Histogram Matching

Xây dựng thuật toán Histogram Matching nhằm biến đổi phân bố mức xám của Source Image sao cho gần với Target Image.

**Manual Histogram Matching:**

1. Tính Histogram của Source và Target.
2. Xác định các mức xám duy nhất và tần suất xuất hiện.
3. Tính CDF của hai ảnh.
4. Ánh xạ Quantile bằng Linear Interpolation.
5. Xây dựng LUT và biến đổi Source Image.

**Phương pháp đối chứng:**

```python
from skimage.exposure import match_histograms

matched = match_histograms(
    source,
    target,
    channel_axis=None
)
```

**Kết quả:**
- Source Image và Target Image.
- Manual Histogram Matching.
- Scikit-image Histogram Matching.
- Biểu đồ Histogram và CDF.
- So sánh CDF giữa các ảnh.
- Bảng LUT ánh xạ mức xám.
- Đánh giá Histogram L1 và Wasserstein-1 Distance.

---

## 3. Các chỉ số đánh giá

Dự án sử dụng các chỉ số sau để phân tích kết quả thực nghiệm:

| Chỉ số | Ý nghĩa |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| PSNR | Peak Signal-to-Noise Ratio |
| SSIM | Structural Similarity Index |
| NCC | Normalized Cross-Correlation |
| Exact Match (%) | Tỷ lệ pixel trùng khớp |
| Max Absolute Error | Sai khác tuyệt đối lớn nhất |
| Histogram L1 | Khoảng cách L1 giữa hai Histogram chuẩn hóa |
| Wasserstein-1 | Khoảng cách giữa hai phân bố mức xám |
| STD | Độ lệch chuẩn của mức xám |



---

## 4. Cấu trúc dự án

```text
BTLXLA_Nhom5/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── Histogram_Image_Processing_Nhom5.ipynb
│
├── images/
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── histogram.py
│   ├── equalization.py
│   ├── adaptive_equalization.py
│   ├── histogram_matching.py
│   ├── metrics.py
│   └── visualization.py
│
└── web_modules/
    ├── __init__.py
    ├── histogram_equalization.py
    ├── adaptive_equalization.py
    └── histogram_matching.py
```

### Vai trò các module

| Module | Chức năng |
|---|---|
| `app.py` | Giao diện chính và điều hướng Streamlit |
| `src/preprocessing.py` | Đọc ảnh, chuyển Grayscale và 8-bit |
| `src/histogram.py` | Tính Histogram, PDF và CDF |
| `src/equalization.py` | Histogram Equalization |
| `src/adaptive_equalization.py` | Manual AHE và OpenCV CLAHE |
| `src/histogram_matching.py` | Histogram Matching |
| `src/metrics.py` | Tính các chỉ số đánh giá |
| `src/visualization.py` | Hiển thị ảnh, Histogram và CDF |
| `web_modules/` | Giao diện Streamlit cho từng Function |
| `Histogram_Image_Processing_Nhom5.ipynb` | Thực nghiệm và phân tích thuật toán |

---

## 5. Cài đặt và chạy dự án

### Yêu cầu

- Python 3.10 trở lên (khuyến nghị).
- Git.
- VS Code hoặc IDE hỗ trợ Python.
- Hệ điều hành Windows, Linux hoặc macOS.

### Bước 1: Clone Repository

```bash
git clone <https://github.com/Khaisaigon/BTL_XLA_Nhom_5_Histogram>
cd BTLXLA_Nhom5
```

Thay `<https://github.com/Khaisaigon/BTL_XLA_Nhom_5_Histogram>` bằng đường dẫn GitHub thực tế.

### Bước 2: Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường ảo.

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Nếu chưa có đầy đủ thư viện:

```bash
pip install numpy opencv-python matplotlib scikit-image streamlit pandas jupyter ipykernel
```

### Bước 4: Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Sau khi khởi chạy, truy cập địa chỉ mặc định:

```text
http://localhost:8501
```

### Bước 5: Sử dụng ứng dụng

1. Mở giao diện Web.
2. Chọn thuật toán trong Sidebar.
3. Upload ảnh đầu vào.
4. Điều chỉnh tham số nếu có.
5. Quan sát ảnh kết quả, Histogram và CDF.
6. Phân tích các chỉ số đánh giá.
7. Tải ảnh kết quả về máy.

---

## 6. Jupyter Notebook

Dự án cung cấp Notebook để giải thích lý thuyết và thực nghiệm các thuật toán.

**File:**

```text
Histogram_Image_Processing_Nhom5.ipynb
```

Notebook bao gồm:

- Histogram, PDF và CDF.
- Histogram Equalization.
- Adaptive Histogram Equalization.
- Histogram Matching.
- Thực nghiệm thay đổi tham số.
- So sánh kết quả bằng các chỉ số đánh giá.
- Nhận xét và kết luận.

**Cách chạy:**

1. Mở Notebook bằng VS Code hoặc Jupyter.
2. Chọn Python Kernel của môi trường `venv`.
3. Chạy lần lượt các cell từ trên xuống.
4. Quan sát và phân tích kết quả.

Notebook sử dụng lại các thuật toán trong thư mục `src/` để bảo đảm thống nhất với ứng dụng Streamlit.

---

---
## 7. Thành viên nhóm

**Môn học:** Xử lý ảnh

**Tên đề tài:** Histogram và Histogram Equalization

**Giảng viên hướng dẫn:** [TS. Võ Tuấn Kiệt]

**Nhóm thực hiện:** Nhóm 5

| STT | Họ và tên | MSSV |
|---|---|---|
| 1 | [Trần Vĩnh Huy] | [2311256] |
| 2 | [Phạm Hoàng Đăng] | [2310732] |
| 3 | [Đàm Lê Khải] | [2311545] |



**Trường:** Trường Đại học Bách khoa – ĐHQG TP.HCM

---


**Histogram & Image Processing – Nhóm 5**
