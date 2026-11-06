
import numpy as np
from skimage.metrics import structural_similarity


def calculate_metrics(image_a, image_b):
    """
    Tính 6 chỉ số đánh giá giữa hai ảnh uint8.

    image_a: Ảnh Manual.
    image_b: Ảnh OpenCV.
    """

    if image_a.shape != image_b.shape:
        raise ValueError("Hai anh phai cung kich thuoc.")

    a = image_a.astype(np.float64)
    b = image_b.astype(np.float64)

    difference = a - b

    # 1. MAE
    mae = np.mean(np.abs(difference))

    # 2. MSE
    mse = np.mean(difference ** 2)

    # 3. RMSE
    rmse = np.sqrt(mse)

    # 4. PSNR
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)

    # 5. SSIM
    ssim = structural_similarity(
        image_a,
        image_b,
        data_range=255
    )

    # 6. NCC - Pearson correlation
    a_flat = a.flatten()
    b_flat = b.flatten()

    if np.std(a_flat) == 0 and np.std(b_flat) == 0:
        ncc = 1.0 if np.array_equal(
            image_a, image_b
        ) else 0.0

    elif np.std(a_flat) == 0 or np.std(b_flat) == 0:
        ncc = 0.0

    else:
        ncc = np.corrcoef(
            a_flat, b_flat
        )[0, 1]

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "PSNR": psnr,
        "SSIM": ssim,
        "NCC": ncc
    }


def calculate_exact_match(image_a, image_b):
    """
    Tính tỷ lệ pixel giống nhau hoàn toàn
    và sai số tuyệt đối lớn nhất.

    Exact Match (%) =
        Số pixel giống nhau / Tổng pixel * 100
    """

    difference = np.abs(
        image_a.astype(np.int16) -
        image_b.astype(np.int16)
    )

    exact_pixels = np.sum(difference == 0)

    total_pixels = difference.size

    exact_match = (
        exact_pixels / total_pixels
    ) * 100

    max_absolute_error = np.max(difference)

    return {
        "Exact Match (%)": exact_match,
        "Max Absolute Error": int(max_absolute_error)
    }
