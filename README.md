# CV-tohe – Nhận diện bột nặn tò he (OpenCV)

Phần **thị giác máy tính (CV)** của dự án *Hộp quà Di sản – Tò He Xuân La*.
Camera hoặc ảnh chụp → tìm cục bột → đo **độ tròn, kích thước so với thân, độ dẹt, độ dài sợi, màu sắc** → chấm sao và nhận xét cho từng bước nặn **chú heo hồng**: Thân → Hai tai → Mõm → Đuôi.

## 📷 Chạy ngay trên trình duyệt, có camera
👉 **https://quangtommy0507-dotcom.github.io/CV-tohe/**

Mở link bằng Chrome/Safari (máy tính hoặc điện thoại) → chờ tải Python + OpenCV (20–60 giây lần đầu) → chọn nhân vật → **📷 Bật camera** → **① Chụp nền** → **② Quét tay** → **③ Chụp từng bộ phận** → nặn và xem máy nhận diện.
Trang này chạy chính file `theo_doi_nan.py` bằng Pyodide; ảnh camera xử lý ngay trên máy, không gửi đi đâu.

## Chạy ngay trên GitHub (không cần cài gì)
- **Actions:** mỗi lần đẩy code lên, GitHub tự tạo 14 ảnh mẫu, chấm điểm và lưu ảnh kết quả. Vào tab **Actions**, chọn lần chạy mới nhất, tải mục **ket-qua-cham-anh**.
- **Codespaces:** bấm **Code → Codespaces → Create codespace on main** để mở VS Code trên trình duyệt (đã cài sẵn OpenCV). Mở Terminal và gõ:
  ```bash
  python nan_heo_cv.py --thu-het --khong-hien
  ```
  Codespaces chạy trên máy chủ nên **không dùng được camera**; chỉ chấm được ảnh.

## Chạy trên máy có camera (Mac / Windows)
```bash
pip install -r requirements.txt
python tao_anh_mau.py                      # tạo ảnh mẫu
python nan_heo_cv.py --thu-het             # chấm thử ảnh mẫu
python nan_heo_cv.py                       # bật camera
python nan_heo_cv.py anh.jpg --buoc tai --than anh_mau/than_tot.jpg
```
Phím trên cửa sổ camera: **1–4** chọn bước · **SPACE** chụp · **A** tự chụp · **M** xem mặt nạ · **S** lưu · **Q** thoát.

## Theo dõi trẻ nặn: nhận diện bàn tay + khung tham chiếu (`theo_doi_nan.py`)
Nhân vật có sẵn: **heo** (con lợn), **hoa** (bông hoa), **trau** (con trâu), **ga** (con gà).
```bash
python theo_doi_nan.py --thu          # thử bằng cảnh giả lập (không cần camera)
python theo_doi_nan.py --nv trau      # camera, nhân vật con trâu
```
**Quy trình chụp / quét để tạo khung tham chiếu:**

| Bước | Phím | Làm gì | Máy học được |
|---|---|---|---|
| ① Nền | **B** | Để tấm lót trống | Đâu là nền |
| ② Tay | **H** | Đặt bàn tay vào giữa khung 2 giây | Màu da của chính trẻ dưới ánh sáng nhà mình |
| ③ Bộ phận | **1–6** | Nặn xong bộ phận nào, đặt riêng lên tấm lót (đủ số lượng, tay để ngoài) | Màu, kích thước, độ dẹt, hình dáng của bộ phận đó |

**Khi theo dõi, mỗi khung hình máy sẽ:**
- Viền **xanh lá** quanh bàn tay.
- Gán nhãn từng cục bột theo khung tham chiếu, ví dụ "Thân 99%", "Tai 94%". Điểm gồm màu 50%, kích thước 30%, hình dáng 20%.
- Báo **bộ phận đang được nặn** trong 3 trường hợp:
  - cục bột chạm tay;
  - bột lộ ra giữa các ngón tay;
  - bộ phận vừa rời tấm lót đúng chỗ bàn tay ("đang cầm trên tay").
- So màu với mẫu của nhân vật, ví dụ "Màu đang là hồng, mẫu cần đen / xám".

Tham chiếu được lưu trong `tham_chieu/<nhân vật>/`, lần sau mở lên dùng lại. Phím **M** để xem mặt nạ (xanh = tay, hồng = bột), **N** để đổi nhân vật, **R** để xoá tham chiếu.

## Cách thuật toán hoạt động
1. Lấy màu nền từ viền ảnh, rồi cân bằng trắng.
2. Đo khoảng cách màu (không gian YCbCr) so với nền; tách bột bằng ngưỡng **Otsu**.
3. Lọc nhiễu, tách các vùng liền nhau, bỏ vùng chạm mép ảnh (thường là tay).
4. Mỗi vùng: tính elip quán tính (trục dài, trục ngắn, góc) và màu trung bình (HSL).
5. So với tỉ lệ chuẩn lấy từ mô hình 3D heo mẫu: tai ≈ 43%, mõm ≈ 46%, đuôi ≈ 90% đường kính thân.

> Bột hồng nhạt hoặc trắng đặt trên giấy trắng thì máy khó thấy. Nên dùng **tấm lót màu xám**.
