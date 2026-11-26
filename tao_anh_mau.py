"""
TẠO ẢNH MẪU GIẢ LẬP để chấm thử khi chưa có bột thật.
Vẽ cục bột (hình elip có bóng đổ, có độ sáng tối) trên nền giấy/tấm lót.
Chạy:  python tao_anh_mau.py   → tạo thư mục anh_mau/ gồm 14 ảnh
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1280, 960
RA = Path(__file__).resolve().parent / "anh_mau"
rng = np.random.default_rng(1)

HONG, DO_DAM, HONG_NHAT = (236, 148, 164), (196, 42, 58), (246, 226, 228)
HONG_SANG, VANG = (242, 170, 182), (235, 190, 60)
GIAY, XAM = (243, 236, 224), (150, 150, 150)


def nen(mau):
    y, x = np.mgrid[0:H, 0:W]
    g = 1 - 0.12 * ((x - W * 0.3) ** 2 + (y - H * 0.2) ** 2) / (W * W)   # ánh sáng không đều
    return np.stack([mau[0] * g, mau[1] * g, mau[2] * g], -1) + rng.normal(0, 3, (H, W, 3))


def cuc_bot(anh, cx, cy, a, b, goc, mau):
    t = np.linspace(0, 2 * np.pi, 200)
    ca, sa = np.cos(goc), np.sin(goc)
    diem = [(cx + a * np.cos(k) * ca - b * np.sin(k) * sa, cy + a * np.cos(k) * sa + b * np.sin(k) * ca) for k in t]
    bong = Image.new("L", (W, H), 0)
    ImageDraw.Draw(bong).polygon([(px + a * 0.12, py + b * 0.18) for px, py in diem], fill=90)
    bong = np.array(bong.filter(ImageFilter.GaussianBlur(12))) / 255.0
    anh *= (1 - bong[..., None] * 0.35)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon(diem, fill=255)
    m = np.array(m.filter(ImageFilter.GaussianBlur(1.2))) / 255.0
    y, x = np.mgrid[0:H, 0:W]
    sang = 1 + 0.18 * np.clip(1 - ((x - cx + a * .3) ** 2 / (a * a + 1) + (y - cy + b * .3) ** 2 / (b * b + 1)), -1, 1)
    anh[:] = anh * (1 - m[..., None]) + np.array(mau)[None, None, :] * sang[..., None] * m[..., None]


def luu(ten, ds, mau_nen=GIAY):
    anh = nen(mau_nen)
    for c in ds:
        cuc_bot(anh, *c)
    Image.fromarray(np.clip(anh, 0, 255).astype("uint8")).save(RA / f"{ten}.jpg", quality=90)
    print("  ✓", ten)


if __name__ == "__main__":
    RA.mkdir(exist_ok=True)
    print("Đang tạo ảnh mẫu vào", RA)
    luu("than_tot", [(640, 470, 120, 116, 0, HONG)])
    luu("than_meo", [(640, 470, 150, 95, 0.3, HONG)])
    luu("than_mau_dam", [(640, 470, 120, 116, 0, DO_DAM)])
    luu("than_mau_vang", [(640, 470, 120, 116, 0, VANG)])
    luu("than_hong_nhat_nen_trang", [(640, 470, 120, 116, 0, HONG_NHAT)])
    luu("than_hong_nhat_nen_xam", [(640, 470, 120, 116, 0, HONG_NHAT)], XAM)
    luu("tai_tot", [(540, 480, 52, 40, 0.4, HONG), (740, 470, 50, 41, -0.3, HONG)])
    luu("tai_thieu", [(600, 480, 52, 40, 0.4, HONG)])
    luu("tai_lech", [(540, 480, 60, 46, 0.4, HONG), (740, 470, 34, 28, -0.3, HONG)])
    luu("tai_to", [(540, 480, 95, 72, 0.4, HONG), (760, 470, 93, 73, -0.3, HONG)])
    luu("mom_tot", [(640, 480, 55, 40, 0.2, HONG_SANG)])
    luu("mom_tron", [(640, 480, 48, 46, 0.2, HONG_SANG)])
    luu("duoi_tot", [(640, 480, 110, 17, 0.5, HONG)])
    luu("duoi_ngan", [(640, 480, 40, 22, 0.5, HONG)])
    print("Xong 14 ảnh.")
