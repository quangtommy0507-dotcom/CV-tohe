"""
NẶN HEO HỒNG – PHẦN NHẬN DIỆN ẢNH (CV) CHẠY RIÊNG BẰNG PYTHON + OPENCV
=====================================================================
Chương trình tìm cục bột trên tấm lót, rồi đo:
  • Độ tròn (thân heo)
  • Kích thước so với thân (tai, mõm, đuôi)
  • Độ dẹt (mõm), độ dài-mảnh (đuôi), hai tai có bằng nhau không
  • Màu: hồng đẹp / hơi đậm / hơi nhạt / chưa phải hồng

CÁCH CHẠY (trong Terminal của VS Code):
  python nan_heo_cv.py                         → dùng camera
  python nan_heo_cv.py --cam 1                 → dùng camera số 1 (nếu camera 0 là iPhone)
  python nan_heo_cv.py anh_mau/than_tot.jpg    → chấm 1 ảnh (mặc định là bước Thân)
  python nan_heo_cv.py anh_mau/tai_tot.jpg --buoc tai --than anh_mau/than_tot.jpg
  python nan_heo_cv.py --thu-het               → chấm thử tất cả ảnh trong anh_mau/

PHÍM TẮT KHI CỬA SỔ CAMERA ĐANG MỞ:
  1 2 3 4   chọn bước: Thân / Tai / Mõm / Đuôi
  SPACE     chụp và chấm điểm
  A         bật / tắt tự chụp khi bột nằm yên
  M         xem "mặt nạ" (vùng máy nhận là bột) – để hiểu máy đang thấy gì
  S         lưu ảnh kết quả vào thư mục ket_qua/
  Q / ESC   thoát
"""
import argparse
import math
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:          # không có Pillow thì vẫn chạy, chữ sẽ không dấu
    Image = None

THU_MUC = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# 1. CẤU HÌNH CÁC BƯỚC – tỉ lệ chuẩn lấy từ mô hình 3D heo mẫu
#    ty_le: kích thước bộ phận / đường kính thân;  khoang: [nhỏ nhất, lớn nhất] chấp nhận
# ----------------------------------------------------------------------
BUOC = {
    "than": {"ten": "Thân heo", "huong_dan": "Xoay một viên bột hồng thật tròn, đặt giữa tấm lót"},
    "tai":  {"ten": "Hai tai",  "huong_dan": "Hai viên nhỏ bằng nhau, ấn dẹt, vuốt nhọn 1 đầu",
             "ty_le": 0.43, "khoang": (0.28, 0.62), "so_khoi": 2},
    "mom":  {"ten": "Mõm",      "huong_dan": "Một viên nhỏ ấn dẹt thành hình bầu dục",
             "ty_le": 0.46, "khoang": (0.30, 0.65)},
    "duoi": {"ten": "Đuôi",     "huong_dan": "Se một sợi bột nhỏ, dài, đặt thẳng",
             "ty_le": 0.90, "khoang": (0.45, 1.50)},
}
THU_TU = ["than", "tai", "mom", "duoi"]

NGUONG_TRON = 0.86      # độ tròn (trục ngắn / trục dài) tối thiểu của thân
NGUONG_DET_MOM = 1.12   # mõm phải dẹt: trục dài / trục ngắn ≥ 1.12
NGUONG_SOI_DUOI = 2.6   # đuôi phải là sợi: dài ≥ 2.6 lần bề ngang
NGUONG_TAI_DEU = 0.70   # diện tích tai nhỏ / tai lớn ≥ 0.7


# ----------------------------------------------------------------------
# 2. TÌM CỤC BỘT TRONG ẢNH
# ----------------------------------------------------------------------
def tim_cuc_bot(anh_bgr, rong_xu_ly=420):
    """Trả về (danh sách khối bột, mặt nạ, hệ số thu nhỏ).

    Các bước:
      a) Thu nhỏ ảnh cho nhanh.
      b) Lấy màu nền = trung vị các điểm ảnh ở viền ảnh (viền thường là tấm lót).
      c) Cân bằng trắng: chỉnh sao cho màu nền thành xám trung tính.
      d) Tính "khoảng cách màu" của từng điểm ảnh so với nền (chú trọng sắc màu,
         giảm ảnh hưởng của bóng đổ tối).
      e) Ngưỡng Otsu tự chọn mức tách bột / nền.
      f) Lọc nhiễu, tách các vùng liền nhau, bỏ vùng quá nhỏ.
      g) Mỗi vùng: diện tích, trọng tâm, elip (trục dài, trục ngắn, góc), màu trung bình.
    """
    h0, w0 = anh_bgr.shape[:2]
    k = min(1.0, rong_xu_ly / w0)
    anh = cv2.resize(anh_bgr, (round(w0 * k), round(h0 * k)), interpolation=cv2.INTER_AREA) if k < 1 else anh_bgr.copy()
    h, w = anh.shape[:2]
    f = anh.astype(np.float32)

    # b) màu nền từ viền ảnh
    bw = max(2, round(min(w, h) * 0.05))
    vien = np.concatenate([f[:bw].reshape(-1, 3), f[-bw:].reshape(-1, 3),
                           f[:, :bw].reshape(-1, 3), f[:, -bw:].reshape(-1, 3)])
    nen = np.median(vien, axis=0) + 1e-3                     # B, G, R

    # c) cân bằng trắng
    he_so = np.clip(nen.mean() / nen, 0.6, 1.7)
    f = np.clip(f * he_so, 0, 255)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]

    # d) khoảng cách màu so với nền (không gian Y-Cb-Cr)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B
    dY = Y - nen.mean()
    trong_so_sang = np.where(dY < 0, 0.35, 0.6)              # bóng tối ít ảnh hưởng hơn
    kc = np.sqrt((dY * trong_so_sang) ** 2 + 2 * Cb ** 2 + 2 * Cr ** 2)
    kc8 = np.clip(kc, 0, 255).astype(np.uint8)

    # e) ngưỡng Otsu (tối thiểu 20 để không bắt nhiễu khi khung trống)
    otsu, _ = cv2.threshold(kc8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nguong = max(otsu, 20)
    mat_na = (kc8 > nguong).astype(np.uint8) * 255

    # f) lọc nhiễu + tách vùng
    mat_na = cv2.medianBlur(mat_na, 3)
    mat_na = cv2.morphologyEx(mat_na, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    so, nhan, tk, _ = cv2.connectedComponentsWithStats(mat_na, connectivity=4)
    dien_tich_min = max(12, w * h * 0.0008)

    khoi = []
    for i in range(1, so):
        x, y, bw_, bh_, n = tk[i]
        if n < dien_tich_min:
            continue
        vung = nhan == i
        ys, xs = np.nonzero(vung)
        cx, cy = xs.mean(), ys.mean()
        cxx, cyy, cxy = ((xs - cx) ** 2).mean(), ((ys - cy) ** 2).mean(), ((xs - cx) * (ys - cy)).mean()
        tr, dt = cxx + cyy, cxx * cyy - cxy * cxy
        disc = math.sqrt(max(0.0, tr * tr / 4 - dt))
        l1, l2 = tr / 2 + disc, max(1e-6, tr / 2 - disc)
        truc_dai, truc_ngan = 4 * math.sqrt(l1), 4 * math.sqrt(l2)   # elip có cùng mô-men
        b_, g_, r_ = f[vung].mean(axis=0)
        khoi.append({
            "dien_tich": int(n), "cx": cx, "cy": cy,
            "truc_dai": truc_dai, "truc_ngan": truc_ngan, "goc": 0.5 * math.atan2(2 * cxy, cxx - cyy),
            "cham_vien": x <= 1 or y <= 1 or x + bw_ >= w - 1 or y + bh_ >= h - 1,
            "hop": (x, y, bw_, bh_), "rgb": (r_, g_, b_),
        })
    khoi.sort(key=lambda b: -b["dien_tich"])
    for b in khoi:
        b["do_det"] = b["truc_dai"] / b["truc_ngan"]
        b["duong_kinh"] = (b["truc_dai"] + b["truc_ngan"]) / 2
        b["hsl"] = rgb_sang_hsl(b["rgb"])
    return khoi, mat_na, k, w


def rgb_sang_hsl(rgb):
    r, g, b = (c / 255 for c in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    l, d = (mx + mn) / 2, mx - mn
    if d < 1e-6:
        return 0.0, 0.0, l
    s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        hh = (g - b) / d + (6 if g < b else 0)
    elif mx == g:
        hh = (b - r) / d + 2
    else:
        hh = (r - g) / d + 4
    return hh * 60, s, l


def kiem_tra_mau(b):
    h, s, l = b["hsl"]
    hong = h >= 315 or h <= 22
    if l > 0.9 or (s < 0.12 and l > 0.72):
        return "mau_nhat"
    if not hong or s < 0.12:
        return "mau_khac"
    if l < 0.5:
        return "mau_dam"
    if l > 0.86:
        return "mau_nhat"
    return "mau_dep"


LOI_NHAN = {
    "khong_thay": "Chưa thấy cục bột. Đặt bột vào giữa tấm lót, tay ra ngoài khung.",
    "than_tron": "Thân heo tròn đều, rất đẹp!",
    "than_meo": "Thân hơi méo, xoay thêm trong lòng bàn tay cho tròn.",
    "tai_thieu": "Mới thấy 1 tai. Đặt cả 2 tai lên tấm lót, cách nhau một chút.",
    "tai_deu": "Hai tai to bằng nhau, giỏi quá!",
    "tai_lech": "Hai tai chưa bằng nhau, chỉnh cho đều.",
    "vua": "Kích thước vừa khít với thân heo.",
    "to_qua": "Hơi to so với thân, bớt bột đi một chút.",
    "nho_qua": "Hơi nhỏ so với thân, thêm chút bột.",
    "mom_dep": "Mõm hình bầu dục xinh xắn.",
    "mom_tron": "Mõm còn tròn quá, ấn dẹt thành bầu dục.",
    "duoi_dep": "Sợi đuôi dài và mảnh, đẹp lắm!",
    "duoi_ngan": "Đuôi còn ngắn và mập, se dài và mảnh hơn.",
    "duoi_dai": "Đuôi hơi dài quá, ngắt bớt một chút.",
    "mau_dep": "Màu hồng rất đẹp.",
    "mau_dam": "Màu hơi đậm, trộn thêm chút bột trắng.",
    "mau_nhat": "Màu hơi nhạt, trộn thêm chút bột đỏ.",
    "mau_khac": "Chưa phải màu hồng: trộn bột đỏ với bột trắng.",
    "chua_co_than": "Chưa đo thân heo nên chưa so được kích thước (làm bước 1 trước).",
}


# ----------------------------------------------------------------------
# 3. CHẤM ĐIỂM MỘT BƯỚC
# ----------------------------------------------------------------------
def chon_khoi(khoi, buoc):
    hop_le = [b for b in khoi if not b["cham_vien"]]          # chạm viền = thường là tay
    if not hop_le:
        return []
    if BUOC[buoc].get("so_khoi") == 2:
        lon = hop_le[0]
        return [b for b in hop_le[:2] if b["dien_tich"] >= lon["dien_tich"] * 0.3]
    return hop_le[:1]


def cham_diem(buoc, khoi, rong, D):
    """D = đường kính thân (phần nghìn chiều rộng ảnh), None nếu chưa đo.
    Trả về (danh sách (đạt?, mã lời nhắn), danh sách số đo, D mới)."""
    cfg, kq, so_do = BUOC[buoc], [], []
    chon = chon_khoi(khoi, buoc)
    if not chon:
        return [(False, "khong_thay")], so_do, D
    b = chon[0]
    chuan = lambda v: v / rong * 1000

    def kt_kich_thuoc(do_dai):
        if not D:
            kq.append((False, "chua_co_than"))
            return
        r = chuan(do_dai) / D
        so_do.append(f"Cỡ so với thân: {r*100:.0f}% (mẫu {cfg['ty_le']*100:.0f}%)")
        lo, hi = cfg["khoang"]
        if r > hi:
            kq.append((False, "duoi_dai" if buoc == "duoi" else "to_qua"))
        elif r < lo:
            kq.append((False, "duoi_ngan" if buoc == "duoi" else "nho_qua"))
        else:
            kq.append((True, "vua"))

    if buoc == "than":
        tron = b["truc_ngan"] / b["truc_dai"]
        D = chuan(b["duong_kinh"])
        so_do.append(f"Độ tròn: {tron*100:.0f}%")
        kq.append((tron >= NGUONG_TRON, "than_tron" if tron >= NGUONG_TRON else "than_meo"))
    elif buoc == "tai":
        if len(chon) < 2:
            kq.append((False, "tai_thieu"))
        else:
            deu = min(chon[0]["dien_tich"], chon[1]["dien_tich"]) / max(chon[0]["dien_tich"], chon[1]["dien_tich"])
            so_do.append(f"Hai tai giống nhau: {deu*100:.0f}%")
            kq.append((deu >= NGUONG_TAI_DEU, "tai_deu" if deu >= NGUONG_TAI_DEU else "tai_lech"))
            kt_kich_thuoc((chon[0]["truc_dai"] + chon[1]["truc_dai"]) / 2)
    elif buoc == "mom":
        so_do.append(f"Độ dẹt: {b['do_det']:.2f}")
        kq.append((b["do_det"] >= NGUONG_DET_MOM, "mom_dep" if b["do_det"] >= NGUONG_DET_MOM else "mom_tron"))
        kt_kich_thuoc(b["truc_dai"])
    elif buoc == "duoi":
        so_do.append(f"Dài gấp {b['do_det']:.1f} lần bề ngang")
        if b["do_det"] < NGUONG_SOI_DUOI:
            kq.append((False, "duoi_ngan"))
        else:
            kq.append((True, "duoi_dep"))
            kt_kich_thuoc(b["truc_dai"])

    m = kiem_tra_mau(b)
    h, s, l = b["hsl"]
    so_do.append(f"Màu: sắc {h:.0f}°, đậm {s*100:.0f}%, sáng {l*100:.0f}%")
    kq.append((m == "mau_dep", m))
    return kq, so_do, D


def so_sao(kq):
    if len(kq) == 1 and kq[0][1] == "khong_thay":
        return 0
    dat = sum(1 for ok, _ in kq if ok)
    return max(1, round(dat / len(kq) * 3))


# ----------------------------------------------------------------------
# 4. VẼ KẾT QUẢ LÊN ẢNH
# ----------------------------------------------------------------------
VANG, TRANG, DO, XANH = (68, 181, 242), (255, 255, 255), (24, 35, 139), (68, 107, 62)


def ve_elip_net_dut(anh, tam, truc, goc_do, mau, day=2):
    pts = cv2.ellipse2Poly((int(tam[0]), int(tam[1])), (int(truc[0]), int(truc[1])), int(goc_do), 0, 360, 6)
    for i in range(0, len(pts) - 1, 2):
        cv2.line(anh, tuple(pts[i]), tuple(pts[i + 1]), mau, day, cv2.LINE_AA)


def ve_lop_phu(anh, khoi, k, buoc, D, rong):
    """Viền VÀNG = cục bột máy thấy.  Nét đứt TRẮNG = khuôn mẫu cần đạt."""
    out = anh.copy()
    cfg = BUOC[buoc]
    for b in chon_khoi(khoi, buoc):
        tam = (b["cx"] / k, b["cy"] / k)
        a, c = b["truc_dai"] / 2 / k, b["truc_ngan"] / 2 / k
        goc = math.degrees(b["goc"])
        cv2.ellipse(out, (int(tam[0]), int(tam[1])), (int(a), int(c)), goc, 0, 360, VANG, 3, cv2.LINE_AA)
        if buoc == "than":
            r = (a + c) / 2
            ve_elip_net_dut(out, tam, (r, r), 0, TRANG)
        elif D and cfg.get("ty_le"):
            muc_tieu = cfg["ty_le"] * D / 1000 * rong / k / 2
            ti_le_det = {"tai": 1.3, "mom": 1.35, "duoi": 4.0}[buoc]
            ve_elip_net_dut(out, tam, (muc_tieu, muc_tieu / ti_le_det), goc, TRANG)
    # vùng chạm viền (thường là tay) – tô mờ đỏ để biết máy đã bỏ qua
    for b in khoi:
        if b["cham_vien"]:
            x, y, w_, h_ = (int(v / k) for v in b["hop"])
            cv2.rectangle(out, (x, y), (x + w_, y + h_), (60, 60, 200), 2)
    return out


def tim_font():
    for p in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return p
    return None


FONT = tim_font()


def bo_dau(t):
    return unicodedata.normalize("NFD", t).encode("ascii", "ignore").decode().replace("Đ", "D").replace("đ", "d")


def ve_bang(anh, dong, rong_bang=420):
    """Ghép bảng chữ bên phải ảnh. dong = [(chữ, màu BGR, cỡ)]."""
    h = anh.shape[0]
    bang = np.full((h, rong_bang, 3), (210, 232, 245), np.uint8)
    if Image is not None and FONT:
        im = Image.fromarray(cv2.cvtColor(bang, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(im)
        y = 16
        for chu, mau, co in dong:
            ft = ImageFont.truetype(FONT, co)
            for doan in cat_dong(chu, ft, rong_bang - 32, dr):
                dr.text((16, y), doan, font=ft, fill=(mau[2], mau[1], mau[0]))
                y += int(co * 1.35)
            y += 4
        bang = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    else:
        y = 28
        for chu, mau, co in dong:
            cv2.putText(bang, bo_dau(chu)[:48], (12, y), cv2.FONT_HERSHEY_SIMPLEX, co / 34, mau, 1, cv2.LINE_AA)
            y += int(co * 1.5)
    return np.hstack([anh, bang])


def cat_dong(chu, ft, rong, dr):
    tu, dong, hien = chu.split(" "), [], ""
    for t in tu:
        thu = (hien + " " + t).strip()
        if dr.textlength(thu, font=ft) <= rong:
            hien = thu
        else:
            dong.append(hien)
            hien = t
    dong.append(hien)
    return dong


def noi_dung_bang(buoc, kq=None, so_do=None, trang_thai="", tu_chup=True):
    NAU, XAM = (16, 24, 43), (95, 105, 128)
    d = [(f"Bước {THU_TU.index(buoc)+1}: {BUOC[buoc]['ten']}", DO, 26), (BUOC[buoc]["huong_dan"], NAU, 17)]
    if trang_thai:
        d.append((trang_thai, XAM, 16))
    if kq:
        sao = so_sao(kq)
        d.append(("★" * sao + "☆" * (3 - sao) if FONT else f"{sao}/3 sao", (54, 138, 200), 30))
        for s in so_do:
            d.append(("• " + s, XAM, 15))
        for ok, ma in kq:
            d.append((("✓ " if ok else "! ") + LOI_NHAN[ma], XANH if ok else (30, 110, 190), 17))
    d.append(("", NAU, 8))
    d.append(("Phím: 1-4 chọn bước · SPACE chụp · A tự chụp " + ("(bật)" if tu_chup else "(tắt)")
              + " · M xem mặt nạ · S lưu · Q thoát", XAM, 14))
    return d


# ----------------------------------------------------------------------
# 5. CHẾ ĐỘ ẢNH
# ----------------------------------------------------------------------
def doc_anh(p):
    anh = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)   # đọc được cả đường dẫn có dấu
    if anh is None:
        sys.exit(f"❌ Không đọc được ảnh: {p}")
    return anh


def in_ket_qua(ten, buoc, kq, so_do):
    sao = so_sao(kq)
    print(f"\n[{ten}] Bước: {BUOC[buoc]['ten']}  →  {sao}/3 sao")
    for s in so_do:
        print("   ·", s)
    for ok, ma in kq:
        print("   ", "✓" if ok else "!", LOI_NHAN[ma])


def che_do_anh(duong_dan, buoc, anh_than, hien_cua_so=True):
    D = None
    if anh_than and buoc != "than":
        khoi, _, _, rong = tim_cuc_bot(doc_anh(anh_than))
        _, _, D = cham_diem("than", khoi, rong, None)
        print(f"Đã đo thân từ {anh_than}: đường kính = {D:.0f} (phần nghìn chiều rộng ảnh)" if D else "Không đo được thân.")
    anh = doc_anh(duong_dan)
    khoi, mat_na, k, rong = tim_cuc_bot(anh)
    kq, so_do, _ = cham_diem(buoc, khoi, rong, D)
    in_ket_qua(Path(duong_dan).name, buoc, kq, so_do)
    hinh = ve_bang(ve_lop_phu(anh, khoi, k, buoc, D, rong), noi_dung_bang(buoc, kq, so_do, tu_chup=False))
    ra = THU_MUC / "ket_qua"
    ra.mkdir(exist_ok=True)
    file_ra = ra / f"{Path(duong_dan).stem}_ket_qua.jpg"
    cv2.imencode(".jpg", hinh)[1].tofile(str(file_ra))
    print("   → Đã lưu ảnh kết quả:", file_ra.relative_to(THU_MUC))
    if hien_cua_so:
        cv2.imshow("Nan heo - CV", thu_nho_vua_man_hinh(hinh))
        print("   (Bấm phím bất kỳ trên cửa sổ ảnh để đóng)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def thu_het(hien_cua_so=False):
    """Chấm thử toàn bộ ảnh mẫu. Tên ảnh bắt đầu bằng tên bước: than_..., tai_..., mom_..., duoi_..."""
    mau = THU_MUC / "anh_mau"
    than_chuan = mau / "than_tot.jpg"
    for p in sorted(mau.glob("*.jpg")):
        buoc = p.stem.split("_")[0]
        if buoc not in BUOC:
            buoc = "than"
        che_do_anh(p, buoc, than_chuan if buoc != "than" else None, hien_cua_so)


def thu_nho_vua_man_hinh(hinh, toi_da=1400):
    h, w = hinh.shape[:2]
    k = min(1.0, toi_da / w)
    return cv2.resize(hinh, (int(w * k), int(h * k))) if k < 1 else hinh


# ----------------------------------------------------------------------
# 6. CHẾ ĐỘ CAMERA
# ----------------------------------------------------------------------
def che_do_camera(so_cam):
    cam = cv2.VideoCapture(so_cam)
    if not cam.isOpened():
        sys.exit("❌ Không mở được camera. Trên Mac: System Settings → Privacy & Security → Camera → "
                 "bật cho Visual Studio Code (hoặc Terminal), rồi mở lại VS Code. "
                 "Nếu máy có nhiều camera, thử: python nan_heo_cv.py --cam 1")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    buoc, D, tu_chup, xem_mat_na = "than", None, True, False
    on_dinh, truoc, dong_bang, ket_qua = 0, None, None, None
    ra = THU_MUC / "ket_qua"
    ra.mkdir(exist_ok=True)
    print("Camera đã mở. Bấm vào cửa sổ camera rồi dùng phím: 1-4, SPACE, A, M, S, Q")

    while True:
        if dong_bang is None:
            ok, khung = cam.read()
            if not ok:
                print("❌ Không đọc được hình từ camera.")
                break
            khoi, mat_na, k, rong = tim_cuc_bot(khung, 240)
            chon = chon_khoi(khoi, buoc)
            # tự chụp: bột nằm yên ~1,3 giây (trọng tâm & diện tích gần như không đổi)
            if chon:
                b = chon[0]
                khoa = (b["cx"] / rong, b["cy"] / rong, b["dien_tich"], len(chon))
                if truoc and abs(khoa[0] - truoc[0]) < 0.025 and abs(khoa[1] - truoc[1]) < 0.025 \
                        and abs(khoa[2] - truoc[2]) / truoc[2] < 0.12 and khoa[3] == truoc[3]:
                    on_dinh += 1
                else:
                    on_dinh = 0
                truoc = khoa
                trang_thai = "Thấy bột! Giữ yên để tự chụp…" if tu_chup else "Thấy bột! Bấm SPACE để chụp"
            else:
                on_dinh, truoc = 0, None
                trang_thai = "Đặt bột vào giữa khung, tay ra ngoài" if khoi else "Đang tìm cục bột…"
            hien = ve_lop_phu(khung, khoi, k, buoc, D, rong)
            if xem_mat_na:
                mn = cv2.resize(mat_na, (khung.shape[1], khung.shape[0]), interpolation=cv2.INTER_NEAREST)
                hien = cv2.cvtColor(mn, cv2.COLOR_GRAY2BGR)
            if chon and tu_chup:
                tien_do = min(1.0, on_dinh / 10)
                b = chon[0]
                cv2.ellipse(hien, (int(b["cx"] / k), int(b["cy"] / k)),
                            (int(b["truc_dai"] / 2 / k) + 18,) * 2, -90, 0, 360 * tien_do, TRANG, 5, cv2.LINE_AA)
            man_hinh = ve_bang(hien, noi_dung_bang(buoc, trang_thai=trang_thai, tu_chup=tu_chup))
            chup = tu_chup and chon and on_dinh >= 10
        else:
            man_hinh, chup = dong_bang, False

        cv2.imshow("Nan heo - CV", thu_nho_vua_man_hinh(man_hinh))
        phim = cv2.waitKey(30) & 0xFF

        if phim in (ord("q"), 27):
            break
        if phim in (ord("1"), ord("2"), ord("3"), ord("4")):
            buoc = THU_TU[phim - ord("1")]
            dong_bang, on_dinh, truoc = None, 0, None
            print(f"\n→ Chuyển sang bước {BUOC[buoc]['ten']}")
        elif phim == ord("a"):
            tu_chup = not tu_chup
        elif phim == ord("m"):
            xem_mat_na = not xem_mat_na
        elif phim == ord("s") and dong_bang is not None:
            p = ra / f"{buoc}_{time.strftime('%H%M%S')}.jpg"
            cv2.imencode(".jpg", dong_bang)[1].tofile(str(p))
            print("   → Đã lưu", p.relative_to(THU_MUC))
        elif phim == 32 and dong_bang is not None:      # SPACE khi đang xem kết quả → chụp lại
            dong_bang, on_dinh, truoc = None, 0, None
            continue

        if (phim == 32 or chup) and dong_bang is None:
            khoi, _, k, rong = tim_cuc_bot(khung, 420)
            kq, so_do, D_moi = cham_diem(buoc, khoi, rong, D)
            if buoc == "than" and D_moi:
                D = D_moi
            in_ket_qua("camera", buoc, kq, so_do)
            dong_bang = ve_bang(ve_lop_phu(khung, khoi, k, buoc, D, rong),
                                noi_dung_bang(buoc, kq, so_do, "Đã chụp. SPACE = chụp lại · phím 1-4 = bước khác", tu_chup))

    cam.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Nặn heo hồng – nhận diện ảnh (CV)")
    ap.add_argument("anh", nargs="?", help="đường dẫn ảnh (bỏ trống = dùng camera)")
    ap.add_argument("--buoc", default="than", choices=THU_TU, help="than / tai / mom / duoi")
    ap.add_argument("--than", help="ảnh thân heo để làm thước đo kích thước")
    ap.add_argument("--cam", type=int, default=0, help="số thứ tự camera (0, 1, 2…)")
    ap.add_argument("--thu-het", action="store_true", help="chấm thử tất cả ảnh trong anh_mau/")
    ap.add_argument("--khong-hien", action="store_true", help="không mở cửa sổ ảnh (chỉ in kết quả)")
    a = ap.parse_args()
    if a.thu_het:
        thu_het(hien_cua_so=not a.khong_hien)
    elif a.anh:
        che_do_anh(a.anh, a.buoc, a.than, hien_cua_so=not a.khong_hien)
    else:
        che_do_camera(a.cam)
