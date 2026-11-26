"""
THEO DÕI TRẺ NẶN TÒ HE – NHẬN DIỆN BÀN TAY + KHUNG THAM CHIẾU TỪNG BỘ PHẬN (OpenCV)
=====================================================================================
Ý tưởng: trước khi theo dõi, máy "học" nhanh 3 thứ bằng cách chụp / quét:
  ① NỀN    – chụp tấm lót trống                 → biết đâu là nền
  ② TAY    – quét bàn tay của trẻ 2 giây          → biết màu da của chính trẻ (ánh sáng nhà mình)
  ③ BỘ PHẬN – nặn xong bộ phận nào, đặt riêng lên tấm lót và chụp
              → lưu "khung tham chiếu": màu (biểu đồ màu), kích thước, độ dẹt, hình dáng
Sau đó ở chế độ THEO DÕI, mỗi khung hình:
  • tách TAY (màu da đã quét) và BỘT (khác nền, không phải da)
  • gán nhãn từng cục bột theo khung tham chiếu: "Thân 86%", "Tai 74%"…
  • phát hiện bộ phận ĐANG ĐƯỢC NẶN: cục bột chạm tay, hoặc bột lộ ra giữa các ngón tay,
    hoặc bộ phận vừa biến khỏi tấm lót khi tay đang ở đó ("đang cầm trên tay")
  • so màu, kích thước với mẫu của nhân vật → nhắc trẻ

Nhân vật có sẵn: heo (con lợn), hoa (bông hoa), trau (con trâu), ga (con gà).

CHẠY:
  python theo_doi_nan.py --nv heo            camera, nhân vật con lợn
  python theo_doi_nan.py --nv ga --cam 1     con gà, camera số 1
  python theo_doi_nan.py --thu               chạy thử bằng cảnh giả lập (không cần camera)

PHÍM:  B chụp nền · H quét tay · 1–6 chụp tham chiếu bộ phận · N đổi nhân vật
       M xem mặt nạ (xanh = tay, hồng = bột) · R xoá tham chiếu · S lưu ảnh · Q thoát
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from nan_heo_cv import ve_bang, thu_nho_vua_man_hinh   # bảng chữ tiếng Việt dùng chung

THU_MUC = Path(__file__).resolve().parent
RONG = 480                      # chiều rộng ảnh khi xử lý (nhỏ cho nhanh)

# ----------------------------------------------------------------------------
# 1. NHÂN VẬT VÀ BỘ PHẬN
#    mau: màu cần có · ty_le: kích thước so với bộ phận chính (bộ phận đầu tiên)
#    det: độ dẹt mong muốn (trục dài/trục ngắn) · so: số lượng
# ----------------------------------------------------------------------------
NHAN_VAT = {
    "heo": {"ten": "Con lợn", "bo_phan": [
        {"ma": "than", "ten": "Thân", "mau": ["hong"], "ty_le": 1.0, "det": 1.0, "so": 1},
        {"ma": "tai", "ten": "Tai", "mau": ["hong"], "ty_le": 0.43, "det": 1.3, "so": 2},
        {"ma": "mom", "ten": "Mõm", "mau": ["hong", "hong_nhat"], "ty_le": 0.46, "det": 1.35, "so": 1},
        {"ma": "duoi", "ten": "Đuôi", "mau": ["hong"], "ty_le": 0.9, "det": 4.0, "so": 1}]},
    "hoa": {"ten": "Bông hoa", "bo_phan": [
        {"ma": "canh", "ten": "Cánh hoa", "mau": ["do", "hong"], "ty_le": 1.0, "det": 1.5, "so": 5},
        {"ma": "nhuy", "ten": "Nhuỵ", "mau": ["vang"], "ty_le": 0.55, "det": 1.0, "so": 1},
        {"ma": "la", "ten": "Lá", "mau": ["xanh_la"], "ty_le": 1.2, "det": 2.2, "so": 2},
        {"ma": "cuong", "ten": "Cuống", "mau": ["xanh_la"], "ty_le": 2.2, "det": 6.0, "so": 1}]},
    "trau": {"ten": "Con trâu", "bo_phan": [
        {"ma": "than", "ten": "Thân", "mau": ["den", "xam"], "ty_le": 1.0, "det": 1.5, "so": 1},
        {"ma": "dau", "ten": "Đầu", "mau": ["den", "xam"], "ty_le": 0.55, "det": 1.2, "so": 1},
        {"ma": "chan", "ten": "Chân", "mau": ["den", "xam"], "ty_le": 0.45, "det": 2.5, "so": 4},
        {"ma": "sung", "ten": "Sừng", "mau": ["trang", "xam"], "ty_le": 0.4, "det": 3.0, "so": 2},
        {"ma": "duoi", "ten": "Đuôi", "mau": ["den", "xam"], "ty_le": 0.5, "det": 5.0, "so": 1}]},
    "ga": {"ten": "Con gà", "bo_phan": [
        {"ma": "than", "ten": "Thân", "mau": ["vang", "nau", "cam"], "ty_le": 1.0, "det": 1.3, "so": 1},
        {"ma": "dau", "ten": "Đầu", "mau": ["vang", "nau", "cam"], "ty_le": 0.45, "det": 1.0, "so": 1},
        {"ma": "mao", "ten": "Mào", "mau": ["do"], "ty_le": 0.3, "det": 1.6, "so": 1},
        {"ma": "mo", "ten": "Mỏ", "mau": ["cam", "vang"], "ty_le": 0.18, "det": 1.6, "so": 1},
        {"ma": "duoi", "ten": "Đuôi lông", "mau": ["do", "xanh_la", "xanh_duong", "tim"], "ty_le": 0.7, "det": 2.5, "so": 1},
        {"ma": "chan", "ten": "Chân", "mau": ["cam", "vang"], "ty_le": 0.35, "det": 3.0, "so": 2}]},
}
TEN_MAU = {"hong": "hồng", "hong_nhat": "hồng nhạt", "do": "đỏ", "cam": "cam", "vang": "vàng", "nau": "nâu",
           "xanh_la": "xanh lá", "xanh_duong": "xanh dương", "tim": "tím", "den": "đen", "xam": "xám", "trang": "trắng"}


def goi_ten_mau(bgr):
    """Gọi tên màu từ màu trung bình (đã cân bằng trắng)."""
    b, g, r = (c / 255 for c in bgr)
    mx, mn = max(r, g, b), min(r, g, b)
    l, d = (mx + mn) / 2, mx - mn
    s = 0 if d < 1e-6 else (d / (2 - mx - mn) if l > 0.5 else d / (mx + mn))
    if d < 1e-6:
        h = 0
    elif mx == r:
        h = ((g - b) / d) % 6 * 60
    elif mx == g:
        h = ((b - r) / d + 2) * 60
    else:
        h = ((r - g) / d + 4) * 60
    if l < 0.2:
        return "den"
    if s < 0.15:
        return "trang" if l > 0.75 else "xam"
    if (h >= 330 or h < 15) and l > 0.62:
        return "hong_nhat" if l > 0.82 else "hong"
    if h >= 300 and l > 0.5:
        return "hong"
    if h >= 330 or h < 15:
        return "do"
    if h < 45:
        return "nau" if l < 0.42 else "cam"
    if h < 70:
        return "vang"
    if h < 170:
        return "xanh_la"
    if h < 255:
        return "xanh_duong"
    return "tim"


# ----------------------------------------------------------------------------
# 2. KHUNG THAM CHIẾU (lưu ra đĩa để lần sau dùng lại)
# ----------------------------------------------------------------------------
def hist_hs(hsv, mask):
    h = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 255, cv2.NORM_MINMAX)
    return h


class ThamChieu:
    def __init__(self, nv):
        self.nv = nv
        self.nen = None            # ảnh nền (BGR, đã thu nhỏ)
        self.tay = None            # biểu đồ màu da tay
        self.bp = {}               # mã bộ phận -> {hist, dien_tich, det, hu, bgr, so}

    @property
    def thu_muc(self):
        return THU_MUC / "tham_chieu" / self.nv

    def luu(self):
        self.thu_muc.mkdir(parents=True, exist_ok=True)
        if self.nen is not None:
            cv2.imencode(".png", self.nen)[1].tofile(str(self.thu_muc / "nen.png"))
        du_lieu = {"tay": self.tay.tolist() if self.tay is not None else None,
                   "bo_phan": {k: {**v, "hist": v["hist"].tolist()} for k, v in self.bp.items()}}
        (self.thu_muc / "tham_chieu.json").write_text(json.dumps(du_lieu), encoding="utf-8")

    def nap(self):
        f = self.thu_muc / "tham_chieu.json"
        if not f.exists():
            return False
        d = json.loads(f.read_text(encoding="utf-8"))
        self.tay = np.array(d["tay"], np.float32) if d.get("tay") else None
        self.bp = {k: {**v, "hist": np.array(v["hist"], np.float32)} for k, v in d["bo_phan"].items()}
        p = self.thu_muc / "nen.png"
        if p.exists():
            self.nen = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        return True


# ----------------------------------------------------------------------------
# 3. TÁCH TAY VÀ BỘT
# ----------------------------------------------------------------------------
def thu_nho(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (RONG, round(h * RONG / w)), interpolation=cv2.INTER_AREA)


def tien_canh(anh, nen):
    """Những điểm ảnh khác nền (Lab: bỏ bớt ảnh hưởng của bóng đổ)."""
    a = cv2.cvtColor(cv2.GaussianBlur(anh, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(nen, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    dL = a[..., 0] - b[..., 0]
    kc = np.sqrt((np.where(dL < 0, 0.35, 0.7) * dL) ** 2 + 2 * (a[..., 1] - b[..., 1]) ** 2 + 2 * (a[..., 2] - b[..., 2]) ** 2)
    m = (kc > 22).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def da_mac_dinh(anh):
    """Khi chưa quét tay: dùng khoảng màu da phổ biến (YCrCb)."""
    ycc = cv2.cvtColor(anh, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycc, (40, 135, 85), (255, 175, 128))


def cac_vung(mask, dien_tich_min):
    so, nhan, tk, tam = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(i, tk[i], tam[i]) for i in range(1, so) if tk[i, 4] >= dien_tich_min], nhan


def tach_tay_bot(anh, tc):
    h, w = anh.shape[:2]
    N = h * w
    hsv = cv2.cvtColor(anh, cv2.COLOR_BGR2HSV)
    fg = tien_canh(anh, tc.nen)

    # mức "giống bột" = lớn nhất trong các bộ phận đã tham chiếu
    bp_bot = np.zeros((h, w), np.float32)
    for v in tc.bp.values():
        bp_bot = np.maximum(bp_bot, cv2.calcBackProject([hsv], [0, 1], v["hist"], [0, 180, 0, 256], 1).astype(np.float32))
    if tc.tay is not None:
        bp_tay = cv2.calcBackProject([hsv], [0, 1], tc.tay, [0, 180, 0, 256], 1).astype(np.float32)
    else:
        bp_tay = da_mac_dinh(anh).astype(np.float32)
    bp_tay = cv2.GaussianBlur(bp_tay, (9, 9), 0)
    bp_bot = cv2.GaussianBlur(bp_bot, (9, 9), 0)

    tay = ((fg > 0) & (bp_tay > 25) & (bp_tay > bp_bot * 1.15)).astype(np.uint8) * 255
    tay = cv2.morphologyEx(tay, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tay = cv2.morphologyEx(tay, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    vung_tay, nhan = cac_vung(tay, N * 0.012)
    tay_sach = np.zeros_like(tay)
    for i, _, _ in vung_tay:
        tay_sach[nhan == i] = 255

    # vùng tiền cảnh lớn chạm mép ảnh mà không giống bột → cũng là tay/cánh tay
    vung_fg, nhan_fg = cac_vung(fg, N * 0.02)
    for i, st, _ in vung_fg:
        x, y, bw, bh, _ = st
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            vm = nhan_fg == i
            if bp_tay[vm].mean() >= bp_bot[vm].mean():
                tay_sach[vm & (bp_tay > bp_bot)] = 255

    tay_no = cv2.dilate(tay_sach, np.ones((7, 7), np.uint8))
    bot = ((fg > 0) & (tay_no == 0)).astype(np.uint8) * 255
    bot = cv2.morphologyEx(bot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # bột lộ ra GIỮA các ngón tay (nằm trong bao lồi của bàn tay, giống bột hơn giống da)
    trong_tay = np.zeros_like(bot)
    cac_tay = []
    for c in cv2.findContours(tay_sach, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) < N * 0.012:
            continue
        bao = cv2.convexHull(c)
        cac_tay.append({"vien": c, "bao": bao})
        mbao = np.zeros_like(bot)
        cv2.drawContours(mbao, [bao], -1, 255, -1)
        trong_tay |= ((mbao > 0) & (fg > 0) & (bp_bot > bp_tay) & (bp_bot > 30)).astype(np.uint8) * 255
    return fg, tay_sach, bot, trong_tay, cac_tay, hsv


# ----------------------------------------------------------------------------
# 4. ĐẶC TRƯNG MỘT CỤC BỘT VÀ SO VỚI THAM CHIẾU
# ----------------------------------------------------------------------------
def dac_trung(mask, anh, hsv):
    ys, xs = np.nonzero(mask)
    n = len(xs)
    cx, cy = xs.mean(), ys.mean()
    cxx, cyy, cxy = ((xs - cx) ** 2).mean(), ((ys - cy) ** 2).mean(), ((xs - cx) * (ys - cy)).mean()
    tr, dt = cxx + cyy, cxx * cyy - cxy * cxy
    disc = math.sqrt(max(0.0, tr * tr / 4 - dt))
    l1, l2 = tr / 2 + disc, max(1e-6, tr / 2 - disc)
    hu = cv2.HuMoments(cv2.moments(mask, True)).flatten()
    hu = [float(-np.sign(v) * np.log10(abs(v) + 1e-30)) for v in hu[:4]]
    bgr = [float(v) for v in anh[mask > 0].mean(axis=0)]
    return {"dien_tich": int(n), "cx": float(cx), "cy": float(cy), "dai": 4 * math.sqrt(l1), "ngan": 4 * math.sqrt(l2),
            "goc": 0.5 * math.atan2(2 * cxy, cxx - cyy), "det": math.sqrt(l1 / l2), "hu": hu, "bgr": bgr,
            "mau": goi_ten_mau(bgr), "hist": hist_hs(hsv, mask)}


def nhan_dien(dt, tc, nv):
    """So một cục bột với khung tham chiếu: màu 50% · kích thước 30% · hình dáng 20%."""
    ket = []
    for bp in NHAN_VAT[nv]["bo_phan"]:
        ref = tc.bp.get(bp["ma"])
        if not ref:
            continue
        mau = 1 - cv2.compareHist(dt["hist"], ref["hist"], cv2.HISTCMP_BHATTACHARYYA)
        co = math.exp(-abs(math.log(dt["dien_tich"] / ref["dien_tich"])))
        dang = math.exp(-abs(math.log(dt["det"] / ref["det"]))) * math.exp(-0.5 * sum(abs(a - b) for a, b in zip(dt["hu"][:2], ref["hu"][:2])))
        ket.append((0.5 * mau + 0.3 * co + 0.2 * dang, bp))
    ket.sort(key=lambda x: -x[0])
    return ket[0] if ket and ket[0][0] >= 0.45 else (ket[0][0] if ket else 0, None)


# ----------------------------------------------------------------------------
# 5. ỨNG DỤNG
# ----------------------------------------------------------------------------
class UngDung:
    def __init__(self, nv="heo"):
        self.doi_nv(nv)
        self.xem_mat_na = False
        self.vet = {}              # theo dõi các cục bột giữa các khung hình
        self.id_tiep = 1
        self.thong_bao = ""

    def doi_nv(self, nv):
        self.nv = nv
        self.tc = ThamChieu(nv)
        self.tc.nap()
        self.vet = {}

    # ---- các bước chụp / quét -------------------------------------------------
    def chup_nen(self, frame):
        self.tc.nen = thu_nho(frame)
        self.tc.luu()
        self.thong_bao = "① Đã chụp nền."

    def quet_tay(self, cac_khung):
        """Gom màu da từ nhiều khung: chỉ lấy vùng khác nền nằm giữa khung hình."""
        if self.tc.nen is None:
            self.thong_bao = "Chụp nền (phím B) trước khi quét tay."
            return False
        tong = None
        for f in cac_khung:
            a = thu_nho(f)
            h, w = a.shape[:2]
            m = tien_canh(a, self.tc.nen)
            khung = np.zeros_like(m)
            cv2.rectangle(khung, (int(w * .25), int(h * .2)), (int(w * .75), int(h * .95)), 255, -1)
            m &= khung
            if cv2.countNonZero(m) < h * w * 0.02:
                continue
            hh = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2HSV)], [0, 1], m, [30, 32], [0, 180, 0, 256])
            tong = hh if tong is None else tong + hh
        if tong is None:
            self.thong_bao = "Chưa thấy tay trong khung giữa. Đặt cả bàn tay vào giữa rồi quét lại."
            return False
        cv2.normalize(tong, tong, 0, 255, cv2.NORM_MINMAX)
        self.tc.tay = tong
        self.tc.luu()
        self.thong_bao = "② Đã quét màu da tay."
        return True

    def dang_ky(self, chi_so, frame):
        """Chụp tham chiếu cho bộ phận thứ chi_so: đặt RIÊNG bộ phận đó (đủ số lượng) lên tấm lót."""
        ds = NHAN_VAT[self.nv]["bo_phan"]
        if chi_so >= len(ds):
            return
        bp = ds[chi_so]
        if self.tc.nen is None:
            self.thong_bao = "Chụp nền (phím B) trước."
            return
        a = thu_nho(frame)
        hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        bot = tien_canh(a, self.tc.nen)            # lúc chụp tham chiếu chỉ có bộ phận trên tấm lót, tay để ngoài
        vung, nhan = cac_vung(bot, a.shape[0] * a.shape[1] * 0.0015)
        hw = a.shape[:2]
        vung = [v for v in vung if not (v[1][0] <= 1 or v[1][1] <= 1 or v[1][0] + v[1][2] >= hw[1] - 1 or v[1][1] + v[1][3] >= hw[0] - 1)]
        vung.sort(key=lambda v: -v[1][4])
        vung = vung[:bp["so"]]
        if not vung:
            self.thong_bao = f"Chưa thấy {bp['ten']} trên tấm lót."
            return
        gop = np.zeros_like(bot)
        cac_dt = []
        for i, _, _ in vung:
            m = (nhan == i).astype(np.uint8) * 255
            cac_dt.append(dac_trung(m, a, hsv))
            gop |= m
        tb = lambda k: float(np.mean([d[k] for d in cac_dt]))
        ht = hist_hs(hsv, gop)
        self.tc.bp[bp["ma"]] = {"hist": ht, "dien_tich": tb("dien_tich"), "det": tb("det"),
                                "hu": [float(np.mean([d["hu"][j] for d in cac_dt])) for j in range(4)],
                                "bgr": [float(np.mean([d["bgr"][j] for d in cac_dt])) for j in range(3)],
                                "so": len(cac_dt), "dai": tb("dai")}
        self.tc.luu()
        cv2.imencode(".jpg", frame)[1].tofile(str(self.tc.thu_muc / f"{bp['ma']}.jpg"))
        self.thong_bao = f"③ Đã lưu tham chiếu {bp['ten']} ({len(cac_dt)} cục). " + " ".join(self.nhan_xet(bp, cac_dt))

    def nhan_xet(self, bp, cac_dt):
        """So với mẫu của nhân vật: màu, số lượng, kích thước so với bộ phận chính."""
        ra = []
        mau = Counter(d["mau"] for d in cac_dt).most_common(1)[0][0]
        if mau not in bp["mau"]:
            ra.append(f"Màu đang là {TEN_MAU[mau]}, mẫu cần {' / '.join(TEN_MAU[m] for m in bp['mau'])}.")
        if len(cac_dt) < bp["so"]:
            ra.append(f"Cần {bp['so']} {bp['ten'].lower()}, mới thấy {len(cac_dt)}.")
        chinh = NHAN_VAT[self.nv]["bo_phan"][0]
        ref = self.tc.bp.get(chinh["ma"])
        if ref and bp is not chinh:
            r = np.mean([d["dai"] for d in cac_dt]) / ref["dai"]
            if r > bp["ty_le"] * 1.4:
                ra.append(f"Hơi to so với {chinh['ten'].lower()}.")
            elif r < bp["ty_le"] * 0.65:
                ra.append(f"Hơi nhỏ so với {chinh['ten'].lower()}.")
        return ra or ["Đúng mẫu!"]

    # ---- theo dõi ------------------------------------------------------------
    def xu_ly(self, frame):
        a = thu_nho(frame)
        k = frame.shape[1] / RONG
        if self.tc.nen is None:
            return self.ve(frame, k, [], [], None, None), {"tay": 0, "bot": [], "dang_nan": None}
        fg, tay, bot, trong_tay, cac_tay, hsv = tach_tay_bot(a, self.tc)
        N = a.shape[0] * a.shape[1]
        tay_gan = cv2.dilate(tay, np.ones((21, 21), np.uint8))

        cuc = []
        vung, nhan = cac_vung(bot, N * 0.0015)
        for i, st, _ in vung:
            m = (nhan == i).astype(np.uint8) * 255
            dt = dac_trung(m, a, hsv)
            if any(cv2.pointPolygonTest(t["bao"], (dt["cx"], dt["cy"]), False) >= 0 for t in cac_tay):
                continue                      # nằm trong lòng bàn tay → là bột đang cầm, không phải trên tấm lót
            diem, bp = nhan_dien(dt, self.tc, self.nv)
            dt.update({"bp": bp, "diem": diem, "cham_tay": cv2.countNonZero(m & tay_gan) > 0, "mask": m})
            cuc.append(dt)

        # bột trong tay
        trong = None
        if cv2.countNonZero(trong_tay) > N * 0.002:
            dt = dac_trung(trong_tay, a, hsv)
            diem, bp = nhan_dien(dt, self.tc, self.nv)
            trong = {**dt, "bp": bp, "diem": diem}

        self.cap_nhat_vet(cuc, cac_tay, a.shape)
        dang_nan = self.chon_dang_nan(cuc, trong, cac_tay)
        return self.ve(frame, k, cuc, cac_tay, trong, dang_nan, fg=fg, tay=tay, bot=bot), \
            {"tay": len(cac_tay), "bot": [(c["bp"]["ten"] if c["bp"] else "?") for c in cuc],
             "dang_nan": dang_nan}

    def cap_nhat_vet(self, cuc, cac_tay, shape):
        bay_gio = time.time()
        dung = set()
        for c in cuc:
            tot, gan = None, 1e9
            for vid, v in self.vet.items():
                d = math.hypot(c["cx"] - v["cx"], c["cy"] - v["cy"])
                ti = c["dien_tich"] / max(1, v.get("dien_tich", c["dien_tich"]))
                if d < gan and d < 45 and 0.6 < ti < 1.67 and vid not in dung:
                    tot, gan = vid, d
            if tot is None:
                tot = self.id_tiep
                self.id_tiep += 1
                self.vet[tot] = {"nhan": deque(maxlen=12)}
            dung.add(tot)
            v = self.vet[tot]
            v.update(cx=c["cx"], cy=c["cy"], luc=bay_gio, dien_tich=c["dien_tich"])
            if c["bp"]:
                v["nhan"].append(c["bp"]["ten"])
            c["id"] = tot
            if v["nhan"] and c["diem"] < 0.6:   # nhận chưa chắc → dùng nhãn xuất hiện nhiều nhất gần đây
                ten = Counter(v["nhan"]).most_common(1)[0][0]
                c["bp"] = next(b for b in NHAN_VAT[self.nv]["bo_phan"] if b["ten"] == ten)
        # cục vừa biến mất gần tay → đang được cầm lên nặn
        for vid, v in list(self.vet.items()):
            if vid in dung:
                v.pop("tren_tay", None)
                continue
            gan_tay = any(cv2.pointPolygonTest(t["bao"], (float(v["cx"]), float(v["cy"])), True) > -80 for t in cac_tay)
            if gan_tay and v["nhan"]:
                v["tren_tay"] = bay_gio
            if bay_gio - v["luc"] > 6:
                del self.vet[vid]

    def chon_dang_nan(self, cuc, trong, cac_tay):
        if not cac_tay:
            return None
        cam = [v for v in self.vet.values() if "tren_tay" in v]
        if cam:                      # bộ phận vừa rời tấm lót ngay chỗ bàn tay → đang được cầm nặn
            v = max(cam, key=lambda v: v["tren_tay"])
            ten = Counter(v["nhan"]).most_common(1)[0][0]
            return {"bp": next(b for b in NHAN_VAT[self.nv]["bo_phan"] if b["ten"] == ten),
                    "kieu": "đang cầm trên tay", "mau": trong["mau"] if trong else None}
        if trong and trong["bp"]:
            return {"bp": trong["bp"], "kieu": "bột lộ giữa các ngón tay", "mau": trong["mau"]}
        cham = [c for c in cuc if c["cham_tay"]]
        if cham:
            c = max(cham, key=lambda c: c["dien_tich"])
            return {"bp": c["bp"], "kieu": "đang chạm tay", "mau": c["mau"]}
        return {"bp": None, "kieu": "tay trống", "mau": None}

    # ---- vẽ ------------------------------------------------------------------
    def ve(self, frame, k, cuc, cac_tay, trong, dang_nan, fg=None, tay=None, bot=None):
        out = frame.copy()
        if self.xem_mat_na and tay is not None:
            mn = np.zeros((*tay.shape, 3), np.uint8)
            mn[tay > 0] = (80, 200, 80)
            mn[bot > 0] = (180, 120, 255)
            out = cv2.resize(mn, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        for t in cac_tay:
            cv2.drawContours(out, [(t["vien"] * k).astype(np.int32)], -1, (80, 200, 80), 3, cv2.LINE_AA)
            cv2.drawContours(out, [(t["bao"] * k).astype(np.int32)], -1, (80, 200, 80), 1, cv2.LINE_AA)
        for c in cuc:
            mau = (60, 200, 255) if c["cham_tay"] else (68, 181, 242)
            cv2.ellipse(out, (int(c["cx"] * k), int(c["cy"] * k)), (int(c["dai"] / 2 * k), int(c["ngan"] / 2 * k)),
                        math.degrees(c["goc"]), 0, 360, mau, 4 if c["cham_tay"] else 2, cv2.LINE_AA)
        dong = [(f"{NHAN_VAT[self.nv]['ten']} · theo dõi nặn", (24, 35, 139), 24)]
        ok = lambda b: "✓" if b else "○"
        dong.append((f"{ok(self.tc.nen is not None)} B  Chụp nền trống", (40, 40, 40), 16))
        dong.append((f"{ok(self.tc.tay is not None)} H  Quét tay", (40, 40, 40), 16))
        for i, bp in enumerate(NHAN_VAT[self.nv]["bo_phan"]):
            dong.append((f"{ok(bp['ma'] in self.tc.bp)} {i+1}  Tham chiếu {bp['ten']} ×{bp['so']} ({'/'.join(TEN_MAU[m] for m in bp['mau'])})", (40, 40, 40), 15))
        if self.tc.nen is None:
            dong.append(("Bắt đầu: để tấm lót TRỐNG rồi bấm B.", (30, 110, 190), 17))
        if dang_nan:
            if dang_nan["bp"]:
                dong.append((f"▶ Con đang nặn: {dang_nan['bp']['ten']} ({dang_nan['kieu']})", (40, 120, 40), 20))
                if dang_nan["mau"] and dang_nan["mau"] not in dang_nan["bp"]["mau"]:
                    dong.append((f"Màu đang là {TEN_MAU[dang_nan['mau']]}, mẫu cần {TEN_MAU[dang_nan['bp']['mau'][0]]}", (30, 110, 190), 16))
            else:
                dong.append((f"▶ Thấy tay ({dang_nan['kieu']})", (40, 120, 40), 18))
        if cuc:
            dem = Counter((c["bp"]["ten"] if c["bp"] else "chưa rõ") for c in cuc)
            dong.append(("Trên tấm lót: " + ", ".join(f"{n} ×{s}" for n, s in dem.items()), (95, 105, 128), 15))
        if self.thong_bao:
            dong.append((self.thong_bao, (30, 110, 190), 15))
        dong.append(("B nền · H quét tay · 1–6 tham chiếu · N nhân vật · M mặt nạ · S lưu · Q thoát", (95, 105, 128), 13))
        # nhãn cạnh từng cục bột
        for c in cuc:
            nh = (c["bp"]["ten"] + f" {c['diem']*100:.0f}%") if c["bp"] else TEN_MAU[c["mau"]]
            ve_nhan(out, nh, int((c["cx"] - c["dai"] / 2) * k), int((c["cy"] - c["ngan"] / 2) * k) - 10)
        for t in cac_tay:
            x, y, _, _ = cv2.boundingRect(t["vien"])
            ve_nhan(out, "Tay", int(x * k), int(y * k) + 30, (60, 140, 60))
        return ve_bang(out, dong)


def ve_nhan(anh, chu, x, y, nen=(43, 24, 16)):
    from nan_heo_cv import FONT, bo_dau
    x, y = max(2, x), max(24, y)
    if FONT:
        from PIL import Image, ImageDraw, ImageFont
        ft = ImageFont.truetype(FONT, 20)
        im = Image.fromarray(cv2.cvtColor(anh, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(im)
        w = dr.textlength(chu, font=ft)
        dr.rounded_rectangle((x, y - 24, x + w + 16, y + 4), 12, fill=(nen[2], nen[1], nen[0]))
        dr.text((x + 8, y - 22), chu, font=ft, fill=(255, 255, 255))
        anh[:] = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(anh, bo_dau(chu), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ----------------------------------------------------------------------------
# 6. CAMERA
# ----------------------------------------------------------------------------
def chay_camera(nv, so_cam):
    cam = cv2.VideoCapture(so_cam)
    if not cam.isOpened():
        sys.exit("❌ Không mở được camera (xem HUONG_DAN_VSCODE.md: cấp quyền Camera cho VS Code, hoặc thử --cam 1).")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    app = UngDung(nv)
    quet = []                 # đang quét tay: gom khung hình trong 2 giây
    dang_quet_den = 0
    print(f"Nhân vật: {NHAN_VAT[nv]['ten']}. Bấm vào cửa sổ camera rồi dùng phím (B, H, 1–6, N, M, S, Q).")
    while True:
        ok, f = cam.read()
        if not ok:
            break
        if dang_quet_den:
            quet.append(f.copy())
            app.thong_bao = f"Đang quét tay… giữ yên bàn tay ở giữa ({max(0, dang_quet_den - time.time()):.1f}s)"
            if time.time() >= dang_quet_den:
                dang_quet_den = 0
                app.quet_tay(quet)
                quet = []
        hinh, tt = app.xu_ly(f)
        if not dang_quet_den and tt["dang_nan"] is None and app.tc.nen is not None and not app.thong_bao:
            pass
        cv2.imshow("Theo doi nan to he", thu_nho_vua_man_hinh(hinh))
        p = cv2.waitKey(1) & 0xFF
        if p in (ord("q"), 27):
            break
        if p == ord("b"):
            app.chup_nen(f)
        elif p == ord("h"):
            dang_quet_den, quet = time.time() + 2, []
        elif ord("1") <= p <= ord("6"):
            app.dang_ky(p - ord("1"), f)
            print(app.thong_bao)
        elif p == ord("n"):
            ds = list(NHAN_VAT)
            app.doi_nv(ds[(ds.index(app.nv) + 1) % len(ds)])
            app.thong_bao = f"Đổi sang {NHAN_VAT[app.nv]['ten']}"
        elif p == ord("m"):
            app.xem_mat_na = not app.xem_mat_na
        elif p == ord("r"):
            app.tc.bp, app.tc.tay = {}, None
            app.tc.luu()
            app.thong_bao = "Đã xoá tham chiếu tay và bộ phận."
        elif p == ord("s"):
            (THU_MUC / "ket_qua").mkdir(exist_ok=True)
            pth = THU_MUC / "ket_qua" / f"theo_doi_{app.nv}_{time.strftime('%H%M%S')}.jpg"
            cv2.imencode(".jpg", hinh)[1].tofile(str(pth))
            print("Đã lưu", pth.name)
    cam.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# 7. CHẠY THỬ BẰNG CẢNH GIẢ LẬP (không cần camera)
# ----------------------------------------------------------------------------
def canh_gia_lap():
    rng = np.random.default_rng(5)
    W, H = 1280, 720
    y, x = np.mgrid[0:H, 0:W]
    g = 1 - 0.1 * ((x - W * .3) ** 2 + (y - H * .2) ** 2) / (W * W)
    nen = np.stack([150 * g, 150 * g, 148 * g], -1)            # tấm lót xám (B, G, R)
    HONG, DA = (164, 148, 236), (140, 172, 222)                 # BGR: bột hồng, da tay

    def anh(cuc=(), tay=None):
        a = nen.copy()
        for (cx, cy, ax, ay, goc, mau) in cuc:
            m = np.zeros((H, W), np.uint8)
            cv2.ellipse(m, (cx, cy), (ax, ay), goc, 0, 360, 255, -1)
            m = cv2.GaussianBlur(m, (3, 3), 0)[..., None] / 255.0
            sang = 1 + 0.15 * np.clip(1 - ((x - cx + ax * .3) ** 2 / (ax * ax) + (y - cy + ay * .3) ** 2 / (ay * ay)), -1, 1)[..., None]
            a = a * (1 - m) + np.array(mau) * sang * m
        if tay is not None:
            tx, ty, cam_bot = tay                               # bàn tay đi vào từ mép dưới
            m = np.zeros((H, W), np.uint8)
            cv2.ellipse(m, (tx, ty), (95, 80), 0, 0, 360, 255, -1)              # lòng bàn tay
            cv2.rectangle(m, (tx - 70, ty), (tx + 70, H), 255, -1)                # cổ tay
            for i, dx in enumerate((-65, -22, 22, 62)):                          # bốn ngón
                cv2.ellipse(m, (tx + dx, ty - 100), (17, 60), 0, 0, 360, 255, -1)
            cv2.ellipse(m, (tx - 115, ty - 20), (18, 55), -50, 0, 360, 255, -1)  # ngón cái
            mm = cv2.GaussianBlur(m, (3, 3), 0)[..., None] / 255.0
            a = a * (1 - mm) + np.array(DA) * mm
            if cam_bot:                                          # bột hồng lộ giữa các ngón
                mb = np.zeros((H, W), np.uint8)
                cv2.ellipse(mb, (tx, ty - 40), (38, 30), 0, 0, 360, 255, -1)
                mb = (mb[..., None] / 255.0) * (m[..., None] > 0)
                a = a * (1 - mb) + np.array(cam_bot) * mb
        return np.clip(a + rng.normal(0, 3, a.shape), 0, 255).astype(np.uint8)

    THAN = (640, 380, 120, 116, 0, HONG)
    TAI = [(560, 380, 52, 40, 20, HONG), (720, 380, 50, 41, -20, HONG)]
    MOM = (640, 380, 55, 40, 10, HONG)
    DUOI = (640, 380, 110, 17, 30, HONG)
    return {
        "nen": anh(), "tay": anh(tay=(640, 520, None)),
        "than": anh([THAN]), "tai": anh(TAI), "mom": anh([MOM]), "duoi": anh([DUOI]),
        # theo dõi: thân + 2 tai trên tấm lót, tay đang chạm mõm
        "f1": anh([(360, 300, 120, 116, 0, HONG), (650, 220, 52, 40, 20, HONG), (780, 230, 50, 41, -20, HONG),
                   (930, 470, 55, 40, 10, HONG)], tay=(980, 640, None)),
        # tay đang cầm đuôi (bột lộ giữa ngón), trên tấm lót chỉ còn thân
        "f4": anh([(360, 300, 120, 116, 0, HONG), (650, 220, 52, 40, 20, HONG), (780, 230, 50, 41, -20, HONG),
                   (930, 300, 55, 40, 10, HONG)], tay=(1010, 640, HONG)),
        # không có tay
        "f3": anh([(360, 300, 120, 116, 0, HONG), (650, 220, 52, 40, 20, HONG), (780, 230, 50, 41, -20, HONG),
                   (930, 300, 55, 40, 10, HONG), (1000, 520, 110, 17, 30, HONG)]),
    }


def chay_thu():
    c = canh_gia_lap()
    app = UngDung("heo")
    app.tc.bp, app.tc.tay, app.tc.nen = {}, None, None
    ra = THU_MUC / "ket_qua"
    ra.mkdir(exist_ok=True)
    app.chup_nen(c["nen"])
    print(app.thong_bao)
    app.quet_tay([c["tay"]] * 3)
    print(app.thong_bao)
    for i, ma in enumerate(["than", "tai", "mom", "duoi"]):
        app.dang_ky(i, c[ma])
        print(app.thong_bao)
    for ten in ("f3", "f4", "f1"):
        for _ in range(3):                      # vài khung để nhãn ổn định
            hinh, tt = app.xu_ly(c[ten])
        dn = tt["dang_nan"]
        print(f"[{ten}] tay: {tt['tay']} · bột: {tt['bot']} · đang nặn: "
              f"{(dn['bp']['ten'] if dn and dn['bp'] else '-') + (' (' + dn['kieu'] + ')' if dn else '')}")
        cv2.imencode(".jpg", hinh)[1].tofile(str(ra / f"theo_doi_{ten}.jpg"))
    print("Ảnh kết quả: ket_qua/theo_doi_f3.jpg, f4, f1")


if __name__ == "__main__":
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Theo dõi trẻ nặn tò he: tay + khung tham chiếu bộ phận")
    ap.add_argument("--nv", default="heo", choices=list(NHAN_VAT), help="heo / hoa / trau / ga")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--thu", action="store_true", help="chạy thử bằng cảnh giả lập")
    a = ap.parse_args()
    chay_thu() if a.thu else chay_camera(a.nv, a.cam)
