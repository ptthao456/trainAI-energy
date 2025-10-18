"""
energy.py — Energy & CO₂ calculator với 2-pha đọc CSV:
1) Đọc header-only để dò đúng tên cột cần (Time/CPU/GPU).
2) Đọc lại file với usecols để CHỈ lấy 2–3 cột cần thiết → giảm I/O & RAM.

Giữ nguyên công thức tính như trước (trapezoid, Δt thực), và nhận start/end
do time_utils.py trả về (chuỗi "HH:MM:SS.mmm").
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import re
import math

import numpy as np
import pandas as pd

# ===== HẰNG SỐ =====
CI_G_PER_WH: float = 0.471  # g/Wh

# Regex/heuristics nhận diện cột
CPU_COL_PATTERNS = [r"^cpu\s*package\s*power\s*\[w\]$"]
GPU_COL_PATTERNS = [r"^gpu\s*power\s*\[w\]$"]
TIME_PATTERNS    = [r"^time$"]


# ====== TIỆN ÍCH ======
def _read_csv_safely(path: str, **kwargs) -> pd.DataFrame:
    """Đọc CSV với nhiều encoding; cho phép truyền nrows/usecols để dùng 2-pha."""
    encodings = ["utf-8-sig", "ISO-8859-1", "cp1252", "utf-16"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", **kwargs)
        except Exception as e:
            last_err = e
            continue
    try:
        return pd.read_csv(path, on_bad_lines="skip", **kwargs)
    except Exception as e:
        raise RuntimeError(f"Không thể đọc CSV. Lỗi cuối: {last_err or e}")


def _find_col(cols: Sequence[str], patterns, fallback_contains=None) -> Optional[str]:
    """Tìm cột bằng regex fullmatch; nếu không có thì fallback contains (case-insensitive)."""
    for col in cols:
        name = str(col).strip()
        for pat in patterns:
            if re.fullmatch(pat, name, flags=re.IGNORECASE):
                return col
    if fallback_contains:
        for col in cols:
            name = str(col).strip().lower()
            if fallback_contains.lower() in name:
                return col
    return None


def _normalize_time_string(s: str) -> Optional[str]:
    """Chuẩn 'HH:MM:SS.mmm' (ms 3 chữ số). Ví dụ '7:5:2.1' -> '07:05:02.100'."""
    if s is None:
        return None
    txt = str(s).strip()
    m = re.match(r"^\s*(\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.(\d{1,6}))?\s*$", txt)
    if not m:
        return None
    H, M, S, ms = m.groups()
    if ms is None:
        ms = "000"
    ms = (ms + "000")[:3]
    return f"{int(H):02d}:{int(M):02d}:{int(S):02d}.{ms}"


def _time_str_to_seconds(hhmmss_mmm: str) -> float:
    """'HH:MM:SS.mmm' -> giây-trong-ngày (float)."""
    hh, mm, rest = hhmmss_mmm.split(":")
    ss, ms = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def _make_time_seconds_column(df: pd.DataFrame, time_col: str) -> pd.Series:
    t_norm = df[time_col].astype(str).map(_normalize_time_string)
    t_sec = t_norm.dropna().map(_time_str_to_seconds)
    return t_sec


def _unwrap_seconds_across_midnight(t_sec: pd.Series) -> pd.Series:
    """Tạo trục thời gian tuyệt đối đơn điệu tăng dần (qua nửa đêm +86400)."""
    t_abs = []
    day_shift = 0.0
    prev = None
    for x in t_sec:
        if prev is not None and x < prev:
            day_shift += 86400.0
        t_abs.append(x + day_shift)
        prev = x
    return pd.Series(t_abs, index=t_sec.index, dtype=float)


def _to_float_watts(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", ".", regex=False)
    out = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return out.clip(lower=0.0)


def _energy_wh_trapezoid(power_w: np.ndarray | None, tabs_sec: np.ndarray) -> float:
    if power_w is None or power_w.size < 2:
        return 0.0
    dt = np.diff(tabs_sec)
    dt = np.where(dt > 0, dt, 0.0)
    p0 = power_w[:-1]
    p1 = power_w[1:]
    e = ((p0 + p1) * 0.5) * (dt / 3600.0)
    return float(e.sum())


# ====== KẾT QUẢ ======
@dataclass
class EnergyResult:
    cpu_wh: float
    gpu_wh: float
    total_wh: float
    co2_g: float
    start_idx: int
    end_idx: int
    duration_clock_s: float
    duration_sum_s: float


# ====== NGỮ CẢNH TÍNH NĂNG LƯỢNG ======
class EnergyContext:
    def __init__(self, df: pd.DataFrame, time_col: str, cpu_col: str, gpu_col: Optional[str]):
        self.df = df
        self.time_col = time_col
        self.cpu_col = cpu_col
        self.gpu_col = gpu_col
        # Lưu lại thông tin CSV để có thể reload
        self._path = getattr(self, "_path", None)
        self._keep_cols = getattr(self, "_keep_cols", None)

        # Các cột đã chuẩn hoá
        self.tsec = df["_tsec"].to_numpy(dtype=float)
        self.tabs = df["_tabs"].to_numpy(dtype=float)
        self.cpu_w = _to_float_watts(df[cpu_col]).to_numpy(dtype=float)
        self.gpu_w = _to_float_watts(df[gpu_col]).to_numpy(dtype=float) if (gpu_col and gpu_col in df.columns) else None

        self.n = self.tabs.size
        if self.n == 0:
            raise ValueError("CSV không có dữ liệu thời gian hợp lệ.")
        if self.n != self.cpu_w.size:
            raise ValueError("Kích thước thời gian và công suất CPU không khớp.")
        if self.gpu_w is not None and self.gpu_w.size != self.n:
            raise ValueError("Kích thước thời gian và công suất GPU không khớp.")

        # Con trỏ tìm kiếm để lần sau bắt đầu từ đây (giúp ổn định với nhiều cặp liên tiếp)
        self.search_cursor = 0

    # ---- Tạo từ CSV bằng 2-pha đọc ----
    @classmethod
    def from_csv(cls, path: str) -> "EnergyContext":
        # Pha 1: đọc header-only để dò cột
        df0 = _read_csv_safely(path, nrows=0)
        time_col = _find_col(df0.columns, TIME_PATTERNS, fallback_contains="time")
        if time_col is None:
            raise ValueError("Không tìm thấy cột 'Time' trong CSV.")
        cpu_col = _find_col(df0.columns, CPU_COL_PATTERNS, fallback_contains="cpu package power")
        if cpu_col is None:
            raise ValueError("Không tìm thấy cột 'CPU PACKAGE POWER [W]' trong CSV.")
        gpu_col = _find_col(df0.columns, GPU_COL_PATTERNS, fallback_contains="gpu power")

        keep_cols = [time_col, cpu_col] + ([gpu_col] if gpu_col else [])

        # Pha 2: đọc lại file, CHỈ giữ 2–3 cột cần thiết
        df = _read_csv_safely(path, usecols=keep_cols)

        # Chuẩn hoá thời gian & tạo trục tuyệt đối
        t_sec = _make_time_seconds_column(df, time_col)
        if t_sec.empty:
            raise ValueError("Không parse được cột 'Time' trong CSV.")
        df = df.loc[t_sec.index].copy()
        df["_tsec"] = t_sec
        df["_tabs"] = _unwrap_seconds_across_midnight(t_sec)

        # Sắp xếp theo _tabs (đề phòng bị đảo dòng)
        df = df.sort_values(by="_tabs", kind="mergesort", ignore_index=True)

        ctx = cls(df=df, time_col=time_col, cpu_col=cpu_col, gpu_col=gpu_col)
        ctx._path = path            # <— lưu path
        ctx._keep_cols = keep_cols  # <— lưu cột
        return ctx

    # ---- Core: tính năng lượng cho một cửa sổ ----
    def compute_between(self, start_str: str, end_str: str) -> EnergyResult:
        s_norm = _normalize_time_string(start_str)
        e_norm = _normalize_time_string(end_str)
        if not s_norm or not e_norm:
            raise ValueError("start/end không đúng định dạng 'HH:MM:SS.mmm'.")

        s_sec = _time_str_to_seconds(s_norm)
        e_sec = _time_str_to_seconds(e_norm)

        # duration theo đồng hồ (qua ngày nếu end < start)
        if e_sec >= s_sec:
            dur_clock = e_sec - s_sec
        else:
            dur_clock = (86400.0 - s_sec) + e_sec
        if dur_clock <= 0:
            dur_clock = 0.0

        tabs = self.tabs
        n = self.n

        # ---- Neo tuyệt đối quanh search_cursor ----
        cur_idx = min(max(self.search_cursor, 0), n - 1)
        cur_abs = tabs[cur_idx]
        cur_day = math.floor(cur_abs / 86400.0) * 86400.0

        start_abs = cur_day + s_sec
        EPS = 1e-6
        while start_abs + EPS < cur_abs and (start_abs + 86400.0) - cur_abs < 3 * 86400.0:
            start_abs += 86400.0
        end_abs = start_abs + dur_clock

        # ---- Tìm index biên bằng searchsorted ----
        i0 = int(np.searchsorted(tabs, start_abs, side="left"))
        i1 = int(np.searchsorted(tabs, end_abs, side="right") - 1)

        if i0 >= n:
            i0 = n - 1
        if i1 < i0:
            i1 = i0

        # ---- Năng lượng ----
        slice_tabs = tabs[i0:i1 + 1]
        cpu_slice = self.cpu_w[i0:i1 + 1]
        gpu_slice = self.gpu_w[i0:i1 + 1] if self.gpu_w is not None else None

        if slice_tabs.size >= 2:
            e_cpu_wh = _energy_wh_trapezoid(cpu_slice, slice_tabs)
            e_gpu_wh = _energy_wh_trapezoid(gpu_slice, slice_tabs) if gpu_slice is not None else 0.0
        else:
            # Fallback: hình chữ nhật với công suất tại i0 và duration_clock
            p_cpu = float(cpu_slice[0]) if cpu_slice.size > 0 else 0.0
            p_gpu = float(gpu_slice[0]) if (gpu_slice is not None and gpu_slice.size > 0) else 0.0
            e_cpu_wh = (p_cpu * (dur_clock / 3600.0)) if dur_clock > 0 else 0.0
            e_gpu_wh = (p_gpu * (dur_clock / 3600.0)) if dur_clock > 0 else 0.0

        total_wh = e_cpu_wh + e_gpu_wh
        co2_g = total_wh * CI_G_PER_WH

        # duration theo dữ liệu (nếu có ≥2 mẫu)
        duration_sum = float(slice_tabs[-1] - slice_tabs[0]) if slice_tabs.size >= 2 else 0.0

        # Cập nhật con trỏ
        self.search_cursor = max(i1, self.search_cursor)

        return EnergyResult(cpu_wh=float(e_cpu_wh),
                            gpu_wh=float(e_gpu_wh),
                            total_wh=float(total_wh),
                            co2_g=float(co2_g),
                            start_idx=i0,
                            end_idx=i1,
                            duration_clock_s=float(dur_clock),
                            duration_sum_s=float(duration_sum))
    def estimate_idle_between(self, start_str: str, end_str: str) -> tuple[float, float]:
        """Ước lượng idle (W) bằng TRUNG VỊ trong cửa sổ [start, end]."""
        s_norm = _normalize_time_string(start_str)
        e_norm = _normalize_time_string(end_str)
        if not s_norm or not e_norm:
            raise ValueError("start/end idle phải dạng 'HH:MM:SS.mmm'")

        s_sec = _time_str_to_seconds(s_norm)
        e_sec = _time_str_to_seconds(e_norm)
        tabs = self.tabs
        cur_day = math.floor(tabs[0] / 86400.0) * 86400.0

        if e_sec < s_sec:  # qua ngày
            e_sec += 86400.0

        start_abs = cur_day + s_sec
        end_abs = cur_day + e_sec

        i0 = int(np.searchsorted(tabs, start_abs, side="left"))
        i1 = int(np.searchsorted(tabs, end_abs, side="right"))

        if i1 <= i0:
            # fallback an toàn: dùng median toàn cục (hoặc quantile thấp)
            cpu_idle = float(np.nanmean(self.cpu_w))
            gpu_idle = float(np.nanmean(self.gpu_w)) if self.gpu_w is not None else 0.0
            return cpu_idle, gpu_idle

        cpu_idle = float(np.nanmean(self.cpu_w[i0:i1]))
        gpu_idle = 0.0
        if self.gpu_w is not None and self.gpu_w.size:
            gpu_idle = float(np.nanmean(self.gpu_w[i0:i1]))

        return cpu_idle, gpu_idle

