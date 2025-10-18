"""
Điều phối chạy 20 cặp (dataset, model) + tính Energy/CO2

Thay đổi chính:
- Pha 1: train xong thu start/end.
- Pha 2: LÚC NÀY mới đọc CSV bằng EnergyContext.from_csv(csv_path) và tính năng lượng.
- Mọi import đặt ở đầu file (đúng yêu cầu).
"""

from __future__ import annotations
import csv
import os
import random
import importlib
from dataclasses import dataclass
from metrics import compute_metrics
import numpy as np
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

# Import tuỳ chọn để đặt seed sâu cho DL/torch nếu có
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import tensorflow as tf  # type: ignore
    _HAS_TF = True
except Exception:
    tf = None  # type: ignore
    _HAS_TF = False

from datasets import load_dataset,load_dataset_dl
from train_algorithms import get_ml_algorithms, get_dl_algorithms
from time_utils import time_section_generic
import time
from time_utils import now_str
from energy import EnergyContext, _energy_wh_trapezoid, CI_G_PER_WH  # dùng lại hàm tích phân & hằng số



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH and torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda"):
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    if _HAS_TF and tf is not None:
        try:
            if hasattr(tf, "random"):
                tf.random.set_seed(seed)
        except Exception:
            pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def _ensure_csv(path: str, header: list[str]):
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if is_new:
        w.writerow(header)
    return f, w


def _format_result(tag, ds_name, model_name, acc, fit, pred, total, start, end, e_total, e_co2):
    return (f"[{tag}] {ds_name} | {model_name} | "
            f"acc={acc:.4f} | fit={fit:.4f}s | pred={pred:.4f}s | total={total:.4f}s | "
            f"start={start} | end={end}  | TOTAL Energy={e_total:.8f}Wh | CO2={e_co2:.8f}g  ")

def measure_idle_window(seconds: int = 30) -> tuple[str, str]:
    """Đặt cửa sổ idle: đánh dấu thời gian, sleep để HWInfo log, rồi đánh dấu kết thúc."""
    idle_start = now_str()
    time.sleep(max(0, int(seconds)))
    idle_end = now_str()
    return idle_start, idle_end


@dataclass
class Record:
    tag: str            # "ML" | "DL"
    ds_key: str         # usps|mnist|fashion_mnist|cifar10
    ds_name: str        # Usps|Mnist|Fashion-MNIST|Cifar-10
    model_name: str
    acc: float
    fit_s: float
    pred_s: float
    total_s: float
    start: str
    end: str

    n_train: int
    n_test: int
    idle_start: str
    idle_end: str
    idle_cpu_w: float
    idle_gpu_w: float

baseline_acc = {
    "usps": 0.10,
    "mnist": 0.10,
    "fashion_mnist": 0.10,
    "cifar10": 0.10,
}

def compute_majority_acc(y):
    counts = Counter(y)
    majority_class, majority_count = counts.most_common(1)[0]
    return majority_count / len(y)



def run_all(csv_path: str):
    # Thứ tự dataset cố định
    ordered_keys = ["usps", "mnist", "fashion_mnist", "cifar10"]
    pretty = {
        "usps": "Usps",
        "mnist": "Mnist",
        "fashion_mnist": "Fashion-MNIST",
        "cifar10": "Cifar-10",
    }

    # Thuật toán
    ml_algos = get_ml_algorithms()               # 4 model: DT, RF, LogReg, MLP(sklearn)
    dl_algos = get_dl_algorithms()               # 1 model: CNN (Keras)

    if not dl_algos:
        raise RuntimeError("Không tìm thấy TensorFlow/Keras cho model 'CNN'. Vui lòng cài TensorFlow để chạy đủ 20 cặp.")

    total_pairs = 20
    pair_idx = 0
    records: list[Record] = []

    # ===================== PHA 1: TRAIN + CHỈ IN PROGRESS =====================
    for key in ordered_keys:
        X_flat, y, X_img = load_dataset(key, need_image=True)
        ds_name_pretty = pretty[key]
        base_acc_value = compute_majority_acc(y)
        baseline_acc[key] = base_acc_value

        # --- 4 model ML ---
        Xtr, Xte, ytr, yte = train_test_split(X_flat, y, test_size=0.2, random_state=42, stratify=y)
        n_tr, n_te = len(Xtr), len(Xte)

        for build in ml_algos:
            pair_idx += 1
            print(f"[progress] Training {pair_idx}/{total_pairs} cặp... (sẽ in kết quả sau khi tính Energy/CO2)")

             # 1) đo idle 30s
            idle_start, idle_end = measure_idle_window(seconds=30)

            # 2) reload để có mẫu idle vừa ghi
            energy_ctx = EnergyContext.from_csv(csv_path)

            # 3) ước lượng idle_median và IN trước train
            idle_cpu_w, idle_gpu_w = energy_ctx.estimate_idle_between(idle_start, idle_end)
            print(f"[idle] CPU≈{idle_cpu_w:.2f}W, GPU≈{idle_gpu_w:.2f}W  ({idle_start} → {idle_end})")

            # 4) tạo “hàng rào” thời gian nhỏ
            time.sleep(0.3)

            # 5) train

            model, model_name = build()
            acc, fit_t, pred_t, total_t, start_str, end_str = time_section_generic(
                model, Xtr, ytr, Xte, yte, task="ml"
            )
            records.append(Record("ML", key, ds_name_pretty, model_name,
                      acc, fit_t, pred_t, total_t, start_str, end_str,
                      n_tr, n_te, idle_start, idle_end, idle_cpu_w, idle_gpu_w))


        X_img_dl, y_dl, X_flat_dl = load_dataset_dl(key)
        # Sanity: 1-NN trên ảnh (flatten). Đúng reshape sẽ >0.90
        try:
            from sklearn.model_selection import train_test_split as _tts
            from sklearn.neighbors import KNeighborsClassifier as _KNN
            Xf = X_img_dl.reshape(len(X_img_dl), -1)
            _Xtr,_Xte,_ytr,_yte = _tts(Xf, y_dl, test_size=0.2, random_state=42, stratify=y_dl)
            _knn = _KNN(n_neighbors=1)
            _knn.fit(_Xtr,_ytr)
            print(f"[sanity] {key} 1-NN(DL path) acc={_knn.score(_Xte,_yte):.4f}")
        except Exception as _e:
            print("[sanity] 1-NN skip:", _e)

        # --- 1 model DL (CNN) ---
        if X_img_dl is None:
            raise RuntimeError(f"Dataset {ds_name_pretty} không cung cấp ảnh cho CNN.")

        Xtr_img, Xte_img, ytr_img, yte_img = train_test_split(
            X_img_dl, y_dl, test_size=0.2, random_state=42, stratify=y_dl
        )

        # (khuyến nghị) log/guard nhanh
        print(f"[run] DL | {ds_name_pretty} | CNN | Xtr={Xtr_img.shape}, Xte={Xte_img.shape}")
        assert Xtr_img.ndim == 4 and Xte_img.ndim == 4

        n_tr_img, n_te_img = len(Xtr_img), len(Xte_img)

        for build in dl_algos[:1]:
            pair_idx += 1
            print(f"[progress] Training {pair_idx}/{total_pairs} cặp... (sẽ in kết quả sau khi tính Energy/CO2)")

            idle_start, idle_end = measure_idle_window(seconds=30)
            energy_ctx = EnergyContext.from_csv(csv_path)
            idle_cpu_w, idle_gpu_w = energy_ctx.estimate_idle_between(idle_start, idle_end)
            print(f"[idle] CPU≈{idle_cpu_w:.2f}W, GPU≈{idle_gpu_w:.2f}W  ({idle_start} → {idle_end})")
            time.sleep(0.3)

            n_classes = int(np.unique(y_dl).size)
            model, model_name = build(input_shape=Xtr_img.shape[1:], n_classes=n_classes)
            acc, fit_t, pred_t, total_t, start_str, end_str = time_section_generic(
                model, Xtr_img, ytr_img, Xte_img, yte_img, task="dl"
            )

            records.append(Record("DL", key, ds_name_pretty, "CNN",
                      acc, fit_t, pred_t, total_t, start_str, end_str,
                      n_tr_img, n_te_img, idle_start, idle_end, idle_cpu_w, idle_gpu_w))

    # ===================== PHA 2: ĐỌC CSV + TÍNH ENERGY =====================
    # Đọc CSV bây giờ (sau khi train xong) và CHỈ giữ 2–3 cột cần thiết.
    energy_ctx = EnergyContext.from_csv(csv_path)

    fcsv, wcsv = _ensure_csv(
        "results_20_pairs.csv",
        ["dataset","model","acc","fit_s","pred_s","total_s","start","end",
     "total_wh","co2_g","baseline","N","idle_cpu_w","idle_gpu_w",
     "j_per_sample","e_per_b","t_per_b"],
    )

    try:
        # In theo nhóm dataset
        for key in ordered_keys:
            ds_name_pretty = pretty[key]
            print(f"\n===== Running on dataset: {ds_name_pretty} =====")
            subset = [r for r in records if r.ds_key == key]
            for r in subset:

               # 1) lát cắt thời gian train (lấy index)
                e_raw = energy_ctx.compute_between(r.start, r.end)
                i0, i1 = e_raw.start_idx, e_raw.end_idx
                tabs = energy_ctx.tabs[i0:i1+1]
                cpu  = energy_ctx.cpu_w[i0:i1+1]
                gpu  = energy_ctx.gpu_w[i0:i1+1] if energy_ctx.gpu_w is not None else None

                # 2) TRỪ idle median per-run
                cpu_net = np.clip(cpu - r.idle_cpu_w, 0.0, None)
                gpu_net = None
                if gpu is not None:
                    gpu_net = np.clip(gpu - r.idle_gpu_w, 0.0, None)

                e_cpu_wh = _energy_wh_trapezoid(cpu_net, tabs)
                e_gpu_wh = _energy_wh_trapezoid(gpu_net, tabs) if gpu_net is not None else 0.0
                total_wh = e_cpu_wh + e_gpu_wh
                co2_g    = total_wh * CI_G_PER_WH

                # baseline accuracy
                b_acc = baseline_acc[r.ds_key]

                # 3) Metrics
                e_joules = total_wh * 3600.0
                N = r.n_train + r.n_test
                if N <= 0: N = 1
                metrics = compute_metrics(e_joules, r.total_s, r.acc, b_acc, N)

                print(
                        f"[{r.tag}] {r.ds_name} | {r.model_name} | acc={r.acc:.4f} | total_s={r.total_s:.2f}s | "
                        f"E_net={e_joules:.1f}J | J/sample={metrics['j_per_sample']:.6f} | "
                        f"E/B={metrics['e_per_b']:.2f} | T/B={metrics['t_per_b']:.2f} "
                        f"(idle_cpu≈{r.idle_cpu_w:.2f}W, idle_gpu≈{r.idle_gpu_w:.2f}W, baseline={b_acc:.3f}, N={N})"
                 )

                wcsv.writerow([
                        r.ds_name, r.model_name,
                        f"{r.acc:.6f}", f"{r.fit_s:.6f}", f"{r.pred_s:.6f}", f"{r.total_s:.6f}",
                        r.start, r.end, f"{total_wh:.8f}", f"{co2_g:.8f}",
                        f"{b_acc:.6f}", f"{N}", f"{r.idle_cpu_w:.6f}", f"{r.idle_gpu_w:.6f}",
                        f"{metrics['j_per_sample']:.8f}",
                        f"{metrics['e_per_b']:.8f}", f"{metrics['t_per_b']:.8f}"
                ])
    finally:
        fcsv.close()
