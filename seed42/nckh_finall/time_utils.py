"""
Đo thời gian train/predict + lấy start/end theo đồng hồ hệ thống.
- Toàn bộ import đặt ở đầu file (đúng yêu cầu).
"""

from __future__ import annotations
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

def now_str() -> str:
    # "HH:MM:SS.mmm"
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def time_section_generic(model, Xtr, ytr, Xte, yte, task="ml"):
    """
    Trả về: acc, fit_s, pred_s, total_s, start_str, end_str
    """
    is_keras = "keras" in type(model).__module__
    if is_keras and task != "dl":
        raise ValueError("Model Keras mà task='ml' → có thể bạn đang feed nhầm. Hãy dùng task='dl'.")
    if (not is_keras) and task == "dl":
        raise ValueError("Model sklearn mà task='dl'.")
    if task == "dl":
        assert Xtr.ndim == 4 and Xte.ndim == 4, f"DL cần 4D (N,H,W,C), nhận {Xtr.shape}, {Xte.shape}"
    else:
        assert Xtr.ndim == 2 and Xte.ndim == 2, f"ML cần 2D (N,D), nhận {Xtr.shape}, {Xte.shape}"

    start_str = now_str()
    t0 = time.perf_counter()

    # FIT
    t_fit0 = time.perf_counter()
    if task == "ml":
        model.fit(Xtr, ytr)
    else:
        # DL (Keras)
        cb = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
        model.fit(
            Xtr, ytr,
            epochs=60,
            batch_size=128,
            validation_split=0.1,
            shuffle=True,              # ← thêm
            verbose=0,
            callbacks=cb
        )
        # log train-acc chỉ cho DL
        try:
            tr_pred = model.predict(Xtr, verbose=0)
            if getattr(tr_pred, "ndim", 1) > 1 and tr_pred.shape[-1] > 1:
                tr_pred = np.argmax(tr_pred, axis=1)
            tr_acc = accuracy_score(ytr, tr_pred)
            print(f"[dl-train] acc≈{tr_acc:.4f}")
        except Exception as _e:
            print("[dl-train] skip:", _e)

    t_fit1 = time.perf_counter()
    fit_s = t_fit1 - t_fit0

    # PRED
    t_pred0 = time.perf_counter()
    if task == "ml":
        y_pred = model.predict(Xte)
        if hasattr(y_pred, "ndim") and getattr(y_pred, "ndim", 1) > 1:
            y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = model.predict(Xte, verbose=0)
        if getattr(y_pred, "ndim", 1) > 1 and y_pred.shape[-1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

    t_pred1 = time.perf_counter()
    pred_s = t_pred1 - t_pred0

    # ACC
    acc = accuracy_score(yte, y_pred)

    t1 = time.perf_counter()
    total_s = t1 - t0
    end_str = now_str()

    return acc, fit_s, pred_s, total_s, start_str, end_str
