"""
Datasets helper (không import thẳng tensorflow/torch cho dữ liệu)
- Ưu tiên OpenML / sklearn; nếu không có mạng → fallback synthetic
- Trả về:
  * ML: (X_flat, y, None)
  * DL: (X_flat, y, X_img) nếu có ảnh, else X_img=None
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    from sklearn.datasets import fetch_openml as _fetch_openml  # type: ignore
except Exception:
    _fetch_openml = None  # type: ignore



def _to_int_labels(y):
   # Luôn ánh xạ về 0..K-1 theo thứ tự tăng dần lớp gốc
    ys = np.asarray(y)
    classes = np.unique(ys)
    # map: giá trị gốc -> chỉ số [0..K-1]
    mapping = {c: i for i, c in enumerate(sorted(classes))}
    return np.array([mapping[v] for v in ys], dtype=int)


def _load_openml(name: str, version: int | None = None):
    if _fetch_openml is None:
        raise RuntimeError("scikit-learn chưa sẵn sàng cho fetch_openml.")
    kw = dict(as_frame=False, parser="auto")
    data = _fetch_openml(name=name, version=version, **kw) if version is not None else _fetch_openml(name=name, **kw)
    X = data.data
    y = _to_int_labels(data.target)
    return X, y


def _load_usps():
    try:
        X, y = _load_openml("usps", version=2)
        y = _to_int_labels(y)  # 0..9

        X = X.astype("float32")
        x_min, x_max = X.min(), X.max()
        if x_min >= -1.0 and x_max <= 1.0:
            X = (X + 1.0) / 2.0
        elif x_max > 1.0:
            X = X / 255.0

        # Tạo 3 ứng viên reshape: C, Fortran, Fortran+transpose
        cand = [
            X.reshape(-1, 16, 16, 1),                              # C-order
            X.reshape(-1, 16, 16, 1, order="F"),                   # Fortran
            np.transpose(X.reshape(-1, 16, 16, 1, order="F"), (0,2,1,3)),  # F + T
        ]

        # Chọn ứng viên tốt nhất bằng baseline 1-NN (nếu sklearn có sẵn), fallback = phương sai
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.neighbors import KNeighborsClassifier
            scores = []
            for imgs in cand:
                Xf = imgs.reshape(len(imgs), -1)
                Xtr, Xte, ytr, yte = train_test_split(
                    Xf, y, test_size=0.2, random_state=42, stratify=y
                )
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(Xtr, ytr)
                scores.append(knn.score(Xte, yte))
            best_idx = int(np.argmax(scores))
        except Exception:
            # Fallback: dùng phương sai trung bình (ảnh “đúng” ít phẳng hơn)
            scores = [np.var(imgs, axis=(1,2,3)).mean() for imgs in cand]
            best_idx = int(np.argmax(scores))

        X_img = cand[best_idx]
        X_flat = X_img.reshape(len(X_img), -1)

        # sanity checks
        assert X_img.shape[1:] == (16, 16, 1)
        u = np.unique(y)
        assert u.min() == 0 and u.max() == 9 and u.size == 10

        return X_flat, y, X_img
    except Exception:
        raise RuntimeError("[WARNING] Không tải được dataset USPS.")


def _load_mnist():
    try:
        X, y = _load_openml("mnist_784", version=1)
        X = X.astype("float32") / 255.0
        X_img = X.reshape(-1, 28, 28, 1)
        X_flat = X_img.reshape(len(X_img), -1)
        return X_flat, y, X_img
    except Exception:
        raise RuntimeError("[WARNING] Không tải được dataset MNIST.")


def _load_fashion():
    try:
        X, y = _load_openml("Fashion-MNIST", version=1)
        X = X.astype("float32") / 255.0
        X_img = X.reshape(-1, 28, 28, 1)
        X_flat = X_img.reshape(len(X_img), -1)
        return X_flat, y, X_img
    except Exception:
        raise RuntimeError("[WARNING] Không tải được dataset Fashion-MNIST.")


def _load_cifar10():
    try:
        X, y = _load_openml("CIFAR_10")
        X = X.astype("float32") / 255.0
        try:
            X_img = X.reshape(-1, 32, 32, 3)
            X_flat = X_img.reshape(len(X_img), -1)
            return X_flat, y, X_img
        except Exception:
            return X, y, None
    except Exception:
        raise RuntimeError("[WARNING] Không tải được dataset CIFAR_10.")


def load_dataset(name: str, need_image: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # "ml": trả về (X_flat, y)
    # "dl": trả về (X_img, y)
    key = (name or "").lower()
    if key in ("mnist",):
        X_flat, y, X_img = _load_mnist()
    elif key in ("fashion_mnist", "fashion-mnist", "fashion"):
        X_flat, y, X_img = _load_fashion()
    elif key == "usps":
        X_flat, y, X_img = _load_usps()
    elif key in ("cifar10", "cifar-10"):
        X_flat, y, X_img = _load_cifar10()
    else:
        raise ValueError(f"Dataset không hỗ trợ: {name}")

    if need_image:
        return X_flat, y, X_img
    return X_flat, y, None


def load_dataset_dl(name: str):
    X_flat, y, X_img = load_dataset(name, need_image=True)
    return X_img, y, X_flat  # đặt X_img lên đầu để dùng cho DL