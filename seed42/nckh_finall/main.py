"""
Entry point: Train + đo năng lượng (HWinfo64)

Thay đổi chính:
- Lúc đầu chỉ hỏi/lưu csv_path, KHÔNG đọc CSV ngay.
- Truyền csv_path vào runner.run_all(); runner sẽ đọc CSV SAU khi train xong.
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

from runner import run_all, set_seed


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train + Energy (HWinfo64)")
    p.add_argument("--deterministic", action="store_true", help="Cố định seed")
    p.add_argument("--seed", type=int, default=42, help="Seed mặc định khi --deterministic")
    return p


def _prompt_csv_path(max_tries: int = 3) -> str:
    for _ in range(max_tries):
        path = input("Nhập đường dẫn CSV HWinfo64: ").strip().strip('"').strip("'")
        if os.path.exists(path) and os.path.isfile(path):
            return path
        print(f"[warn] Không tìm thấy file: {path}")
    print("[error] Hết số lần thử. Thoát.")
    sys.exit(2)


def main():
    args = build_argparser().parse_args()
    if args.deterministic:
        set_seed(args.seed)

    # Chỉ hỏi/lưu đường dẫn ở đây, CHƯA đọc file
    csv_path = _prompt_csv_path()

    # Toàn bộ train (Pha 1) → xong xuôi mới đọc CSV và đo (Pha 2) bên trong runner
    run_all(csv_path=csv_path)
    print("\n[done] Đã in Energy/CO2 cho 20 cặp.")


if __name__ == "__main__":
    main()
