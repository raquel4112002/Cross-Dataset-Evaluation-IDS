from __future__ import annotations
import os, glob, shutil
import kagglehub

def _copy_csvs(src_dir: str, dst_dir: str) -> int:
    os.makedirs(dst_dir, exist_ok=True)
    c = 0
    for p in glob.glob(os.path.join(src_dir, "**", "*.csv"), recursive=True):
        try:
            shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))
            c += 1
        except Exception:
            pass
    return c

def download_unsw(dst_dir: str) -> str:
    path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
    if _copy_csvs(path, dst_dir) == 0:
        raise RuntimeError("UNSW-NB15: no CSVs found after download.")
    return dst_dir

def download_cicids(dst_dir: str) -> str:
    path = kagglehub.dataset_download("dhoogla/cicids2017")
    if _copy_csvs(path, dst_dir) == 0:
        raise RuntimeError("CICIDS2017: no CSVs found after download.")
    return dst_dir
