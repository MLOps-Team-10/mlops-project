from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
from typing import Iterable

import kagglehub


def _iter_files(root: Path, exts: tuple[str, ...] = (".jpg", ".jpeg")) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _compute_manifest_hash(root: Path) -> str:
    """
    Compute a stable hash over the dataset contents without reading full file bytes.
    We hash relative path + size, which is usually good enough as a lightweight integrity check.
    """
    items = []
    for p in sorted(_iter_files(root), key=lambda x: x.as_posix()):
        rel = p.relative_to(root).as_posix()
        st = p.stat()
        items.append(f"{rel}|{st.st_size}")

    h = hashlib.sha256()
    for line in items:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def ensure_eurosat_rgb(
    download_root: str = "data/raw",
    target_name: str = "eurosat_rgb",
    sentinel_name: str = ".READY",
    kaggle_dataset: str = "apollo2506/eurosat-dataset",
    rgb_folder_name: str = "EuroSAT",
) -> Path:
    """
    Ensure EuroSAT RGB (JPG) dataset exists under data/raw/eurosat_rgb.

    Behavior:
    - If target_dir/.READY exists, do nothing and return target_dir
    - Otherwise download via kagglehub (cache), copy only RGB folder into target_dir
    - Write .READY with basic metadata + a manifest hash

    This keeps training deterministic and avoids mixing TIFF/all-bands data.
    """
    target_dir = Path(download_root).expanduser().resolve() / target_name
    sentinel = target_dir / sentinel_name

    # Fast path: dataset is ready
    if sentinel.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    # Download to kagglehub cache
    cache_path = Path(kagglehub.dataset_download(kaggle_dataset)).resolve()
    print(f"kagglehub cache path: {cache_path}")

    # Find RGB folder inside cache
    rgb_src = cache_path / rgb_folder_name
    if not rgb_src.exists() or not rgb_src.is_dir():
        # Some Kaggle bundles nest folders; fallback to a search
        candidates = [p for p in cache_path.rglob(rgb_folder_name) if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find RGB folder '{rgb_folder_name}' inside kaggle cache: {cache_path}"
            )
        rgb_src = candidates[0]

    # Copy RGB data into target_dir (classes folders with JPGs)
    # target_dir will contain directly: AnnualCrop/, Forest/, ... (ImageFolder-friendly)
    for item in rgb_src.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            # Normally there should be no files at this level, but keep safe
            shutil.copy2(item, dest)

    # Minimal validation: must contain at least some JPG files
    jpg_count = sum(1 for _ in _iter_files(target_dir))
    if jpg_count == 0:
        raise RuntimeError(f"No JPG files found after copying RGB dataset into {target_dir}")

    # Write sentinel with useful metadata
    manifest_hash = _compute_manifest_hash(target_dir)
    meta = {
        "dataset": kaggle_dataset,
        "source_rgb_folder": rgb_src.as_posix(),
        "jpg_count": jpg_count,
        "manifest_sha256": manifest_hash,
    }
    sentinel.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"EuroSAT RGB dataset ready at: {target_dir}")
    return target_dir


if __name__ == "__main__":
    ensure_eurosat_rgb()