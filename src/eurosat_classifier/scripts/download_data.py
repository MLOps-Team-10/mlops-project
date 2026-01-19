from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
from typing import Iterable
import zipfile

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
            raise FileNotFoundError(f"Could not find RGB folder '{rgb_folder_name}' inside kaggle cache: {cache_path}")
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


def ensure_eurosat_rgb_cloud(
    download_root: str = "data/raw",
    target_name: str = "eurosat_rgb",
    sentinel_name: str = ".READY",
) -> Path:
    """Ensure EuroSAT RGB exists under ``{download_root}/{target_name}`` using a DVC-pulled zip.

    Behavior:
        - If ``{download_root}/{target_name}/{sentinel_name}`` exists, return immediately.
        - Otherwise run ``uv run dvc pull`` to fetch ``./data_zip/eurosat_rgb.zip``.
        - The zip must contain a top-level folder named ``eurosat_rgb``.
        - Copy that folder to ``./data/raw/`` so the final path is ``./data/raw/eurosat_rgb``.
        - Write a sentinel file inside the dataset folder to indicate readiness.

    Args:
        download_root: Root directory where the dataset folder will live.
        target_name: Dataset folder name (also expected as the top-level folder in the zip).
        sentinel_name: Sentinel filename written inside the dataset folder.

    Returns:
        Path to the ready dataset directory.

    Raises:
        RuntimeError: If DVC pull fails or no JPG files are found after extraction.
        FileNotFoundError: If the zip or expected folder inside the zip is missing.
    """
    target_dir = Path(download_root).expanduser().resolve() / target_name
    sentinel = target_dir / sentinel_name
    if sentinel.exists():
        print(f"Dataset already ready at: {target_dir}")
        return target_dir

    project_root = Path.cwd().resolve()
    zip_path = project_root / "data_zip" / f"{target_name}.zip"

    print("Starting DVC pull...")
    try:
        subprocess.run(["uv", "run", "dvc", "pull"], check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run `uv run dvc pull` (exit code {e.returncode}).") from e
    print("DVC pull completed.")

    if not zip_path.exists():
        raise FileNotFoundError(f"Expected zip not found at: {zip_path}")

    print(f"Unzipping dataset from: {zip_path}")
    with tempfile.TemporaryDirectory(prefix=f"{target_name}_") as tmp:
        tmp_dir = Path(tmp).resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        print(f"Unzipping completed to: {tmp_dir}")

        extracted_root = tmp_dir / target_name
        if not extracted_root.is_dir():
            raise FileNotFoundError(
                f"Zip did not contain expected folder '{target_name}'. Extracted to: {tmp_dir}"
            )

        if target_dir.exists():
            shutil.rmtree(target_dir)

        print(f"Copying extracted folder to: {target_dir}")
        shutil.copytree(extracted_root, target_dir)
        print("Copy completed.")

    jpg_count = sum(1 for _ in _iter_files(target_dir))
    if jpg_count == 0:
        raise RuntimeError(f"No JPG files found after extracting dataset into {target_dir}")

    manifest_hash = _compute_manifest_hash(target_dir)
    meta = {
        "dataset": f"dvc:data_zip/{target_name}.zip",
        "source_zip": zip_path.as_posix(),
        "jpg_count": jpg_count,
        "manifest_sha256": manifest_hash,
    }
    sentinel.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Dataset ready at: {target_dir}")
    return target_dir


if __name__ == "__main__":
    ensure_eurosat_rgb_cloud()
