from pathlib import Path
import shutil

import kagglehub


def ensure_eurosat(download_root: str = "data/raw") -> Path:
    """
    Downloads EuroSAT via kagglehub (if needed) and puts it under data/raw/eurosat.
    Returns the path to the dataset folder.
    """
    target_dir = Path(download_root) / "eurosat"
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Dataset already present at: {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    # kagglehub downloads to a cache location and returns that path
    cache_path = Path(kagglehub.dataset_download("apollo2506/eurosat-dataset"))
    print("kagglehub cache path:", cache_path)

    # Copy from cache -> project data dir (so it's stable and mountable)
    # If the dataset already contains the folder structure you need, copy everything
    for item in cache_path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    print(f"Dataset copied to: {target_dir}")
    return target_dir


if __name__ == "__main__":
    ensure_eurosat()