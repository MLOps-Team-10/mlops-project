import subprocess

from eurosat_classifier.scripts.download_data import ensure_eurosat


def main() -> None:
    data_dir = ensure_eurosat("data/raw")

    cmd = [
        "python",
        "src/eurosat_classifier/train.py",
        f"data.data_dir={data_dir.as_posix()}",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()