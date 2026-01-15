import subprocess

from eurosat_classifier.scripts.download_data import ensure_eurosat_rgb


def main() -> None:

    cmd = [
        "python",
        "src/eurosat_classifier/train.py"
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()