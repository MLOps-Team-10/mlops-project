import subprocess


def main() -> None:
    cmd = ["python", "src/eurosat_classifier/train.py"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
