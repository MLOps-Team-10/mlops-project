import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from eurosat_classifier.data import get_dataloaders, DataConfig


def benchmark():
    worker_counts = [0, 1, 2, 4, 8, 12]
    num_runs = 5
    batches_to_check = 50

    final_means = []
    final_stds = []

    print(f"Starting benchmark ({num_runs} runs per setting)...")

    for workers in worker_counts:
        times = []
        for run in range(num_runs):
            config = DataConfig(data_dir="data/raw", batch_size=64, num_workers=workers)
            train_loader, _ = get_dataloaders(config)

            it = iter(train_loader)
            next(it)

            start = time.time()
            for i, batch in enumerate(it):
                if i >= batches_to_check:
                    break
            end = time.time()
            times.append(end - start)

        mean_time = np.mean(times)
        std_time = np.std(times)
        final_means.append(mean_time)
        final_stds.append(std_time)
        print(f"Workers: {workers:2} | Mean: {mean_time:.2f}s | Std: {std_time:.2f}s")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(worker_counts, final_means, yerr=final_stds, fmt="-o", capsize=5, ecolor="red", label="Timing")
    plt.xlabel("Number of Workers")
    plt.ylabel(f"Time (s) for {batches_to_check} batches")
    plt.title("M29: DataLoader Performance (Mean & Std Dev)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plot_path = "reports/figures/dataloader_performance.png"
    plt.savefig(plot_path)
    print(f"\n Success! Plot saved to: {plot_path}")


if __name__ == "__main__":
    benchmark()
