# src/generate_report_visuals.py
import os
import json
import matplotlib.pyplot as plt

def build_visuals(artifacts_dir="./artifacts", output_dir="./artifacts/report_figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    runs = {
        "Baseline (AdamW, LR=1e-3, BS=64)": "baseline_adamw_history.json",
        "Champion (SGD+Momentum, LR=1e-2, BS=64)": "optimizer_sgd_momentum_history.json",
        "Suboptimal LR (AdamW, LR=1e-4, BS=64)": "lr_0001_adamw_history.json",
        "Large Batch (AdamW, LR=1e-3, BS=256)": "batchsize_256_adamw_history.json"
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (title, filename) in enumerate(runs.items()):
        filepath = os.path.join(artifacts_dir, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r") as f:
            d = json.load(f)

        ax = axes[idx]
        epochs = range(1, len(d["train_acc"]) + 1)
        ax.plot(epochs, [x * 100 for x in d["train_acc"]], label="Train Accuracy", color="#1f77b4", lw=2)
        ax.plot(epochs, [x * 100 for x in d["val_acc"]], label="Val Accuracy", color="#ff7f0e", lw=2, ls="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(35, 95)
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_convergence_matrix.png"), dpi=300)
    plt.close()

    # Fig 2: Champion Detailed Convergence
    with open(os.path.join(artifacts_dir, "optimizer_sgd_momentum_history.json"), "r") as f:
        champ = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    epochs = range(1, len(champ["train_loss"]) + 1)

    ax1.plot(epochs, champ["train_loss"], label="Train Loss", color="#2ca02c", lw=2)
    ax1.plot(epochs, champ["val_loss"], label="Val Loss", color="#d62728", lw=2, ls="--")
    ax1.set_title("SGD+Momentum Loss Decay", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy")
    ax1.legend()

    ax2.plot(epochs, [x * 100 for x in champ["train_acc"]], label="Train Acc", color="#1f77b4", lw=2)
    ax2.plot(epochs, [x * 100 for x in champ["val_acc"]], label="Val Acc", color="#ff7f0e", lw=2, ls="--")
    ax2.set_title("SGD+Momentum Accuracy Progression", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_champion_curves.png"), dpi=300)
    plt.close()
    print(f"Visuals written to: {output_dir}")

if __name__ == "__main__":
    build_visuals()