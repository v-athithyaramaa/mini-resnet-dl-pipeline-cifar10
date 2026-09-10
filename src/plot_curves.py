# src/plot_curves.py
import json
import os
import matplotlib.pyplot as plt

def generate_report_plots(artifacts_dir="./artifacts", output_dir="./artifacts/plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    experiments = {
        "Baseline (AdamW, lr=0.001, bs=64)": "baseline_adamw_history.json",
        "Optimizer Study (SGD+Momentum, lr=0.01, bs=64)": "optimizer_sgd_momentum_history.json",
        "Learning Rate Study (AdamW, lr=0.0001, bs=64)": "lr_0001_adamw_history.json",
        "Batch Size Study (AdamW, lr=0.001, bs=256)": "batchsize_256_adamw_history.json"
    }

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: Training vs Validation Accuracy across all experiments
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for idx, (label, json_file) in enumerate(experiments.items()):
        path = os.path.join(artifacts_dir, json_file)
        if not os.path.exists(path):
            print(f"Skipping {json_file} (not found)")
            continue
            
        with open(path, "r") as f:
            data = json.load(f)
            
        epochs = range(1, len(data["train_acc"]) + 1)
        ax = axes[idx]
        ax.plot(epochs, [a * 100 for a in data["train_acc"]], label="Train Acc", color="#1f77b4", linewidth=2)
        ax.plot(epochs, [a * 100 for a in data["val_acc"]], label="Val Acc", color="#ff7f0e", linewidth=2, linestyle="--")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.set_ylim(30, 95)

    plt.tight_layout()
    acc_path = os.path.join(output_dir, "hyperparameter_accuracy_curves.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved accuracy curves to {acc_path}")

    # Figure 2: Champion Model Convergence (Loss & Accuracy)
    champ_path = os.path.join(artifacts_dir, "optimizer_sgd_momentum_history.json")
    with open(champ_path, "r") as f:
        champ_data = json.load(f)

    epochs = range(1, len(champ_data["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, champ_data["train_loss"], label="Train Loss", color="#2ca02c", linewidth=2)
    ax1.plot(epochs, champ_data["val_loss"], label="Val Loss", color="#d62728", linewidth=2, linestyle="--")
    ax1.set_title("Champion Model Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in champ_data["train_acc"]], label="Train Acc", color="#1f77b4", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in champ_data["val_acc"]], label="Val Acc", color="#ff7f0e", linewidth=2, linestyle="--")
    ax2.set_title("Champion Model Accuracy Progression", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    champ_plot_path = os.path.join(output_dir, "champion_model_curves.png")
    plt.savefig(champ_plot_path, dpi=300)
    plt.close()
    print(f"Saved champion convergence plots to {champ_plot_path}")

if __name__ == "__main__":
    generate_report_plots()