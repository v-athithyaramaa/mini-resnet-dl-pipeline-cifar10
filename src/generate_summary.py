# src/generate_summary.py
import os
import json

def generate_markdown_summary(artifacts_dir="./artifacts", output_md="./artifacts/EXPERIMENT_SUMMARY.md"):
    experiments = [
        {
            "id": "Exp 1",
            "name": "Baseline (AdamW, LR=1e-3, BS=64)",
            "file": "baseline_adamw_history.json",
            "param_changed": "Baseline reference",
            "hypothesis": "Rapid early convergence via adaptive 2nd moments"
        },
        {
            "id": "Exp 2",
            "name": "SGD + Nesterov Momentum (LR=1e-2, BS=64)",
            "file": "optimizer_sgd_momentum_history.json",
            "param_changed": "Optimizer -> SGD (m=0.9, wd=5e-4)",
            "hypothesis": "Overcome sharp minima, achieve flatter basin generalization"
        },
        {
            "id": "Exp 3",
            "name": "Suboptimal LR (AdamW, LR=1e-4, BS=64)",
            "file": "lr_0001_adamw_history.json",
            "param_changed": "Learning Rate -> 1e-4 (10x smaller)",
            "hypothesis": "Test gradient update sensitivity; expect severe underfitting"
        },
        {
            "id": "Exp 4",
            "name": "Large Batch (AdamW, LR=1e-3, BS=256)",
            "file": "batchsize_256_adamw_history.json",
            "param_changed": "Batch Size -> 256 (4x larger)",
            "hypothesis": "Higher hardware throughput, reduced gradient stochasticity"
        }
    ]

    rows = []
    baseline_acc = None

    for exp in experiments:
        filepath = os.path.join(artifacts_dir, exp["file"])
        if not os.path.exists(filepath):
            print(f"Warning: {exp['file']} not found. Skipping.")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        # Identify the best epoch by validation accuracy
        best_val_acc = max(data["val_acc"])
        best_idx = data["val_acc"].index(best_val_acc)

        val_acc_pct = best_val_acc * 100
        val_f1 = data["val_f1"][best_idx]
        val_prec = data["val_precision"][best_idx]
        val_rec = data["val_recall"][best_idx]

        if baseline_acc is None:
            baseline_acc = val_acc_pct
            delta_str = "REF"
        else:
            delta = val_acc_pct - baseline_acc
            delta_str = f"{delta:+.2f}%"

        rows.append(
            f"| **{exp['id']}** | {exp['name']} | `{exp['param_changed']}` | "
            f"**{val_acc_pct:.2f}%** | `{delta_str}` | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f} |"
        )

    markdown_content = f"""# CIFAR-10 Hyperparameter Sensitivity Matrix

| Exp ID | Experiment Setup | Parameter Alteration | Peak Val Acc | Delta vs Base | Macro Precision | Macro Recall | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
""" + "\n".join(rows) + "\n"

    with open(output_md, "w") as f:
        f.write(markdown_content)

    print(f"Summary table successfully written to {output_md}")
    print("\n" + markdown_content)

if __name__ == "__main__":
    generate_markdown_summary()