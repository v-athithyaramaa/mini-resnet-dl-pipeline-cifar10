# src/train.py
import os
import json
import time
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.dataset import get_dataloaders
from src.model import MiniResNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes one full pass over the training set (50,000 images).
    """
    model.train()  # Enables training mode (enables BatchNorm running stats updates)
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on unseen test/validation data (10,000 images).
    Calculates Loss, Accuracy, Precision, Recall, and F1 Score.
    """
    model.eval()  # Disables training mode (freezes BatchNorm)
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():  # Crucial: Disable gradient tracking to save RAM & compute
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    
    # Calculate Precision, Recall, and F1 (Macro-averaged across all 10 classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    return val_loss, acc, precision, recall, f1


def run_experiment(
    exp_name="baseline_adamw",
    optimizer_type="AdamW",
    lr=0.001,
    batch_size=64,
    epochs=15,
    save_dir="./artifacts"
):
    """
    Runs an isolated hyperparameter experiment and saves artifacts.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Starting Experiment: {exp_name}")
    print(f"Device: {device} | Optimizer: {optimizer_type} | LR: {lr} | Batch Size: {batch_size}")
    print(f"{'='*60}")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = MiniResNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    # Configure Optimizer based on Experiment Settings
    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "SGD_Momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type == "SGD_Vanilla":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Learning rate scheduler: Cosine Annealing to gently decay learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": []
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, precision, recall, f1 = validate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)

        epoch_duration = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {f1:.4f}")

        # Save Best Model Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(save_dir, f"{exp_name}_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best model to {checkpoint_path} (Acc: {val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} mins. Peak Val Accuracy: {best_val_acc*100:.2f}%")

    # Save History to JSON (Human-readable, opens in VS Code)
    history_path = os.path.join(save_dir, f"{exp_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved history log to {history_path}")

    return history


if __name__ == "__main__":
    # Local Smoke Test: Run 1 quick epoch to verify end-to-end pipeline
    print("Running a 1-epoch Smoke Test on local CPU...")
    run_experiment(
        exp_name="local_smoke_test",
        optimizer_type="AdamW",
        lr=0.001,
        batch_size=128,
        epochs=1
    )