# ============================================================
# Baseline training script for CIFAR-10
# Context: src/train_baseline.py
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
import time
import os
import sys
import csv  # ✅ For per-epoch logging

# ------------------------------------------------------------
# 1. Add project root to Python path
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# 2. Import project modules
# ------------------------------------------------------------
from src.config import CIFAR10_CONFIG, LOGS_DIR, DATA_DIR, CARBON_CONFIG
from src.utils import set_seed, save_metrics, get_device_interactive, get_model_size_mb
from src.models.cnn import SimpleCNN


# ------------------------------------------------------------
# 3. Dataset loader
# ------------------------------------------------------------
def get_cifar10_loaders(batch_size=128):
    """Load CIFAR-10 dataset with standard preprocessing"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ------------------------------------------------------------
# 4. Training and evaluation functions
# ------------------------------------------------------------
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(test_loader), 100. * correct / total


# ------------------------------------------------------------
# 5. Main training workflow
# ------------------------------------------------------------
def main():
    print("=" * 60)
    print("🌱 CIFAR-10 Baseline Training — GreenAI Lifecycle Tracker")
    print("=" * 60)

    # Setup
    set_seed(CIFAR10_CONFIG['seed'])
    device, device_type = get_device_interactive()
    print(f"\nDevice Selected: {device_type}\n")

    # Data
    print("📦 Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(CIFAR10_CONFIG['batch_size'])

    # Model
    print("🧠 Initializing model...")
    model = SimpleCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CIFAR10_CONFIG['learning_rate'])

    # Energy tracking
    print("\n🌿 Starting energy tracking...")
    tracker = EmissionsTracker(
        project_name=f"cifar10_baseline_{device_type.lower()}",
        output_dir=LOGS_DIR,
        **CARBON_CONFIG
    )
    tracker.start()
    start_time = time.time()

    # CSV logging setup
    log_path = os.path.join(LOGS_DIR, f"training_log_baseline_{device_type.lower()}.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time_sec"])

    print(f"\n🚀 Training for {CIFAR10_CONFIG['epochs']} epochs...")
    print("-" * 60)

    best_acc = 0
    for epoch in range(CIFAR10_CONFIG['epochs']):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1:02d} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")

        # Save epoch log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc, epoch_time])

        # Flush CodeCarbon per epoch (for better CO₂ tracking)
        tracker.flush()

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(LOGS_DIR, f"cifar10_baseline_best_{device_type.lower()}.pth"))

    # Stop tracking
    emissions = tracker.stop()
    training_time = time.time() - start_time

    print("-" * 60)
    print(f"\n✅ Best Accuracy: {best_acc:.2f}%")
    print(f"⏱️ Training Time: {training_time:.2f}s")
    print(f"🌍 CO₂ Emissions: {emissions:.6f} kg")

    # Save summary metrics
    model_path = os.path.join(LOGS_DIR, f"cifar10_baseline_best_{device_type.lower()}.pth")
    metrics = {
        "experiment": "cifar10_baseline",
        "experiment_type": "baseline",
        "accuracy": float(best_acc),
        "training_time_sec": float(training_time),
        "co2_kg": float(emissions),
        "model_size_mb": float(get_model_size_mb(model_path)),
        "total_params": int(total_params),
        "device_used": device_type
    }

    save_metrics(metrics, os.path.join(LOGS_DIR, f"cifar10_baseline_metrics_{device_type.lower()}.json"))
    print(f"\n📊 Metrics and logs saved to: {LOGS_DIR}/")
    print("🏁 Baseline training completed successfully!\n")


# ------------------------------------------------------------
# 6. Script entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
