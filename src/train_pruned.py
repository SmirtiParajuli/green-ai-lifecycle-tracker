"""
train_pruned.py
Train a pruned version of the CIFAR-10 baseline model and track energy/emissions.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import json
from codecarbon import EmissionsTracker

# ------------------------------------------------------------
# 0. Fix import path BEFORE importing from src
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import LOGS_DIR, MODELS_DIR, RESULTS_DIR


# ------------------------------------------------------------
# 1. Define CNN model (same as baseline)
# ------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ------------------------------------------------------------
# 2. Training and evaluation functions
# ------------------------------------------------------------
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, device, test_loader, criterion):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return loss / len(test_loader), 100. * correct / total


# ------------------------------------------------------------
# 3. Main function
# ------------------------------------------------------------
def main(device_type="gpu", num_epochs=20, batch_size=128, prune_amount=0.3):
    print(f"🚀 Training pruned CIFAR-10 model on {device_type.upper()}...")

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() and device_type == "gpu" else "cpu")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ------------------------------------------------------------
    # Apply pruning
    # ------------------------------------------------------------
    print(f"🔧 Applying {prune_amount*100:.0f}% pruning to convolutional layers...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=prune_amount)
            prune.remove(module, 'weight')

    # ------------------------------------------------------------
    # Initialize CodeCarbon tracker
    # ------------------------------------------------------------
    tracker = EmissionsTracker(project_name="cifar10_pruned", output_dir=LOGS_DIR)
    tracker.start()

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    train_log = []
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        train_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time_sec": epoch_time
        })

    total_time = time.time() - start_time
    emissions = tracker.stop()

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, f"cifar10_pruned_best_{device_type.lower()}.pth")
    torch.save(model.state_dict(), model_path)

    log_path = os.path.join(LOGS_DIR, f"training_log_pruned_{device_type.lower()}.csv")
    pd.DataFrame(train_log).to_csv(log_path, index=False)

    metrics = {
        "experiment": "cifar10_pruned",
        "experiment_type": "pruned",
        "accuracy": test_acc,
        "training_time_sec": total_time,
        "co2_kg": emissions,
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "total_params": sum(p.numel() for p in model.parameters()),
        "device_used": device_type.upper(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

    metrics_path = os.path.join(LOGS_DIR, f"cifar10_pruned_metrics_{device_type.lower()}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n✅ Pruned training complete on {device_type.upper()}")
    print(f"📊 Metrics saved to {metrics_path}")
    print(f"💾 Model saved to {model_path}")
    print(f"📈 Logs saved to {log_path}")


# ------------------------------------------------------------
# 4. Run in auto mode (for VS Code play button)
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        choice = input("💻 Choose device (gpu/cpu): ").strip().lower() or "gpu"
        sys.argv += ["--device", choice]

    device = "gpu" if "gpu" in sys.argv else "cpu"
    main(device_type=device)
