"""
train_quantized.py
Quantization experiment for CIFAR-10 with CodeCarbon energy tracking.
Evaluates baseline vs quantized model efficiency (accuracy, size, speed, CO₂).
"""

import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
import time
import os
import sys
import json
import random
import numpy as np

# ============================================================
# 0️⃣ Project Path Setup
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import CIFAR10_CONFIG, LOGS_DIR, DATA_DIR, CARBON_CONFIG
from src.models.cnn import SimpleCNN
from src.train_baseline import get_cifar10_loaders, evaluate


# ============================================================
# 1️⃣ Helper Utilities (inline — replaces src.utils)
# ============================================================
def set_seed(seed: int = 42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics(metrics_dict: dict, save_path: str):
    """Save experiment metrics to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"📁 Metrics saved to: {save_path}")


def get_model_size_mb(path: str) -> float:
    """Return model size in MB."""
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


# ============================================================
# 2️⃣ Quantization Helper Functions
# ============================================================
def quantize_model_dynamic(model):
    """Apply dynamic quantization to model layers."""
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model


def measure_inference_time(model, test_loader, device, num_batches=100):
    """Measure average inference time per batch."""
    model.eval()
    times = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            if i < 10:  # Warm-up phase
                _ = model(inputs.to(device))
                continue

            inputs = inputs.to(device)
            start = time.time()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)

    return sum(times) / len(times) if times else 0.0


# ============================================================
# 3️⃣ Main Quantization Experiment
# ============================================================
def run_quantization_experiment(device_name="cpu"):
    """Run quantization experiment on the given device (CPU/GPU)."""

    print("=" * 70)
    print(f"CIFAR-10 QUANTIZATION EXPERIMENT - {device_name.upper()}")
    print("=" * 70)

    # Setup
    set_seed(CIFAR10_CONFIG["seed"])
    device = torch.device("cuda" if device_name == "gpu" else "cpu")
    print(f"\n🖥️ Device: {device}")

    # Load data
    print("\n📊 Loading CIFAR-10 dataset...")
    _, test_loader = get_cifar10_loaders(CIFAR10_CONFIG["batch_size"])

    # Locate baseline model file
    baseline_path = os.path.join(LOGS_DIR, f"cifar10_baseline_best_{device_name}.pth")
    if not os.path.exists(baseline_path) and device_name == "gpu":
        print("⚠️ GPU baseline not found — using CPU baseline instead.")
        baseline_path = os.path.join(LOGS_DIR, "cifar10_baseline_best_cpu.pth")

    if not os.path.exists(baseline_path):
        print(f"❌ Baseline model not found at: {baseline_path}")
        print("   Please run train_baseline.py first!")
        return

    # Load baseline model
    print("\n📥 Loading baseline model...")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(baseline_path, map_location=device))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # PHASE 1: Baseline Evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE MODEL EVALUATION")
    print("=" * 70)

    print("\n📊 Measuring baseline accuracy...")
    _, baseline_acc = evaluate(model, test_loader, criterion, device)
    print(f"   ✅ Baseline Accuracy: {baseline_acc:.2f}%")

    print("\n⏱️ Measuring baseline inference speed...")
    baseline_inference_time = measure_inference_time(model, test_loader, device)
    print(f"   ✅ Baseline Inference: {baseline_inference_time*1000:.2f} ms/batch")

    print("\n⚡ Measuring baseline inference energy...")
    tracker = EmissionsTracker(
        project_name=f"cifar10_baseline_inference_{device_name}",
        output_dir=LOGS_DIR,
        **CARBON_CONFIG,
    )
    tracker.start()
    _, _ = evaluate(model, test_loader, criterion, device)
    baseline_co2 = tracker.stop()
    print(f"   ✅ Baseline CO₂: {baseline_co2:.8f} kg")

    baseline_size = get_model_size_mb(baseline_path)
    print(f"   ✅ Baseline Size: {baseline_size:.2f} MB")

    # ============================================================
    # PHASE 2: Quantization
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: MODEL QUANTIZATION")
    print("=" * 70)

    print("\n🔄 Applying dynamic quantization...")
    model_cpu = SimpleCNN()
    model_cpu.load_state_dict(torch.load(baseline_path, map_location="cpu"))
    model_cpu.eval()

    quantized_model = quantize_model_dynamic(model_cpu)

    quantized_path = os.path.join(LOGS_DIR, f"cifar10_quantized_best_{device_name}.pth")
    torch.save(quantized_model.state_dict(), quantized_path)
    print(f"   ✅ Quantized model saved to {quantized_path}")

    quantized_size = get_model_size_mb(quantized_path)
    print(f"   ✅ Quantized Size: {quantized_size:.2f} MB")
    print(f"   📉 Size Reduction: {((baseline_size - quantized_size) / baseline_size * 100):.1f}%")

    # ============================================================
    # PHASE 3: Quantized Evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: QUANTIZED MODEL EVALUATION")
    print("=" * 70)

    quantized_model_eval = quantized_model
    eval_device = torch.device("cpu")
    _, test_loader_cpu = get_cifar10_loaders(CIFAR10_CONFIG["batch_size"])

    print("\n📊 Measuring quantized accuracy...")
    _, quantized_acc = evaluate(quantized_model_eval, test_loader_cpu, criterion, eval_device)
    print(f"   ✅ Quantized Accuracy: {quantized_acc:.2f}%")

    print("\n⏱️ Measuring quantized inference speed...")
    quantized_inference_time = measure_inference_time(
        quantized_model_eval, test_loader_cpu, eval_device
    )
    print(f"   ✅ Quantized Inference: {quantized_inference_time*1000:.2f} ms/batch")

    print("\n⚡ Measuring quantized inference energy...")
    tracker = EmissionsTracker(
        project_name=f"cifar10_quantized_inference_{device_name}",
        output_dir=LOGS_DIR,
        **CARBON_CONFIG,
    )
    tracker.start()
    _, _ = evaluate(quantized_model_eval, test_loader_cpu, criterion, eval_device)
    quantized_co2 = tracker.stop()
    print(f"   ✅ Quantized CO₂: {quantized_co2:.8f} kg")

    # ============================================================
    # PHASE 4: Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    acc_drop = baseline_acc - quantized_acc
    size_reduction = ((baseline_size - quantized_size) / baseline_size) * 100
    speed_improvement = ((baseline_inference_time - quantized_inference_time) / baseline_inference_time) * 100
    co2_reduction = ((baseline_co2 - quantized_co2) / baseline_co2) * 100 if baseline_co2 > 0 else 0

    print(f"\n📊 Accuracy Drop: {acc_drop:.2f}%")
    print(f"💾 Size Reduction: {size_reduction:.1f}%")
    print(f"⚡ CO₂ Reduction: {co2_reduction:.1f}%")
    print(f"⏱️ Speed Improvement: {speed_improvement:.1f}%")

    metrics = {
        "experiment": "cifar10_quantized",
        "device": device_name,
        "baseline": {
            "accuracy": baseline_acc,
            "model_size_mb": baseline_size,
            "inference_time_ms": baseline_inference_time * 1000,
            "co2_kg": baseline_co2,
        },
        "quantized": {
            "accuracy": quantized_acc,
            "model_size_mb": quantized_size,
            "inference_time_ms": quantized_inference_time * 1000,
            "co2_kg": quantized_co2,
        },
        "improvements": {
            "accuracy_drop_pct": acc_drop,
            "size_reduction_pct": size_reduction,
            "speed_improvement_pct": speed_improvement,
            "co2_reduction_pct": co2_reduction,
        },
    }

    metrics_file = os.path.join(LOGS_DIR, f"cifar10_quantized_metrics_{device_name}.json")
    save_metrics(metrics, metrics_file)

    print(f"\n✅ Experiment complete! Results saved to {metrics_file}")
    print("=" * 70)


# ============================================================
# 4️⃣ Run Experiment
# ============================================================
if __name__ == "__main__":
    print("\n🚀 Starting Quantization Experiment...\n")
    run_quantization_experiment("cpu")

    if torch.cuda.is_available():
        print("\n🚀 Starting GPU Quantization Experiment...\n")
        run_quantization_experiment("gpu")
    else:
        print("\n⚠️ GPU not available, skipping GPU experiment.")
