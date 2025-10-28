import torch
import numpy as np
import random
import json
import os
from datetime import datetime

# ------------------------------------------------------------
# 1. Seed setting for reproducibility
# ------------------------------------------------------------
def set_seed(seed=42):
    """Set all random seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------
# 2. Interactive device selector (CPU or GPU)
# ------------------------------------------------------------
def get_device_interactive():
    """
    Let the user choose whether to train on GPU or CPU interactively.
    If GPU is not available, defaults to CPU automatically.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name}")
        choice = input("💡 Do you want to use GPU for training? (y/n): ").strip().lower()
        if choice == 'y':
            print("🚀 Using GPU (CUDA)...")
            return torch.device('cuda'), "GPU"
        else:
            print("⚙️ Using CPU as selected.")
            return torch.device('cpu'), "CPU"
    else:
        print("⚠️ No GPU detected. Using CPU.")
        return torch.device('cpu'), "CPU"


# ------------------------------------------------------------
# 3. Save metrics to JSON file
# ------------------------------------------------------------
def save_metrics(metrics, filename):
    """Save metrics to JSON file"""
    metrics['timestamp'] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


# ------------------------------------------------------------
# 4. Model size calculator
# ------------------------------------------------------------
def get_model_size_mb(filepath):
    """Get model file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)
