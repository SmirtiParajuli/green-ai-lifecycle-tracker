# src/hpo_energy.py
# 🌱 Energy-Aware Multi-Objective HPO for CIFAR-10 with CNNs

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker
import time
import pandas as pd
import sys, os

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.cnn import SimpleCNN
from src.train_baseline import get_cifar10_loaders, train_epoch, evaluate
from src.config import LOGS_DIR, CARBON_CONFIG
from src.utils import set_seed, save_metrics


def objective(trial):
    """Energy-aware multi-objective function: maximize accuracy, minimize CO₂."""
    
    # 🎯 Hyperparameter search space
    config = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.3, 0.7),
    }

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CIFAR-10
    train_loader, test_loader = get_cifar10_loaders(config['batch_size'])

    # Model setup
    model = SimpleCNN(dropout=config['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 🌿 Start tracking CO₂
    tracker = EmissionsTracker(
        project_name=f"hpo_trial_{trial.number}",
        output_dir=LOGS_DIR,
        **CARBON_CONFIG
    )
    tracker.start()
    start_time = time.time()

    best_acc = 0
    epochs = 10  # Short training for HPO speed

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc

        # ❌ Remove trial.report() and pruning (not supported in multi-objective)
        # trial.report(test_acc, epoch)
        # if trial.should_prune():
        #     tracker.stop()
        #     raise optuna.TrialPruned()

    co2_kg = tracker.stop()
    training_time = time.time() - start_time

    # 🧠 Log metrics to trial
    trial.set_user_attr('co2_kg', co2_kg)
    trial.set_user_attr('accuracy', best_acc)
    trial.set_user_attr('training_time', training_time)

    # 🎯 Multi-objective return: maximize accuracy, minimize CO₂
    return best_acc, co2_kg


def run_hpo_study(n_trials=30, device='cuda'):
    """Run Pareto-based multi-objective HPO."""
    
    print("="*70)
    print(f"ENERGY-AWARE MULTI-OBJECTIVE HPO - {device.upper()}")
    print("="*70)

    # ⚙️ Multi-objective study setup (maximize accuracy, minimize CO₂)
    study = optuna.create_study(
        study_name=f"cifar10_energy_hpo_{device}",
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler()
    )

    print(f"\n🚀 Starting {n_trials} trials with NSGA-II Pareto optimization...")
    print(f"   Each trial: 10 epochs with energy tracking")
    print(f"   Objectives: Maximize Accuracy, Minimize CO₂ Emissions\n")
    print("-"*70)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n" + "="*70)
    print("🌟 HPO COMPLETE — Pareto Optimization Finished!")
    print("="*70)

    # 📊 Pareto front summary
    pareto_trials = study.best_trials
    print(f"\n🏆 Found {len(pareto_trials)} Pareto-optimal trials:")

    for t in pareto_trials:
        print(f" - Trial {t.number}: Accuracy={t.values[0]:.2f}%, CO₂={t.values[1]*1000:.3f} g")

    # 💾 Save study
    import joblib
    study_path = f"{LOGS_DIR}/hpo_study_{device}.pkl"
    joblib.dump(study, study_path)
    print(f"\n💾 Study saved → {study_path}")

    # 💾 Save Pareto front CSV
    pareto_df = pd.DataFrame(
        [(t.number, t.values[0], t.values[1]) for t in pareto_trials],
        columns=["trial", "accuracy", "co2_kg"]
    )
    pareto_csv = f"{LOGS_DIR}/hpo_pareto_{device}.csv"
    pareto_df.to_csv(pareto_csv, index=False)
    print(f"📈 Pareto front saved → {pareto_csv}")

    # 💾 Save summary metrics
    summary = {
        "n_trials": len(study.trials),
        "n_pareto": len(pareto_trials),
        "best_accuracy": max(t.values[0] for t in pareto_trials),
        "lowest_co2_kg": min(t.values[1] for t in pareto_trials),
    }
    save_metrics(summary, f"{LOGS_DIR}/hpo_summary_{device}.json")

    print("\n✅ HPO results and Pareto front successfully saved!")
    return study


if __name__ == "__main__":
    import sys
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    print("=" * 70)
    print(f"🎯 AUTO-RUN MODE: Pareto HPO on GPU + CPU ({n_trials} trials each)")
    print("=" * 70)

    # GPU
    if torch.cuda.is_available():
        print("\n🚀 Running GPU HPO...")
        study_gpu = run_hpo_study(n_trials=n_trials, device="cuda")
    else:
        print("\n⚠️ GPU not available. Skipping CUDA.")
        study_gpu = None

    # CPU
    print("\n🧠 Running CPU HPO...")
    study_cpu = run_hpo_study(n_trials=n_trials, device="cpu")

    print("\n✅ All HPO runs complete!")
    print("📂 Results stored in:")
    print("   - logs/hpo_study_cuda.pkl")
    print("   - logs/hpo_pareto_cuda.csv")
    print("   - logs/hpo_summary_cuda.json")
    print("=" * 70)
