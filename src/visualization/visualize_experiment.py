# %% [markdown]
"""
# 🧠 visualize_experiment_interactive.py  
Visualize any experiment (baseline, pruned, quantized, etc.) dynamically.  
Works perfectly in VS Code interactive mode (with Python + Jupyter extensions).
"""

# %% ------------------------------------------------------------
# 0. Setup Imports & Project Path
# ------------------------------------------------------------
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config import LOGS_DIR

# %% ------------------------------------------------------------
# 1. Choose Experiment and Device (interactive)
# ------------------------------------------------------------
EXPERIMENT = input("Enter experiment name (baseline / pruned / quantized): ").strip().lower() or "baseline"
DEVICE = input("Choose device (gpu / cpu): ").strip().lower() or "gpu"

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", EXPERIMENT, f"{EXPERIMENT}_{DEVICE}")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"✅ Selected: {EXPERIMENT.upper()} on {DEVICE.upper()}")
print(f"📁 Saving figures to: {RESULTS_DIR}")


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", EXPERIMENT, f"{EXPERIMENT}_{DEVICE}")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"📁 Saving figures to: {RESULTS_DIR}")

# %% ------------------------------------------------------------
# 2. Load JSON Metrics Dynamically
# ------------------------------------------------------------
possible_names = [
    f"cifar10_{EXPERIMENT}_metrics_{DEVICE}.json",
    f"cifar10_{EXPERIMENT.replace('ion','ed')}_metrics_{DEVICE}.json",
    f"cifar10_{EXPERIMENT.replace('ed','ion')}_metrics_{DEVICE}.json"
]

metrics = None
for fname in possible_names:
    path = os.path.join(LOGS_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            metrics = json.load(f)
        print(f"✅ Loaded metrics from {fname}")
        break
if not metrics:
    print(f"⚠️ No metrics file found for {EXPERIMENT} ({DEVICE})")
else:
    print(f"✅ Summary keys: {list(metrics.keys())}")

# %% ------------------------------------------------------------
# 3. Load Emissions and Training Logs
# ------------------------------------------------------------
emissions_path = os.path.join(LOGS_DIR, "emissions.csv")
emissions_df = pd.read_csv(emissions_path) if os.path.exists(emissions_path) else None
if emissions_df is not None:
    print(f"✅ Loaded emissions data ({len(emissions_df)} rows)")

train_path = os.path.join(LOGS_DIR, f"training_log_{EXPERIMENT}_{DEVICE}.csv")
train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
if train_df is not None:
    train_df.columns = [c.strip().lower().replace(" ", "_") for c in train_df.columns]
    if "epoch" not in train_df.columns:
        train_df.insert(0, "epoch", range(1, len(train_df)+1))
    print(f"✅ Loaded training log ({len(train_df)} epochs)")
else:
    print("⚠️ No training log found.")

# %% ------------------------------------------------------------
# 4. Plot CO₂ Emissions Over Time
# ------------------------------------------------------------
if emissions_df is not None and not emissions_df.empty:
    plt.figure(figsize=(10,5))
    plt.plot(emissions_df["duration"], emissions_df["emissions"], marker="o")
    plt.title(f"CO₂ Emissions Over Time ({EXPERIMENT.upper()} - {DEVICE.upper()})")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("CO₂ Emissions (kg)")
    plt.grid(True)
    plt.show()

# %% ------------------------------------------------------------
# 5. Plot Accuracy and Loss
# ------------------------------------------------------------
if train_df is not None:
    acc_cols = [c for c in train_df.columns if "acc" in c]
    loss_cols = [c for c in train_df.columns if "loss" in c]
    if acc_cols and loss_cols:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        for c in acc_cols:
            plt.plot(train_df["epoch"], train_df[c], label=c)
        plt.legend(); plt.title("Accuracy per Epoch"); plt.grid(True)

        plt.subplot(1,2,2)
        for c in loss_cols:
            plt.plot(train_df["epoch"], train_df[c], label=c)
        plt.legend(); plt.title("Loss per Epoch"); plt.grid(True)
        plt.tight_layout()
        plt.show()

# %% ------------------------------------------------------------
# 6. Efficiency Summary
# ------------------------------------------------------------
if metrics:
    if "quantized" in metrics and "baseline" in metrics:
        acc = metrics["quantized"].get("accuracy", 0)
        co2_total = metrics["quantized"].get("co2_kg", 0)
    else:
        acc = metrics.get("accuracy", 0)
        co2_total = metrics.get("co2_kg", 0)

    eff = acc / co2_total if co2_total else 0
    labels = ["Accuracy (%)", "CO₂ Efficiency (Acc/kg)"]
    values = [acc, eff]

    plt.figure(figsize=(6,4))
    plt.barh(labels, values, color=["#4CAF50", "#009688"])
    plt.title(f"Green AI Efficiency Summary ({EXPERIMENT.upper()} - {DEVICE.upper()})")
    for i, v in enumerate(values):
        plt.text(v, i, f"{v:.2f}", color="black", va="center")
    plt.show()
# %% ------------------------------------------------------------
# 🌿 Fixed Green AI Lifecycle Comparison (Dynamic)
# Handles both flat and nested metric JSONs + auto-scales log axis
# ------------------------------------------------------------
experiments = ["baseline", "pruned", "quantized"]
devices = ["gpu", "cpu"]

data = []
for exp in experiments:
    for dev in devices:
        path = os.path.join(LOGS_DIR, f"cifar10_{exp}_metrics_{dev}.json")
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)

            # ✅ Handle nested structure (quantized JSONs)
            if exp == "quantized" and "quantized" in m:
                acc = m["quantized"].get("accuracy", np.nan)
                co2 = m["quantized"].get("co2_kg", np.nan)
            elif "accuracy" in m and "co2_kg" in m:
                acc = m["accuracy"]
                co2 = m["co2_kg"]
            elif "baseline" in m:  # fallback: take baseline block if present
                acc = m["baseline"].get("accuracy", np.nan)
                co2 = m["baseline"].get("co2_kg", np.nan)
            else:
                acc, co2 = np.nan, np.nan

            data.append({
                "Model": f"{exp.capitalize()} {dev.upper()}",
                "Accuracy": acc,
                "CO2": co2
            })
from IPython.display import display
df = pd.DataFrame(data)
if not df.empty:
    # 🧮 Derived metrics
    df["Efficiency"] = df["Accuracy"] / df["CO2"]
    print("✅ Loaded comparison metrics:")
    display(df)

    # 🌈 Color by CO₂ Efficiency Level
    def color_by_eff(v):
        if v < 0.001: return "#2ECC71"  # Green = very efficient
        elif v < 0.02: return "#F39C12"  # Orange = moderate
        else: return "#E74C3C"          # Red = high emission

    co2_colors = df["CO2"].apply(color_by_eff)
    x = np.arange(len(df))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(13, 6))

    # 🎯 Accuracy Bars
    acc_bars = ax1.bar(x - width / 2, df["Accuracy"], width,
                       color="#3498DB", label="Accuracy (%)", zorder=3)
    ax1.set_ylabel("Accuracy (%)", color="#3498DB")
    ax1.tick_params(axis="y", labelcolor="#3498DB")
    ax1.set_ylim(0, max(df["Accuracy"].fillna(0)) + 10)

    # 🌿 CO₂ Bars (Log scale) — dynamic lower bound
    min_co2 = df["CO2"].min(skipna=True)
    lower_limit = min_co2 * 0.5 if min_co2 > 0 else 1e-7

    ax2 = ax1.twinx()
    co2_bars = ax2.bar(x + width / 2, df["CO2"], width,
                       color=co2_colors, alpha=0.8, label="CO₂ Emissions (kg)", zorder=2)
    ax2.set_yscale("log")
    ax2.set_ylabel("CO₂ Emissions (kg, log scale)", color="#16A085")
    ax2.tick_params(axis="y", labelcolor="#16A085")
    ax2.set_ylim(lower_limit, max(df["CO2"].fillna(1e-5)) * 10)

    # 🏷 Annotate both Accuracy and CO₂ values
    for i, (a_bar, c_bar) in enumerate(zip(acc_bars, co2_bars)):
        # accuracy label
        ax1.text(a_bar.get_x() + a_bar.get_width() / 2,
                 a_bar.get_height() + 0.5,
                 f"{df['Accuracy'][i]:.2f}%",
                 ha='center', va='bottom', fontsize=8, color="#2C3E50")

        # co2 label (never drops below visible range)
        text_y = max(c_bar.get_height() * 1.5, lower_limit * 3)
        ax2.text(c_bar.get_x() + c_bar.get_width() / 2,
                 text_y,
                 f"{df['CO2'][i]:.6f}",
                 ha='center', va='bottom', fontsize=8, rotation=90, color='black')

    # 🧭 X labels and title
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax1.set_title("🌿 Green AI Lifecycle Comparison — CIFAR-10 (Dynamic)")

    # 🌈 Custom Legend
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color="#3498DB", label="Accuracy (%)"),
        mpatches.Patch(color="#2ECC71", label="Low CO₂ (Efficient)"),
        mpatches.Patch(color="#F39C12", label="Moderate CO₂"),
        mpatches.Patch(color="#E74C3C", label="High CO₂")
    ]
    ax1.legend(handles=legend_items, loc="upper right")

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

else:
    print("⚠️ No comparison data available.")




# %% ------------------------------------------------------------
# Done
# ------------------------------------------------------------
print(f"\n✅ Visualization complete for {EXPERIMENT.upper()} ({DEVICE.upper()})")
