import joblib
import pandas as pd
import os

LOGS_DIR = "logs"

# Export both CUDA and CPU studies if available
for device in ["cuda", "cpu"]:
    pkl_path = os.path.join(LOGS_DIR, f"hpo_study_{device}.pkl")
    csv_path = os.path.join(LOGS_DIR, f"hpo_trials_{device}.csv")

    if os.path.exists(pkl_path):
        print(f"📂 Loading study: {pkl_path}")
        study = joblib.load(pkl_path)
        df = study.trials_dataframe()
        df.to_csv(csv_path, index=False)
        print(f"✅ Exported {csv_path} ({len(df)} trials)")
    else:
        print(f"⚠️ {pkl_path} not found, skipping.")

print("\n🎉 CSV export complete!")
