"""CodeCarbon integration and energy tracking helpers."""
from codecarbon import OfflineEmissionsTracker
import json
import os

def start_tracker(run_name: str, output_dir: str = 'logs'):
    os.makedirs(output_dir, exist_ok=True)
    tracker = OfflineEmissionsTracker(project_name=run_name, output_dir=output_dir)
    tracker.start()
    return tracker

def stop_and_save(tracker, run_id: str):
    emissions: dict = tracker.stop()
    out_path = os.path.join('logs', f'emissions_{run_id}.json')
    with open(out_path, 'w') as f:
        json.dump(emissions, f, indent=2)
    return out_path
