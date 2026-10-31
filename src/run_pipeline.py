# src/run_pipeline.py
import subprocess
import sys

steps = [
    ["python", "-m", "src.ingestion"],
    ["python", "-m", "src.train"],
    ["python", "-m", "src.anomaly_pipeline"],
]

for step in steps:
    print(f"\nRunning: {' '.join(step)}")
    subprocess.run(step, check=True)

print("\nData, model, and anomaly pipeline completed.")