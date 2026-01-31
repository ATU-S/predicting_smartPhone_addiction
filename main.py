from src.experiment import run_experiment
from src.config import RESULTS_DIR, RESULTS_FILE
import os

def main():
    print("Starting Smartphone Addiction ML Experiment...")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_experiment()
    results.to_csv(RESULTS_FILE, index=False)

    print("\nExperiment completed successfully")
    print(results)

if __name__ == "__main__":
    main()
