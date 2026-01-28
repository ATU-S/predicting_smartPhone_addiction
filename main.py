from src.experiment import run_experiment
import os

def main():
    print("Starting Smartphone Addiction ML Experiment...")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Run experiment
    results_df = run_experiment()

    # Save results
    output_path = "results/feature_comparison_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nExperiment completed successfully.")
    print(f"Results saved to: {output_path}\n")

    print(results_df)

if __name__ == "__main__":
    main()
