
import pandas as pd
import numpy as np

results_df = pd.read_csv('/Users/pranayvure/E_distance_benchmarking/e_distance_results.csv') 

results_df['E_Distance'] = np.sqrt(results_df['Squared_E_Distance'].clip(lower=0))

print("\n--- Squared E-distance values ---")
print(results_df['Squared_E_Distance'].describe())

print("\n--- Square-Root E-distance values ---")
print(results_df['E_Distance'].describe())


# Calculate summary statistics on the square-rooted values
mean_e_distance = results_df['E_Distance'].mean()
median_e_distance = results_df['E_Distance'].median()

print(f"\nOverall Model Performance Summary (based on sqrt E-distance):")
print(f"Mean E-distance across {len(results_df)} perturbations: {mean_e_distance:.6f}")
print(f"Median E-distance across {len(results_df)} perturbations: {median_e_distance:.6f}")

# Optional: Save results to CSV including both values
# results_df.to_csv("e_distance_results.csv")
# print("\nResults saved to e_distance_results.csv")
