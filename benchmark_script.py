import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import time

# --- Function Definition (Ensure it handles layers/sparsity) ---
def compute_e_distance_sqeuclidean(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the squared Energy Distance based on average squared Euclidean distances.
    Handles potential sparse matrix input and checks for empty arrays.
    """
    # Convert sparse to dense if necessary
    # Check for sparse matrix format scipy.sparse instead of just hasattr
    from scipy.sparse import issparse
    if issparse(X):
        X = X.toarray()
    if issparse(Y):
        Y = Y.toarray()

    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Check if dimensions match AFTER potential conversion
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
         raise ValueError(f"Feature dimensions invalid or do not match: X shape {X.shape}, Y shape {Y.shape}")
    if X.shape[0] == 0 or Y.shape[0] == 0:
        print("Warning: One or both input arrays for E-distance calculation are empty. Returning NaN.")
        return np.nan

    # Check memory usage potential BEFORE calculation
    mem_needed_gb = (X.shape[0]**2 + Y.shape[0]**2 + X.shape[0]*Y.shape[0]) * X.itemsize * 8 / (1024**3)
    print(f"Calculating pairwise distances for X ({X.shape[0]}x{X.shape[1]}) and Y ({Y.shape[0]}x{Y.shape[1]}). Est. peak memory: ~{mem_needed_gb:.2f} GB")

    try:
        # Use n_jobs=-1 to utilize all available CPU cores, can speed up large calculations
        sigma_X = pairwise_distances(X, X, metric='sqeuclidean', n_jobs=-1).mean()
        sigma_Y = pairwise_distances(Y, Y, metric='sqeuclidean', n_jobs=-1).mean()
        delta = pairwise_distances(X, Y, metric='sqeuclidean', n_jobs=-1).mean()
    except MemoryError:
        print("\nERROR: MemoryError occurred during pairwise distance calculation.")
        print("Consider using dimensionality reduction (PCA) first, or subsampling cells.")
        return np.nan
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during pairwise_distances: {e}")
        return np.nan


    e_distance_sq = 2 * delta - sigma_X - sigma_Y
    # Clamp result to be non-negative
    return max(0.0, e_distance_sq)

# --- Configuration ---
PATH_SRIVATSAN_ADATA = "/Users/pranayvure/E_distance_benchmarking/srivatsan20_processed.h5ad" # <-- CHANGE THIS
PATH_PREDICTION_ADATA = "/Users/pranayvure/E_distance_benchmarking/lam_srivatsan20_test_predictions.h5ad" # <-- Used your provided path

# --- Load Data ---
print(f"Loading ground truth data: {PATH_SRIVATSAN_ADATA}")
try:
    adata_gt = ad.read_h5ad(PATH_SRIVATSAN_ADATA)
    print(f"Ground truth data loaded: {adata_gt.n_obs} cells x {adata_gt.n_vars} genes")
except Exception as e:
    print(f"Error loading ground truth file: {e}")
    exit()

print(f"\nLoading prediction data: {PATH_PREDICTION_ADATA}")
try:
    adata_pred = ad.read_h5ad(PATH_PREDICTION_ADATA)
    print(f"Prediction data loaded: {adata_pred.n_obs} cells x {adata_pred.n_vars} genes")
    # Check if the required layer exists
    if 'lam_pred' not in adata_pred.layers:
         print("\nERROR: Prediction file does not contain the required layer 'lam_pred'.")
         exit()
except Exception as e:
    print(f"Error loading prediction file: {e}")
    exit()

# --- Step 1: Align Features (Genes) ---
print("\nAligning features (genes)...")
common_genes = adata_gt.var_names.intersection(adata_pred.var_names)
n_common_genes = len(common_genes)

if n_common_genes == 0:
    print("ERROR: No common genes found between the two files. Cannot proceed.")
    exit()
elif n_common_genes < adata_gt.n_vars or n_common_genes < adata_pred.n_vars:
    print(f"Found {n_common_genes} common genes.")
    print(f"Filtering ground truth data from {adata_gt.n_vars} to {n_common_genes} genes.")
    adata_gt = adata_gt[:, common_genes].copy() # Use .copy() to avoid view warnings
    print(f"Filtering prediction data from {adata_pred.n_vars} to {n_common_genes} genes.")
    adata_pred = adata_pred[:, common_genes].copy()
else:
    print("Feature sets already seem aligned.")

# --- Step 2: Identify Common Perturbations (excluding control) ---
print("\nIdentifying common perturbations...")
gt_pert_set = set(adata_gt.obs['perturbation'].unique())
pred_pert_set = set(adata_pred.obs['perturbation'].unique())

common_perts = list(gt_pert_set.intersection(pred_pert_set))

# Exclude 'control' from the list we iterate over
if 'control' in common_perts:
    common_perts.remove('control')

print(f"Found {len(common_perts)} common perturbations (excluding 'control') to compare.")
if not common_perts:
    print("ERROR: No common perturbation labels found (excluding 'control'). Check 'perturbation' columns.")
    exit()

# --- Step 3: Loop, Subset, and Calculate E-distance ---
results = {}
print("\nCalculating E-distance for each perturbation:")
calculation_start_time = time.time()

for pert_name in sorted(common_perts): # Sort for consistent output order
    print(f"\n--- Processing: {pert_name} ---")

    # Get True Perturbed Data (from adata_gt.X)
    true_mask = adata_gt.obs['perturbation'] == pert_name
    adata_true_subset = adata_gt[true_mask, :]
    X_true = adata_true_subset.X # Keep as potentially sparse

    # Get Predicted Perturbed Data (from adata_pred.layers['lam_pred'])
    pred_mask = adata_pred.obs['perturbation'] == pert_name
    adata_pred_subset = adata_pred[pred_mask, :]
    # IMPORTANT: Access the correct layer!
    Y_pred = adata_pred_subset.layers['lam_pred'] # Keep as potentially sparse

    print(f"True cells: {adata_true_subset.n_obs}, Predicted cells: {adata_pred_subset.n_obs}")

    if adata_true_subset.n_obs == 0 or adata_pred_subset.n_obs == 0:
        print("Skipping E-distance calculation due to zero cells in one or both subsets.")
        results[pert_name] = np.nan
        continue

    # Calculate distance
    iter_start_time = time.time()
    e_dist = compute_e_distance_sqeuclidean(X_true, Y_pred) # Function handles sparsity check
    iter_end_time = time.time()

    if np.isnan(e_dist):
         print(f"E-distance calculation failed for {pert_name}.")
    else:
         print(f"Squared E-distance for {pert_name}: {e_dist:.6f} (took {iter_end_time - iter_start_time:.2f}s)")
    results[pert_name] = e_dist

calculation_end_time = time.time()
print(f"\n--- Finished all calculations in {calculation_end_time - calculation_start_time:.2f} seconds ---")

# --- Display Results ---
print("\nE-distance Results Summary:")
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Squared_E_Distance'])
print(results_df.sort_values(by='Squared_E_Distance')) # Sort by distance

# Optional: Save results to CSV
results_df.to_csv("e_distance_results.csv")
print("\nResults saved to e_distance_results.csv")