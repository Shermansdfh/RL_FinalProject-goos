"""Discretize a saved SPINS-B power-splitter design.

Reads a design pickle (e.g., from optimization steps), extracts the design variable 
(the optimization parameters, not the full epsilon mesh), discretizes it into 
binary values (silicon/core vs. silica/background), and saves a new pickle 
that preserves the structure needed for re-simulation (e.g. by quick_sim_test.py).

Outputs
-------
1. `<basename>_discrete.pkl`: Contains the 'variable_data' structure with discretized values.
2. `<basename>_binary_center.txt`: 50x50 binary mask (1=core, 0=bg).
"""

import argparse
import os
import pickle
import sys
import numpy as np

def find_design_var_in_pickle(data):
    """Extract the design variable array directly from optimization data."""
    if isinstance(data, dict):
        # Primary path: optimization variable state
        if "variable_data" in data and "design_var" in data["variable_data"]:
            return data["variable_data"]["design_var"]["value"]
    
    raise ValueError("Could not locate 'design_var' in 'variable_data' within the pickle.")

def kmeans2_1d(values, max_iters=50, tol=1e-6):
    """Simple k-means clustering for 1D data (k=2) to find background/core levels."""
    v = values.reshape(-1)
    c0, c1 = v.min(), v.max()
    
    for _ in range(max_iters):
        d0 = np.abs(v - c0)
        d1 = np.abs(v - c1)
        mask = d1 < d0
        
        if mask.sum() == 0 or (~mask).sum() == 0:
            break
            
        new_c0 = v[~mask].mean()
        new_c1 = v[mask].mean()
        
        if abs(new_c0 - c0) < tol and abs(new_c1 - c1) < tol:
            c0, c1 = new_c0, new_c1
            break
        c0, c1 = new_c0, new_c1
        
    return np.sort(np.array([c0, c1]))

def main():
    parser = argparse.ArgumentParser(description="Discretize design variable from .pkl")
    parser.add_argument("pkl", help="Input optimization .pkl file")
    parser.add_argument("--out-pkl", help="Path for discretized output pickle")
    parser.add_argument("--out-binary", help="Path for binary center TXT export")
    parser.add_argument("--threshold", type=float, help="Manual threshold (0-1). Default: auto-detected via k-means.")
    
    args = parser.parse_args()

    if not os.path.isfile(args.pkl):
        raise FileNotFoundError(args.pkl)

    # 1. Load original data
    with open(args.pkl, "rb") as fp:
        data = pickle.load(fp)

    # 2. Extract design variable (the 50x50 parameter array)
    # This is typically in range [0, 1] for continuous optimization
    design_vals = find_design_var_in_pickle(data)
    
    print(f"Loaded design variable. Shape: {design_vals.shape}, Range: [{design_vals.min():.3f}, {design_vals.max():.3f}]")

    # 3. Determine Threshold
    if args.threshold is not None:
        thresh = args.threshold
        val_low, val_high = 0.0, 1.0
    else:
        # Auto-detect levels (usually 0 and 1)
        centers = kmeans2_1d(design_vals)
        val_low, val_high = centers[0], centers[1]
        thresh = 0.5 * (val_low + val_high)
        print(f"Auto-detected levels: {val_low:.3f} / {val_high:.3f}. Threshold: {thresh:.3f}")

    # 4. Discretize
    # For the variable_data structure, we want to preserve the optimized bounds [0, 1]
    # usually, so we snap to 0 or 1 exactly.
    binary_mask = (design_vals >= thresh).astype(int)
    
    # 5. Prepare Output Data Structure
    # We clone the input data structure so it remains compatible with loading scripts
    # that expect 'variable_data' -> 'design_var'
    out_data = data.copy()
    
    # Update the value
    # Note: The optimization script expects float arrays for variables
    out_data["variable_data"]["design_var"]["value"] = binary_mask.astype(float)
    
    # We can also strip heavy monitor data if we want a lighter file, 
    # but keeping it might be useful for provenance. 
    # Let's update metadata to indicate modification.
    out_data["discretized"] = True
    out_data["original_pkl"] = os.path.abspath(args.pkl)

    # 6. Save Outputs
    base = os.path.splitext(os.path.basename(args.pkl))[0]
    out_pkl_path = args.out_pkl or (base + "_discrete.pkl")
    out_txt_path = args.out_binary or (base + "_binary_center.txt")

    # Ensure directories exist
    for path in [out_pkl_path, out_txt_path]:
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # Save Pickle
    with open(out_pkl_path, "wb") as fp:
        pickle.dump(out_data, fp)
    print(f"Saved discretized pickle to: {out_pkl_path}")
    print("  (Compatible with quick_sim_test.py)")

    # Save Binary TXT (center slice is just the array itself here, assuming 50x50 is the center design)
    # The design_vals IS the center design region in this optimization formulation.
    # If dimensions > 2, we take the middle z-slice.
    if binary_mask.ndim == 3:
        z_mid = binary_mask.shape[2] // 2
        txt_arr = binary_mask[:, :, z_mid].T  # Transpose for visual alignment (y, x)
    else:
        txt_arr = binary_mask.T

    np.savetxt(out_txt_path, txt_arr, fmt="%d")
    print(f"Saved binary mask text to:   {out_txt_path}")

if __name__ == "__main__":
    main()
