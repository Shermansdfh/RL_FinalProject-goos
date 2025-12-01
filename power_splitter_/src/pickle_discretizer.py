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
    parser.add_argument("--optimize-threshold", action="store_true", help="Run binary search to find optimal threshold using quick_sim_test.py")
    
    args = parser.parse_args()

    if not os.path.isfile(args.pkl):
        raise FileNotFoundError(args.pkl)

    # 1. Load original data
    with open(args.pkl, "rb") as fp:
        data = pickle.load(fp)

    # 2. Extract design variable
    design_vals = find_design_var_in_pickle(data)
    print(f"Loaded design variable. Shape: {design_vals.shape}, Range: [{design_vals.min():.3f}, {design_vals.max():.3f}]")

    # 3. Determine Threshold
    best_thresh = None
    if args.optimize_threshold:
        print("\n--- Optimizing Threshold (using GOOS simulation) ---")
        # Import local modules
        # We need create_simulation and create_objective from the optimizer script
        # Assuming this script is in the same folder or we can import it.
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            import power_splitter_cont_opt as opt_script
            from spins import goos
            from spins.goos_sim import maxwell
        except ImportError:
             print("Error: Could not import 'power_splitter_cont_opt.py' or 'spins'.")
             print("Make sure you are running this script from the correct environment.")
             sys.exit(1)

        # Reconstruct config (using defaults or loading from file if possible)
        # For simplicity, we use defaults but try to match the grid size from pickle data
        config = opt_script.SplitterConfig()
        
        # Infer design dimensions from data shape if possible, or trust defaults
        # design_vals shape is (Nx, Ny, Nz) or (Nx, Ny)
        # Default pixel_size is 100nm. Design is 2000nm -> 20 pixels.
        # If design_vals is 20x20, pixel_size=100 is correct.
        nx = design_vals.shape[0]
        config.design.pixel_size = config.design.width / nx
        
        # Create base shapes (waveguides)
        _, wg_in, wg_up, wg_down, _, _ = opt_script.create_design(config)

        def evaluate_threshold(thresh):
            # 1. Discretize
            mask = (design_vals >= thresh).astype(float) # 0.0 or 1.0
            
            # FIX: Ensure mask is 3D (Nx, Ny, Nz)
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]

            # 2. Create Shape
            # We need to wrap this numpy array into a goos.Shape
            # Use pixelated_cont_shape logic but with fixed array
            var = goos.Constant(mask)
            
            # Reconstruct the PixelatedContShape manually or via helper
            # We'll use the low-level constructor to be safe
            design_shape = opt_script.goos.PixelatedContShape(
                array=var,
                pixel_size=[config.design.pixel_size, config.design.pixel_size, config.design.thickness], 
                pos=goos.Constant([0, 0, 0]),
                extents=[config.design.width, config.design.height, config.design.thickness],
                material=goos.material.Material(index=config.material.background_index),
                material2=goos.material.Material(index=config.material.core_index)
            )
            
            # 3. Build Simulation
            eps_struct = goos.GroupShape([wg_in, wg_up, wg_down, design_shape])
            sim = opt_script.create_simulation(eps_struct, config, name="sim_thresh_eval")
            
            # 4. Build Objective
            # We only need the values, not the symbolic graph for gradients
            # But create_objective returns symbolic nodes.
            # We need to EVALUATE them.
            obj, ratio_term, penalty_term, total_power_term, power_up, power_down = \
                opt_script.create_objective(sim, config, name_prefix="obj_thresh_eval")

            # 5. Run Simulation (Evaluate Graph)
            # We can use a temporary plan to evaluate
            with goos.OptimizationPlan(save_path=None) as plan:
                 # We just want to evaluate the objective nodes
                 # The easiest way in GOOS is often to ask for the value
                 # But since we are outside a standard run loop, we need to trigger evaluation.
                 # We can try plan.eval_node() or just compute the outputs.
                 
                 # Actually, GOOS simulation evaluation happens when we request the output.
                 # Let's compute total power and ratio error.
                 
                 # We need to run the simulation first.
                 # In GOOS, 'sim' is a Simulation object. Accessing its outputs usually triggers solve if needed.
                 # However, without an OptimizationPlan.run(), it might not execute.
                 # Let's just use a simple plan.run() with 0 iters or just evaluation.
                 
                 # A trick: define a dummy optimization with 0 iters to force evaluation?
                 # Or just directly compute if possible.
                 # Let's try to evaluate the 'obj' node.
                 
                 val_ratio_mse = plan.eval_node(ratio_term).array
                 val_total_power = plan.eval_node(total_power_term).array
                 val_power_up = plan.eval_node(power_up).array
                 val_power_down = plan.eval_node(power_down).array
                 
                 # Cost function: same as optimization target
                 # (ratio_mse + penalty)
                 val_obj = plan.eval_node(obj).array
                 
                 # Extract scalar
                 cost = float(val_obj.real) if np.iscomplexobj(val_obj) else float(val_obj)
                 up = float(val_power_up.real)
                 down = float(val_power_down.real)
                 total = float(val_total_power.real)
                 
                 eff = total
                 ratio = up / (total + 1e-12)
                 
                 print(f"Thresh: {thresh:.4f} | Ratio: {ratio:.4f} (Up={up:.3f}, Dn={down:.3f}) | Eff: {eff:.3f} | Cost: {cost:.6f}")
                 return cost

        # ... (Search logic remains similar) ...
        # Coarse search
        candidates = np.linspace(0.2, 0.8, 7)
        best_cost = float('inf')
        best_thresh = 0.5
        
        for th in candidates:
            try:
                c = evaluate_threshold(th)
                if c < best_cost:
                    best_cost = c
                    best_thresh = th
            except Exception as e:
                print(f"Sim failed for thresh {th}: {e}")
        
        # Fine search
        delta = 0.05
        for _ in range(3):
            low, high = max(0, best_thresh - delta), min(1, best_thresh + delta)
            sub_candidates = np.linspace(low, high, 5)
            for th in sub_candidates:
                try:
                    c = evaluate_threshold(th)
                    if c < best_cost:
                        best_cost = c
                        best_thresh = th
                except Exception: pass
            delta /= 2
            
        print(f"--- Optimal Threshold Found: {best_thresh:.4f} ---\n")
        thresh = best_thresh
    elif args.threshold is not None:
        thresh = args.threshold
    else:
        # Auto-detect levels (k-means)
        centers = kmeans2_1d(design_vals)
        thresh = 0.5 * (centers[0] + centers[1])
        print(f"Auto-detected threshold: {thresh:.3f}")

    # 4. Discretize using final threshold
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

    txt_arr_to_save = np.flipud(txt_arr)

    np.savetxt(out_txt_path, txt_arr_to_save, fmt="%d")
    print(f"Saved binary mask text to:   {out_txt_path}")

if __name__ == "__main__":
    main()
