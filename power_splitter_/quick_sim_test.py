import pickle
import sys
import numpy as np

sys.path.append('spins-b')
sys.path.append('spins-b/power_splitter_')

from spins import goos
import power_splitter_cont_opt as opt_module

# 1. Load the optimized variable data
pkl_path = 'step99.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

design_vals = data['variable_data']['design_var']['value']

# 2. Reconstruct graph and run simulation
# The context manager handles the graph state
with goos.OptimizationPlan(save_path='.') as plan:
    config = opt_module.SplitterConfig()
    
    # Rebuild the design and simulation graph
    # var is the goos.Variable we need to update
    var, wg_in, wg_up, wg_down, design, eps_render = opt_module.create_design(config)
    eps_struct = goos.GroupShape([wg_in, wg_up, wg_down, design])
    sim = opt_module.create_simulation(eps_struct, config)

    # 3. Set the variable to the optimized values
    var.set(design_vals)
    
    # 4. Execute the simulation node to get results
    # sim.get() triggers the evaluation of the graph up to that node
    # It returns a list of results corresponding to the 'outputs' list in create_simulation
    results = sim.get()

    # 5. Extract specific outputs (order matches the 'outputs' list in create_simulation)
    # [eps, field, overlap_up, overlap_down]
    eps_data = results[0].array
    field_data = results[1].array
    overlap_up = results[2].array
    overlap_down = results[3].array

    print(f"Overlap Up (Power): {np.abs(overlap_up)**2}")
    print(f"Overlap Down (Power): {np.abs(overlap_down)**2}")