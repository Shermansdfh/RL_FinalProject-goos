import pickle
import sys
import numpy as np

sys.path.append('spins-b')
sys.path.append('spins-b/power_splitter_')

from spins import goos
import power_splitter_cont_opt as opt_module

# Load the pickle
pkl_path = 'spins-b/power_splitter_/step6.pkl'  # or step160.pkl
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

design_vals = data['variable_data']['design_var']['value']

# Create simulation graph
with goos.OptimizationPlan(save_path='.') as plan:
    config = opt_module.SplitterConfig()
    var, wg_in, wg_up, wg_down, design, eps_render = opt_module.create_design(config)
    eps_struct = goos.GroupShape([wg_in, wg_up, wg_down, design])
    sim = opt_module.create_simulation(eps_struct, config)

    # CORRECT way to inject variable values
    from spins.goos import flows, graph_executor
    const_flags = flows.NumericFlow.ConstFlags()
    frozen_flags = flows.NumericFlow.ConstFlags(False)
    context = goos.NodeFlags(const_flags=const_flags, frozen_flags=frozen_flags)
    override_map = {var: (flows.NumericFlow(design_vals), context)}

    # Evaluate with the override
    flow_results = graph_executor.eval_fun([sim["overlap_up"], sim["overlap_down"]], override_map)
    
    overlap_up = flow_results[0].array
    overlap_down = flow_results[1].array
    
    power_up = np.abs(overlap_up)**2
    power_down = np.abs(overlap_down)**2
    
    print(f"Correct Overlap Up (Power): {power_up}")
    print(f"Correct Overlap Down (Power): {power_down}")