import argparse
import os
import pickle
import sys
from typing import Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("spins-b")
sys.path.append("spins-b/power_splitter_")

from spins import goos
from spins.goos import flows, graph_executor

import power_splitter_cont_opt as opt_module


def detect_stage(data: dict) -> str:
    """Return 'sig' if the pickle corresponds to the discrete (sigmoid) phase."""
    monitors = data.get("monitor_data", {}) or {}
    for key in monitors:
        if "sig" in key:
            return "sig"
    # Fallback: check optimizer metadata
    optim_logs = data.get("optimization_logs", {}) or {}
    for key in optim_logs:
        if "discrete" in key or "sig" in key:
            return "sig"
    return "cont"


def extract_design_and_factor(data: dict) -> Tuple[np.ndarray, float | None]:
    design_vals = np.array(
        data["variable_data"]["design_var"]["value"], dtype=float
    )
    discr = None
    discr_entry = data.get("variable_data", {}).get("discr_factor")
    if discr_entry is not None and "value" in discr_entry:
        discr = float(np.asarray(discr_entry["value"]).item())
    return design_vals, discr


def build_simulation(config: opt_module.SplitterConfig, stage: str, discr_factor: float | None):
    """Create the correct simulation graph (continuous or sigmoid/discrete)."""
    var, wg_in, wg_up, wg_down, design, _ = opt_module.create_design(config)

    if stage == "sig":
        # Use a sigmoid layer, matching how discrete optimization runs.
        sig_value = discr_factor if discr_factor is not None else config.optimization.sigmoid_factors[-1]
        sigmoid_factor = goos.Variable(sig_value, parameter=True, name="discr_factor_eval")
        design_shape = goos.cast(goos.Sigmoid(sigmoid_factor * (2 * design - 1)), goos.Shape)
        eps_struct = goos.GroupShape([wg_in, wg_up, wg_down, design_shape])
        sim = opt_module.create_simulation(eps_struct, config, name="sim_splitter_sig_eval")
    else:
        eps_struct = goos.GroupShape([wg_in, wg_up, wg_down, design])
        sim = opt_module.create_simulation(eps_struct, config, name="sim_splitter_cont_eval")

    return var, sim


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved splitter design (continuous or sigmoid).")
    parser.add_argument("pkl", help="Path to checkpoint .pkl (stepXX.pkl)")
    parser.add_argument(
        "--stage",
        choices=["auto", "cont", "sig"],
        default="auto",
        help="Specify whether the checkpoint is from the continuous or sigmoid phase. Default: auto-detect.",
    )
    parser.add_argument("--save-path", default=".", help="Temporary GOOS plan save folder.")
    
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show(), only print powers.")
    parser.add_argument("--fig-prefix", default="quick_sim", help="Prefix for saved figures (PNG).")
    args = parser.parse_args()

    if not os.path.isfile(args.pkl):
        raise FileNotFoundError(args.pkl)

    with open(args.pkl, "rb") as fp:
        data = pickle.load(fp)

    design_vals, discr_factor = extract_design_and_factor(data)
    print(
        f"Loaded design_var with shape {design_vals.shape} "
        f"and range [{design_vals.min():.3f}, {design_vals.max():.3f}]"
    )

    stage = args.stage
    if stage == "auto":
        stage = detect_stage(data)
    print(f"Evaluation stage: {stage.upper()} (TE mode in create_simulation)")

    config = opt_module.SplitterConfig()

    with goos.OptimizationPlan(save_path=args.save_path) as plan:
        var, sim = build_simulation(config, stage, discr_factor)

        const_flags = flows.NumericFlow.ConstFlags()
        frozen_flags = flows.NumericFlow.ConstFlags(False)
        context = goos.NodeFlags(const_flags=const_flags, frozen_flags=frozen_flags)
        override_map = {var: (flows.NumericFlow(design_vals), context)}

        flow_results = graph_executor.eval_fun(
            [sim["overlap_up"], sim["overlap_down"], sim["eps"], sim["field"]],
            override_map,
        )

        overlap_up = flow_results[0].array
        overlap_down = flow_results[1].array
        eps_raw = flow_results[2].array
        field_raw = flow_results[3].array

        power_up = np.abs(overlap_up) ** 2
        power_down = np.abs(overlap_down) ** 2
        total = power_up + power_down

        print(f"Power Up  : {power_up:.6f}")
        print(f"Power Down: {power_down:.6f}")
        print(f"Total     : {total:.6f}")

        # ================================================
        eps = np.real(eps_raw[2])
        field_mag = np.linalg.norm(field_raw, axis=0)

        z_slice_idx = eps.shape[2] // 2
        eps_slice = eps[:, :, z_slice_idx]
        field_slice = field_mag[:, :, z_slice_idx]

        eps_plot = eps_slice.T
        field_plot = field_slice.T

        nm_to_um = 1.0 / 1000.0
        x_extent = config.simulation.region * nm_to_um
        y_extent = config.simulation.region * nm_to_um
        extent = [-x_extent / 2, x_extent / 2, -y_extent / 2, y_extent / 2]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax_eps = axes[0]
        ax_field = axes[1]

        im1 = ax_eps.imshow(
            eps_plot,
            cmap="viridis",
            aspect="equal",
            extent=extent,
            origin="lower",
        )
        ax_eps.set_title("Permittivity (ε)", fontsize=12, fontweight="bold")
        ax_eps.set_xlabel("x (μm)")
        ax_eps.set_ylabel("y (μm)")
        ax_eps.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig.colorbar(im1, ax=ax_eps, label="|ε|")

        im2 = ax_field.imshow(
            field_plot,
            cmap="hot",
            aspect="equal",
            extent=extent,
            origin="lower",
        )
        ax_field.set_title("|E| Field Magnitude", fontsize=12, fontweight="bold")
        ax_field.set_xlabel("x (μm)")
        ax_field.set_ylabel("y (μm)")
        ax_field.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig.colorbar(im2, ax=ax_field, label="|E|")

        plt.tight_layout()

        base = os.path.splitext(os.path.basename(args.pkl))[0]
        fig_name = f"{args.fig_prefix}_{base}.png"
        out_path = os.path.join(args.save_path, fig_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved permittivity/field figure to: {out_path}")

        if not args.no_show:
            plt.show()

        # ================================================

if __name__ == "__main__":
    main()