"""Single-simulation sanity check for straight waveguide transmission.

This utility reuses the shared configuration dataclasses from
`power_splitter_cont_opt.py` to run a single FDFD solve with a straight
waveguide spanning the full simulation region. It prints raw overlap amplitude,
power, dB loss, and a simple pass/fail verdict so you can confirm that the
simulation stack (mesh, sources, monitors, etc.) is healthy before debugging
inverse-design runs.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from spins import goos
from spins.goos_sim import maxwell

from power_splitter_cont_opt import (
    DesignConfig,
    MaterialConfig,
    SimulationConfig,
    WaveguideConfig,
)


def build_waveguide_core(
    design_cfg: DesignConfig,
    wg_cfg: WaveguideConfig,
    mat_cfg: MaterialConfig,
    sim_cfg: SimulationConfig,
):
    """Create a single straight silicon waveguide spanning the simulation box."""
    # Extend slightly beyond sim region to ensure continuity through PML.
    length = sim_cfg.region + 2 * sim_cfg.pml_thickness
    return goos.Cuboid(
        pos=goos.Constant([0, 0, 0]),
        extents=goos.Constant([length, wg_cfg.width, design_cfg.thickness]),
        material=goos.material.Material(index=mat_cfg.core_index),
    )


def source_center_x(design_cfg: DesignConfig, wg_cfg: WaveguideConfig, sim_cfg: SimulationConfig) -> float:
    input_start = -design_cfg.width / 2 - wg_cfg.input_length
    return input_start * sim_cfg.source_shift


def create_straight_simulation(
    waveguide_shape: goos.Shape,
    design_cfg: DesignConfig,
    wg_cfg: WaveguideConfig,
    sim_cfg: SimulationConfig,
    mat_cfg: MaterialConfig,
):
    """Construct a single FDFD simulation with the straight core waveguide.

    In addition to the overlap monitor, we also record the full electric field
    so that we can inspect mode components (Ex/Ey/Ez).
    """
    return maxwell.fdfd_simulation(
        name="straight_waveguide",
        wavelength=sim_cfg.wavelength,
        eps=waveguide_shape,
        solver="local_direct",
        sources=[
            maxwell.WaveguideModeSource(
                center=[source_center_x(design_cfg, wg_cfg, sim_cfg), 0, 0],
                extents=[0, wg_cfg.width * 3, design_cfg.thickness * 4],
                normal=[1, 0, 0],
                mode_num=1,
                power=1,
            )
        ],
        simulation_space=maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=design_cfg.resolution),
            sim_region=goos.Box3d(
                center=[0, 0, 0],
                extents=[sim_cfg.region, sim_cfg.region, sim_cfg.z_extent],
            ),
            pml_thickness=[
                sim_cfg.pml_thickness,
                sim_cfg.pml_thickness,
                sim_cfg.pml_thickness,
                sim_cfg.pml_thickness,
                0,
                0,
            ],
        ),
        background=goos.material.Material(index=mat_cfg.background_index),
        outputs=[
            maxwell.WaveguideModeOverlap(
                name="straight_overlap",
                center=[sim_cfg.monitor_position, 0, 0],
                extents=[0, wg_cfg.width * 2, design_cfg.thickness * 4],
                normal=[1, 0, 0],
                mode_num=1,
                power=1,
            ),
            maxwell.ElectricField(name="field"),
        ],
    )


def compute_metrics(overlap_flow) -> Tuple[float, float, float]:
    """Return (amplitude, power, dB loss) from a complex overlap flow."""
    value = complex(np.asarray(overlap_flow.array).item())
    amplitude = abs(value)
    power = amplitude ** 2
    loss_db = 10 * math.log10(max(power, 1e-12))
    return amplitude, power, loss_db


def run_straight_check(save_folder: str | None):
    design_cfg = DesignConfig()
    wg_cfg = WaveguideConfig()
    sim_cfg = SimulationConfig()
    mat_cfg = MaterialConfig()

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        goos.util.setup_logging(save_folder)

    with goos.OptimizationPlan(save_path=save_folder) as plan:
        straight_core = build_waveguide_core(design_cfg, wg_cfg, mat_cfg, sim_cfg)
        sim = create_straight_simulation(straight_core, design_cfg, wg_cfg, sim_cfg, mat_cfg)
        overlap = sim["straight_overlap"]
        field_flow = sim["field"]

        # First run the simulation once, retrieving the full electric field.
        field_result = field_flow.get(run=True)
        # Then reuse the same run to get the overlap (no need to rerun).
        overlap_flow = overlap.get(run=False)
        amplitude, power, loss_db = compute_metrics(overlap_flow)

        print(f"Raw overlap amplitude : {amplitude:.6f}")
        print(f"Calculated power      : {power:.6f}")
        print(f"dB loss               : {loss_db:.2f} dB")

        if power > 0.95:
            print("[PASS] Simulation setup is healthy.")
        else:
            print("[FAIL] Baseline transmission is too low. Check mesh resolution, PML settings, or monitor mode mismatch.")

        # ------------------------------------------------------------------
        # Analyze and visualize mode components Ex/Ey/Ez for this straight WG.
        # ------------------------------------------------------------------
        field_raw = np.asarray(field_result.array)

        if field_raw.ndim != 4 or field_raw.shape[0] != 3:
            raise RuntimeError(
                f"Unexpected ElectricField shape {field_raw.shape}; "
                "expected (3, Nx, Ny, Nz) for (Ex, Ey, Ez)."
            )

        Ex = field_raw[0]
        Ey = field_raw[1]
        Ez = field_raw[2]

        Ex_energy = np.sum(np.abs(Ex) ** 2)
        Ey_energy = np.sum(np.abs(Ey) ** 2)
        Ez_energy = np.sum(np.abs(Ez) ** 2)
        total_energy = Ex_energy + Ey_energy + Ez_energy + 1e-30

        print("\nField component energy (unnormalized ∑|E|^2 over grid):")
        print(f"  Ex: {Ex_energy:.4e}  ({Ex_energy / total_energy:.2%} of total)")
        print(f"  Ey: {Ey_energy:.4e}  ({Ey_energy / total_energy:.2%} of total)")
        print(f"  Ez: {Ez_energy:.4e}  ({Ez_energy / total_energy:.2%} of total)")

        # For visualization, take a central z-slice of Ey / Ez.
        z_slice_idx = Ey.shape[2] // 2
        Ey_slice = np.abs(Ey[:, :, z_slice_idx]) ** 2
        Ez_slice = np.abs(Ez[:, :, z_slice_idx]) ** 2

        # Transpose to (y, x) for imshow (rows, cols) = (y, x).
        Ey_plot = Ey_slice.T
        Ez_plot = Ez_slice.T

        nm_to_um = 1.0 / 1000.0
        x_extent = sim_cfg.region * nm_to_um
        y_extent = sim_cfg.region * nm_to_um
        extent = [-x_extent / 2, x_extent / 2, -y_extent / 2, y_extent / 2]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im_ey = axes[0].imshow(
            Ey_plot,
            cmap="plasma",
            aspect="equal",
            extent=extent,
            origin="lower",
        )
        axes[0].set_title("|Ey|^2 (central z-slice)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("x (μm)")
        axes[0].set_ylabel("y (μm)")
        axes[0].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig.colorbar(im_ey, ax=axes[0], label="|Ey|^2")

        im_ez = axes[1].imshow(
            Ez_plot,
            cmap="plasma",
            aspect="equal",
            extent=extent,
            origin="lower",
        )
        axes[1].set_title("|Ez|^2 (central z-slice)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("x (μm)")
        axes[1].set_ylabel("y (μm)")
        axes[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig.colorbar(im_ez, ax=axes[1], label="|Ez|^2")

        plt.tight_layout()
        plt.show()

        if save_folder:
            fig_path = os.path.join(save_folder, "straight_mode_components.png")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Saved field component visualization to: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline straight waveguide transmission check.")
    parser.add_argument(
        "--save-folder",
        default=None,
        help="Optional folder to store spins logs/checkpoints.",
    )
    args = parser.parse_args()
    run_straight_check(args.save_folder)


if __name__ == "__main__":
    main()
