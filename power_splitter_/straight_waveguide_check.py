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
    """Construct a single FDFD simulation with the straight core waveguide."""
    return maxwell.fdfd_simulation(
        name="straight_waveguide",
        wavelength=sim_cfg.wavelength,
        eps=waveguide_shape,
        solver="local_direct",
        sources=[
            maxwell.WaveguideModeSource(
                center=[source_center_x(design_cfg, wg_cfg, sim_cfg), 0, 0],
                extents=[0, wg_cfg.width * 6, design_cfg.thickness * 4],
                normal=[1, 0, 0],
                mode_num=0,
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
                extents=[0, wg_cfg.width * 4, design_cfg.thickness * 4],
                normal=[1, 0, 0],
                mode_num=0,
                power=1,
            )
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

        overlap_flow = overlap.get(run=True)
        amplitude, power, loss_db = compute_metrics(overlap_flow)

        print(f"Raw overlap amplitude : {amplitude:.6f}")
        print(f"Calculated power      : {power:.6f}")
        print(f"dB loss               : {loss_db:.2f} dB")

        if power > 0.95:
            print("[PASS] Simulation setup is healthy.")
        else:
            print("[FAIL] Baseline transmission is too low. Check mesh resolution, PML settings, or monitor mode mismatch.")


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

