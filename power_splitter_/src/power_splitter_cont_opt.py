"""GOOS power splitter optimization targeting 60% / 40% split.

This example follows the `bend90` workflow but tailors the objective so that
an input excitation from the left couples 60% of the power into the upper arm
and 40% into the lower arm.

Usage
-----
Run optimization:

    $ python power_splitter_60_40.py run path/to/save_folder

Inspect a saved step:

    $ python power_splitter_cont_opt.py view path/to/save_folder --step 10
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use("Agg")

from spins import goos
from spins.goos_sim import maxwell


@dataclasses.dataclass
class DesignConfig:
    width: float = 2000
    height: float = 2000
    thickness: float = 220
    pixel_size: float = 100
    resolution: float = 50


@dataclasses.dataclass
class WaveguideConfig:
    width: float = 400
    input_length: float = 1500
    output_length: float = 1500
    offset: float = 600
    output_center: float = 1750


@dataclasses.dataclass
class MaterialConfig:
    background_index: float = 1.45
    core_index: float = 3.45


@dataclasses.dataclass
class SimulationConfig:
    wavelength: float = 1550
    region: float = 6000
    z_extent: float = 40
    pml_thickness: float = 200
    source_shift: float = 0.95
    monitor_position: float = 1800


@dataclasses.dataclass
class OptimizationConfig:
    max_iters: int = 100
    target_ratio: float = 0.65  # power in upper arm
    power_loss_weight: float = 0.1
    sigmoid_factors: Tuple[int, ...] = (4, 8, 16, 24, 32, 48, 64, 96, 128)


@dataclasses.dataclass
class SplitterConfig:
    design: DesignConfig = dataclasses.field(default_factory=DesignConfig)
    waveguide: WaveguideConfig = dataclasses.field(default_factory=WaveguideConfig)
    material: MaterialConfig = dataclasses.field(default_factory=MaterialConfig)
    simulation: SimulationConfig = dataclasses.field(default_factory=SimulationConfig)
    optimization: OptimizationConfig = dataclasses.field(default_factory=OptimizationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def geometry_summary(config: SplitterConfig) -> Dict[str, float]:
    """Convenience geometry values shared across helpers."""
    design = config.design
    waveguide = config.waveguide
    sim = config.simulation

    input_start = -design.width / 2 - waveguide.input_length # -2500
    input_center = input_start + waveguide.input_length / 2 # -1750
    source_x = input_start * sim.source_shift # -2375
    design_bounds = (
        -design.width / 2,
        design.width / 2,
        -design.height / 2,
        design.height / 2,
    )
    sim_bounds = (
        -sim.region / 2,
        sim.region / 2,
        -sim.region / 2,
        sim.region / 2,
    )
    return {
        "input_start": input_start,
        "input_center": input_center,
        "source_x": source_x,
        "design_bounds": design_bounds,
        "sim_bounds": sim_bounds,
        "output_start": design.width / 2,
        "monitor_x": sim.monitor_position,
    }


def validate_geometry(config: SplitterConfig) -> List[str]:
    """Return human-readable warnings when geometry looks suspicious."""
    issues: List[str] = []
    design = config.design
    wg = config.waveguide
    summary = geometry_summary(config)

    if summary["monitor_x"] < design.width / 2:
        issues.append(
            "Monitor plane lies inside the design region; consider moving it "
            "further to the right so modal overlaps are well-defined."
        )
    if wg.output_center < design.width / 2:
        issues.append(
            "Output waveguides do not fully clear the design area (output_center "
            "< design.width / 2)."
        )
    if wg.offset + wg.width / 2 > design.height / 2:
        issues.append(
            "Waveguide offset pushes outputs outside the design window (offset too large)."
        )
    if design.width <= 0 or design.height <= 0:
        issues.append("Design region dimensions must be positive.")
    if wg.width <= 0:
        issues.append("Waveguide width must be positive.")
    if config.optimization.target_ratio <= 0 or config.optimization.target_ratio >= 1:
        issues.append("Target ratio should be in (0, 1).")
    return issues


def plot_geometry_layout(
    config: SplitterConfig,
    summary: Dict[str, float] = None,
    save_path: str = None,
):
    """Compact geometry visualization + textual report."""
    summary = summary or geometry_summary(config)
    design = config.design
    wg = config.waveguide
    sim = config.simulation

    nm_to_um = 1e-3
    fig, ax = plt.subplots(figsize=(9, 6))

    # Simulation boundary
    sim_bounds = summary["sim_bounds"]
    sim_rect = patches.Rectangle(
        (sim_bounds[0] * nm_to_um, sim_bounds[2] * nm_to_um),
        sim.region * nm_to_um,
        sim.region * nm_to_um,
        linewidth=1.5,
        edgecolor="gray",
        facecolor="none",
        linestyle="--",
        label="Simulation",
    )
    ax.add_patch(sim_rect)

    # Design region
    design_rect = patches.Rectangle(
        (summary["design_bounds"][0] * nm_to_um, summary["design_bounds"][2] * nm_to_um),
        design.width * nm_to_um,
        design.height * nm_to_um,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
        label="Design",
    )
    ax.add_patch(design_rect)

    # Input waveguide
    ax.add_patch(
        patches.Rectangle(
            (summary["input_start"] * nm_to_um, -wg.width / 2 * nm_to_um),
            wg.input_length * nm_to_um,
            wg.width * nm_to_um,
            color="royalblue",
            alpha=0.4,
            label="Input WG",
        )
    )

    # Output waveguides
    ax.add_patch(
        patches.Rectangle(
            (summary["output_start"] * nm_to_um, (wg.offset - wg.width / 2) * nm_to_um),
            wg.output_length * nm_to_um,
            wg.width * nm_to_um,
            color="seagreen",
            alpha=0.4,
            label="Output WG",
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (summary["output_start"] * nm_to_um, (-wg.offset - wg.width / 2) * nm_to_um),
            wg.output_length * nm_to_um,
            wg.width * nm_to_um,
            color="seagreen",
            alpha=0.4,
        )
    )

    # Source / monitor markers
    ax.axvline(summary["source_x"] * nm_to_um, color="red", linestyle=":", label="Source")
    ax.axvline(summary["monitor_x"] * nm_to_um, color="purple", linestyle="-.", label="Monitor")

    ax.set_aspect("equal")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Power Splitter Geometry Overview")

    span_x = max(abs(sim_bounds[0]), abs(sim_bounds[1])) * nm_to_um
    span_y = max(abs(sim_bounds[2]), abs(sim_bounds[3])) * nm_to_um
    ax.set_xlim(-span_x, span_x)
    ax.set_ylim(-span_y, span_y)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax


def _scalar_from_any(value: Any) -> float | None:
    """Convert stored monitor data to a python float."""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    try:
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def _load_dict_from_path(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML file describing overrides."""
    _, ext = os.path.splitext(path.lower())
    with open(path, "r", encoding="utf-8") as fp:
        if ext in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "PyYAML is required to parse YAML config files; install via `pip install pyyaml`."
                ) from exc
            return yaml.safe_load(fp) or {}
        return json.load(fp)


def _merge_dataclass(instance: Any, updates: Dict[str, Any]):
    """Recursively apply overrides to a dataclass instance."""
    for field in dataclasses.fields(instance):
        if field.name not in updates:
            continue
        value = getattr(instance, field.name)
        override = updates[field.name]
        if dataclasses.is_dataclass(value):
            if not isinstance(override, dict):
                raise ValueError(f"Expected mapping for nested config '{field.name}'.")
            _merge_dataclass(value, override or {})
        else:
            setattr(instance, field.name, override)


def build_config_from_file(path: str | None, target_ratio: float, max_iters: int) -> SplitterConfig:
    """Create SplitterConfig optionally overriding from a structured file."""
    config = SplitterConfig()
    config.optimization.target_ratio = target_ratio
    config.optimization.max_iters = max_iters
    if path:
        overrides = _load_dict_from_path(path)
        if not isinstance(overrides, dict):
            raise ValueError("Top-level config overrides must be a dictionary.")
        _merge_dataclass(config, overrides)
    return config


def resolve_save_folder(path: str, action: str) -> str:
    """Derive a timestamped save directory for optimization runs."""
    if action != "run":
        return path

    abs_path = os.path.abspath(path)
    if os.path.isdir(abs_path):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        abs_path = os.path.join(abs_path, timestamp)

    os.makedirs(abs_path, exist_ok=True)
    print(f"Using save folder: {abs_path}")
    return abs_path


def create_design(config: SplitterConfig):
    """Creates design variables and static structures."""

    design_cfg = config.design
    wg_cfg = config.waveguide
    mat_cfg = config.material

    def initializer(size):
        np.random.seed(42)
        return np.random.random(size) * 0.2 + 0.5

    var, design = goos.pixelated_cont_shape(
        initializer=initializer,
        pos=goos.Constant([0, 0, 0]),
        extents=[design_cfg.width, design_cfg.height, design_cfg.thickness],
        material=goos.material.Material(index=mat_cfg.background_index),
        material2=goos.material.Material(index=mat_cfg.core_index),
        pixel_size=[design_cfg.pixel_size, design_cfg.pixel_size, design_cfg.thickness],
        var_name="design_var",
    )

    geom = geometry_summary(config)
    input_center_x = geom["input_start"] + wg_cfg.input_length / 2 # -2500 + 1500 / 2 = -1750

    wg_in = goos.Cuboid(
        pos=goos.Constant([input_center_x, 0, 0]),
        extents=goos.Constant([wg_cfg.input_length, wg_cfg.width, design_cfg.thickness]), # 1500, 400, 220
        material=goos.material.Material(index=mat_cfg.core_index),
    )

    wg_up = goos.Cuboid(
        pos=goos.Constant([wg_cfg.output_center, wg_cfg.offset, 0]),
        extents=goos.Constant([wg_cfg.output_length, wg_cfg.width, design_cfg.thickness]),
        material=goos.material.Material(index=mat_cfg.core_index),
    )

    wg_down = goos.Cuboid(
        pos=goos.Constant([wg_cfg.output_center, -wg_cfg.offset, 0]),
        extents=goos.Constant([wg_cfg.output_length, wg_cfg.width, design_cfg.thickness]),
        material=goos.material.Material(index=mat_cfg.core_index),
    )

    eps_struct = goos.GroupShape([wg_in, wg_up, wg_down, design])

    eps_render = maxwell.RenderShape(
        design,
        region=goos.Box3d(
            center=[0, 0, 0],
            extents=[design_cfg.width, design_cfg.height, 0],
        ),
        mesh=maxwell.UniformMesh(dx=design_cfg.pixel_size),
        wavelength=config.simulation.wavelength,
        name="eps_rendered",
    )

    return var, wg_in, wg_up, wg_down, design, eps_render


def create_simulation(eps: goos.Shape, config: SplitterConfig, name: str = "sim_splitter"):
    """Sets up the FDFD simulation and monitors."""
    geom = geometry_summary(config)
    design = config.design
    wg = config.waveguide
    sim_cfg = config.simulation
    mat_cfg = config.material

    sim = maxwell.fdfd_simulation(
        name=name,
        wavelength=sim_cfg.wavelength,
        eps=eps,
        solver="local_direct",
        sources=[
            maxwell.WaveguideModeSource(
                center=[geom["source_x"], 0, 0],
                extents=[0, wg.width * 3, design.thickness * 4],
                normal=[1, 0, 0],
                mode_num=0,
                power=1,
            )
        ],
        simulation_space=maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=design.resolution),
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
            maxwell.Epsilon(name="eps"),
            maxwell.ElectricField(name="field"),
            maxwell.WaveguideModeOverlap(
                name="overlap_up",
                center=[sim_cfg.monitor_position, wg.offset, 0],
                extents=[0, wg.width * 1.5, design.thickness * 2],
                normal=[1, 0, 0],
                mode_num=0,
                power=1,
            ),
            maxwell.WaveguideModeOverlap(
                name="overlap_down",
                center=[sim_cfg.monitor_position, -wg.offset, 0],
                extents=[0, wg.width * 1.5, design.thickness * 2],
                normal=[1, 0, 0],
                mode_num=0,
                power=1,
            ),
        ],
    )

    return sim


def create_objective(sim, config: SplitterConfig, name_prefix: str = "obj_splitter"):
    """Constructs the power-ratio objective."""

    def named(expr, suffix):
        return goos.rename(expr, name=f"{name_prefix}.{suffix}")

    power_up_raw = goos.abs(sim["overlap_up"]) ** 2
    power_down_raw = goos.abs(sim["overlap_down"]) ** 2

    BASELINE_POWER = 1.009828  # from straight_waveguide_check.py

    power_up = named(power_up_raw / BASELINE_POWER, "power_up")
    power_down = named(power_down_raw / BASELINE_POWER, "power_down")
    total_power = power_up + power_down + 1e-12

    ratio_up = power_up / total_power
    ratio_down = power_down / total_power
    target_ratio = goos.Constant(config.optimization.target_ratio)
    ratio_mse = (ratio_up - target_ratio) ** 2 + (ratio_down - (1 - target_ratio)) ** 2
    power_penalty = config.optimization.power_loss_weight * (1 - total_power) ** 2

    ratio_term = named(ratio_mse, "ratio_mse")
    penalty_term = named(power_penalty, "power_penalty")
    total_power_term = named(total_power, "total_power")
    obj = goos.rename(ratio_term + penalty_term, name=name_prefix)
    return obj, ratio_term, penalty_term, total_power_term, power_up, power_down

def run(save_folder: str, config: SplitterConfig, visualize: bool = False, plot_geometry: bool = False):
    goos.util.setup_logging(save_folder)

    issues = validate_geometry(config)
    if issues:
        print("Geometry sanity warnings:")
        for issue in issues:
            print(f"  - {issue}")

    if plot_geometry:
        geometry_plot_path = os.path.join(save_folder, "geometry_sanity_check.png")
        plot_geometry_layout(config, save_path=geometry_plot_path)

    with goos.OptimizationPlan(save_path=save_folder) as plan:
        var, wg_in, wg_up, wg_down, design, eps_render = create_design(config)

        eps_struct = goos.GroupShape([wg_in, wg_up, wg_down, design])
        sim = create_simulation(eps_struct, config, name="sim_splitter_cont")
        obj, ratio_term, penalty_term, total_power_term, power_up, power_down = create_objective(
            sim, config, name_prefix="obj_splitter_cont"
        )

        goos.opt.scipy_minimize(
            obj,
            "L-BFGS-B",
            max_iters=config.optimization.max_iters,
            monitor_list=[
                sim["eps"],
                sim["field"],
                power_up,
                power_down,
                total_power_term,
                ratio_term,
                penalty_term,
                obj,
            ],
            name="optimize_splitter_cont",
        )

        sigmoid_factor = goos.Variable(4, parameter=True, name="discr_factor")
        design_sig = goos.cast(goos.Sigmoid(sigmoid_factor * (2 * design - 1)), goos.Shape)

        eps_struct_sig = goos.GroupShape([wg_in, wg_up, wg_down, design_sig])
        sim_sig = create_simulation(eps_struct_sig, config, name="sim_splitter_sig")
        (
            obj_sig,
            ratio_term_sig,
            penalty_term_sig,
            total_power_sig,
            power_up_sig,
            power_down_sig,
        ) = create_objective(sim_sig, config, name_prefix="obj_splitter_sig")

        for factor in config.optimization.sigmoid_factors:
            sigmoid_factor.set(factor)
            goos.opt.scipy_minimize(
                obj_sig,
                "L-BFGS-B",
                max_iters=min(20, config.optimization.max_iters),
                monitor_list=[
                    sim_sig["eps"],
                    sim_sig["field"],
                    power_up_sig,
                    power_down_sig,
                    total_power_sig,
                    ratio_term_sig,
                    penalty_term_sig,
                    obj_sig,
                ],
                name=f"optimize_splitter_discrete_{factor}",
            )

        plan.save()
        plan.run()

        if visualize:
            goos.util.visualize_eps(sim_sig["eps"].get().array[2])


def view(
    save_folder: str,
    step: int,
    config: SplitterConfig | None = None,
    components: bool = False,
):
    if step is None:
        raise ValueError("Must specify --step when viewing results.")

    with open(os.path.join(save_folder, f"step{step}.pkl"), "rb") as fp:
        data = pickle.load(fp)

    monitor_data = data.get("monitor_data", {})

    def pick(first_choice: List[str]):
        for candidate in first_choice:
            if candidate in monitor_data:
                return monitor_data[candidate]
        return None

    eps_raw = pick(["sim_splitter_cont.eps", "sim_splitter_sig.eps"])
    field_raw = pick(["sim_splitter_cont.field", "sim_splitter_sig.field"])
    if eps_raw is None or field_raw is None:
        raise KeyError("Could not find epsilon/field monitors in the selected step.")

    # eps = np.linalg.norm(eps_raw, axis=0)
    eps = np.real(eps_raw[2])
    field = np.linalg.norm(field_raw, axis=0)
    
    # Take z-slice at the middle
    z_slice_idx = eps.shape[2] // 2
    eps_slice = eps[:, :, z_slice_idx]
    field_slice = field[:, :, z_slice_idx]
    
    # Transpose to get correct orientation: (y, x) for imshow
    # imshow expects (rows, cols) = (y, x) for proper orientation
    eps_plot = eps_slice.T
    field_plot = field_slice.T
    
    if config is None:
        config = SplitterConfig()
    
    nm_to_um = 1.0 / 1000.0
    x_extent = config.simulation.region * nm_to_um
    y_extent = config.simulation.region * nm_to_um
    extent = [-x_extent/2, x_extent/2, -y_extent/2, y_extent/2]

    power_up_val = _scalar_from_any(
        pick(["obj_splitter_cont.power_up", "obj_splitter_sig.power_up"])
    )
    power_down_val = _scalar_from_any(
        pick(["obj_splitter_cont.power_down", "obj_splitter_sig.power_down"])
    )
    total_power_val = _scalar_from_any(
        pick(["obj_splitter_cont.total_power", "obj_splitter_sig.total_power"])
    )
    ratio_val = _scalar_from_any(
        pick(["obj_splitter_cont.ratio_mse", "obj_splitter_sig.ratio_mse"])
    )
    penalty_val = _scalar_from_any(
        pick(["obj_splitter_cont.power_penalty", "obj_splitter_sig.power_penalty"])
    )

    if any(val is not None for val in (total_power_val, power_up_val, power_down_val)):
        metrics_text = []
        if power_up_val is not None:
            metrics_text.append(f"up={power_up_val:.4f}")
        if power_down_val is not None:
            metrics_text.append(f"down={power_down_val:.4f}")
        if total_power_val is not None:
            metrics_text.append(f"total={total_power_val:.4f}")
        print(f"[Step {step}] Waveguide powers: {', '.join(metrics_text)}")

    if ratio_val is not None or penalty_val is not None:
        ratio_text = f"ratio_mse={ratio_val:.4e}" if ratio_val is not None else None
        penalty_text = (
            f"power_penalty={penalty_val:.4e}" if penalty_val is not None else None
        )
        text_items = [t for t in (ratio_text, penalty_text) if t]
        if text_items:
            print(f"[Step {step}] Objective terms: {', '.join(text_items)}")

    # Extract design variables if available
    design_vals = None
    if "variable_data" in data and "design_var" in data["variable_data"]:
        design_vals = np.array(data["variable_data"]["design_var"]["value"])

    design_plot = None
    if design_vals is not None:
        if design_vals.ndim == 3:
            # Take z-slice at the middle
            z_slice_idx = design_vals.shape[2] // 2
            design_slice = design_vals[:, :, z_slice_idx]
            design_plot = design_slice.T
        elif design_vals.ndim == 2:
            design_plot = design_vals.T

    cols = 3
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))

    ax_eps = axes[0]
    ax_field = axes[1]
    ax_design = axes[2]

    im1 = ax_eps.imshow(
        eps_plot, cmap="viridis", aspect="equal", extent=extent, origin="lower"
    )
    ax_eps.set_title(f"Permittivity (Step {step})", fontsize=12, fontweight="bold")
    ax_eps.set_xlabel("x (μm)")
    ax_eps.set_ylabel("y (μm)")
    ax_eps.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    fig.colorbar(im1, ax=ax_eps, label="|ε|")

    im2 = ax_field.imshow(
        field_plot, cmap="hot", aspect="equal", extent=extent, origin="lower"
    )
    ax_field.set_title(f"Field Magnitude (Step {step})", fontsize=12, fontweight="bold")
    ax_field.set_xlabel("x (μm)")
    ax_field.set_ylabel("y (μm)")
    ax_field.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    fig.colorbar(im2, ax=ax_field, label="|E|")

    if design_plot is not None:
        # Use design region bounds for the design plot
        design_extent = [
            config.design.width * nm_to_um / -2,
            config.design.width * nm_to_um / 2,
            config.design.height * nm_to_um / -2,
            config.design.height * nm_to_um / 2,
        ]
        im3 = ax_design.imshow(
            design_plot, cmap="Greys", aspect="equal", extent=design_extent, origin="lower", vmin=0, vmax=1
        )
        ax_design.set_title(f"Design Variables (Step {step})", fontsize=12, fontweight="bold")
        ax_design.set_xlabel("x (μm)")
        ax_design.set_ylabel("y (μm)")
        ax_design.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig.colorbar(im3, ax=ax_design, label="Value")
    else:
        ax_design.text(0.5, 0.5, "No design data found", ha="center", va="center")
        ax_design.axis("off")

    plt.tight_layout()
    plt.show()

    if save_folder:
        save_path = os.path.join(save_folder, f"step{step}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    # Optional: plot individual field components (Ey, Ez) to inspect quasi-TE / TM.
    if components:
        # field_raw is expected to have shape (3, Nx, Ny, Nz) for (Ex, Ey, Ez).
        Ex = field_raw[0]
        Ey = field_raw[1]
        Ez = field_raw[2]

        Ey_slice = np.abs(Ey[:, :, z_slice_idx]) ** 2
        Ez_slice = np.abs(Ez[:, :, z_slice_idx]) ** 2

        Ey_plot = Ey_slice.T
        Ez_plot = Ez_slice.T

        fig_comp, axes_comp = plt.subplots(1, 2, figsize=(12, 5))

        im_ey = axes_comp[0].imshow(
            Ey_plot, cmap="plasma", aspect="equal", extent=extent, origin="lower"
        )
        axes_comp[0].set_title(
            f"|Ey|^2 (Step {step})", fontsize=12, fontweight="bold"
        )
        axes_comp[0].set_xlabel("x (μm)")
        axes_comp[0].set_ylabel("y (μm)")
        axes_comp[0].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig_comp.colorbar(im_ey, ax=axes_comp[0], label="|Ey|^2")

        im_ez = axes_comp[1].imshow(
            Ez_plot, cmap="plasma", aspect="equal", extent=extent, origin="lower"
        )
        axes_comp[1].set_title(
            f"|Ez|^2 (Step {step})", fontsize=12, fontweight="bold"
        )
        axes_comp[1].set_xlabel("x (μm)")
        axes_comp[1].set_ylabel("y (μm)")
        axes_comp[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig_comp.colorbar(im_ez, ax=axes_comp[1], label="|Ez|^2")

        plt.tight_layout()
        plt.show()

        if save_folder:
            comp_path = os.path.join(save_folder, f"step{step}_components.png")
            fig_comp.savefig(comp_path, dpi=150, bbox_inches="tight")
            print(f"Saved field components visualization to: {comp_path}")


def main():
    parser = argparse.ArgumentParser(description="60/40 power splitter optimizer.")
    parser.add_argument("action", choices=("run", "view"))
    parser.add_argument("save_folder")
    parser.add_argument("--step", type=int, help="Checkpoint step for view.")
    parser.add_argument("--visualize", action="store_true", help="Render permittivity.")
    parser.add_argument(
        "--plot-geometry",
        action="store_true",
        help="Plot geometry sanity check before optimization.",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.6,
        help="Desired power ratio for the upper arm.",
    )
    parser.add_argument("--max-iters", type=int, default=60)
    parser.add_argument(
        "--config",
        help="Optional JSON/YAML config overrides (nested keys match dataclass structure).",
        default=None,
    )
    parser.add_argument(
        "--components",
        action="store_true",
        help="In view mode, also plot Ey/Ez components to inspect quasi-TE/TM.",
    )

    args = parser.parse_args()
    save_folder = resolve_save_folder(args.save_folder, args.action)
    config = build_config_from_file(args.config, args.target_ratio, args.max_iters)

    if args.action == "run":
        run(
            save_folder,
            config,
            visualize=args.visualize,
            plot_geometry=args.plot_geometry,
        )
    elif args.action == "view":
        view(
            save_folder,
            args.step,
            config if args.config else None,
            components=args.components,
        )


if __name__ == "__main__":
    main()

