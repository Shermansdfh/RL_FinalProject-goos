"""Discretize a saved SPINS-B power-splitter design by thresholding epsilon.

Outputs
-------
1. `<basename>_discrete.pkl` containing the full (≈150×150) real-permittivity array,
   including PMLs and waveguide feeds, snapped to background/core values.
2. `<basename>_binary_center.txt` containing a 50×50 center crop encoded as
   binary (1 = silicon/core, 0 = silica/background) for the design window only.
3. When `--export-original` is passed, `<basename>_full_real.txt` stores the
   full-resolution real-valued permittivity slice (same extent as #1).
"""
import argparse
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from pathlib import Path

# Try to import GOOS / maxwell if available (used for optional simulate/run)
try:
    from spins import goos
    from spins.goos_sim import maxwell
except Exception:
    goos = None
    maxwell = None


def find_eps_in_pickle(data):
    # Similar heuristic used elsewhere in this repo
    if isinstance(data, dict):
        if "variable_data" in data and "design_var" in data["variable_data"]:
             # Extract design variable directly if available (shape agnostic)
             return data["variable_data"]["design_var"]["value"]
        if "monitor_data" in data and isinstance(data["monitor_data"], dict):
            md = data["monitor_data"]
            for k in md:
                if k.lower().endswith(".eps") or "eps" in k.lower():
                    return md[k]
            if len(md) == 1:
                return list(md.values())[0]
        if "eps" in data:
            return data["eps"]
    raise ValueError("Could not locate eps data in pickle file")


def middle_slice(eps_data):
    eps = np.linalg.norm(eps_data, axis=0)
    if eps.ndim < 3:
        return eps
    z_idx = eps.shape[2] // 2
    return eps[:, :, z_idx]


def kmeans2_1d(values, max_iters=50, tol=1e-6):
    # values: 1D array
    v = values.reshape(-1)
    # initialize centers
    c0 = v.min()
    c1 = v.max()
    for _ in range(max_iters):
        # assign
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
    return np.array([c0, c1])


def crop_center_pixels(arr2d, px_w=50, px_h=50):
    nx, ny = arr2d.shape
    px_w = max(1, min(px_w, nx))
    px_h = max(1, min(px_h, ny))
    sx = max(0, nx // 2 - px_w // 2)
    ex = min(nx, sx + px_w)
    sy = max(0, ny // 2 - px_h // 2)
    ey = min(ny, sy + px_h)
    return arr2d[sx:ex, sy:ey]


def main():
    parser = argparse.ArgumentParser(description="Discretize epsilon from .pkl by thresholding")
    parser.add_argument("pkl", help="Input optimization .pkl file")
    parser.add_argument("--out-pkl", help="Path for discretized full-resolution pickle (default: <pkl>_discrete.pkl)")
    parser.add_argument("--out-binary", help="Path for 50x50 binary center TXT (default: <pkl>_binary_center.txt)")
    parser.add_argument("--bg-index", type=float, help="Background refractive index (silica)")
    parser.add_argument("--core-index", type=float, help="Core refractive index (silicon)")
    parser.add_argument("--export-original", action="store_true", help="Also export 50x50 center real permittivity slice")
    parser.add_argument("--original-txt", help="Path for real-valued center TXT when --export-original is used")

    args = parser.parse_args()

    if not os.path.isfile(args.pkl):
        raise FileNotFoundError(args.pkl)

    with open(args.pkl, "rb") as fp:
        data = pickle.load(fp)

    eps_data = find_eps_in_pickle(data)
    eps2d = middle_slice(eps_data)

    provided_bg_n = args.bg_index
    provided_core_n = args.core_index
    if (provided_bg_n is None) ^ (provided_core_n is None):
        raise ValueError("Provide both --bg-index and --core-index or neither.")

    if provided_bg_n is not None:
        bg_eps = float(provided_bg_n) ** 2
        core_eps = float(provided_core_n) ** 2
    else:
        centers = np.sort(kmeans2_1d(eps2d.reshape(-1)))
        bg_eps, core_eps = float(centers[0]), float(centers[1])
    thresh = 0.5 * (bg_eps + core_eps)

    discrete_full = np.where(eps2d >= thresh, core_eps, bg_eps)
    binary_full = (eps2d >= thresh).astype(int)

    base = os.path.splitext(os.path.basename(args.pkl))[0]
    out_pkl = args.out_pkl or (base + "_discrete.pkl")
    out_binary = args.out_binary or (base + "_binary_center.txt")
    orig_full_path = None
    if args.export_original:
        orig_full_path = args.original_txt or (base + "_full_real.txt")

    # Ensure parent directories exist
    for path in [out_pkl, out_binary, orig_full_path]:
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

    binary_center = crop_center_pixels(binary_full, px_w=50, px_h=50).T
    np.savetxt(out_binary, binary_center, fmt="%d")
    print(f"Saved binary center mask to: {out_binary}")

    out_data = {
        "source_pkl": os.path.abspath(args.pkl),
        "discrete_eps": discrete_full,
        "binary_mask": binary_full,
        "bg_eps": bg_eps,
        "core_eps": core_eps,
        "threshold_eps": thresh,
    }
    if provided_bg_n is not None:
        out_data["bg_index"] = float(provided_bg_n)
        out_data["core_index"] = float(provided_core_n)

    with open(out_pkl, "wb") as fp:
        pickle.dump(out_data, fp)
    print(f"Saved discretized design pickle to: {out_pkl}")

    if args.export_original and orig_full_path:
        full_real = eps2d.T
        np.savetxt(orig_full_path, full_real, fmt="%.6e")
        print(f"Saved real-valued full slice to: {orig_full_path}")


    


def view_pkl(pkl_path, out_png=None, show=True):
    """Load a .pkl (or choose one from a directory) and display/save a 2D view.

    - If `pkl_path` is a directory and `--step` is not used, the newest .pkl is chosen.
    - If the .pkl contains `discrete_eps` it is used; otherwise the script will try
      to locate eps monitor data and extract a middle slice.
    """
    # resolve path
    if os.path.isdir(pkl_path):
        pkl_files = [os.path.join(pkl_path, f) for f in os.listdir(pkl_path) if f.endswith('.pkl')]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in directory: {pkl_path}")
        # pick latest by mtime
        pkl_files.sort(key=lambda p: os.path.getmtime(p))
        full_path = pkl_files[-1]
    else:
        full_path = pkl_path

    with open(full_path, 'rb') as fp:
        data = pickle.load(fp)

    # Prefer discrete_eps if present
    if isinstance(data, dict) and 'discrete_eps' in data:
        arr = data['discrete_eps']
    else:
        # try to find eps monitor data
        try:
            eps = find_eps_in_pickle(data)
        except Exception as e:
            raise RuntimeError(f"Could not locate eps data in {full_path}: {e}")
        arr = middle_slice(eps)

    arr = np.array(arr)
    # If accidentally 3D, reduce to 2D middle slice
    if arr.ndim >= 3:
        arr = middle_slice(arr)

    # Simple plotting
    plt.figure(figsize=(6, 6))
    # choose cmap depending on discrete vs continuous
    uniq = np.unique(arr)
    if np.issubdtype(arr.dtype, np.integer) or (uniq.size <= 4):
        cmap = 'gray'
    else:
        cmap = 'viridis'
    # show with origin='lower' so array indices map visually as expected
    plt.imshow(arr.T, origin='lower', cmap=cmap)
    plt.colorbar()
    plt.title(os.path.basename(full_path))

    if out_png:
        out_dir = os.path.dirname(os.path.abspath(out_png))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png, bbox_inches='tight', dpi=150)
        print(f"Saved view PNG to: {out_png}")

    if show:
        plt.show()
    else:
        plt.close()

    # Try to extract overlap monitor values (power in output waveguides) and annotate
    # Look for typical monitor keys like 'overlap_up' / 'overlap_down' or keys containing 'overlap'
    try:
        up = None
        down = None
        if isinstance(data, dict) and 'monitor_data' in data and isinstance(data['monitor_data'], dict):
            for k, v in data['monitor_data'].items():
                key_lower = k.lower()
                if 'overlap_up' in key_lower or ('overlap' in key_lower and 'up' in key_lower):
                    # monitor value might be scalar or array; try to extract numeric
                    try:
                        up = float(v)
                    except Exception:
                        # if array-like, take last or sum
                        try:
                            up = float(np.array(v).ravel().sum())
                        except Exception:
                            up = None
                if 'overlap_down' in key_lower or ('overlap' in key_lower and 'down' in key_lower):
                    try:
                        down = float(v)
                    except Exception:
                        try:
                            down = float(np.array(v).ravel().sum())
                        except Exception:
                            down = None

        # Some pickles may store overlap monitors at top-level
        if up is None and isinstance(data, dict) and 'overlap_up' in data:
            try:
                up = float(data['overlap_up'])
            except Exception:
                up = None
        if down is None and isinstance(data, dict) and 'overlap_down' in data:
            try:
                down = float(data['overlap_down'])
            except Exception:
                down = None

        if up is not None and down is not None:
            total = up + down
            ratio_up = up / (total + 1e-12)
            # Print and annotate on the existing plot (save an annotated PNG if asked)
            info = f"Power Up: {up:.4g}  Power Down: {down:.4g}  Total: {total:.4g}  Ratio Up: {ratio_up:.3f}"
            print(info)
            # draw a small summary figure
            fig2, ax2 = plt.subplots(figsize=(6, 2))
            ax2.axis('off')
            ax2.text(0.01, 0.5, info, fontsize=12, va='center')
            if out_png:
                base, ext = os.path.splitext(out_png)
                summary_png = base + '_summary.png'
                fig2.savefig(summary_png, bbox_inches='tight', dpi=150)
                print(f"Saved summary PNG to: {summary_png}")
            if show:
                plt.show()
            else:
                plt.close(fig2)
        else:
            # If overlaps not found, gently inform the user how to produce them
            if show:
                print("Overlap monitors not found in .pkl. To get input/output power ratios, run the optimization or a single FDFD sim that records 'overlap_up' and 'overlap_down' monitors (see power_splitter_60_40.py).")
            else:
                # non-interactive: just print
                print("No overlap monitors found in .pkl; cannot compute power ratios.")
    except Exception as e:
        print(f"Warning: failed to extract overlap monitors: {e}")


def _view_mode(argv):
    parser = argparse.ArgumentParser(prog="discretize_pkl.py view", description="View a .pkl or directory of .pkl files")
    parser.add_argument('pkl_or_dir', nargs='?', default='.', help='Path to a .pkl file or a directory containing .pkl files (optional if --pkl-file is used)')
    parser.add_argument('--pkl-file', dest='pkl_file', help='Path to a specific .pkl file to view (overrides positional)')
    parser.add_argument('--step', type=int, help='If given and a directory is provided, prefer files containing the step number in their name')
    parser.add_argument('--out-png', help='Optional path to save a PNG view')
    parser.add_argument('--simulate', action='store_true', help='Run a quick FDFD simulation using discrete_eps and Options defaults from power_splitter_60_40.py')
    parser.add_argument('--no-show', action='store_true', help='Do not pop up an interactive window')
    args = parser.parse_args(argv)
    # If user provided --pkl-file, prefer it. Otherwise use positional which may be a file or directory.
    if args.pkl_file:
        target = args.pkl_file
    else:
        target = args.pkl_or_dir

    if os.path.isdir(target) and args.step is not None:
        # try to find a file with the given step in its name
        candidates = [os.path.join(target, f) for f in os.listdir(target) if f.endswith('.pkl') and f"step{args.step}" in f]
        if not candidates:
            raise FileNotFoundError(f"No .pkl file with step{args.step} found in {target}")
        # choose the newest among matches
        candidates.sort(key=lambda p: os.path.getmtime(p))
        target = candidates[-1]

    # If target is still a directory, view_pkl will pick the newest .pkl inside it
    # Optionally run a quick simulate using power_splitter defaults
    if args.simulate:
        if goos is None or maxwell is None:
            raise RuntimeError("GOOS / maxwell modules not available in this Python environment; cannot run simulation.")

        # Load power_splitter Options from local example file
        mod_path = Path(__file__).parent / 'power_splitter_60_40.py'
        if not mod_path.exists():
            raise FileNotFoundError(f"Could not find power_splitter_60_40.py at expected location: {mod_path}")

        spec = importlib.util.spec_from_file_location('power_splitter_60_40', str(mod_path))
        pp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pp)

        # Use defaults
        Params = getattr(pp, 'Options', None)
        if Params is None:
            raise RuntimeError('Could not find Options in power_splitter_60_40.py')
        params = Params()

        # Load pkl data and pick discrete_eps if present, otherwise try to use arr
        with open(target, 'rb') as fp:
            data = pickle.load(fp)

        if isinstance(data, dict) and 'discrete_eps' in data:
            eps_full = np.array(data['discrete_eps'])
        else:
            # fall back to extracting eps via existing helper
            eps_data = find_eps_in_pickle(data)
            eps_full = middle_slice(eps_data)

        # Attempt to build a quick simulation node using the discrete eps array
        try:
            # Wrap eps into a GOOS array flow
            eps_flow = goos.ArrayFlow([eps_full])

            input_start_x = 0 - params.design_width / 2 - params.wg_len_in
            src = maxwell.WaveguideModeSource(
                center=[input_start_x * params.source_pos_shift_coeff, 0, 0],
                extents=[0, params.wg_width * 6, params.thickness * 4],
                normal=[1, 0, 0],
                mode_num=0,
                power=1,
            )

            outputs = [
                maxwell.Epsilon(name='eps'),
                maxwell.ElectricField(name='field'),
                maxwell.WaveguideModeOverlap(
                    name='overlap_up',
                    center=[params.monitor_pos, params.wg_offset, 0],
                    extents=[0, params.wg_width, params.thickness * 2],
                    normal=[1, 0, 0],
                    mode_num=0,
                    power=1,
                ),
                maxwell.WaveguideModeOverlap(
                    name='overlap_down',
                    center=[params.monitor_pos, -params.wg_offset, 0],
                    extents=[0, params.wg_width, params.thickness * 2],
                    normal=[1, 0, 0],
                    mode_num=0,
                    power=1,
                ),
            ]

            simspace = maxwell.SimulationSpace(
                mesh=maxwell.UniformMesh(dx=params.resolution),
                sim_region=goos.Box3d(center=[0, 0, 0], extents=[params.sim_region, params.sim_region, params.sim_z_extent]),
                pml_thickness=[params.pml_thickness, params.pml_thickness, params.pml_thickness, params.pml_thickness, 0, 0],
            )

            sim = maxwell.fdfd_simulation(
                wavelength=params.wlen,
                eps=eps_flow,
                background=goos.material.Material(index=params.background_index),
                simulation_space=simspace,
                sources=[src],
                outputs=outputs,
                solver='local_direct',
            )

            # Try to fetch outputs (these calls may raise if GOOS runtime isn't in the expected state)
            try:
                up = sim['overlap_up'].get()
                down = sim['overlap_down'].get()
                eps_out = sim['eps'].get()
                field_out = sim['field'].get()

                # unwrap numeric flows
                try:
                    up_val = float(up)
                except Exception:
                    up_val = float(np.array(up).ravel().sum())
                try:
                    down_val = float(down)
                except Exception:
                    down_val = float(np.array(down).ravel().sum())

                print(f"Simulated power: up={up_val:.6g}, down={down_val:.6g}, ratio={up_val/(up_val+down_val+1e-12):.3f}")
            except Exception as e:
                print(f"Simulation constructed but failed to evaluate outputs: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to construct/run simulation: {e}")

        # After simulation attempt, still call view_pkl to show image
        view_pkl(target, out_png=args.out_png, show=not args.no_show)
    else:
        view_pkl(target, out_png=args.out_png, show=not args.no_show)


if __name__ == "__main__":
    # keep backward-compatible behavior: allow running `python discretize_pkl.py view <path>`
    if len(sys.argv) > 1 and sys.argv[1] == 'view':
        _view_mode(sys.argv[2:])
    else:
        main()
