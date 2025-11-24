"""Live plotter for metrics recorded in SPINS-B `spins.log`.

Usage
-----
    python log_monitor.py --log path/to/save_folder/spins.log

The script watches the log file, extracts metrics such as
`obj_splitter_cont.ratio_mse` and `obj_splitter_cont.total_power`, and
continuously updates a matplotlib plot. It works offline (no wandb needed).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import deque
from typing import Deque, Dict, List

import matplotlib.pyplot as plt

METRIC_PATTERN = re.compile(
    r"(?P<metric>obj_splitter_(?:cont|sig)\.(?:ratio_mse|total_power))\s*[:=]\s*(?P<value>[-+]?[\d\.eE+-]+)"
)


def parse_new_lines(buffer: str) -> List[tuple[str, float]]:
    """Extract (metric, value) tuples from new log content."""
    updates: List[tuple[str, float]] = []
    for line in buffer.splitlines():
        match = METRIC_PATTERN.search(line)
        if not match:
            continue
        metric = match.group("metric")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        updates.append((metric, value))
    return updates


def watch_log(
    log_path: str,
    refresh: float,
    max_points: int,
    metrics: List[str],
):
    """Tail the log file and update matplotlib lines."""
    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Live Metrics from spins.log")

    axis_map = {
        "ratio_mse": axes[0],
        "total_power": axes[1],
    }
    axes[0].set_ylabel("Ratio MSE")
    axes[1].set_ylabel("Total Power")
    axes[1].set_xlabel("Updates")

    def pick_axis(metric_name: str):
        for suffix, axis in axis_map.items():
            if metric_name.endswith(suffix):
                return axis
        return axes[0]

    data_store: Dict[str, Deque[float]] = {
        metric: deque(maxlen=max_points) for metric in metrics
    }
    x_store: Dict[str, Deque[int]] = {
        metric: deque(maxlen=max_points) for metric in metrics
    }

    lines = {}
    for metric in metrics:
        axis = pick_axis(metric)
        line_label = metric.replace("obj_splitter_", "")
        lines[metric] = axis.plot([], [], label=line_label)[0]

    for axis in axes:
        if axis.lines:
            axis.legend(loc="upper right")

    last_pos = 0
    if not os.path.exists(log_path):
        print(f"Waiting for log file {log_path} to appear...")
        while not os.path.exists(log_path):
            time.sleep(refresh)

    while True:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as fp:
            fp.seek(last_pos)
            chunk = fp.read()
            last_pos = fp.tell()

        updates = parse_new_lines(chunk)
        for metric, value in updates:
            if metric not in data_store:
                continue
            data_store[metric].append(value)
            x_store[metric].append(len(x_store[metric]) + 1)

        for metric, line in lines.items():
            line.set_data(list(x_store[metric]), list(data_store[metric]))

        for axis in axes:
            axis.relim()
            axis.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(refresh)


def main():
    parser = argparse.ArgumentParser(description="Local live plotter for spins.log metrics")
    parser.add_argument(
        "--log",
        required=True,
        help="Path to spins.log inside an optimization save folder.",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Polling interval in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Max buffered points per metric (default: 2000).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "obj_splitter_cont.ratio_mse",
            "obj_splitter_cont.total_power",
            "obj_splitter_sig.ratio_mse",
            "obj_splitter_sig.total_power",
        ],
        help="Which metrics to track (default: cont/sig ratio_mse & total_power).",
    )

    args = parser.parse_args()
    watch_log(
        log_path=args.log,
        refresh=args.refresh,
        max_points=args.max_points,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()

