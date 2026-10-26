"""Benchmark the lane detection node with different model configs.

Launches the node once per model config (with a temporary copy of the params
file, the installed package is not modified), waits for the warmup, collects the
'total' and 'inference' times the node logs for every image and prints/saves the
averages. The node needs images, e.g. from a running camera or a rosbag.

Usage (in a sourced workspace, the configs must be installed in the package):
    python3 scripts/benchmark_models.py config/model_a.py config/model_b.py \
        --samples 100 --output results/benchmark.json --plot results/benchmark.png
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import yaml

PACKAGE = "lane_detection_ai"
TIME_PATTERNS = {
    "total": re.compile(r"'total':\s*([\d.]+)"),
    "inference": re.compile(r"'inference':\s*([\d.]+)"),
}


def _write_params_file(params_file: Path, model_config: str, output: Path) -> None:
    """Copy the ROS params file with model_config_path replaced."""
    params = yaml.safe_load(params_file.read_text())
    node_params = params.setdefault(f"{PACKAGE}_node", {}).setdefault(
        "ros__parameters", {}
    )
    node_params["model_config_path"] = model_config
    output.write_text(yaml.safe_dump(params))


def _stop(process: subprocess.Popen) -> None:
    """Stop the launch process group (SIGINT like Ctrl+C, then SIGKILL)."""
    if process.poll() is not None:
        return
    # The group may already be gone if the watchdog stopped it concurrently
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def benchmark(
    params_file: Path, warmup_s: float, samples: int, timeout_s: float
) -> dict[str, float]:
    """Run the node with the params file and return the average times in ms.

    Raises:
        RuntimeError: If the node stops or logs no times before the timeout.
    """
    ros2 = shutil.which("ros2")
    if ros2 is None:
        msg = "ros2 not found, source the ROS 2 workspace first"
        raise RuntimeError(msg)

    process = subprocess.Popen(  # noqa: S603 (fixed command, no shell)
        [
            ros2,
            "launch",
            PACKAGE,
            f"{PACKAGE}.launch.py",
            f"params_file:={params_file}",
            "debug:=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # own process group, to stop only this launch
    )
    if process.stdout is None:
        msg = "No stdout pipe of the launch process"
        raise RuntimeError(msg)
    # Stops the launch on timeout, which ends the read loop below
    watchdog = threading.Timer(timeout_s, _stop, args=[process])
    watchdog.start()

    times: dict[str, list[float]] = {key: [] for key in TIME_PATTERNS}
    start = time.monotonic()
    try:
        for line in process.stdout:
            elapsed = time.monotonic() - start
            print(f"[{elapsed:6.1f}s] {line.rstrip()}")

            matches = {k: p.search(line) for k, p in TIME_PATTERNS.items()}
            if elapsed >= warmup_s and all(matches.values()):
                for key, match in matches.items():
                    times[key].append(float(match.group(1)))
                if len(times["total"]) >= samples:
                    break
    finally:
        watchdog.cancel()
        _stop(process)

    if len(times["total"]) < samples:
        msg = (
            f"Node stopped or timed out ({timeout_s}s) after "
            f"{len(times['total'])} of {samples} samples"
        )
        raise RuntimeError(msg)

    return {key: sum(values) / len(values) for key, values in times.items()}


def _plot(results: dict[str, dict[str, float]], output: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415 (optional dependency)

    names = list(results)
    positions = range(len(names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(names)), 5))
    ax.bar(
        [p - width / 2 for p in positions],
        [results[n]["inference"] for n in names],
        width,
        label="Inference",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [results[n]["total"] for n in names],
        width,
        label="Total",
    )
    ax.set_xticks(list(positions), names, rotation=30, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Lane detection runtime per model")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Plot saved to {output}")


def main() -> None:
    """Benchmark the model configs given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "model_configs",
        nargs="+",
        help="Model config paths relative to the package (e.g. config/model.py)",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "ros_params.yaml",
        help="ROS params file to start from (default: config/ros_params.yaml)",
    )
    parser.add_argument("--warmup", type=float, default=30.0, help="Warmup in s")
    parser.add_argument("--samples", type=int, default=100, help="Samples per model")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Max. seconds per model"
    )
    parser.add_argument("--output", type=Path, help="Save the results as JSON")
    parser.add_argument("--plot", type=Path, help="Save a bar plot (matplotlib)")
    args = parser.parse_args()

    results: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for model_config in args.model_configs:
            print(f"\n=== {model_config} ===")
            params_file = Path(tmp_dir) / "ros_params.yaml"
            _write_params_file(args.params_file, model_config, params_file)
            try:
                results[model_config] = benchmark(
                    params_file, args.warmup, args.samples, args.timeout
                )
            except RuntimeError as e:
                print(f"Benchmark of {model_config} failed: {e}")
                continue
            print(
                f"{model_config}: total {results[model_config]['total']:.2f} ms, "
                f"inference {results[model_config]['inference']:.2f} ms"
            )

    print("\nResults (average ms):")
    for model_config, times in results.items():
        print(
            f"  {model_config}: total {times['total']:.2f}, "
            f"inference {times['inference']:.2f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")
    if args.plot and results:
        _plot(results, args.plot)


if __name__ == "__main__":
    main()
