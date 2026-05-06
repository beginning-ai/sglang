#!/usr/bin/env python3
"""Collect and plot Qwen3-Omni soak telemetry."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROCM_SMI_ARGS = (
    "--showuse",
    "--showmemuse",
    "--showmeminfo",
    "vram",
    "--json",
)

SERVER_TS_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"
)
SNAPSHOT_RE = re.compile(
    r"Qwen3-Omni runtime snapshot: event=(?P<event>\S+) "
    r"stats=(?P<stats>\{.*\})"
)


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.replace("%", ""))
    raise TypeError(f"Cannot convert {value!r} to float")


def _sample_rocm_smi(rocm_smi: str) -> dict[str, float]:
    raw = subprocess.check_output(
        [rocm_smi, *ROCM_SMI_ARGS],
        text=True,
    )
    payload = json.loads(raw)
    if "card0" not in payload:
        raise RuntimeError(f"rocm-smi JSON did not contain card0: {payload}")
    card = payload["card0"]
    used_b = _as_float(card["VRAM Total Used Memory (B)"])
    total_b = _as_float(card["VRAM Total Memory (B)"])
    return {
        "gpu_use_pct": _as_float(card["GPU use (%)"]),
        "vram_used_gib": used_b / 1024**3,
        "vram_total_gib": total_b / 1024**3,
        "vram_used_pct": _as_float(card["GPU Memory Allocated (VRAM%)"]),
        "mem_activity_pct": _as_float(card["GPU Memory Read/Write Activity (%)"]),
    }


def monitor(args: argparse.Namespace) -> None:
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    with output_path.open("w") as out:
        while True:
            now = time.time()
            out.write(
                json.dumps(
                    {
                        "time": now,
                        "elapsed_s": now - started_at,
                        "gpu": _sample_rocm_smi(args.rocm_smi),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            out.flush()
            if args.duration_s is not None and now - started_at >= args.duration_s:
                break
            time.sleep(args.interval_s)


def _load_gpu_samples(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_runtime_snapshots(path: Path) -> list[dict[str, Any]]:
    snapshots = []
    with path.open(errors="ignore") as f:
        for line in f:
            match = SNAPSHOT_RE.search(line)
            if match is None:
                continue
            ts_match = SERVER_TS_RE.match(line)
            if ts_match is None:
                raise RuntimeError(f"Runtime snapshot line has no timestamp: {line}")
            snapshots.append(
                {
                    "time": datetime.strptime(
                        ts_match.group("ts"), "%Y-%m-%d %H:%M:%S"
                    )
                    .replace(tzinfo=timezone.utc)
                    .timestamp(),
                    "event": match.group("event"),
                    "stats": json.loads(match.group("stats")),
                }
            )
    return snapshots


def _series(
    samples: list[dict[str, Any]],
    key,
) -> tuple[list[float], list[float]]:
    if not samples:
        return [], []
    start = samples[0]["time"]
    xs = [sample["time"] - start for sample in samples]
    ys = [float(key(sample)) for sample in samples]
    return xs, ys


def _write_svg_plot(
    output_path: str,
    panels: list[tuple[str, list[tuple[str, list[float], list[float]]]]],
) -> None:
    width = 1400
    panel_height = 260
    left = 90
    right = 40
    top = 30
    bottom = 45
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
    height = top + bottom + panel_height * len(panels)

    def points_for(xs: list[float], ys: list[float], panel_top: int) -> str:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_min == x_max:
            x_max = x_min + 1
        if y_min == y_max:
            y_max = y_min + 1
        plot_w = width - left - right
        plot_h = panel_height - top - bottom
        pts = []
        for x, y in zip(xs, ys):
            sx = left + (x - x_min) / (x_max - x_min) * plot_w
            sy = panel_top + top + (1 - (y - y_min) / (y_max - y_min)) * plot_h
            pts.append(f"{sx:.1f},{sy:.1f}")
        return " ".join(pts)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:14px;fill:#111827}.small{font-size:12px;fill:#374151}</style>',
    ]
    for panel_index, (title, series_list) in enumerate(panels):
        panel_top = panel_index * panel_height
        plot_top = panel_top + top
        plot_h = panel_height - top - bottom
        plot_w = width - left - right
        lines.append(
            f'<text x="{left}" y="{panel_top + 20}" font-weight="700">{title}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>'
        )
        non_empty = [(label, xs, ys) for label, xs, ys in series_list if xs and ys]
        all_ys = [y for _, _, ys in non_empty for y in ys]
        y_label = f"{min(all_ys):.1f} - {max(all_ys):.1f}" if all_ys else "no data"
        lines.append(f'<text class="small" x="10" y="{plot_top + 18}">{y_label}</text>')
        for idx, (label, xs, ys) in enumerate(non_empty):
            color = colors[idx % len(colors)]
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points_for(xs, ys, panel_top)}"/>'
            )
            legend_x = left + 20 + idx * 210
            legend_y = panel_top + panel_height - 16
            lines.append(
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>'
            )
            lines.append(
                f'<text class="small" x="{legend_x + 30}" y="{legend_y}">{label}</text>'
            )
    lines.append("</svg>")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))


def plot(args: argparse.Namespace) -> None:
    gpu_samples = _load_gpu_samples(Path(args.telemetry_jsonl))
    snapshots = _load_runtime_snapshots(Path(args.server_log))
    if gpu_samples:
        start_time = gpu_samples[0]["time"] - args.snapshot_slop_s
        end_time = gpu_samples[-1]["time"] + args.snapshot_slop_s
        snapshots = [
            snapshot
            for snapshot in snapshots
            if start_time <= snapshot["time"] <= end_time
        ]
    panels = [
        (
            "GPU and HBM activity",
            [
                ("GPU use %", *_series(gpu_samples, lambda s: s["gpu"]["gpu_use_pct"])),
                (
                    "HBM activity %",
                    *_series(gpu_samples, lambda s: s["gpu"]["mem_activity_pct"]),
                ),
            ],
        ),
        (
            "VRAM",
            [
                (
                    "VRAM used GiB",
                    *_series(gpu_samples, lambda s: s["gpu"]["vram_used_gib"]),
                )
            ],
        ),
        (
            "Main KV and radix",
            [
                (
                    "main KV available",
                    *_series(
                        snapshots, lambda s: s["stats"]["tree_cache"]["available"]
                    ),
                ),
                (
                    "main KV used",
                    *_series(snapshots, lambda s: s["stats"]["tree_cache"]["used"]),
                ),
                (
                    "radix protected",
                    *_series(
                        snapshots, lambda s: s["stats"]["tree_cache"]["protected"]
                    ),
                ),
                (
                    "radix evictable",
                    *_series(
                        snapshots, lambda s: s["stats"]["tree_cache"]["evictable"]
                    ),
                ),
            ],
        ),
        (
            "Component KV",
            [
                (
                    "thinker KV used",
                    *_series(
                        snapshots,
                        lambda s: s["stats"]["component_kv"]["thinker"]["used"],
                    ),
                ),
                (
                    "talker KV used",
                    *_series(
                        snapshots,
                        lambda s: s["stats"]["component_kv"]["talker"]["used"],
                    ),
                ),
                (
                    "code_predictor KV used",
                    *_series(
                        snapshots,
                        lambda s: s["stats"]["component_kv"]["code_predictor"][
                            "used"
                        ],
                    ),
                ),
                (
                    "streaming held tokens",
                    *_series(
                        snapshots,
                        lambda s: s["stats"]["streaming_session"]["held_tokens"],
                    ),
                ),
            ],
        ),
    ]
    if args.output_png.endswith(".svg"):
        _write_svg_plot(args.output_png, panels)
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)

    x, y = _series(gpu_samples, lambda sample: sample["gpu"]["gpu_use_pct"])
    axes[0].plot(x, y, label="GPU use %")
    x, y = _series(gpu_samples, lambda sample: sample["gpu"]["mem_activity_pct"])
    axes[0].plot(x, y, label="HBM activity %")
    axes[0].set_ylabel("%")
    axes[0].legend(loc="upper right")

    x, y = _series(gpu_samples, lambda sample: sample["gpu"]["vram_used_gib"])
    axes[1].plot(x, y, label="VRAM used GiB")
    axes[1].set_ylabel("GiB")
    axes[1].legend(loc="upper right")

    x, y = _series(
        snapshots, lambda sample: sample["stats"]["tree_cache"]["available"]
    )
    axes[2].plot(x, y, label="main KV available")
    x, y = _series(snapshots, lambda sample: sample["stats"]["tree_cache"]["used"])
    axes[2].plot(x, y, label="main KV used")
    x, y = _series(
        snapshots, lambda sample: sample["stats"]["tree_cache"]["protected"]
    )
    axes[2].plot(x, y, label="radix protected")
    x, y = _series(
        snapshots, lambda sample: sample["stats"]["tree_cache"]["evictable"]
    )
    axes[2].plot(x, y, label="radix evictable")
    axes[2].set_ylabel("tokens/pages")
    axes[2].legend(loc="upper right")

    for component in ("thinker", "talker", "code_predictor"):
        x, y = _series(
            snapshots,
            lambda sample, component=component: sample["stats"]["component_kv"][
                component
            ]["used"],
        )
        axes[3].plot(x, y, label=f"{component} KV used")
    x, y = _series(
        snapshots,
        lambda sample: sample["stats"]["streaming_session"]["held_tokens"],
    )
    axes[3].plot(x, y, label="streaming held tokens")
    axes[3].set_ylabel("tokens")
    axes[3].set_xlabel("seconds since first sample")
    axes[3].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--output-jsonl", required=True)
    monitor_parser.add_argument("--interval-s", type=float, default=5.0)
    monitor_parser.add_argument("--duration-s", type=float)
    monitor_parser.add_argument("--rocm-smi", default="/opt/rocm/bin/rocm-smi")

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--telemetry-jsonl", required=True)
    plot_parser.add_argument("--server-log", required=True)
    plot_parser.add_argument("--output-png", required=True)
    plot_parser.add_argument("--snapshot-slop-s", type=float, default=10.0)

    args = parser.parse_args()
    if args.command == "monitor":
        monitor(args)
    elif args.command == "plot":
        plot(args)
    else:
        raise RuntimeError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
