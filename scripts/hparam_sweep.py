#!/usr/bin/env python3
"""
Hyperparameter sweep driver for train_model_rbf.py.

The script prints the hyperparameter ranges being explored, the probe-training
duration used to accept/reject a configuration, and then sequentially launches
train_model_rbf.py runs for each trial (optionally truncated via --max-trials).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ParamSpec:
    name: str
    note: str
    values: Sequence[Any]
    arg: str | None = None
    kind: str = "value"  # value | flag | group
    tag_prefix: str | None = None


PARAM_SPECS: Sequence[ParamSpec] = (
    ParamSpec(
        name="learning_rate",
        arg="--lr",
        values=[1e-4, 3e-4, 1e-3, 3e-3],
        note="Adam learning rate controlling update magnitude.",
        tag_prefix="lr",
    ),
    ParamSpec(
        name="batch_size",
        arg="--batch_size",
        values=[16, 20, 24],
        note=(
            "Mini-batch size (smaller batches stabilize noisy updates "
            "when memory is tight)."
        ),
        tag_prefix="bs",
    ),
    ParamSpec(
        name="temperature",
        note="Constant temperature controlling soft alignment sharpness.",
        values=(
            {
                "label": "temp1",
                "args": {"--T_start": 1.0, "--T_end": 1.0},
                "note": "Temperature fixed at 1.0.",
            },
            {
                "label": "temp2",
                "args": {"--T_start": 2.0, "--T_end": 1.0},
                "note": "Temperature starts at 2.0.",
            },
            {
                "label": "temp3",
                "args": {"--T_start": 3.0, "--T_end": 1.0},
                "note": "Temperature starts at 3.0.",
            },
            {
                "label": "temp4",
                "args": {"--T_start": 4.0, "--T_end": 1.0},
                "note": "Temperature starts at 4.0.",
            },
            {
                "label": "temp5",
                "args": {"--T_start": 5.0, "--T_end": 1.0},
                "note": "Temperature starts at 5.0.",
            },
        ),
        kind="group",
    ),
)

RESULTS_HEADER = [
    "run_tag",
    "status",
    "learning_rate",
    "batch_size",
    "temperature_label",
    "T_start",
    "T_end",
    "final_val_loss",
    "best_val_loss",
    "steps_completed",
]


def format_value_for_display(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if 0.1 <= abs(value) < 100:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.0e}".replace("+", "")
    return str(value)


def describe_spec_values(spec: ParamSpec) -> str:
    if spec.kind == "group":
        parts = []
        for item in spec.values:
            label = item.get("label", "option")
            args_desc = ", ".join(
                f"{k}={format_value_for_display(v)}"
                for k, v in item["args"].items()
            )
            parts.append(f"{label} ({args_desc})")
        return "; ".join(parts)
    values = ", ".join(format_value_for_display(v) for v in spec.values)
    return f"[{values}]"


def sanitize_token(token: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in token)
    return safe.strip("_")


def format_tag_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        if value == 0:
            return "0"
        text = f"{value:.0e}".replace("+", "")
        if "e" not in text:
            text = text.replace(".", "p")
        return text.replace(".", "p")
    return str(value)


def build_tag_token(spec: ParamSpec, value: Any) -> str:
    if spec.kind == "group":
        label = value.get("label")
        return sanitize_token(label) if label else ""
    if spec.kind == "flag":
        return sanitize_token(spec.tag_prefix or spec.name) if value else ""
    prefix = spec.tag_prefix or spec.name
    return sanitize_token(f"{prefix}{format_tag_value(value)}")


def calc_total_combinations(specs: Sequence[ParamSpec]) -> int:
    total = 1
    for spec in specs:
        total *= len(spec.values)
    return total


def print_search_space(
    specs: Sequence[ParamSpec], epochs: int, max_steps: int | None
) -> None:
    print("Hyperparameter search space (per-parameter ranges):")
    for spec in specs:
        print(f"  - {spec.name}: {describe_spec_values(spec)}")
        print(f"      {spec.note}")
    if max_steps:
        print(
            (
                "Probe training duration per trial: {epochs} epoch(s),"
                f" capped at {max_steps} step(s)"
            )
        )
    else:
        print(f"Probe training duration per trial: {epochs} epoch(s)")


def extract_metrics(out_dir: str, run_tag: str) -> Dict[str, Any]:
    metrics_path = os.path.join(out_dir, f"metrics_{run_tag}.json")
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, "r") as f:
        data = json.load(f)
    val_loss = data.get("val_loss", [])
    train_loss = data.get("train_loss", [])
    metrics = {
        "final_val_loss": val_loss[-1] if val_loss else None,
        "best_val_loss": min(val_loss) if val_loss else None,
        "final_train_loss": train_loss[-1] if train_loss else None,
        "steps_completed": data.get("steps_completed"),
    }
    return metrics


def build_results_row(
    metadata: Dict[str, Any], metrics: Dict[str, Any], status: str
) -> Dict[str, Any]:
    return {
        "run_tag": metadata.get("run_tag"),
        "status": status,
        "learning_rate": metadata.get("learning_rate"),
        "batch_size": metadata.get("batch_size"),
        "temperature_label": metadata.get("temperature"),
        "T_start": metadata.get("T_start"),
        "T_end": metadata.get("T_end"),
        "final_val_loss": metrics.get("final_val_loss"),
        "best_val_loss": metrics.get("best_val_loss"),
        "steps_completed": metrics.get("steps_completed"),
    }


def append_results_row(csv_path: str, row: Dict[str, Any]) -> None:
    path = Path(csv_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def iter_trials(
    specs: Sequence[ParamSpec],
) -> Iterable[List[Tuple[ParamSpec, Any]]]:
    value_product = itertools.product(*(spec.values for spec in specs))
    for combo in value_product:
        yield list(zip(specs, combo))


def build_cli_args(
    spec: ParamSpec, value: Any
) -> Tuple[List[str], Dict[str, Any]]:
    cli_args: List[str] = []
    metadata: Dict[str, Any] = {}
    if spec.kind == "group":
        metadata[spec.name] = value.get("label", value.get("args"))
        note = value.get("note")
        if note:
            metadata[f"{spec.name}_note"] = note
        for arg_name, arg_value in value["args"].items():
            cli_args.extend([arg_name, str(arg_value)])
            metadata[arg_name.lstrip("-")] = arg_value
        return cli_args, metadata
    if spec.kind == "flag":
        metadata[spec.name] = bool(value)
        if value:
            cli_args.append(spec.arg)
        return cli_args, metadata
    cli_args.extend([spec.arg, str(value)])
    metadata[spec.name] = value
    return cli_args, metadata


def run_trial(
    trial_idx: int,
    planned: int,
    base_cmd: List[str],
    trial: List[Tuple[ParamSpec, Any]],
    tag_prefix: str,
    out_dir: str,
    results_csv: str | None,
    dry_run: bool,
    stop_on_error: bool,
) -> int:
    trial_cli: List[str] = []
    metadata: Dict[str, Any] = {}
    tag_tokens: List[str] = []

    for spec, value in trial:
        args, meta = build_cli_args(spec, value)
        trial_cli.extend(args)
        metadata.update(meta)
        token = build_tag_token(spec, value)
        if token:
            tag_tokens.append(token)

    run_tag_parts = [tag_prefix, f"{trial_idx:03d}"] + tag_tokens
    run_tag = "_".join(filter(None, run_tag_parts))
    metadata["run_tag"] = run_tag

    cmd = base_cmd + ["--run_tag", run_tag] + trial_cli
    printable_cmd = " ".join(json.dumps(part) for part in cmd)
    print(f"\n[{trial_idx}/{planned}] Launching: {printable_cmd}")
    if dry_run:
        return 0

    result = subprocess.run(cmd, check=False)
    status = "ok" if result.returncode == 0 else f"failed({result.returncode})"
    metrics: Dict[str, Any] = {}
    should_stop = False
    if result.returncode != 0:
        print(
            f"Trial {trial_idx} failed with exit code {result.returncode}. "
            f"{'Stopping early.' if stop_on_error else 'Continuing...'}"
        )
        should_stop = stop_on_error
    else:
        metrics = extract_metrics(out_dir, run_tag)
        final_val = metrics.get("final_val_loss")
        best_val = metrics.get("best_val_loss")
        steps = metrics.get("steps_completed")
        if final_val is not None:
            best_text = f"{best_val:.4f}" if best_val is not None else "n/a"
            step_text = steps if steps is not None else "n/a"
            print(
                f"Validation summary ({run_tag}): final={final_val:.4f} "
                f"best={best_text} steps={step_text}"
            )
        else:
            print(f"Validation summary ({run_tag}): metrics file missing.")

    if results_csv:
        row = build_results_row(metadata, metrics, status)
        append_results_row(results_csv, row)

    if should_stop:
        sys.exit(result.returncode)

    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter sweep for train_model_rbf.py."
    )
    parser.add_argument(
        "--train-script",
        default="train_model_rbf.py",
        help="Path to train_model_rbf.py (default: %(default)s)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory passed through to the training script.",
    )
    parser.add_argument(
        "--out-dir",
        default="checkpoints",
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train per trial.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15000,
        help="Maximum number of training steps per trial (0 disables the cap).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=24,
        help="Number of configurations to launch (0 runs the full grid).",
    )
    parser.add_argument(
        "--tag-prefix",
        default="rbf",
        help="Prefix to add to every generated run_tag.",
    )
    parser.add_argument(
        "--results-csv",
        default="sweep_results.csv",
        help="Validation summary path (empty string to disable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands that would be executed.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the sweep immediately if a trial fails.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments to append (use `-- --flag value`).",
    )
    args = parser.parse_args()

    steps_cap = args.max_steps if args.max_steps > 0 else None
    results_csv = args.results_csv or None

    total_combos = calc_total_combinations(PARAM_SPECS)
    planned = (
        total_combos
        if args.max_trials <= 0
        else min(args.max_trials, total_combos)
    )

    print_search_space(PARAM_SPECS, args.epochs, steps_cap)
    print(
        f"Total candidate configurations: {total_combos}. "
        f"Scheduled to run: {planned} (--max-trials 0 runs the entire grid)."
    )

    base_cmd = [
        args.python,
        args.train_script,
        "--data_dir",
        args.data_dir,
        "--out_dir",
        args.out_dir,
        "--epochs",
        str(args.epochs),
        "--max_steps",
        str(args.max_steps),
        "--use_rbf",
    ]
    if args.extra:
        base_cmd.extend(args.extra)

    trials_run = 0
    for idx, trial in enumerate(iter_trials(PARAM_SPECS), start=1):
        if args.max_trials > 0 and trials_run >= args.max_trials:
            break
        run_trial(
            trial_idx=idx,
            planned=planned,
            base_cmd=base_cmd,
            trial=trial,
            tag_prefix=args.tag_prefix,
            out_dir=args.out_dir,
            results_csv=results_csv,
            dry_run=args.dry_run,
            stop_on_error=args.stop_on_error,
        )
        trials_run += 1

    print(f"\nCompleted {trials_run} trial(s).")
    if results_csv and not args.dry_run:
        print(f"Validation summaries appended to {results_csv}")


if __name__ == "__main__":
    main()
