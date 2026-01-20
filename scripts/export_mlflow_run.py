"""Export the latest MLflow run metrics to a small JSON summary.

This is handy for CV/docs: it lets you point to a stable artifact without
relying on screenshots.

Usage:
  python3 scripts/export_mlflow_run.py --tracking-uri file:./mlruns --experiment attention_is_all_you_need_cpu

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--experiment", default="attention_is_all_you_need_cpu")
    p.add_argument("--out", default="docs/assets/latest_run_summary.json")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"No experiment named {args.experiment!r} at {args.tracking_uri}")

    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        raise SystemExit(f"No runs found for experiment {args.experiment!r}")

    row = runs.iloc[0].to_dict()

    # Keep it small and stable.
    summary = {
        "run_id": row.get("run_id"),
        "experiment": args.experiment,
        "metrics": {k.replace("metrics.", ""): v for k, v in row.items() if k.startswith("metrics.")},
        "params": {k.replace("params.", ""): v for k, v in row.items() if k.startswith("params.")},
        "tags": {k.replace("tags.", ""): v for k, v in row.items() if k.startswith("tags.")},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
