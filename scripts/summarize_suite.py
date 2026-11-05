"""Aggregate existing server results; never starts training or evaluates a policy."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    rows, curves = [], []
    commands = json.loads((args.suite / "commands.json").read_text())
    for entry in commands:
        directory = args.suite / f"seed_{entry['seed']}" / entry["method"]
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "missing_or_running"}
        evaluation_paths = sorted(directory.glob("evaluations/step_*/evaluation.json"))
        row = {"method": entry["method"], "seed": entry["seed"], "status": status["status"],
               "iteration": status.get("iteration"), "train_env_transitions": status.get("train_env_transitions"),
               "total_wall_s": status.get("total_wall_s"),
               **{f"time_{k}": v for k, v in status.get("times", {}).items()},
               **status.get("counters", {})}
        for path in evaluation_paths:
            evaluation = json.loads(path.read_text())
            summary = evaluation["summary"]
            curves.append({"method": entry["method"], "seed": entry["seed"],
                           "iteration": evaluation["iteration"], "norm_return": summary["mean_norm_return"],
                           "window_norm_return": summary["mean_window_norm_return"],
                           "completion_fraction": summary["mean_completion_fraction"],
                           "failure_rate": summary["failed"] / max(summary["clips"], 1),
                           "capped": summary["capped"], **summary["metrics"]})
        if evaluation_paths:
            last = curves[-1]
            row.update({k: v for k, v in last.items() if k not in ("method", "seed", "iteration")})
            row["last_eval_iteration"] = last["iteration"]
        rows.append(row)

    def save_csv(path, values):
        if not values:
            return
        columns = list(dict.fromkeys(k for row in values for k in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)

    save_csv(args.suite / "summary.csv", rows)
    save_csv(args.suite / "evaluation_curves.csv", curves)
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "completed" and row.get("norm_return") is not None:
            grouped[(row["method"], row.get("last_eval_iteration"))].append(row)
    aggregate = []
    for (method, iteration), values in grouped.items():
        scores = [v["norm_return"] for v in values]
        aggregate.append({"method": method, "iteration": iteration, "n_completed_seeds": len(scores),
                          "mean_norm_return": statistics.mean(scores),
                          "std_norm_return": statistics.stdev(scores) if len(scores) > 1 else None,
                          "seeds": [v["seed"] for v in values]})
    # Paired contrasts use the same seed AND training iteration; never compare unmatched final budgets.
    by_seed = {(r["method"], r["seed"], r.get("last_eval_iteration")): r for r in rows
               if r["status"] == "completed" and r.get("norm_return") is not None}
    paired = []
    for (method, iteration), values in grouped.items():
        if method == "supermimic":
            continue
        differences = []
        for value in values:
            baseline = by_seed.get(("supermimic", value["seed"], iteration))
            if baseline is not None:
                differences.append({"seed": value["seed"],
                                    "norm_return_delta": value["norm_return"] - baseline["norm_return"],
                                    "failure_rate_delta": value["failure_rate"] - baseline["failure_rate"]})
        paired.append({"method": method, "iteration": iteration, "pairs": differences,
                       "n_pairs": len(differences),
                       "mean_norm_return_delta": statistics.mean(
                           d["norm_return_delta"] for d in differences) if differences else None})
    (args.suite / "aggregate.json").write_text(json.dumps({
        "groups": aggregate, "paired_vs_supermimic": paired,
        "note": "Only completed seeds grouped at the same evaluation iteration; "
        "failures remain in summary.csv. Capped episodes are not successful full-clip completions."}, indent=2))
    print(f"Saved summary.csv, evaluation_curves.csv and aggregate.json in {args.suite}")


if __name__ == "__main__":
    main()
