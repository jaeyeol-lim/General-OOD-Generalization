"""Shared subprocess-based grid search launcher."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence


def _value_name(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_sweep(
    method: str,
    parameter: str | None = None,
    values: Sequence[float | None] = (None,),
    argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=f"Grid search for {method} on DrugOOD IC50")
    parser.add_argument("--domains", nargs="+", choices=("assay", "scaffold", "size"), default=["assay"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--subset", choices=("core", "general", "refined"), default="core")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "sweeps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")

    script = Path(__file__).resolve().parent / f"{method}.py"
    jobs = []
    for domain, seed, value in itertools.product(args.domains, args.seeds, values):
        output_dir = (
            args.output_root
            / method
            / domain
            / (
                f"{parameter.lstrip('-').replace('-', '_')}_{_value_name(value)}"
                if parameter is not None and value is not None
                else "default"
            )
            / f"seed_{seed}"
        )
        command = [
            sys.executable,
            str(script),
            "--domain",
            domain,
            "--subset",
            args.subset,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--output-dir",
            str(output_dir),
        ]
        if parameter is not None and value is not None:
            command.extend((parameter, str(value)))
        if args.data_root is not None:
            command.extend(("--data-root", str(args.data_root)))
        command.extend(extra)
        jobs.append((command, output_dir, domain, value))

    print(f"method={method} jobs={len(jobs)} max_parallel={args.max_parallel}")
    for command, _, _, _ in jobs:
        print(" ".join(command))
    if args.dry_run:
        return

    def launch(job):
        command, _, _, _ = job
        completed = subprocess.run(command, check=False)
        return job, completed.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            job, returncode = future.result()
            if returncode:
                failures.append((job[0], returncode))

    if failures:
        details = "\n".join(f"exit={code}: {' '.join(command)}" for command, code in failures)
        raise SystemExit(f"{len(failures)}/{len(jobs)} sweep jobs failed:\n{details}")

    grouped = {}
    for _, output_dir, domain, value in jobs:
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        value_label = "default" if value is None else f"{value:g}"
        key = f"{domain}/{parameter or 'default'}={value_label}"
        grouped.setdefault(key, []).append(summary)

    def stats(values):
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return {"mean": None, "std": None}
        return {
            "mean": statistics.fmean(finite),
            "std": statistics.pstdev(finite),
        }

    aggregate = {"method": method, "seeds": args.seeds, "groups": {}}
    for key, summaries in sorted(grouped.items()):
        aggregate["groups"][key] = {
            "runs": len(summaries),
            "best_ood_val_accuracy": stats(
                summary["best_ood_val_accuracy"] for summary in summaries
            ),
            "ood_test_accuracy": stats(
                summary["metrics"]["ood_test"]["accuracy"] for summary in summaries
            ),
            "ood_test_roc_auc": stats(
                summary["metrics"]["ood_test"]["roc_auc"] for summary in summaries
            ),
        }
    aggregate_path = args.output_root / method / "aggregate.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"aggregate={aggregate_path}")
    print(f"completed {len(jobs)} jobs")
