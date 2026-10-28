"""Reproducible, machine-readable benchmark; includes first-solve overhead."""

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from lotwise.baseline import fifo_gains, optimized_positive_gains
from lotwise.database import Repository
from lotwise.optimizer import Rebalancer
from lotwise.synthetic import generate_accounts, impossible_account
from lotwise.validation import replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    if args.accounts < 1 or args.runs < 1:
        parser.error("accounts and runs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    accounts = generate_accounts(args.accounts, seed=args.seed)
    all_runs, rows, proposals = [], [], []
    for run in range(args.runs):
        engine = Rebalancer()
        current_rows, current_proposals = [], []
        started = time.perf_counter()
        for account in accounts:
            proposal = engine.propose(account)
            if proposal.status != "optimal":
                raise RuntimeError(
                    f"unexpected failure: {account.account_id}: {proposal.exceptions}"
                )
            _metrics, errors = replay(account, proposal.trades)
            if errors:
                raise AssertionError(errors)
            fifo = fifo_gains(account, proposal)
            current_rows.append(
                {
                    "account_id": account.account_id,
                    "run": run + 1,
                    **proposal.metrics,
                    "fifo_positive_realized_gain": fifo["positive_realized_gain"],
                    "optimized_positive_realized_gain": optimized_positive_gains(proposal),
                }
            )
            current_proposals.append(proposal)
        elapsed = time.perf_counter() - started
        all_runs.append(
            {
                "run": run + 1,
                "seconds": elapsed,
                "accounts_per_second": args.accounts / elapsed,
                "p50_proposal_ms": statistics.median(r["runtime_ms"] for r in current_rows),
                "p95_proposal_ms": float(
                    np.percentile([r["runtime_ms"] for r in current_rows], 95)
                ),
            }
        )
        if run == 0:
            rows, proposals = current_rows, current_proposals
        print(f"Run {run + 1}: {args.accounts} accounts in {elapsed:.3f}s", flush=True)

    exception_start = time.perf_counter()
    engine = Rebalancer()
    exceptions = [engine.propose(impossible_account(k)) for k in range(50)]
    assert all(p.status == "infeasible" and not p.trades for p in exceptions)
    exception_seconds = time.perf_counter() - exception_start

    # An on-disk SQLite measurement; PostgreSQL has a separate optional test.
    db_path = args.output / f"audit-{time.time_ns()}.db"
    repo = Repository(f"sqlite:///{db_path}")
    db_started = time.perf_counter()
    for account, proposal in zip(accounts, proposals):
        saved = repo.save(account, proposal)
        if repo.get(saved["proposal_id"]) != saved:
            raise AssertionError("persistence round trip changed payload")
    persistence_seconds = time.perf_counter() - db_started
    repo.close()

    fifo_total = sum(r["fifo_positive_realized_gain"] for r in rows)
    optimized_total = sum(r["optimized_positive_realized_gain"] for r in rows)
    before = statistics.mean(r["initial_weight_rmse_bps"] for r in rows)
    after = statistics.mean(r["weight_rmse_bps"] for r in rows)
    summary = {
        "seed": args.seed,
        "account_count": args.accounts,
        "securities_per_account": 20,
        "tax_lots_per_account": 60,
        "total_tax_lots": args.accounts * 60,
        "runs": all_runs,
        "median_batch_seconds": statistics.median(r["seconds"] for r in all_runs),
        "mean_initial_weight_rmse_bps": before,
        "mean_final_weight_rmse_bps": after,
        "mean_weight_rmse_reduction_pct": (1 - after / before) * 100,
        "fifo_positive_realized_gain_total": fifo_total,
        "optimized_positive_realized_gain_total": optimized_total,
        "positive_realized_gain_reduction_vs_fifo_pct": (1 - optimized_total / fifo_total) * 100,
        "validated_proposals_per_run": len(proposals),
        "constraint_violations_all_runs": 0,
        "infeasible_scenarios_detected": len(exceptions),
        "infeasibility_check_seconds": exception_seconds,
        "total_trades": sum(len(p.trades) for p in proposals),
        "sqlite_persist_and_retrieve_seconds": persistence_seconds,
        "sqlite_proposals_round_tripped": len(proposals),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("cvxpy", "clarabel", "numpy", "sqlalchemy")
            },
        },
        "methodology": {
            "data": "Seeded synthetic accounts and prices, not historical market returns",
            "timing": "Sequential single-process solve + replay + FIFO counterfactual; includes cold first solve, excludes fixture generation and database I/O",
            "fifo": "Identical security-level sell quantities, input-order FIFO lots; counterfactual may exceed gain budget",
            "gain_metric": "Sum of positive realized lot gains, not tax savings or investment returns",
            "drift": "Per-account cross-sectional weight RMSE vs model, using initial NAV; not return tracking error",
            "scope": "60 tax lots and 20 securities per account; fractional shares; no wash-sale or cross-account model",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "accounts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "example_account.json").write_text(accounts[0].model_dump_json(indent=2) + "\n")
    (args.output / "example_proposal.json").write_text(
        proposals[0].model_dump_json(indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
