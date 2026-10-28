"""Train once on one seed; evaluate untouched accounts from a different seed."""

import argparse
import csv
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from lotwise.models import Account
from lotwise.optimizer import Rebalancer
from lotwise.synthetic import generate_accounts
from lotwise.triage import FEATURES, features
from lotwise.validation import fingerprint


def dataset(count, seed):
    rng = np.random.default_rng(seed + 1000)
    engine = Rebalancer()
    x, y, identities, rejected = [], [], [], 0
    for account in generate_accounts(count, seed=seed):
        data = account.model_dump()
        nav = account.cash + sum(
            l.shares * next(s.price for s in account.securities if s.symbol == l.symbol)
            for l in account.lots
        )
        # Independent economic stress knobs, shared distribution across seeds.
        data["restricted_symbols"] = list(
            rng.choice(
                [s.symbol for s in account.securities], size=int(rng.integers(0, 6)), replace=False
            )
        )
        data["realized_gain_budget"] = nav * float(rng.uniform(0, 0.035))
        data["max_position_weight"] = float(rng.uniform(0.04, 0.24))
        data["min_cash_weight"] = float(rng.uniform(0.01, 0.65))
        for lot in data["lots"]:
            price = next(s.price for s in account.securities if s.symbol == lot["symbol"])
            lot["cost_basis_per_share"] = price * float(rng.uniform(0.45, 1.30))
        account = Account.model_validate(data)
        proposal = engine.propose(account)
        if proposal.status == "rejected":
            rejected += 1
            continue
        x.append(features(account))
        y.append(int(proposal.status == "infeasible"))
        identities.append(fingerprint(account))
    return np.array(x), np.array(y), identities, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1500)
    parser.add_argument("--test", type=int, default=500)
    args = parser.parse_args()
    started = time.perf_counter()
    x_train, y_train, train_ids, train_rejected = dataset(args.train, 2027)
    x_test, y_test, test_ids, test_rejected = dataset(args.test, 2028)
    assert not set(train_ids) & set(test_ids)
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(C=1.0, max_iter=3000, random_state=42).fit(
        scaler.transform(x_train), y_train
    )
    probabilities = classifier.predict_proba(scaler.transform(x_test))[:, 1]
    predicted = probabilities >= 0.5
    evaluation = {
        "training_accounts": len(y_train),
        "test_accounts": len(y_test),
        "training_seed": 2027,
        "test_seed": 2028,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1": float(f1_score(y_test, predicted)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "majority_baseline_accuracy": float(max(np.mean(y_test), 1 - np.mean(y_test))),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "test_conflicts": int(y_test.sum()),
        "training_rejected": train_rejected,
        "test_rejected": test_rejected,
        "fit_and_evaluation_seconds": time.perf_counter() - started,
        "scope": "Synthetic, same-distribution holdout from a separate seed; not production generalization or investment performance",
    }
    model = {
        "version": "synthetic-logistic-v1",
        "feature_names": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "evaluation": evaluation,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
    }
    path = Path("src/lotwise/artifacts/triage_model.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(model, indent=2) + "\n")
    results = Path("benchmarks/ai-results")
    results.mkdir(exist_ok=True)
    (results / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    with (results / "holdout.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["input_hash", "actual_conflict", "predicted_conflict", "conflict_score"])
        writer.writerows(zip(test_ids, y_test, predicted.astype(int), probabilities))
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
