#!/usr/bin/env python3
"""Platt / logistic calibration of CM scores on dev (bonafide=1, spoof=0)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
AASIST = ROOT / "Baseline-AASIST"
sys.path.insert(0, str(AASIST))
from eval.calculate_metrics import calculate_minDCF_EER_CLLR  # noqa: E402
from eval.calculate_modules import compute_eer  # noqa: E402


def load_score_lines(path: Path) -> list[list[str]]:
    rows = []
    with path.open() as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 4:
                rows.append(p)
    return rows


def rows_to_arrays(rows: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    bonafide, spoof = [], []
    for p in rows:
        s = float(p[2])
        if p[3].lower() == "bonafide":
            bonafide.append(s)
        else:
            spoof.append(s)
    return np.array(bonafide, dtype=np.float64), np.array(spoof, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out-scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="calibrated")
    args = parser.parse_args()

    rows = load_score_lines(args.scores)
    bonafide, spoof = rows_to_arrays(rows)
    x = np.concatenate([bonafide, spoof]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(bonafide)), np.zeros(len(spoof))])

    clf = LogisticRegression(max_iter=1000, C=1e6, solver="lbfgs")
    clf.fit(x, y)

    args.out_scores.parent.mkdir(parents=True, exist_ok=True)
    with args.out_scores.open("w") as fh:
        for p in rows:
            raw = float(p[2])
            cal = float(clf.predict_proba([[raw]])[0, 1])
            fh.write(f"{p[0]} {p[1]} {cal:.6f} {p[3]}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    min_dcf, eer, cllr = calculate_minDCF_EER_CLLR(
        cm_scores_file=str(args.out_scores),
        output_file=str(args.out_dir / f"{args.tag}_dcf_eer.txt"),
        printout=False,
    )
    b2, s2 = rows_to_arrays(load_score_lines(args.out_scores))
    eer2, _, _, _ = compute_eer(b2, s2)

    from sklearn.metrics import roc_auc_score

    y_true = np.concatenate([np.ones(len(b2)), np.zeros(len(s2))])
    y_score = np.concatenate([b2, s2])
    auc = float(roc_auc_score(y_true, y_score))

    metrics = {
        "tag": args.tag,
        "scores_file": str(args.out_scores),
        "method": "logistic_platt",
        "coef": float(clf.coef_.ravel()[0]),
        "intercept": float(clf.intercept_.ravel()[0]),
        "EER": float(eer),
        "EER_verify": float(eer2),
        "minDCF": float(min_dcf),
        "CLLR": float(cllr),
        "auc": auc,
        "n_bonafide": int(len(bonafide)),
        "n_spoof": int(len(spoof)),
    }
    out_json = args.out_dir / f"{args.tag}_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out_scores} and {out_json}")


if __name__ == "__main__":
    main()
