#!/usr/bin/env python3
"""Experiment A: EER/minDCF/CLLR + accuracy/F1/ROC from CM score file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AASIST = ROOT / "Baseline-AASIST"
sys.path.insert(0, str(AASIST))

from eval.calculate_metrics import calculate_minDCF_EER_CLLR  # noqa: E402
from eval.calculate_modules import compute_eer  # noqa: E402


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse AASIST score file: spk utt score bonafide|spoof."""
    bonafide, spoof, labels = [], [], []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            score = float(parts[2])
            key = parts[3].lower()
            if key == "bonafide":
                bonafide.append(score)
                labels.append(1)
            else:
                spoof.append(score)
                labels.append(0)
    return (
        np.array(bonafide, dtype=np.float64),
        np.array(spoof, dtype=np.float64),
        np.array(labels, dtype=np.int32),
    )


def sklearn_metrics(bonafide: np.ndarray, spoof: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        roc_curve,
    )

    y_true = np.concatenate(
        [np.ones(len(bonafide)), np.zeros(len(spoof))],
    )
    y_score = np.concatenate([bonafide, spoof])
    y_pred = (y_score >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def _maybe_swanlab_init(tag: str, results: dict, roc_path: Path) -> None:
    import os

    if os.environ.get("USE_SWANLAB", "").lower() not in ("1", "true", "yes", "on"):
        return
    import swanlab

    exp = os.environ.get("SWANLAB_EXPERIMENT", f"exp_a_{tag}")
    if not getattr(_maybe_swanlab_init, "_inited", False):
        swanlab.init(
            project=os.environ.get("SWANLAB_PROJECT", "ASVspoof5"),
            experiment_name=exp,
            description=f"Experiment A metrics — {tag}",
            mode=os.environ.get("SWANLAB_MODE", "cloud"),
            logdir=os.environ.get("SWANLAB_LOGDIR", str(ROOT / "logs" / "swanlab")),
        )
        _maybe_swanlab_init._inited = True  # type: ignore[attr-defined]
    swanlab.log({
        f"{tag}/EER": results["EER"],
        f"{tag}/minDCF": results["minDCF"],
        f"{tag}/CLLR": results["CLLR"],
        f"{tag}/accuracy": results["accuracy"],
        f"{tag}/f1": results["f1"],
        f"{tag}/auc": results["auc"],
    })
    swanlab.log({f"{tag}/roc": swanlab.Image(str(roc_path))})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="aasist")
    parser.add_argument("--no-swanlab", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bonafide, spoof, _ = load_scores(args.scores)

    metrics_txt = args.out_dir / f"{args.tag}_dcf_eer.txt"
    min_dcf, eer, cllr = calculate_minDCF_EER_CLLR(
        cm_scores_file=str(args.scores),
        output_file=str(metrics_txt),
        printout=False,
    )
    eer2, _, _, _ = compute_eer(bonafide, spoof)

    sk = sklearn_metrics(bonafide, spoof)
    results = {
        "tag": args.tag,
        "scores_file": str(args.scores),
        "n_bonafide": int(len(bonafide)),
        "n_spoof": int(len(spoof)),
        "minDCF": float(min_dcf),
        "EER": float(eer),
        "EER_verify": float(eer2),
        "CLLR": float(cllr),
        "accuracy": sk["accuracy"],
        "f1": sk["f1"],
        "auc": sk["auc"],
    }

    out_json = args.out_dir / f"{args.tag}_metrics.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    plt.figure(figsize=(6, 5))
    plt.plot(sk["fpr"], sk["tpr"], label=f"AUC={sk['auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC — {args.tag}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_path = args.out_dir / f"{args.tag}_roc.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_json} and {roc_path}")

    if not args.no_swanlab:
        _maybe_swanlab_init(args.tag, results, roc_path)


if __name__ == "__main__":
    main()
