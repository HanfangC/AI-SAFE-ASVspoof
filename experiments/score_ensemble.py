#!/usr/bin/env python3
"""Fuse two score files: score = alpha * s1 + (1-alpha) * s2; search alpha on dev."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AASIST = ROOT / "Baseline-AASIST"
sys.path.insert(0, str(AASIST))
from eval.calculate_modules import compute_eer  # noqa: E402


def load_scores(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 4:
                continue
            rows.append({"utt": p[1], "score": float(p[2]), "label": p[3].lower()})
    return pd.DataFrame(rows)


def pooled_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    bonafide = scores[labels == 1]
    spoof = scores[labels == 0]
    eer, _, _, _ = compute_eer(bonafide, spoof)
    return float(eer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-a", type=Path, required=True)
    parser.add_argument("--scores-b", type=Path, required=True)
    parser.add_argument("--metainfor", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="ensemble")
    parser.add_argument("--n-grid", type=int, default=51)
    args = parser.parse_args()

    a = load_scores(args.scores_a).rename(columns={"score": "score_a"})
    b = load_scores(args.scores_b).rename(columns={"score": "score_b"})
    df = a.merge(b, on=["utt", "label"], how="inner")
    labels = (df["label"] == "bonafide").astype(int).to_numpy()

    base_a = pooled_eer(df["score_a"].to_numpy(), labels)
    base_b = pooled_eer(df["score_b"].to_numpy(), labels)

    best = {"alpha": 0.5, "pooled_eer": 1.0, "eer_a": base_a, "eer_b": base_b}
    for alpha in np.linspace(0.0, 1.0, args.n_grid):
        fused = alpha * df["score_a"].to_numpy() + (1.0 - alpha) * df["score_b"].to_numpy()
        eer = pooled_eer(fused, labels)
        if eer < best["pooled_eer"]:
            best["alpha"] = float(alpha)
            best["pooled_eer"] = eer

    df["score_fused"] = (
        best["alpha"] * df["score_a"] + (1.0 - best["alpha"]) * df["score_b"]
    )
    best["improvement_pp_vs_a"] = (base_a - best["pooled_eer"]) * 100
    best["improvement_pp_vs_b"] = (base_b - best["pooled_eer"]) * 100
    best["scores_a"] = str(args.scores_a)
    best["scores_b"] = str(args.scores_b)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spk_map = {}
    with args.metainfor.open() as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2:
                spk_map[p[1]] = p[0]

    out_scores = args.out_dir / f"eval_scores_dev_full_{args.tag}.txt"
    with out_scores.open("w") as fh:
        for _, row in df.iterrows():
            spk = spk_map.get(row["utt"], "unknown")
            fh.write(
                f"{spk} {row['utt']} {row['score_fused']:.6f} {row['label']}\n"
            )
    best["fused_scores_file"] = str(out_scores)

    out_json = args.out_dir / f"{args.tag}.json"
    out_json.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"Wrote {out_scores}")


if __name__ == "__main__":
    main()
