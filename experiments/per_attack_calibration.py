#!/usr/bin/env python3
"""Per-attack-ID score shifts (B1) to minimize pooled dev EER."""
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

from run_exp_b_attack_compare import load_scores_with_meta  # noqa: E402


def pooled_eer(df: pd.DataFrame, col: str) -> float:
    bonafide = df.loc[df["label"] == "bonafide", col].to_numpy()
    spoof = df.loc[df["label"] != "bonafide", col].to_numpy()
    if len(bonafide) == 0 or len(spoof) == 0:
        return float("nan")
    eer, _, _, _ = compute_eer(bonafide, spoof)
    return float(eer)


def apply_attack_shifts(df: pd.DataFrame, shifts: dict) -> pd.DataFrame:
    out = df.copy()
    out["score_adj"] = out["score"].astype(float)
    for atk, shift in shifts.items():
        mask = (out["label"] != "bonafide") & (out["attack"] == atk)
        out.loc[mask, "score_adj"] += shift
    unknown = (out["label"] != "bonafide") & ~out["attack"].isin(shifts.keys())
    if unknown.any() and shifts:
        out.loc[unknown, "score_adj"] += float(np.mean(list(shifts.values())))
    return out


def coordinate_search(
    df: pd.DataFrame,
    attacks: list,
    n_grid: int = 15,
    max_rounds: int = 3,
) -> dict:
    spoof_all = df.loc[df["label"] != "bonafide", "score"].to_numpy()
    bonafide = df.loc[df["label"] == "bonafide", "score"].to_numpy()
    base_eer, _, _, _ = compute_eer(bonafide, spoof_all)
    std = float(np.std(spoof_all)) if len(spoof_all) else 0.1
    grid = np.linspace(-0.12 * std, 0.12 * std, n_grid)

    shifts = {a: 0.0 for a in attacks}
    best_eer = float(base_eer)

    for _ in range(max_rounds):
        improved = False
        for atk in attacks:
            local_best = shifts[atk]
            local_eer = best_eer
            for s in grid:
                trial = dict(shifts)
                trial[atk] = float(s)
                adj = apply_attack_shifts(df, trial)
                eer = pooled_eer(adj, "score_adj")
                if eer < local_eer:
                    local_eer = eer
                    local_best = float(s)
            if local_eer < best_eer - 1e-8:
                shifts[atk] = local_best
                best_eer = local_eer
                improved = True
        if not improved:
            break

    return {
        "baseline_pooled_eer": float(base_eer),
        "pooled_eer": float(best_eer),
        "shifts": shifts,
        "improvement_pp": (float(base_eer) - float(best_eer)) * 100,
    }


def write_scores(
    df: pd.DataFrame,
    shifts: dict,
    metainfor: Path,
    out_path: Path,
) -> None:
    adj = apply_attack_shifts(df, shifts)
    spk_map = {}
    with metainfor.open() as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2:
                spk_map[p[1]] = p[0]
    with out_path.open("w") as fh:
        for _, row in adj.iterrows():
            spk = spk_map.get(row["utt"], "unknown")
            fh.write(f"{spk} {row['utt']} {row['score_adj']:.6f} {row['label']}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--metainfor", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-grid", type=int, default=15)
    parser.add_argument("--write-scores", action="store_true")
    args = parser.parse_args()

    df = load_scores_with_meta(args.scores, args.metainfor)
    attacks = sorted(df.loc[df["label"] != "bonafide", "attack"].dropna().unique())
    results = coordinate_search(df, attacks, n_grid=args.n_grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.write_scores:
        out_scores = args.out_dir / "eval_scores_dev_full_per_attack.txt"
        write_scores(df, results["shifts"], args.metainfor, out_scores)
        results["adjusted_scores_file"] = str(out_scores)

    out_json = args.out_dir / "per_attack_calibration.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
