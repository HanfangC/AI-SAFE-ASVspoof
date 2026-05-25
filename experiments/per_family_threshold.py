#!/usr/bin/env python3
"""Per-attack-family score shifts to minimize pooled dev EER."""
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

from run_exp_b_attack_compare import attack_family, load_scores_with_meta  # noqa: E402


def apply_shifts(df: pd.DataFrame, shift_tts: float, shift_vc: float) -> pd.DataFrame:
    out = df.copy()
    out["score_adj"] = out["score"].astype(float)
    out.loc[(out["label"] != "bonafide") & (out["family"] == "tts"), "score_adj"] += shift_tts
    out.loc[(out["label"] != "bonafide") & (out["family"] == "vc"), "score_adj"] += shift_vc
    other = (out["label"] != "bonafide") & (~out["family"].isin(["tts", "vc"]))
    out.loc[other, "score_adj"] += (shift_tts + shift_vc) / 2.0
    return out


def pooled_eer_from_df(df: pd.DataFrame, score_col: str = "score_adj") -> float:
    bonafide = df.loc[df["label"] == "bonafide", score_col].to_numpy()
    spoof = df.loc[df["label"] != "bonafide", score_col].to_numpy()
    if len(bonafide) == 0 or len(spoof) == 0:
        return float("nan")
    eer, _, _, _ = compute_eer(bonafide, spoof)
    return float(eer)


def family_eer(df: pd.DataFrame, fam: str, score_col: str = "score_adj") -> float:
    sub = df[df["family"] == fam]
    bonafide = df.loc[df["label"] == "bonafide", score_col].to_numpy()
    spoof = sub.loc[sub["label"] != "bonafide", score_col].to_numpy()
    if len(spoof) == 0:
        return float("nan")
    eer, _, _, _ = compute_eer(bonafide, spoof)
    return float(eer)


def search_shifts(df: pd.DataFrame, n_grid: int = 25) -> dict:
    bonafide = df.loc[df["label"] == "bonafide", "score"].to_numpy()
    spoof_all = df.loc[df["label"] != "bonafide", "score"].to_numpy()
    base_eer, _, _, _ = compute_eer(bonafide, spoof_all)

    std = float(np.std(spoof_all)) if len(spoof_all) else 0.1
    grid = np.linspace(-0.15 * std, 0.15 * std, n_grid)

    best = {
        "baseline_pooled_eer": float(base_eer),
        "pooled_eer": float(base_eer),
        "shift_tts": 0.0,
        "shift_vc": 0.0,
    }
    for st in grid:
        for sv in grid:
            adj = apply_shifts(df, st, sv)
            eer = pooled_eer_from_df(adj)
            if eer < best["pooled_eer"]:
                best["pooled_eer"] = eer
                best["shift_tts"] = float(st)
                best["shift_vc"] = float(sv)

    adj = apply_shifts(df, best["shift_tts"], best["shift_vc"])
    best["eer_tts"] = family_eer(adj, "tts")
    best["eer_vc"] = family_eer(adj, "vc")
    best["improvement_pp"] = (best["baseline_pooled_eer"] - best["pooled_eer"]) * 100
    return best


def write_adjusted_scores(
    df: pd.DataFrame,
    shift_tts: float,
    shift_vc: float,
    metainfor: Path,
    out_path: Path,
) -> None:
    adj = apply_shifts(df, shift_tts, shift_vc)
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
    parser.add_argument("--groups", type=Path,
                        default=Path(__file__).parent / "attack_groups.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-grid", type=int, default=25)
    parser.add_argument("--write-scores", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.groups.read_text(encoding="utf-8"))
    df = load_scores_with_meta(args.scores, args.metainfor)
    df["family"] = df["attack"].map(lambda a: attack_family(a, cfg["rules"]))

    results = search_shifts(df, n_grid=args.n_grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.write_scores:
        out_scores = args.out_dir / "eval_scores_dev_full_family_adjusted.txt"
        write_adjusted_scores(
            df, results["shift_tts"], results["shift_vc"],
            args.metainfor, out_scores,
        )
        results["adjusted_scores_file"] = str(out_scores)

    out_json = args.out_dir / "per_family_threshold.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
