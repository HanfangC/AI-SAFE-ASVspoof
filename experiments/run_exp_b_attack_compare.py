#!/usr/bin/env python3
"""Experiment B: EER by attack family (TTS / VC / Adversarial)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AASIST = ROOT / "Baseline-AASIST"
sys.path.insert(0, str(AASIST))
from eval.calculate_modules import compute_eer  # noqa: E402


def attack_family(attack: str, rules: list) -> str:
    attack = str(attack).strip()
    for rule in rules:
        if attack.startswith(rule["prefix"]):
            return rule["family"]
    return "unknown"


def load_scores_with_meta(scores_path: Path, metainfor_path: Path) -> pd.DataFrame:
    rows = []
    with scores_path.open() as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 4:
                continue
            rows.append({"utt": p[1], "score": float(p[2]), "label": p[3].lower()})
    scores = pd.DataFrame(rows)

    meta_rows = []
    with metainfor_path.open() as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6:
                continue
            meta_rows.append({"utt": p[1], "attack": p[4]})
    meta = pd.DataFrame(meta_rows)
    return scores.merge(meta, on="utt", how="left")


def eer_for_subset(df: pd.DataFrame, spoof_mask: pd.Series) -> float:
    bonafide = df.loc[df["label"] == "bonafide", "score"].to_numpy()
    spoof = df.loc[spoof_mask & (df["label"] != "bonafide"), "score"].to_numpy()
    if len(bonafide) == 0 or len(spoof) == 0:
        return float("nan")
    eer, _, _, _ = compute_eer(bonafide, spoof)
    return float(eer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--metainfor", type=Path, required=True)
    parser.add_argument("--groups", type=Path,
                        default=Path(__file__).parent / "attack_groups.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.groups.read_text(encoding="utf-8"))
    rules = cfg["rules"]

    df = load_scores_with_meta(args.scores, args.metainfor)
    df["family"] = df["attack"].map(lambda a: attack_family(a, rules))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # pooled spoof EER (all attacks)
    pooled_eer = eer_for_subset(df, df["label"] != "bonafide")

    family_eers = {}
    for fam in ["tts", "vc", "adv", "unknown"]:
        mask = df["family"] == fam
        if mask.any():
            family_eers[fam] = eer_for_subset(df, mask)

    results = {"pooled_eer": pooled_eer, "family_eer": family_eers}
    out_json = args.out_dir / "exp_b_attack_eer.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    labels = list(family_eers.keys())
    values = [family_eers[k] * 100 for k in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"][:len(labels)])
    plt.ylabel("EER (%)")
    plt.title("Detection EER by attack family (dev)")
    plt.tight_layout()
    fig_path = args.out_dir / "exp_b_attack_eer_bar.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(json.dumps(results, indent=2))
    print(f"Wrote {out_json} and {fig_path}")


if __name__ == "__main__":
    main()
