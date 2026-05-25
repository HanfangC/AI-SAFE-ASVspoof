#!/usr/bin/env python3
"""Apply family shifts then optional per-attack shifts on a score file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from per_family_threshold import apply_shifts, search_shifts, write_adjusted_scores  # noqa: E402
from per_attack_calibration import apply_attack_shifts, coordinate_search, write_scores  # noqa: E402
from run_exp_b_attack_compare import attack_family, load_scores_with_meta  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--metainfor", type=Path, required=True)
    parser.add_argument("--groups", type=Path, default=ROOT / "experiments" / "attack_groups.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="stacked")
    parser.add_argument("--skip-per-attack", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.groups.read_text(encoding="utf-8"))
    df = load_scores_with_meta(args.scores, args.metainfor)
    df["family"] = df["attack"].map(lambda a: attack_family(a, cfg["rules"]))

    fam_res = search_shifts(df)
    df_fam = apply_shifts(df, fam_res["shift_tts"], fam_res["shift_vc"])
    df_fam["score"] = df_fam["score_adj"]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fam_path = out_dir / f"eval_scores_{args.tag}_family.txt"
    write_adjusted_scores(df, fam_res["shift_tts"], fam_res["shift_vc"],
                          args.metainfor, fam_path)

    results = {"family": fam_res, "family_scores": str(fam_path)}

    if not args.skip_per_attack:
        attacks = sorted(df_fam.loc[df_fam["label"] != "bonafide", "attack"].dropna().unique())
        atk_res = coordinate_search(df_fam, attacks)
        atk_path = out_dir / f"eval_scores_{args.tag}_family_attack.txt"
        write_scores(df_fam, atk_res["shifts"], args.metainfor, atk_path)
        results["per_attack"] = atk_res
        results["final_scores"] = str(atk_path)
    else:
        results["final_scores"] = str(fam_path)

    (out_dir / f"{args.tag}_stack.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
