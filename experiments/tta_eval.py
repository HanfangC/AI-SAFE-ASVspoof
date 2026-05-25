#!/usr/bin/env python3
"""TTA: multiple random crops per utterance, average bonafide-class scores."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
AASIST = ROOT / "Baseline-AASIST"
sys.path.insert(0, str(AASIST))

from data_utils import genSpoof_list, pad_random  # noqa: E402
from main import get_model  # noqa: E402


class TTADataset(Dataset):
    def __init__(self, list_IDs, base_dir: Path, n_aug: int, seed: int):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.n_aug = n_aug
        self.cut = 64600
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"))
        crops = []
        for _ in range(self.n_aug):
            crops.append(pad_random(X, self.cut))
        return torch.stack([torch.tensor(c, dtype=torch.float32) for c in crops]), key


def collate_tta(batch):
    xs, keys = zip(*batch)
    return torch.stack(xs), keys


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=AASIST / "config" / "AASIST_ASVspoof5_eval_fast.conf")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get(
                            "ASVSPOOF5_DATA_DIR",
                            "/path/to/ASVspoof5",
                        )))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-utts", type=int, default=5000,
                        help="0 = full dev set")
    parser.add_argument("--full-dev", action="store_true",
                        help="Evaluate all dev utterances (same as --max-utts 0)")
    parser.add_argument("--n-aug", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-metrics", action="store_true",
                        help="Run run_exp_a_eval on output scores")
    args = parser.parse_args()

    if args.full_dev:
        args.max_utts = 0

    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config["model_config"], device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dev_meta = args.data_dir / "ASVspoof5.dev.metainfor.txt"
    dev_flac = args.data_dir / "flac_D"
    _, file_dev = genSpoof_list(dev_meta, is_train=False, is_eval=False)
    if args.max_utts > 0:
        file_dev = file_dev[: args.max_utts]

    ds = TTADataset(file_dev, dev_flac, args.n_aug, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_tta, pin_memory=True)

    meta = {}
    with dev_meta.open() as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 6:
                meta[p[1]] = (p[0], p[5])

    scores = {}
    for batch_x, keys in tqdm(loader, desc="tta"):
        b, n_aug, _ = batch_x.shape
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(b * n_aug, -1)
        _, out = model(batch_x)
        sc = out[:, 1].view(b, n_aug).mean(dim=1).cpu().numpy().ravel()
        for k, s in zip(keys, sc):
            scores[k] = float(s)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "dev_scores_tta.txt"
    with out_path.open("w") as fh:
        for utt, (spk, key) in meta.items():
            if utt in scores:
                fh.write(f"{spk} {utt} {scores[utt]} {key}\n")
    meta_path = {"n_aug": args.n_aug, "n_utts": len(scores), "max_utts": args.max_utts}
    (args.out_dir / "tta_meta.json").write_text(
        json.dumps(meta_path, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path} ({len(scores)} utts, n_aug={args.n_aug})")

    if args.run_metrics:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "run_exp_a_eval.py"),
                "--scores", str(out_path),
                "--out-dir", str(args.out_dir),
                "--tag", f"tta_n{args.n_aug}",
                "--no-swanlab",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
