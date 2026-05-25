#!/usr/bin/env python3
"""Experiment C: robustness — MP3 / noise / resample on dev, then AASIST inference."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
AASIST = ROOT / "Baseline-AASIST"
sys.path.insert(0, str(AASIST))

from data_utils import TestDataset, genSpoof_list, pad_random  # noqa: E402
from main import get_model  # noqa: E402


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_pow = np.mean(x ** 2) + 1e-12
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = rng.standard_normal(x.shape).astype(np.float32)
    noise = noise * np.sqrt(noise_pow / (np.mean(noise ** 2) + 1e-12))
    return np.clip(x + noise, -1.0, 1.0)


def resample_chain(x: np.ndarray, sr: int, mid_sr: int) -> np.ndarray:
    import torchaudio

    t = torch.from_numpy(x).float().unsqueeze(0)
    t_mid = torchaudio.functional.resample(t, sr, mid_sr)
    t_back = torchaudio.functional.resample(t_mid, mid_sr, sr)
    return t_back.squeeze(0).numpy()


def mp3_degrade(x: np.ndarray, sr: int, bitrate: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav_in = td / "in.wav"
        mp3 = td / "out.mp3"
        wav_out = td / "out.wav"
        sf.write(str(wav_in), x, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_in),
             "-b:a", bitrate, str(mp3)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
             "-ar", str(sr), "-ac", "1", str(wav_out)],
            check=True,
        )
        y, _ = sf.read(str(wav_out), dtype="float32")
    return y


class RobustDataset(TestDataset):
    """TestDataset with on-the-fly degradation."""

    def __init__(self, list_IDs, base_dir, condition: str, seed: int = 42):
        super().__init__(list_IDs, base_dir)
        self.condition = condition
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, sr = sf.read(str(self.base_dir / f"{key}.flac"))
        if self.condition == "noise_20":
            X = add_noise(X, 20.0, self.rng)
        elif self.condition == "noise_10":
            X = add_noise(X, 10.0, self.rng)
        elif self.condition == "noise_5":
            X = add_noise(X, 5.0, self.rng)
        elif self.condition == "resample_8k":
            X = resample_chain(X, sr, 8000)
        elif self.condition == "mp3_128":
            X = mp3_degrade(X, sr, "128k")
        elif self.condition == "mp3_64":
            X = mp3_degrade(X, sr, "64k")
        elif self.condition != "clean":
            raise ValueError(f"unknown condition: {self.condition}")

        X_pad = pad_random(X, self.cut)
        return torch.tensor(X_pad, dtype=torch.float32), key


@torch.no_grad()
def run_eval(model, loader, device, trial_path: Path, save_path: Path) -> None:
    model.eval()
    meta = {}
    with trial_path.open() as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 6:
                meta[p[1]] = (p[0], p[5])

    scores = {}
    for batch_x, utt_id in tqdm(loader, desc=save_path.name):
        batch_x = batch_x.to(device)
        _, out = model(batch_x)
        sc = out[:, 1].cpu().numpy().ravel()
        for u, s in zip(utt_id, sc):
            scores[u] = float(s)

    with save_path.open("w") as fh:
        for utt, (spk, key) in meta.items():
            if utt in scores:
                fh.write(f"{spk} {utt} {scores[utt]} {key}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=AASIST / "config" / "AASIST_ASVspoof5.conf")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get(
                            "ASVSPOOF5_DATA_DIR",
                            "/path/to/ASVspoof5",
                        )))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-utts", type=int, default=5000,
                        help="Max dev utterances (0 = all)")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--conditions", nargs="*", default=None,
        help="Subset of conditions (default: all). Example: mp3_128 noise_20",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip conditions whose dev_scores_*.txt already exist",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config["model_config"], device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    dev_meta = args.data_dir / "ASVspoof5.dev.metainfor.txt"
    dev_flac = args.data_dir / "flac_D"
    _, file_dev = genSpoof_list(dev_meta, is_train=False, is_eval=False)
    if args.max_utts > 0:
        file_dev = file_dev[: args.max_utts]

    all_conditions = [
        "clean", "mp3_128", "mp3_64",
        "noise_20", "noise_10", "noise_5", "resample_8k",
    ]
    conditions = args.conditions if args.conditions else all_conditions
    if args.skip_existing:
        conditions = [
            c for c in conditions
            if not (args.out_dir / f"dev_scores_{c}.txt").is_file()
        ]
        if not conditions:
            print("All score files exist; nothing to run.")
            return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for cond in conditions:
        print(f"=== condition: {cond} ===")
        ds = RobustDataset(file_dev, dev_flac, cond)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
        score_path = args.out_dir / f"dev_scores_{cond}.txt"
        run_eval(model, loader, device, dev_meta, score_path)
        summary[cond] = str(score_path)

    meta_path = args.out_dir / "score_files.json"
    if meta_path.is_file():
        try:
            summary = {**json.loads(meta_path.read_text(encoding="utf-8")), **summary}
        except json.JSONDecodeError:
            pass
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done. Run run_exp_a_eval.py on each score file.")


if __name__ == "__main__":
    main()
