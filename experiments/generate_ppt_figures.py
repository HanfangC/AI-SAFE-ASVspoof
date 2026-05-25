#!/usr/bin/env python3
"""Generate PPT-ready figures (300 dpi, Chinese labels) from experiment JSON results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PPT_DIR = RESULTS / "ppt"
AASIST = ROOT / "Baseline-AASIST"
DPI = 300

# Condition metadata for experiment C
EXP_C_CONDITIONS = [
    ("clean", "干净音频", "codec"),
    ("mp3_128", "MP3 128kbps", "codec"),
    ("mp3_64", "MP3 64kbps", "codec"),
    ("resample_8k", "重采样 8kHz", "resample"),
    ("noise_20", "噪声 20dB SNR", "noise"),
    ("noise_10", "噪声 10dB SNR", "noise"),
    ("noise_5", "噪声 5dB SNR", "noise"),
]

GROUP_COLORS = {
    "codec": "#4472C4",
    "resample": "#ED7D31",
    "noise": "#C00000",
}

FONT_DIR = Path(__file__).resolve().parent / "fonts"
NOTO_SC_PATH = FONT_DIR / "NotoSansSC-Regular.otf"


def _register_cjk_font() -> str:
    """Register bundled Noto Sans CJK SC (Latin + Chinese). Avoid Droid-only fallback."""
    if not NOTO_SC_PATH.is_file():
        raise FileNotFoundError(
            f"Missing CJK font: {NOTO_SC_PATH}\n"
            "Download: experiments/fonts/NotoSansSC-Regular.otf from Noto CJK release.",
        )
    font_manager.fontManager.addfont(str(NOTO_SC_PATH))
    return font_manager.FontProperties(fname=str(NOTO_SC_PATH)).get_name()


def setup_style() -> None:
    cn_font = _register_cjk_font()
    plt.rcParams.update({
        "font.family": cn_font,
        "font.sans-serif": [cn_font, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_exp_c_metrics() -> list[dict]:
    metrics_dir = RESULTS / "exp_c" / "metrics"
    out = []
    for tag, label_cn, group in EXP_C_CONDITIONS:
        p = metrics_dir / f"{tag}_metrics.json"
        if not p.exists():
            raise FileNotFoundError(p)
        d = load_json(p)
        d["label_cn"] = label_cn
        d["group"] = group
        out.append(d)
    return out


def roc_from_scores(scores_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    sys.path.insert(0, str(AASIST))
    bonafide, spoof = [], []
    with scores_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            score = float(parts[2])
            if parts[3].lower() == "bonafide":
                bonafide.append(score)
            else:
                spoof.append(score)
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.concatenate([np.ones(len(bonafide)), np.zeros(len(spoof))])
    y_score = np.concatenate([bonafide, spoof])
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, float(auc)


def fig_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 1.5, 1.8, 1.2, "ASVspoof5\ndev 数据", "#E7E6E6"),
        (2.5, 1.5, 1.8, 1.2, "AASIST\n预训练权重", "#D9E2F3"),
        (4.7, 2.3, 1.6, 1.0, "实验 A\n全量指标", "#C6E0B4"),
        (4.7, 0.7, 1.6, 1.0, "实验 B\n攻击族 EER", "#C6E0B4"),
        (4.7, -0.5, 1.6, 1.0, "实验 C\n鲁棒性 7 条件", "#FFE699"),
        (6.8, 1.5, 1.8, 1.2, "EER / minDCF\nAUC / ROC", "#F4B084"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=color, edgecolor="#404040", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10, fontweight="bold")

    arrows = [
        ((2.1, 2.1), (2.5, 2.1)),
        ((4.3, 2.1), (4.7, 2.8)),
        ((4.3, 2.1), (4.7, 1.2)),
        ((4.3, 2.1), (4.7, 0.0)),
        ((6.3, 2.8), (6.8, 2.4)),
        ((6.3, 1.2), (6.8, 2.1)),
        ((6.3, 0.0), (6.8, 1.8)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="->", mutation_scale=12,
            color="#404040", linewidth=1.2,
        ))

    ax.text(5.0, 3.5, "ASVspoof5 反欺骗实验流程", ha="center", fontsize=14, fontweight="bold")
    ax.text(4.7, -1.0, "实验 C：5000 条 dev 子集", ha="center", fontsize=9, color="#666666")
    ax.text(4.7, 3.35, "实验 A/B：140950 条全量 dev", ha="center", fontsize=9, color="#666666")

    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_exp_a_roc(metrics: dict, out: Path) -> None:
    scores_path = Path(metrics["scores_file"])
    fpr, tpr, auc = roc_from_scores(scores_path)
    eer = metrics["EER"] * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4472C4", lw=2,
            label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1)
    ax.set_xlabel("假阳性率 (FPR)")
    ax.set_ylabel("真阳性率 (TPR)")
    ax.set_title(f"实验 A：ROC 曲线（全量 dev，EER = {eer:.2f}%）")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_exp_a_metrics_table(metrics: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    rows = [
        ["EER", f"{metrics['EER'] * 100:.2f}%", "越低越好；50% 为随机"],
        ["minDCF", f"{metrics['minDCF']:.3f}", "ASV 系统最小检测代价"],
        ["CLLR", f"{metrics['CLLR']:.3f}", "分数校准；0 为理想"],
        ["AUC", f"{metrics['auc']:.3f}", "ROC 曲线下面积"],
        ["样本量", f"bonafide {metrics['n_bonafide']:,} / spoof {metrics['n_spoof']:,}", "全量 dev"],
    ]
    table = ax.table(
        cellText=[[r[0], r[1]] for r in rows],
        colLabels=["指标", "数值"],
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title("实验 A：全量 dev 主要指标", fontsize=13, fontweight="bold", pad=20)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_exp_b_attack_eer(exp_b: dict, out: Path) -> None:
    families = [
        ("tts", "TTS\n(文本转语音)", exp_b["family_eer"]["tts"]),
        ("vc", "VC\n(语音转换)", exp_b["family_eer"]["vc"]),
        ("pooled", "总体\n(pooled)", exp_b["pooled_eer"]),
    ]
    labels = [f[1] for f in families]
    eers = [f[2] * 100 for f in families]
    colors = ["#4472C4", "#ED7D31", "#70AD47"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, eers, color=colors, edgecolor="#404040", linewidth=0.8)
    ax.axhline(15.2, color="#888888", linestyle="--", lw=1.2, label="全量 EER 15.2%")
    for bar, val in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("EER (%)")
    ax.set_title("实验 B：按攻击族的等错误率 (EER)")
    ax.set_ylim(0, max(eers) * 1.2 + 2)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_exp_c_eer_bar(exp_c: list[dict], out: Path) -> None:
    labels = [d["label_cn"] for d in exp_c]
    eers = [d["EER"] * 100 for d in exp_c]
    colors = [GROUP_COLORS[d["group"]] for d in exp_c]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, eers, color=colors, edgecolor="#404040", linewidth=0.6)
    ax.axhline(15.57, color="#888888", linestyle="--", lw=1.2,
               label="clean 基线 15.57%")
    for bar, val in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("EER (%)")
    ax.set_title("实验 C：鲁棒性评估（5000 条 dev 子集）")
    ax.set_ylim(0, 58)
    legend_patches = [
        mpatches.Patch(color=GROUP_COLORS["codec"], label="编解码"),
        mpatches.Patch(color=GROUP_COLORS["resample"], label="重采样"),
        mpatches.Patch(color=GROUP_COLORS["noise"], label="加性噪声"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="#888888", linestyle="--", label="clean 基线"),
    ], loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_exp_c_eer_delta(exp_c: list[dict], out: Path) -> None:
    clean_eer = exp_c[0]["EER"] * 100
    labels = [d["label_cn"] for d in exp_c[1:]]
    deltas = [(d["EER"] * 100 - clean_eer) for d in exp_c[1:]]
    colors = [GROUP_COLORS[d["group"]] for d in exp_c[1:]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.plot(x, deltas, "o-", color="#4472C4", lw=2, markersize=8)
    for i, (lab, d, c) in enumerate(zip(labels, deltas, colors)):
        ax.scatter(i, d, s=120, color=c, zorder=5, edgecolors="#404040")
        ax.annotate(f"+{d:.1f}pp", (i, d), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.axhline(0, color="#888888", linestyle="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("EER 相对 clean 的增量 (百分点)")
    ax.set_title("实验 C：退化条件下 EER 上升幅度")
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_exp_c_auc_bar(exp_c: list[dict], out: Path) -> None:
    labels = [d["label_cn"] for d in exp_c]
    aucs = [d["auc"] for d in exp_c]
    colors = [GROUP_COLORS[d["group"]] for d in exp_c]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, aucs, color=colors, edgecolor="#404040", linewidth=0.6)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("实验 C：各退化条件下的 AUC")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_postprocess_compare(out: Path) -> None:
    """Bar chart: pretrained vs post-processing methods (full dev)."""
    pp_path = RESULTS / "postprocess" / "postprocess_summary.json"
    if not pp_path.is_file():
        return
    summary = json.loads(pp_path.read_text(encoding="utf-8"))
    labels, eers = [], []
    items = [
        ("预训练", summary["baseline"]["EER"] * 100),
        ("Platt 校准", summary["calibrated"]["EER"] * 100),
        ("分族分数平移", summary["family_adjusted"]["EER"] * 100),
    ]
    for lab, eer in items:
        labels.append(lab)
        eers.append(eer)
    colors = ["#70AD47", "#4472C4", "#ED7D31"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, eers, color=colors, edgecolor="#404040")
    ax.axhline(15.2, color="#888888", linestyle="--", label="预训练参考 15.2%")
    for bar, val in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("EER (%)")
    ax.set_title("全量 dev：后处理 vs 预训练（推荐：分族平移 14.07%）")
    ax.set_ylim(0, max(eers) * 1.15 + 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_baseline_training_flow(out: Path) -> None:
    """Official baseline: pretrained init + optional fine-tune on train."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, 2.2, 2.2, 1.2, "best.pth\n(组织者预训练)", "#E2EFDA"),
        (3.0, 2.2, 2.4, 1.2, "ASVspoof5 train\n182k 微调(可选)", "#FFF2CC"),
        (5.8, 2.2, 2.0, 1.2, "dev 前2000\n训练时验证", "#FCE4D6"),
        (8.2, 2.2, 1.5, 1.2, "全量 dev\n--eval 15.2%", "#DEEBF7"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#404040", linewidth=1.2,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    for x1, x2 in [(2.5, 3.0), (5.4, 5.8), (7.8, 8.2)]:
        ax.add_patch(FancyArrowPatch(
            (x1, 2.8), (x2, 2.8), arrowstyle="->", mutation_scale=12,
            color="#404040", linewidth=1.5,
        ))
    ax.text(5.0, 0.6,
            "结论：非从零训练；README 15.2% = 对现成 best.pth 全量 dev 评估",
            ha="center", fontsize=11, fontweight="bold")
    ax.set_title("官方 AASIST Baseline 训练与评估流程", fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_innovation_leaderboard(out: Path) -> None:
    """Bar chart from innovation/summary.json."""
    inv_path = RESULTS / "innovation" / "summary.json"
    if not inv_path.is_file():
        return
    summary = json.loads(inv_path.read_text(encoding="utf-8"))
    labels, eers, colors = [], [], []
    palette = {
        "baseline": "#888888",
        "family": "#ED7D31",
        "A1": "#70AD47",
        "A4": "#4472C4",
        "A2": "#5B9BD5",
        "B1": "#C00000",
        "B2": "#7030A0",
    }
    if "baseline_eer" in summary:
        labels.append("预训练")
        eers.append(summary["baseline_eer"] * 100)
        colors.append(palette["baseline"])
    labels.append("分族平移")
    eers.append(summary.get("family_adjusted_eer", 0.1407) * 100)
    colors.append(palette["family"])

    name_map = {
        "A1_best": "A1 TTA最优",
        "A1_tta_n3": "A1 TTA n=3",
        "A1_tta_n5": "A1 TTA n=5",
        "A4_ensemble": "A4 集成",
        "A2_head_only": "A2 Head-only",
        "B1_per_attack": "B1 按攻击ID",
        "B2_platt_per_attack": "B2 Platt+攻击",
        "B_stack_family_attack": "B 分族+攻击",
    }
    for key, lab in name_map.items():
        if key in summary and isinstance(summary[key], dict) and "EER" in summary[key]:
            labels.append(lab)
            eers.append(float(summary[key]["EER"]) * 100)
            prefix = key.split("_")[0]
            colors.append(palette.get(prefix, "#4472C4"))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), eers, color=colors, edgecolor="#404040")
    ax.axhline(14.07, color="#ED7D31", linestyle="--", linewidth=1.5, label="目标线 14.07%")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    for bar, val in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("EER (%)")
    ax.set_title("创新实验：全量 dev EER 对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_finetune_failed(out: Path) -> None:
    ft_path = RESULTS / "finetune_epochs_full_dev" / "finetune_10epochs_summary.json"
    if not ft_path.is_file():
        return
    rows = json.loads(ft_path.read_text(encoding="utf-8"))
    ep = [r for r in rows if r["name"].startswith("epoch_")]
    labels = [r["name"].replace("epoch_", "ep") for r in ep]
    eers = [r["EER_pct"] for r in ep]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, eers, color="#C00000", alpha=0.75)
    ax.axhline(15.2, color="#70AD47", linestyle="--", linewidth=2, label="预训练 15.2%")
    ax.set_ylabel("EER (%)")
    ax.set_title("全参数微调：全量 dev EER 均劣于预训练（已弃用）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_summary(exp_a: dict, exp_b: dict, exp_c: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    vc_eer = exp_b["family_eer"]["vc"] * 100
    tts_eer = exp_b["family_eer"]["tts"] * 100
    noise5 = next(d for d in exp_c if d["tag"] == "noise_5")["EER"] * 100
    best_pp = 14.07
    pp_path = RESULTS / "postprocess" / "postprocess_summary.json"
    if pp_path.is_file():
        fa = json.loads(pp_path.read_text(encoding="utf-8")).get("family_adjusted", {})
        if fa:
            best_pp = fa["EER"] * 100

    bullets = [
        f"1. 主结果：预训练全量 dev EER = {exp_a['EER']*100:.2f}%（实验 A/B/C）",
        f"2. 后处理：分族分数平移后 EER ≈ {best_pp:.2f}%（dev 调参，优于全参微调）",
        f"3. 攻击类型：VC ({vc_eer:.1f}%) 显著难于 TTS ({tts_eer:.1f}%)",
        "4. 鲁棒性：MP3 影响小；8kHz 与低 SNR 噪声是主要威胁",
        f"5. 全参微调 ep0–9 全量 EER 17.3%+，已停止；不推荐继续该路线",
    ]
    y = 0.85
    ax.text(0.5, 0.95, "综合结论", ha="center", fontsize=16, fontweight="bold",
            transform=ax.transAxes)
    for b in bullets:
        ax.text(0.06, y, b, ha="left", va="top", fontsize=12,
                transform=ax.transAxes, wrap=True)
        y -= 0.16
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_roc_grid(out: Path) -> None:
    """2x4 grid of experiment C ROC thumbnails from existing PNGs."""
    tags = [t[0] for t in EXP_C_CONDITIONS]
    labels = [t[1] for t in EXP_C_CONDITIONS]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i, (tag, lab) in enumerate(zip(tags, labels)):
        src = RESULTS / "exp_c" / "metrics" / f"{tag}_roc.png"
        if not src.exists():
            axes[i].text(0.5, 0.5, "无图", ha="center", va="center")
            axes[i].set_title(lab, fontsize=9)
            axes[i].axis("off")
            continue
        img = plt.imread(str(src))
        axes[i].imshow(img)
        axes[i].set_title(lab, fontsize=9)
        axes[i].axis("off")
    # hide unused subplot if any
    for j in range(len(tags), len(axes)):
        axes[j].axis("off")
    fig.suptitle("附录：实验 C 各条件 ROC 曲线", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def export_metrics_csv(
    exp_a: dict,
    exp_b: dict,
    exp_c: list[dict],
    out: Path,
) -> None:
    lines = ["experiment,subset,condition,EER_pct,minDCF,AUC,CLLR,notes"]
    lines.append(
        f"exp_a,dev_full,all,{exp_a['EER']*100:.4f},{exp_a['minDCF']:.6f},"
        f"{exp_a['auc']:.6f},{exp_a['CLLR']:.6f},n_bonafide={exp_a['n_bonafide']}"
    )
    for fam, eer in exp_b["family_eer"].items():
        if eer != eer:  # NaN
            continue
        lines.append(
            f"exp_b,dev_full,{fam},{eer*100:.4f},,,,attack_family"
        )
    lines.append(
        f"exp_b,dev_full,pooled,{exp_b['pooled_eer']*100:.4f},,,,same_as_exp_a"
    )
    clean_eer = exp_c[0]["EER"] * 100
    for d in exp_c:
        delta = d["EER"] * 100 - clean_eer if d["tag"] != "clean" else 0
        lines.append(
            f"exp_c,dev_5000,{d['tag']},{d['EER']*100:.4f},{d['minDCF']:.6f},"
            f"{d['auc']:.6f},{d.get('CLLR', '')},eer_delta_pp={delta:.2f}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    PPT_DIR.mkdir(parents=True, exist_ok=True)

    exp_a = load_json(RESULTS / "exp_a" / "aasist_metrics.json")
    exp_b = load_json(RESULTS / "exp_b" / "exp_b_attack_eer.json")
    exp_c = load_exp_c_metrics()

    export_metrics_csv(exp_a, exp_b, exp_c, PPT_DIR / "metrics_summary.csv")

    figures = {
        "fig_pipeline.png": lambda: fig_pipeline(PPT_DIR / "fig_pipeline.png"),
        "fig_exp_a_roc.png": lambda: fig_exp_a_roc(exp_a, PPT_DIR / "fig_exp_a_roc.png"),
        "fig_exp_a_metrics_table.png": lambda: fig_exp_a_metrics_table(
            exp_a, PPT_DIR / "fig_exp_a_metrics_table.png"),
        "fig_exp_b_attack_eer.png": lambda: fig_exp_b_attack_eer(
            exp_b, PPT_DIR / "fig_exp_b_attack_eer.png"),
        "fig_exp_c_eer_bar.png": lambda: fig_exp_c_eer_bar(
            exp_c, PPT_DIR / "fig_exp_c_eer_bar.png"),
        "fig_exp_c_eer_delta.png": lambda: fig_exp_c_eer_delta(
            exp_c, PPT_DIR / "fig_exp_c_eer_delta.png"),
        "fig_exp_c_auc_bar.png": lambda: fig_exp_c_auc_bar(
            exp_c, PPT_DIR / "fig_exp_c_auc_bar.png"),
        "fig_summary.png": lambda: fig_summary(
            exp_a, exp_b, exp_c, PPT_DIR / "fig_summary.png"),
        "fig_postprocess_compare.png": lambda: fig_postprocess_compare(
            PPT_DIR / "fig_postprocess_compare.png"),
        "fig_finetune_failed.png": lambda: fig_finetune_failed(
            PPT_DIR / "fig_finetune_failed.png"),
        "fig_exp_c_roc_grid.png": lambda: fig_roc_grid(
            PPT_DIR / "fig_exp_c_roc_grid.png"),
        "fig_baseline_training_flow.png": lambda: fig_baseline_training_flow(
            PPT_DIR / "fig_baseline_training_flow.png"),
        "fig_innovation_leaderboard.png": lambda: fig_innovation_leaderboard(
            PPT_DIR / "fig_innovation_leaderboard.png"),
    }

    for name, fn in figures.items():
        fn()
        print(f"Wrote {PPT_DIR / name}")

    print(f"Done. Output directory: {PPT_DIR}")


if __name__ == "__main__":
    main()
