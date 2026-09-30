"""README charts rendered from results/*.json (no model or data needed)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Light-mode design tokens (fixed categorical order: slot 1 blue, 2 orange, 3 aqua).
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FONT = ["DejaVu Sans"]  # ships with matplotlib, so charts look the same everywhere


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w=6.4, h=3.9):
    plt.rcParams["font.family"] = FONT
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _title(fig, title, subtitle):
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, subtitle, ha="left", va="top", fontsize=9, color=INK2)


def pr_curves(results: Path, out: Path) -> Path:
    curves = json.loads((results / "pr_curves.json").read_text())
    report = json.loads((results / "train_report.json").read_text())
    ap = {k: v["pr_auc"] for k, v in report["comparison_test"].items()}
    fig, ax = _fig()
    spec = [("gradient_boosting", "Gradient boosting", BLUE, (0.60, 0.985)),
            ("logistic_regression", "Logistic regression", ORANGE, (0.40, 0.90)),
            ("rules", "Hand-written rules", AQUA, (0.10, 0.64))]
    for key, label, colour, _ in spec:
        c = curves[key]
        ax.plot(c["recall"], c["precision"], color=colour, linewidth=2, label=f"{label} (PR-AUC {ap[key]:.3f})")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall (share of fraud caught)", color=INK2, fontsize=9)
    ax.set_ylabel("Precision (share of alerts that are fraud)", color=INK2, fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig, "Precision-recall on the held-out final window", "Higher and further right is better. Synthetic data; 353 fraud rows.")
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.11, right=0.97)
    path = out / "pr_curves.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _bars(labels, values, title, subtitle, xlabel, path, fmt, xmax=None):
    fig, ax = _fig(6.4, 0.55 * len(labels) + 1.6)
    ypos = list(range(len(labels)))[::-1]
    ax.barh(ypos, values, color=BLUE, height=0.55)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, xmax or max(values) * 1.15)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    for y, v in zip(ypos, values):
        ax.text(v + (xmax or max(values)) * 0.015, y, fmt(v), va="center", fontsize=9, color=INK)
    _title(fig, title, subtitle)
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.32, right=0.96)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def make_plots(results: str = "results", out: str = "docs") -> list[Path]:
    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((results / "train_report.json").read_text())
    paths = [pr_curves(results, out)]

    typ = report["recall_by_typology_at_review"]
    names = {"card_testing": "Card testing", "account_takeover": "Account takeover",
             "atm_cashout": "ATM cash-out", "low_and_slow": "Low-and-slow"}
    order = sorted(typ, key=lambda k: -typ[k]["recall"])
    paths.append(_bars([f"{names[k]} (n={typ[k]['n']})" for k in order], [typ[k]["recall"] for k in order],
                       "Recall by fraud type at the review threshold",
                       "Low-and-slow looks like normal behaviour, so it is the hardest.",
                       "Recall", out / "recall_by_typology.png", lambda v: f"{v:.0%}", xmax=1.12))

    gi = report["group_importance_mean_logodds_on_test_fraud"]
    keys = list(gi)
    paths.append(_bars([k.capitalize() for k in keys], [gi[k] for k in keys],
                       "What pushes fraud scores up (feature-group occlusion)",
                       "Mean log-odds drop when a group is replaced by typical values, on test fraud.",
                       "Mean log-odds contribution", out / "factor_importance.png", lambda v: f"{v:.1f}"))
    return paths
