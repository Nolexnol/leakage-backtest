"""Draw the README figures from results/results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots (light mode) + neutral inks.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_validation(res: dict, path: Path) -> None:
    rows = res["validation"]
    schemes = list(dict.fromkeys(r["scheme"] for r in rows))
    models = list(dict.fromkeys(r["model"] for r in rows))
    colors = {"ridge": BLUE, "gbm": ORANGE}
    names = {"ridge": "Ridge regression", "gbm": "Gradient-boosted trees"}
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    h = 0.36
    y = np.arange(len(schemes))[::-1]
    for j, m in enumerate(models):
        vals = [next(r for r in rows if r["model"] == m and r["scheme"] == s) for s in schemes]
        ics = [v["mean_ic"] for v in vals]
        pos = y + (h / 2 if j == 0 else -h / 2) * (1 if len(models) > 1 else 0)
        ax.barh(pos, ics, height=h - 0.04, color=colors.get(m, BLUE), label=names.get(m, m), edgecolor=SURFACE, linewidth=1)
        for p, v in zip(pos, vals):
            x = v["mean_ic"]
            ax.text(x + (0.0006 if x >= 0 else -0.0006), p, f"t = {v['ic_tstat_nw']:+.1f}", va="center",
                    ha="left" if x >= 0 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(schemes, color=INK, fontsize=9)
    ax.set_xlabel("Mean daily rank IC on the same test dates (5-day forward return)", color=INK2, fontsize=9)
    lim = max(abs(r["mean_ic"]) for r in rows) * 1.45
    ax.set_xlim(-lim, lim)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    ax.set_title("Same model, same data: the validation scheme decides whether it \"works\"", loc="left",
                 fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_multiple_testing(res: dict, path: Path) -> None:
    mt = res["multiple_testing"]
    trials = mt["trials"]
    best = mt["best"]
    xs = np.array([t["is_sharpe"] for t in trials])
    ys = np.array([t["oos_sharpe"] for t in trials])
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.scatter(xs, ys, s=34, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=1.2, label=f"{mt['n_trials']} candidate strategies")
    ax.scatter([best["is_sharpe"]], [best["oos_sharpe"]], s=90, color=ORANGE, edgecolor=SURFACE, linewidth=2,
               label="Best in-sample (the one you'd publish)", zorder=3)
    ax.annotate(f"Best in-sample: {best['feature']}\n"
                f"Sharpe {best['is_sharpe']:.2f} in-sample → {best['oos_sharpe']:.2f} out-of-sample\n"
                f"P(real) {mt['best_psr_is']:.2f} naive → {mt['best_dsr_is']:.2f} after deflating for {mt['n_trials']} tries",
                (best["is_sharpe"], best["oos_sharpe"]), xytext=(0.98, 0.04), textcoords="axes fraction", ha="right", va="bottom",
                fontsize=8, color=INK, bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID),
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    lo, hi = min(xs.min(), ys.min()) - 0.2, max(xs.max(), ys.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1, linestyle="--", zorder=0)
    ax.set_ylim(lo - 0.9, hi)  # room under the cloud for the annotation
    ax.set_xlabel(f"In-sample Sharpe (through {mt['split_date']}, before costs)", color=INK2, fontsize=9)
    ax.set_ylabel("Out-of-sample Sharpe (before costs)", color=INK2, fontsize=9)
    ax.set_title("Picking the best of 48 backtests: the in-sample winner doesn't hold up", loc="left", fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def make_all(out_dir: Path) -> None:
    res = json.loads((out_dir / "results.json").read_text())
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig_validation(res, fig_dir / "validation_ic.png")
    fig_multiple_testing(res, fig_dir / "multiple_testing.png")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    make_all(Path(__file__).resolve().parents[1] / "results")
