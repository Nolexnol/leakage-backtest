"""Run every experiment and write results/results.json, results/summary.md and figures.

    python experiments/run_all.py            # full run (a few minutes)
    python experiments/run_all.py --quick    # ridge only, fewer bootstrap draws
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from leakbt import portfolio, stats
from leakbt.audit import format_audit, truncation_audit
from leakbt.data import load_panel
from leakbt.features import (
    BUGGY_FEATURES, FEATURE_NAMES, FEATURES, build_features, cross_sectional_rank, demean, forward_return, to_long,
)
from leakbt.models import predict_by_date_splits, predict_shuffled_kfold, to_wide
from leakbt.validation import blocked_kfold, purged_walk_forward

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"

MIN_TRAIN = 252     # one year of history before the first prediction
TEST_SIZE = 21      # retrain monthly
EMBARGO = 5
COST_BPS = 5.0      # one-way cost per unit turnover


def mean_ic(pred_wide: pd.DataFrame, target_wide: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[float, float]:
    p = pred_wide.reindex(index=dates, columns=target_wide.columns)
    t = target_wide.reindex(index=dates)
    ic = portfolio.rank_ic(p.to_numpy(), t.to_numpy())
    return float(np.nanmean(ic)), stats.newey_west_tstat(ic, lags=10)


def experiment_validation(data5, target5_wide, models):
    """Same model, same features, h=5 overlapping target: how much does the CV scheme change the IC?"""
    dates = data5.index.get_level_values("date").unique().sort_values()
    wf_test_dates = pd.DatetimeIndex(np.concatenate([t for _, t in purged_walk_forward(dates, 5, MIN_TRAIN, TEST_SIZE, EMBARGO)]))
    rows = []
    for model in models:
        schemes = {
            "Shuffled K-fold (rows)": lambda: predict_shuffled_kfold(data5, FEATURE_NAMES, k=5, model=model),
            "Blocked K-fold (no purge)": lambda: predict_by_date_splits(data5, FEATURE_NAMES, blocked_kfold(dates, 5), model),
            "Walk-forward, no purge": lambda: predict_by_date_splits(
                data5, FEATURE_NAMES, purged_walk_forward(dates, horizon=-1, min_train=MIN_TRAIN, test_size=TEST_SIZE, embargo=0), model),
            "Purged walk-forward": lambda: predict_by_date_splits(
                data5, FEATURE_NAMES, purged_walk_forward(dates, horizon=5, min_train=MIN_TRAIN, test_size=TEST_SIZE, embargo=EMBARGO), model),
        }
        for name, fn in schemes.items():
            t0 = time.time()
            pred = to_wide(fn())
            ic, tstat = mean_ic(pred, target5_wide, wf_test_dates)
            rows.append({"model": model, "scheme": name, "mean_ic": ic, "ic_tstat_nw": tstat})
            print(f"  [{model}] {name:28s} IC={ic:+.4f} t={tstat:+.2f} ({time.time() - t0:.0f}s)")
    return rows


def experiment_backtests(panel, feats, buggy_feats, models):
    """Daily long-short backtests: honest vs. three kinds of look-ahead."""
    honest_tgt = demean(forward_return(panel, 1, "next_open"))
    sameclose_tgt = demean(forward_return(panel, 1, "same_close"))
    raw_next = forward_return(panel, 1, "next_open")
    raw_same = forward_return(panel, 1, "same_close")

    data_h = to_long(feats, honest_tgt)
    dates = data_h.index.get_level_values("date").unique().sort_values()
    wf = lambda: purged_walk_forward(dates, horizon=1, min_train=MIN_TRAIN, test_size=TEST_SIZE, embargo=EMBARGO)

    results, curves = {}, {}

    def record(key, label, scores, returns):
        bt = portfolio.run_backtest(scores, returns, quantile=0.1, cost_bps=COST_BPS)
        s = bt.summary()
        s["label"] = label
        results[key] = s
        curves[key] = pd.Series(bt.net, index=bt.dates)
        print(f"  {label:55s} Sharpe gross={s['sharpe_gross']:+.2f} net={s['sharpe_net']:+.2f} IC={s['mean_ic']:+.4f}")
        return bt

    honest_preds = {}
    for model in models:
        pred = to_wide(predict_by_date_splits(data_h, FEATURE_NAMES, wf(), model))
        honest_preds[model] = pred
        record(f"{model}_honest", f"{model}: honest (purged walk-forward, next-open)", pred, raw_next)
        # Cost-aware variant: smooth the predictions over ~5 days (past values only)
        # so positions change less often. Same information, much lower turnover.
        record(f"{model}_honest_smoothed", f"{model}: honest, 5-day smoothed signal", pred.ewm(span=5, min_periods=1).mean(), raw_next)

        # Look-ahead 1: execute at the same close that produced the signal.
        data_sc = to_long(feats, sameclose_tgt)
        pred_sc = to_wide(predict_by_date_splits(data_sc, FEATURE_NAMES, wf(), model))
        record(f"{model}_same_close", f"{model}: trade at the signal's own close", pred_sc, raw_same)

        # Look-ahead 2 and 3: one buggy feature added to an otherwise honest pipeline.
        for bug in BUGGY_FEATURES:
            f2 = dict(feats)
            f2[bug] = buggy_feats[bug]
            data_b = to_long(f2, honest_tgt)
            pred_b = to_wide(predict_by_date_splits(data_b, FEATURE_NAMES + [bug], wf(), model))
            record(f"{model}_{bug}", f"{model}: honest pipeline + `{bug}`", pred_b, raw_next)

    # Cost sensitivity for the honest models
    cost_rows = []
    for model in models:
        pred = honest_preds[model]
        for bps in (0, 2, 5, 10):
            bt = portfolio.run_backtest(pred, raw_next, quantile=0.1, cost_bps=bps)
            cost_rows.append({"model": model, "cost_bps": bps, "sharpe_net": stats.sharpe(bt.net),
                              "ann_return_net": float(np.nanmean(bt.net) * 252)})
    return results, curves, cost_rows


def experiment_multiple_testing(panel, feats):
    """Try 48 simple single-feature strategies (gross of costs), pick the best in-sample, then look out-of-sample."""
    raw_next = forward_return(panel, 1, "next_open")
    ranked = {n: cross_sectional_rank(f) for n, f in feats.items()}
    common = ranked["mom_252_21"].dropna(how="all").index
    split = common[int(len(common) * 0.6)]
    trials = []
    for name, r in ranked.items():
        for sign in (+1, -1):
            for q in (0.1, 0.2, 0.3):
                sc = (sign * r).reindex(common)
                # Gross of costs: isolates selection bias from the (persistent) drag of turnover.
                bt = portfolio.run_backtest(sc, raw_next, quantile=q, cost_bps=0.0)
                net = pd.Series(bt.gross, index=bt.dates)
                ins, oos = net[net.index < split], net[net.index >= split]
                trials.append({
                    "feature": name, "sign": sign, "quantile": q,
                    "is_sharpe": stats.sharpe(ins.to_numpy()), "oos_sharpe": stats.sharpe(oos.to_numpy()),
                    "is_returns": ins.to_numpy(),
                })
    is_daily_sr = np.array([np.nanmean(t["is_returns"]) / np.nanstd(t["is_returns"], ddof=1) for t in trials])
    best = max(trials, key=lambda t: t["is_sharpe"])
    out = {
        "n_trials": len(trials),
        "split_date": str(split.date()),
        "best": {k: best[k] for k in ("feature", "sign", "quantile", "is_sharpe", "oos_sharpe")},
        "best_psr_is": stats.probabilistic_sharpe_ratio(best["is_returns"]),
        "best_dsr_is": stats.deflated_sharpe_ratio(best["is_returns"], is_daily_sr),
        "expected_max_sharpe_annualized": stats.expected_max_sharpe(is_daily_sr) * np.sqrt(252),
        "trials": [{k: t[k] for k in ("feature", "sign", "quantile", "is_sharpe", "oos_sharpe")} for t in trials],
    }
    b = out["best"]
    print(f"  best IS: {b['feature']} sign={b['sign']} q={b['quantile']} IS Sharpe={b['is_sharpe']:.2f} "
          f"OOS Sharpe={b['oos_sharpe']:.2f} PSR={out['best_psr_is']:.3f} DSR={out['best_dsr_is']:.3f}")
    return out


def experiment_speed(panel):
    """C++ kernel vs. NumPy fallback on the full panel."""
    rng = np.random.default_rng(0)
    T, N = panel.close.shape
    s = rng.standard_normal((T, N))
    r = rng.standard_normal((T, N)) * 0.01
    s[rng.random((T, N)) < 0.02] = np.nan
    timings = {}
    for label, use_cpp in (("numpy", False), ("cpp", True)):
        t0 = time.perf_counter()
        for _ in range(3):
            portfolio.long_short(s, r, 0.1, use_cpp=use_cpp)
            portfolio.rank_ic(s, r, use_cpp=use_cpp)
        timings[label] = (time.perf_counter() - t0) / 3
    a = portfolio.long_short(s, r, 0.1, use_cpp=False)
    b = portfolio.long_short(s, r, 0.1, use_cpp=True)
    timings["max_abs_diff_pnl"] = float(np.nanmax(np.abs(a[0] - b[0])))
    timings["speedup"] = timings["numpy"] / timings["cpp"]
    timings["shape"] = [T, N]
    print(f"  numpy {timings['numpy']*1e3:.0f} ms, C++ {timings['cpp']*1e3:.1f} ms, speedup {timings['speedup']:.0f}x")
    return timings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    models = ["ridge"] if args.quick else ["ridge", "gbm"]
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    panel = load_panel()
    print(f"Universe: {panel.close.shape[1]} stocks, {panel.dates[0].date()} to {panel.dates[-1].date()}")
    feats = build_features(panel)
    buggy = build_features(panel, names=list(BUGGY_FEATURES), include_buggy=True)

    print("1) Truncation audit")
    registry = dict(FEATURES)
    registry.update(BUGGY_FEATURES)
    audit = truncation_audit(panel, registry, n_dates=20)

    print("2) Validation schemes (h=5)")
    t5 = demean(forward_return(panel, 5, "next_open"))
    data5 = to_long(feats, t5)
    validation = experiment_validation(data5, t5, models)

    print("3) Backtests")
    bt_results, curves, cost_rows = experiment_backtests(panel, feats, buggy, models)

    print("4) Multiple testing")
    mt = experiment_multiple_testing(panel, feats)

    print("5) C++ vs NumPy")
    speed = experiment_speed(panel)

    payload = {
        "universe": {"stocks": int(panel.close.shape[1]), "start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date())},
        "settings": {"min_train_days": MIN_TRAIN, "retrain_every_days": TEST_SIZE, "embargo_days": EMBARGO, "cost_bps": COST_BPS},
        "audit": [r.__dict__ for r in audit],
        "audit_table": format_audit(audit),
        "validation": validation,
        "backtests": bt_results,
        "cost_sensitivity": cost_rows,
        "multiple_testing": mt,
        "speed": speed,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=lambda o: o if not isinstance(o, tuple) else list(o)))
    pd.DataFrame(curves).to_csv(OUT / "daily_net_returns.csv")
    print(f"Wrote {OUT / 'results.json'}")

    from experiments.make_figures import make_all  # noqa: E402
    make_all(OUT)


if __name__ == "__main__":
    main()
