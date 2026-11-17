#!/usr/bin/env python
"""Compare two sets of round-trip runs: is the head better or worse than the baseline?

    python tests/roundtrip/compare_runs.py \
        --head head_runs/*/roundtrip_metrics.json \
        --base base_runs/*/roundtrip_metrics.json \
        --out comparison.json [--summary-md FILE] [--plot comparison.png]

or, against the runs stored at the last calibration:

    python tests/roundtrip/compare_runs.py --head ... --base-stored tests/roundtrip/thresholds.json ...

A single 25-epoch run varies a lot from run to run (per-variable rms/std spreads
up to ~6x over 20 runs), so one run per side says nothing. Each side is a set of
independent runs, and each quantity is compared by:

  - ratio = median(head) / median(base); lower rms/std and loss are better;
  - a bootstrap confidence interval on that ratio (both sides resampled);
  - a two-sided Mann-Whitney U p-value (normal approximation, tie-corrected),
    reported alongside, not used for the verdict.

Verdict: "better" if the whole interval is below 1, "worse" if it is above 1,
otherwise "no significant change". When the baseline median is 0 (rare-count
variables), the ratio is undefined and the difference of medians is used
instead, with the same rule against 0.

This script never fails on worse results: it exits 0 whenever it could make the
comparison, and 2 if an input is unusable. Runs with structural failures
(missing files, wrong shapes, non-finite values) are left out; runs that only
exceeded the thresholds.json limits are kept, since a comparison has to be able
to see a change that is worse than the limits allow.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from calibrate_thresholds import check_kind  # noqa: E402

OVERALL = "median over variables"
LOSS = "final train loss"


# ---------------------------------------------------------------- loading


def load_run(path):
    """One roundtrip_metrics.json -> {quantity: value}, or None if the run had a
    structural failure. Raises ValueError if the file is not a metrics file."""
    with open(path) as f:
        m = json.load(f)
    if "checks" not in m:
        raise ValueError(f"{path}: not a roundtrip_metrics.json (no 'checks')")
    structural = [
        r for r in m["checks"] if not r["passed"] and check_kind(r) != "threshold"
    ]
    if structural or "per_variable" not in m or "final_train_loss" not in m:
        return None
    values = {
        name: pv["rms_over_std"]
        for name, pv in m["per_variable"].items()
        if pv["rms_over_std"] is not None
    }
    values[OVERALL] = m["rms_over_std_summary"]["median"]
    values[LOSS] = m["final_train_loss"]
    return values


def load_runs(paths, label):
    runs, skipped = [], []
    for p in paths:
        r = load_run(p)
        (skipped if r is None else runs).append(p if r is None else r)
    if skipped:
        print(f"[{label}] left out {len(skipped)} run(s) with structural failures:")
        for p in skipped:
            print(f"    {p}")
    return runs, skipped


def load_stored(thresholds_path):
    """The per-run values stored under "_raw" at the last calibration."""
    with open(thresholds_path) as f:
        t = json.load(f)
    raw = t["_raw"]
    per_var = raw["rms_over_std"]
    n = len(raw["final_train_loss"])
    runs = []
    for i in range(n):
        values = {name: vals[i] for name, vals in per_var.items()}
        values[OVERALL] = float(np.median(list(values.values())))
        values[LOSS] = raw["final_train_loss"][i]
        runs.append(values)
    return runs, t.get("_source", "")


def by_quantity(runs):
    """[{q: v}] -> {q: np.array}; quantity order kept from the first run."""
    out = {}
    for q in runs[0]:
        vals = [r[q] for r in runs if q in r and r[q] is not None]
        out[q] = np.asarray(vals, dtype=float)
    return out


# ---------------------------------------------------------------- statistics


def mann_whitney_p(a, b):
    """Two-sided p-value of the Mann-Whitney U test, normal approximation with
    tie correction. Adequate for n >= ~8 per side."""
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    # average ranks for ties
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    _, counts = np.unique(allv, return_counts=True)
    n = n1 + n2
    tie = (counts**3 - counts).sum() / (n * (n - 1)) if n > 1 else 0.0
    var = n1 * n2 / 12 * ((n + 1) - tie)
    if var <= 0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)  # continuity correction
    z = max(z, 0.0)
    return math.erfc(z / math.sqrt(2))


def compare_quantity(h, b, rng, n_boot, level):
    """Head vs base for one quantity (lower is better)."""
    hm, bm = float(np.median(h)), float(np.median(b))
    hb = rng.choice(h, size=(n_boot, len(h)), replace=True)
    bb = rng.choice(b, size=(n_boot, len(b)), replace=True)
    hmed, bmed = np.median(hb, axis=1), np.median(bb, axis=1)
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    if bm > 0 and np.all(bmed > 0):
        stat, null = "ratio", 1.0
        est = hm / bm
        lo, hi = np.percentile(hmed / bmed, [lo_q, hi_q])
    else:
        stat, null = "difference", 0.0
        est = hm - bm
        lo, hi = np.percentile(hmed - bmed, [lo_q, hi_q])
    if hi < null:
        verdict = "better"
    elif lo > null:
        verdict = "worse"
    else:
        verdict = "no significant change"
    return {
        "head_median": hm,
        "base_median": bm,
        "n_head": int(len(h)),
        "n_base": int(len(b)),
        "statistic": stat,
        "estimate": float(est),
        "ci": [float(lo), float(hi)],
        "mann_whitney_p": mann_whitney_p(h, b),
        "verdict": verdict,
    }


def compare(head_runs, base_runs, n_boot=10000, level=0.95, seed=0):
    rng = np.random.default_rng(seed)  # fixed: the same inputs give the same report
    h, b = by_quantity(head_runs), by_quantity(base_runs)
    results = {}
    for q in h:
        if q not in b or len(h[q]) < 2 or len(b[q]) < 2:
            continue
        results[q] = compare_quantity(h[q], b[q], rng, n_boot, level)
    return results


# ---------------------------------------------------------------- reporting

ICON = {"better": "🟢 better", "worse": "🔴 worse", "no significant change": "⚪ —"}


def short(q):
    return q if q in (OVERALL, LOSS) else q.split(".")[-1]


def fmt_row(q, r):
    if r["statistic"] == "ratio":
        est = f"×{r['estimate']:.3f}"
        ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
    else:
        est = f"{r['estimate']:+.2g}"
        ci = f"[{r['ci'][0]:+.2g}, {r['ci'][1]:+.2g}] (diff)"
    return (
        f"| {short(q)} | {r['base_median']:.4g} | {r['head_median']:.4g} | {est} "
        f"| {ci} | {r['mann_whitney_p']:.3g} | {ICON[r['verdict']]} |"
    )


def markdown(results, meta):
    level = int(round(meta["level"] * 100))
    head = "| quantity | base median | head median | head/base | " + (
        f"{level}% CI | U-test p | verdict |"
    )
    sep = "|---|---|---|---|---|---|---|"
    variables = [q for q in results if q not in (OVERALL, LOSS)]
    counts = {v: 0 for v in ICON}
    for q in variables:
        counts[results[q]["verdict"]] += 1
    expected_fp = (1 - meta["level"]) * len(variables)
    lines = [
        "## Round-trip comparison: head vs baseline",
        "",
        f"Head: {meta['head_desc']} ({meta['n_head']} runs"
        + (f", {meta['n_head_skipped']} left out" if meta["n_head_skipped"] else "")
        + f"). Baseline: {meta['base_desc']} ({meta['n_base']} runs"
        + (f", {meta['n_base_skipped']} left out" if meta["n_base_skipped"] else "")
        + ").",
        "",
        "Lower is better for every row. rms/std = rms(decompressed − original) / "
        "std(original), per variable.",
        "",
        "**Summary**",
        "",
        head,
        sep,
    ]
    for q in (OVERALL, LOSS):
        if q in results:
            lines.append(fmt_row(q, results[q]))
    lines += [
        "",
        f"Per variable: {counts['better']} better, {counts['worse']} worse, "
        f"{counts['no significant change']} no significant change, out of "
        f"{len(variables)}. With no real change, about {expected_fp:.1f} of "
        f"{len(variables)} would be flagged by chance at {level}%.",
        "",
        "<details><summary>Per-variable rms/std</summary>",
        "",
        head,
        sep,
    ]
    lines += [fmt_row(q, results[q]) for q in variables]
    lines += ["", "</details>", ""]
    return "\n".join(lines) + "\n"


def plot(head_runs, base_runs, results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, b = by_quantity(head_runs), by_quantity(base_runs)
    qs = [q for q in results if q != LOSS]
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(qs) + 1.5))
    rng = np.random.default_rng(1)
    # Log axis: values of exactly 0 (rare-count variables) cannot be drawn where
    # they are, so they go at a fixed left edge with a distinct marker. Medians
    # come from `results`, i.e. over all runs including the zeros, so the bars
    # agree with the JSON/markdown report.
    positive = [v for q in qs for v in (*h[q], *b[q]) if v > 0]
    zero_x = min(positive) / 3 if positive else 1e-3
    for i, q in enumerate(qs):
        for vals, med, off, color, label in (
            (b[q], results[q]["base_median"], -0.15, "tab:gray", "baseline"),
            (h[q], results[q]["head_median"], 0.15, "tab:blue", "head"),
        ):
            pos, n_zero = vals[vals > 0], int(np.sum(vals <= 0))
            y = i + off + rng.uniform(-0.06, 0.06, len(pos))
            ax.scatter(
                pos, y, s=8, color=color, alpha=0.6, label=label if i == 0 else None
            )
            if n_zero:
                ax.scatter([zero_x], [i + off], s=18, marker="<", color=color)
                ax.annotate(
                    f"{n_zero}×0",
                    (zero_x, i + off),
                    xytext=(4, -2),
                    textcoords="offset points",
                    fontsize=6,
                    color=color,
                )
            ax.plot(
                [med if med > 0 else zero_x] * 2,
                [i + off - 0.12, i + off + 0.12],
                color=color,
                linestyle="-" if med > 0 else ":",
            )
    ax.axvline(zero_x * 1.6, color="0.85", linewidth=0.8)
    ax.set_yticks(range(len(qs)))
    ax.set_yticklabels([short(q) for q in qs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(
        "rms(residual) / std(original)  (lower is better; bars = medians over all runs)\n"
        "◁ n×0: n runs with value 0, drawn at the left edge; dotted bar: median is 0"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--head", nargs="+", required=True, help="head roundtrip_metrics.json files"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", nargs="+", help="baseline roundtrip_metrics.json files")
    g.add_argument(
        "--base-stored",
        metavar="THRESHOLDS_JSON",
        help="use the per-run values stored in thresholds.json (_raw) as the baseline",
    )
    p.add_argument("--head-desc", default="head")
    p.add_argument("--base-desc", default=None)
    p.add_argument("--level", type=float, default=0.95, help="confidence level")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", required=True, help="comparison JSON")
    p.add_argument("--summary-md", help="append the markdown report to this file")
    p.add_argument("--plot", help="write a per-variable distribution plot (png/pdf)")
    args = p.parse_args(argv)

    try:
        head_runs, head_skipped = load_runs(args.head, "head")
        if args.base_stored:
            base_runs, source = load_stored(args.base_stored)
            base_skipped = []
            base_desc = (
                args.base_desc or f"stored calibration runs ({source.split(':')[0]})"
            )
        else:
            base_runs, base_skipped = load_runs(args.base, "base")
            base_desc = args.base_desc or "baseline"
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for label, runs in (("head", head_runs), ("baseline", base_runs)):
        if len(runs) < 2:
            print(
                f"error: {label} has {len(runs)} usable run(s); need >= 2",
                file=sys.stderr,
            )
            return 2

    results = compare(head_runs, base_runs, n_boot=args.n_boot, level=args.level)
    meta = {
        "head_desc": args.head_desc,
        "base_desc": base_desc,
        "n_head": len(head_runs),
        "n_base": len(base_runs),
        "n_head_skipped": len(head_skipped),
        "n_base_skipped": len(base_skipped),
        "level": args.level,
        "n_boot": args.n_boot,
    }
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    md = markdown(results, meta)
    print(md)
    if args.summary_md:
        with open(args.summary_md, "a") as f:
            f.write(md)
    if args.plot:
        plot(head_runs, base_runs, results, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
