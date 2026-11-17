import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "compare_runs", os.path.join(HERE, "roundtrip", "compare_runs.py")
)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

VARS = [f"v{i}" for i in range(6)]


def synthetic_runs(n, scale=1.0, seed=0):
    """n runs with log-normal per-variable rms/std around different levels, like
    the real runs (spread of a few x between runs)."""
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n):
        values = {
            v: scale * 0.05 * (i + 1) * float(np.exp(rng.normal(0, 0.25)))
            for i, v in enumerate(VARS)
        }
        values[cr.OVERALL] = float(np.median([values[v] for v in VARS]))
        values[cr.LOSS] = scale * 0.003 * float(np.exp(rng.normal(0, 0.05)))
        runs.append(values)
    return runs


def verdicts(results):
    return {q: r["verdict"] for q, r in results.items()}


def test_same_distribution_no_change():
    res = cr.compare(synthetic_runs(15, seed=1), synthetic_runs(15, seed=2))
    assert res[cr.OVERALL]["verdict"] == "no significant change"
    assert res[cr.LOSS]["verdict"] == "no significant change"


def test_halved_is_better():
    res = cr.compare(synthetic_runs(15, 0.5, seed=1), synthetic_runs(15, seed=2))
    assert set(verdicts(res).values()) == {"better"}
    assert res[cr.OVERALL]["estimate"] == pytest.approx(0.5, rel=0.2)


def test_doubled_is_worse():
    res = cr.compare(synthetic_runs(15, 2.0, seed=1), synthetic_runs(15, seed=2))
    assert set(verdicts(res).values()) == {"worse"}


def test_zero_baseline_median_uses_difference():
    base = synthetic_runs(10, seed=1)
    head = synthetic_runs(10, seed=2)
    for r in base:
        r["v0"] = 0.0
    for r in head:
        r["v0"] = 0.02
    res = cr.compare(head, base)
    assert res["v0"]["statistic"] == "difference"
    assert res["v0"]["verdict"] == "worse"


def test_report_is_deterministic():
    a = cr.compare(synthetic_runs(10, seed=1), synthetic_runs(10, seed=2))
    b = cr.compare(synthetic_runs(10, seed=1), synthetic_runs(10, seed=2))
    assert a == b


def test_mann_whitney_p():
    a = np.arange(10.0)
    assert cr.mann_whitney_p(a, a) == pytest.approx(1.0, abs=0.05)
    assert cr.mann_whitney_p(a, a + 100) < 0.001
    assert cr.mann_whitney_p(np.zeros(5), np.zeros(5)) == 1.0  # all tied


def write_metrics(path, values, structural_fail=False, threshold_fail=False):
    checks = [
        {
            "check": "decompressed values finite",
            "passed": not structural_fail,
            "detail": "",
            "kind": "structural",
        },
        {
            "check": "per-variable rms/std within thresholds",
            "passed": not threshold_fail,
            "detail": "",
            "kind": "threshold",
        },
    ]
    m = {
        "checks": checks,
        "per_variable": {v: {"rms_over_std": values[v]} for v in VARS},
        "rms_over_std_summary": {"median": values[cr.OVERALL]},
        "final_train_loss": values[cr.LOSS],
    }
    with open(path, "w") as f:
        json.dump(m, f)
    return str(path)


def test_cli_keeps_threshold_failures_drops_structural(tmp_path):
    head = [
        write_metrics(tmp_path / f"h{i}.json", r, threshold_fail=(i == 0))
        for i, r in enumerate(synthetic_runs(6, seed=1))
    ]
    head.append(
        write_metrics(
            tmp_path / "broken.json", synthetic_runs(1)[0], structural_fail=True
        )
    )
    base = [
        write_metrics(tmp_path / f"b{i}.json", r)
        for i, r in enumerate(synthetic_runs(6, seed=2))
    ]
    out, md = tmp_path / "cmp.json", tmp_path / "summary.md"
    rc = cr.main(
        [
            "--head",
            *head,
            "--base",
            *base,
            "--out",
            str(out),
            "--summary-md",
            str(md),
            "--n-boot",
            "500",
        ]
    )
    assert rc == 0
    meta = json.loads(out.read_text())["meta"]
    assert (meta["n_head"], meta["n_head_skipped"], meta["n_base"]) == (6, 1, 6)
    assert "Round-trip comparison" in md.read_text()


def test_cli_worse_still_exits_zero(tmp_path):
    head = [
        write_metrics(tmp_path / f"h{i}.json", r)
        for i, r in enumerate(synthetic_runs(8, 3.0, seed=1))
    ]
    base = [
        write_metrics(tmp_path / f"b{i}.json", r)
        for i, r in enumerate(synthetic_runs(8, seed=2))
    ]
    out = tmp_path / "cmp.json"
    assert (
        cr.main(
            ["--head", *head, "--base", *base, "--out", str(out), "--n-boot", "500"]
        )
        == 0
    )
    assert json.loads(out.read_text())["results"][cr.OVERALL]["verdict"] == "worse"


def test_cli_too_few_runs(tmp_path):
    head = [write_metrics(tmp_path / "h.json", synthetic_runs(1)[0])]
    base = [
        write_metrics(tmp_path / f"b{i}.json", r)
        for i, r in enumerate(synthetic_runs(3))
    ]
    assert (
        cr.main(["--head", *head, "--base", *base, "--out", str(tmp_path / "x.json")])
        == 2
    )


def test_stored_baseline_matches_thresholds_json():
    path = os.path.join(HERE, "roundtrip", "thresholds.json")
    runs, _ = cr.load_stored(path)
    with open(path) as f:
        raw = json.load(f)["_raw"]
    assert len(runs) == len(raw["final_train_loss"])
    assert set(raw["rms_over_std"]) <= set(runs[0])


def test_plot_with_zero_heavy_variable(tmp_path):
    base = synthetic_runs(10, seed=1)
    head = synthetic_runs(10, seed=2)
    for i, r in enumerate(base):
        r["v0"] = 0.0 if i < 8 else 0.02  # median 0, like mElectronMultiplicity
    res = cr.compare(head, base, n_boot=200)
    assert res["v0"]["base_median"] == 0.0
    out = tmp_path / "p.png"
    cr.plot(head, base, res, str(out))
    assert out.stat().st_size > 0


def test_loss_uses_reconstruction_part_when_present(tmp_path):
    r = synthetic_runs(1)[0]
    path = write_metrics(tmp_path / "m.json", r)
    m = json.loads(open(path).read())
    m["final_train_loss"] = 1.0  # total incl. an L1 term
    m["final_reconstruction_loss"] = 0.003
    with open(path, "w") as f:
        json.dump(m, f)
    assert cr.load_run(path)[cr.LOSS] == 0.003
