import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "check_roundtrip", os.path.join(HERE, "roundtrip", "check_roundtrip.py")
)
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)

WS, PROJ = "W", "P"
N_ROWS, N_COLS, RATIO = 50, 8, 2.0


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """A minimal, valid round-trip output tree (check_roundtrip resolves
    workspaces/... from the working directory)."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    proj = tmp_path / "workspaces" / WS / PROJ
    out = proj / "output"
    for d in (
        "config",
        "compressed_output",
        "decompressed_output",
        "training",
        "plotting",
    ):
        (proj / d if d == "config" else out / d).mkdir(parents=True)
    (proj / "config" / f"{PROJ}_config.py").write_text(
        "def set_config(c):\n"
        f"    c.input_path = 'orig.npz'\n    c.compression_ratio = {RATIO}\n"
        "    c.epochs = 3\n"
    )
    data = rng.normal(size=(N_ROWS, N_COLS))
    names = np.array([f"v{i}" for i in range(N_COLS)])
    np.savez(tmp_path / "orig.npz", data=data, names=names)
    latent = int(np.ceil(N_COLS / RATIO))
    np.savez(
        out / "compressed_output" / "compressed.npz",
        data=rng.normal(size=(N_ROWS, latent)),
        names=names,
        normalization_features=np.zeros(2),
    )
    (out / "compressed_output" / "model.pt").write_bytes(b"x")
    np.savez(
        out / "decompressed_output" / "decompressed.npz",
        data=data + 0.01 * rng.normal(size=data.shape),
        names=names,
    )
    np.save(out / "training" / "loss_data.npy", np.full((2, 3), 0.001))
    np.save(out / "training" / "loss_components.npy", np.full((2, 3), 0.0005))
    for pdf in ("comparison.pdf", "Loss_plot.pdf"):
        (out / "plotting" / pdf).write_bytes(b"%PDF")
    return out


def run(tmp_path, *extra):
    metrics = tmp_path / "m.json"
    rc = chk.main(
        ["--workspace", WS, "--project", PROJ, "--metrics-out", str(metrics), *extra]
    )
    return rc, json.loads(metrics.read_text())


def test_valid_tree_passes(fake_project, tmp_path):
    rc, m = run(tmp_path)
    assert rc == 0, [r for r in m["checks"] if not r["passed"]]
    assert m["final_reconstruction_loss"] == pytest.approx(0.0005)
    assert m["final_regularization_term"] == pytest.approx(0.0005)


def test_nan_loss_component_is_structural(fake_project, tmp_path):
    comp = np.full((2, 3), 0.0005)
    comp[0, -1] = np.nan  # finite loss_data, NaN reconstruction part
    np.save(fake_project / "training" / "loss_components.npy", comp)
    rc, m = run(tmp_path, "--thresholds-informational")
    assert rc == 1
    failed = {r["check"]: r["kind"] for r in m["checks"] if not r["passed"]}
    assert failed == {"loss_components values finite": "structural"}
    assert m["structural_passed"] is False
