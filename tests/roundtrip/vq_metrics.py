#!/usr/bin/env python
"""Reconstruction and compression metrics for the VQ-VAE backend sweep.

Deliberately SEPARATE from tests/roundtrip/check_roundtrip.py. That script is a
calibrated CI gate for the AE at 25 epochs on CI hardware; its thresholds are
not meaningful for these benchmark configs, and other branches depend on it, so
it is left untouched.

This script only measures -- it has no pass/fail thresholds. It reports:

  * per-variable rms(residual)/std(original), using the SAME definition as
    check_roundtrip.py (verified against its recorded AE numbers)
  * measured on-disk compressed.npz size and the achieved compression ratio
  * latent mode, bits/row, and the stored dtype
  * codebook utilisation and perplexity over the whole dataset

Run from the repo root:

    python tests/roundtrip/vq_metrics.py --workspace CMS_workspace \\
        --project CMS_vqvae_bench --metrics-out metrics.json
"""

import argparse
import importlib
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from baler.modules import data_processing, models  # noqa: E402


def load_config(workspace, project):
    mod = importlib.import_module(
        f"workspaces.{workspace}.{project}.config.{project}_config"
    )

    class C:
        pass

    mod.set_config(C)
    return C


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workspace", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--metrics-out", required=True)
    p.add_argument(
        "--codebook-batch",
        type=int,
        default=8192,
        help="batch size for the codebook-usage pass",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.workspace, args.project)
    base = os.path.join("workspaces", args.workspace, args.project, "output")
    comp_path = os.path.join(base, "compressed_output", "compressed.npz")
    dec_path = os.path.join(base, "decompressed_output", "decompressed.npz")
    model_path = os.path.join(base, "compressed_output", "model.pt")
    loss_path = os.path.join(base, "training", "loss_data.npy")

    m = {
        "workspace": args.workspace,
        "project": args.project,
        "model_name": cfg.model_name,
    }

    # ---- reconstruction quality -------------------------------------------
    orig = np.load(cfg.input_path)
    orig_data = orig["data"].astype(np.float64)
    names = [str(n) for n in orig["names"]]
    dec = np.load(dec_path)["data"].astype(np.float64)

    m["original_shape"] = list(orig_data.shape)
    m["decompressed_shape"] = list(dec.shape)
    m["all_finite"] = bool(np.isfinite(dec).all())
    if dec.shape != orig_data.shape:
        raise SystemExit(f"shape mismatch: {dec.shape} vs {orig_data.shape}")

    # Same definition as check_roundtrip.py lines ~330-333.
    residual = dec - orig_data
    rms = np.sqrt(np.mean(residual**2, axis=0))
    std = np.std(orig_data, axis=0)
    per_var = {}
    for name, r, s in zip(names, rms, std):
        per_var[name] = {
            "rms_residual": float(r),
            "std_original": float(s),
            "rms_over_std": float(r / s) if s > 0 else None,
        }
    ratios = [
        v["rms_over_std"] for v in per_var.values() if v["rms_over_std"] is not None
    ]
    m["per_variable"] = per_var
    m["rms_over_std_summary"] = {
        "median": float(np.median(ratios)),
        "max": float(np.max(ratios)),
        "argmax": max(per_var, key=lambda k: per_var[k]["rms_over_std"] or -1),
        "n_at_or_above_0.95": int(sum(r >= 0.95 for r in ratios)),
    }

    # ---- storage ----------------------------------------------------------
    comp = np.load(comp_path)
    orig_bytes = os.path.getsize(cfg.input_path)
    comp_bytes = os.path.getsize(comp_path)
    index_mode = "latent_mode" in comp and str(comp["latent_mode"]) == "indices"
    m["latent_mode"] = "indices" if index_mode else "float"
    m["compressed_shape"] = list(comp["data"].shape)
    m["compressed_dtype"] = str(comp["data"].dtype)
    m["original_bytes"] = orig_bytes
    m["compressed_bytes"] = comp_bytes
    m["compression_ratio"] = orig_bytes / comp_bytes

    if index_mode:
        k = int(comp["num_embeddings"])
        groups = int(comp["num_groups"])
        m["num_embeddings"] = k
        m["num_groups"] = groups
        m["z_dim"] = int(comp["z_dim"])
        m["bits_per_row_nominal"] = groups * math.ceil(math.log2(k))
    else:
        m["bits_per_row_nominal"] = int(comp["data"].shape[1]) * 64

    # ---- training curve ---------------------------------------------------
    if os.path.exists(loss_path):
        loss = np.load(loss_path)
        m["epochs_run"] = int(loss.shape[1])
        m["final_train_loss"] = float(loss[0, -1])
        m["final_val_loss"] = float(loss[1, -1])
        m["best_val_loss"] = float(loss[1].min())
        m["best_val_epoch"] = int(loss[1].argmin()) + 1

    # ---- codebook usage over the whole dataset ----------------------------
    if cfg.model_name == "VQVAE" and os.path.exists(model_path):
        kwargs = data_processing.model_kwargs_from_config(models.VQVAE, cfg)
        model = data_processing.load_model(
            models.VQVAE,
            model_path=model_path,
            n_features=orig_data.shape[1],
            z_dim=math.ceil(orig_data.shape[1] / cfg.compression_ratio),
            **kwargs,
        )
        model.eval()
        norm = np.apply_along_axis(
            data_processing.normalize, 0, orig_data, custom_norm=cfg.custom_norm
        )
        x = torch.tensor(norm, dtype=torch.float64)
        chunks = []
        with torch.no_grad():
            for i in range(0, x.shape[0], args.codebook_batch):
                chunks.append(model.encode_indices(x[i : i + args.codebook_batch]))
        idx = torch.cat(chunks)  # (N, M)
        q = model.quantizer
        counts = torch.stack(
            [
                torch.bincount(idx[:, g], minlength=q.num_embeddings)
                for g in range(q.num_groups)
            ]
        ).double()
        probs = counts / counts.sum(dim=1, keepdim=True)
        ent = -(probs * torch.log2(probs + 1e-12)).sum(dim=1)
        m["codebook"] = {
            "num_embeddings": q.num_embeddings,
            "num_groups": q.num_groups,
            "entries_used_per_group": [int(c) for c in (counts > 0).sum(dim=1)],
            "utilisation": float((counts > 0).double().mean()),
            "perplexity_per_group": [float(v) for v in torch.exp(ent * math.log(2))],
            "perplexity_mean": float(torch.exp(ent * math.log(2)).mean()),
            "entropy_bits_per_row": float(ent.sum()),
        }
        # What an entropy coder could achieve on these indices.
        m["bits_per_row_entropy"] = float(ent.sum())
        m["ideal_entropy_bytes"] = (
            float(orig_data.shape[0] * ent.sum() / 8) + q.codebook.numel() * 8
        )

    with open(args.metrics_out, "w") as f:
        json.dump(m, f, indent=2)

    s = m["rms_over_std_summary"]
    print(
        f"{args.project}: mode={m['latent_mode']} "
        f"compressed={comp_bytes/1e6:.2f} MB ratio={m['compression_ratio']:.1f}x "
        f"bits/row={m['bits_per_row_nominal']}"
    )
    print(
        f"  rms/std median={s['median']:.4f} max={s['max']:.4f} "
        f"n>=0.95: {s['n_at_or_above_0.95']}/{len(ratios)}"
    )
    if "codebook" in m:
        c = m["codebook"]
        print(
            f"  codebook: utilisation={100*c['utilisation']:.1f}% "
            f"perplexity={c['perplexity_mean']:.1f} "
            f"entropy={c['entropy_bits_per_row']:.2f} bits/row"
        )
    print(f"  metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
