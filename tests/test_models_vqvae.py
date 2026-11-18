# Copyright 2022 Baler Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the VQ-VAE backend (models.VectorQuantizer / models.VQVAE).

Reference: van den Oord, Vinyals & Kavukcuoglu, "Neural Discrete
Representation Learning", NeurIPS 2017 (arXiv:1711.00937).
"""

import math

import numpy as np
import pytest
import torch

from baler.modules import data_processing, models

# --------------------------------------------------------------------------
# VectorQuantizer: nearest-neighbour lookup
# --------------------------------------------------------------------------


def _fixed_quantizer(codebook, commitment_cost=0.25):
    """A single-group quantizer whose codebook is set to known values."""
    codebook = torch.tensor(codebook, dtype=torch.float64)
    vq = models.VectorQuantizer(
        num_embeddings=codebook.shape[0],
        embedding_dim=codebook.shape[1],
        commitment_cost=commitment_cost,
    )
    with torch.no_grad():
        vq.codebook.copy_(codebook.unsqueeze(0))
    return vq


def test_quantizer_selects_nearest_codebook_entry():
    vq = _fixed_quantizer([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    z_e = torch.tensor(
        [[0.1, 0.1], [0.9, 0.9], [4.0, 4.0], [-3.0, -3.0]], dtype=torch.float64
    )
    _, _, indices = vq(z_e)
    # Nearest entries are 0, 1, 2 and (for the negative row) 0.
    assert indices.squeeze(1).tolist() == [0, 1, 2, 0]


def test_quantizer_output_equals_selected_codebook_vector():
    vq = _fixed_quantizer([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    z_e = torch.tensor([[0.1, 0.2], [4.9, 5.2]], dtype=torch.float64)
    z_q, _, indices = vq(z_e)
    # Forward value is exactly the codebook entry, despite the
    # straight-through reparameterisation.
    expected = vq.codebook[0][indices.squeeze(1)]
    torch.testing.assert_close(z_q, expected)


def test_quantizer_matches_bruteforce_distance():
    """The expanded ||z||^2 - 2 z.e^T + ||e||^2 form must agree with the
    literal squared distance."""
    torch.manual_seed(0)
    vq = models.VectorQuantizer(num_embeddings=16, embedding_dim=4)
    z_e = torch.randn(32, 4, dtype=torch.float64)
    _, _, indices = vq(z_e)

    brute = torch.stack(
        [((z_e[i] - vq.codebook[0]) ** 2).sum(dim=1) for i in range(z_e.shape[0])]
    )
    assert torch.equal(indices.squeeze(1), brute.argmin(dim=1))


# --------------------------------------------------------------------------
# Product quantization
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_groups", [1, 3, 5, 15])
def test_product_quantization_shapes_and_rate(num_groups):
    k, z_dim = 512, 15
    vq = models.VectorQuantizer(
        num_embeddings=k, embedding_dim=z_dim, num_groups=num_groups
    )
    z_e = torch.randn(7, z_dim, dtype=torch.float64)
    z_q, loss, indices = vq(z_e)

    assert vq.sub_dim == z_dim // num_groups
    assert z_q.shape == z_e.shape
    assert indices.shape == torch.Size([7, num_groups])
    assert indices.dtype == torch.long
    assert torch.isfinite(loss)
    # The whole point: bits/row scales linearly in num_groups.
    assert vq.num_groups * math.log2(k) == num_groups * 9


def test_product_quantization_matches_per_group_bruteforce():
    """Each group must independently pick its own nearest entry."""
    torch.manual_seed(0)
    m, k, sub = 3, 8, 2
    vq = models.VectorQuantizer(num_embeddings=k, embedding_dim=m * sub, num_groups=m)
    z_e = torch.randn(16, m * sub, dtype=torch.float64)
    _, _, indices = vq(z_e)

    z_g = z_e.view(16, m, sub)
    for g in range(m):
        brute = torch.stack(
            [((z_g[i, g] - vq.codebook[g]) ** 2).sum(dim=1) for i in range(16)]
        )
        assert torch.equal(indices[:, g], brute.argmin(dim=1))


def test_num_groups_must_divide_embedding_dim():
    with pytest.raises(ValueError) as e:
        models.VectorQuantizer(num_embeddings=8, embedding_dim=15, num_groups=4)
    msg = str(e.value)
    # The error must tell the user what they *can* use.
    assert "15" in msg and "[1, 3, 5, 15]" in msg


def test_lookup_reconstructs_the_quantized_latent():
    torch.manual_seed(0)
    vq = models.VectorQuantizer(num_embeddings=16, embedding_dim=6, num_groups=3)
    z_e = torch.randn(11, 6, dtype=torch.float64)
    z_q, _, indices = vq(z_e)
    # lookup() sees only the integers, yet must rebuild the same vector.
    torch.testing.assert_close(vq.lookup(indices), z_q)


def test_decode_indices_equals_decode_encode():
    torch.manual_seed(0)
    model = models.VQVAE(n_features=24, z_dim=15, num_embeddings=64, num_groups=5)
    model.eval()
    x = torch.randn(13, 24, dtype=torch.float64)
    with torch.no_grad():
        a = model.decode(model.encode(x))
        b = model.decode_indices(model.encode_indices(x))
    torch.testing.assert_close(a, b)


# --------------------------------------------------------------------------
# VectorQuantizer: straight-through estimator
# --------------------------------------------------------------------------


def test_straight_through_passes_gradient_unchanged():
    """argmin is not differentiable; the estimator must make d(z_q)/d(z_e) = I.

    Without it, z_e.grad from a loss defined purely on z_q would be zero.
    """
    vq = _fixed_quantizer([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    z_e = torch.tensor(
        [[0.1, 0.1], [4.0, 4.0]], dtype=torch.float64, requires_grad=True
    )

    z_q, _, _ = vq(z_e)
    z_q.sum().backward()

    # d(sum z_q)/d(z_e) is exactly 1 for every element.
    assert z_e.grad is not None
    torch.testing.assert_close(z_e.grad, torch.ones_like(z_e))


def test_gradient_reaches_encoder_through_quantizer():
    """End-to-end: a loss on the reconstruction alone (no vq_loss term) must
    still produce encoder gradients, which can only happen via the estimator."""
    torch.manual_seed(0)
    model = models.VQVAE(n_features=6, z_dim=3, num_embeddings=8)
    x = torch.randn(10, 6, dtype=torch.float64)

    reconstruction = model(x)
    reconstruction.sum().backward()

    assert model.en1.weight.grad is not None
    assert model.en1.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------
# VectorQuantizer: loss terms (paper eq. 3)
# --------------------------------------------------------------------------


def test_loss_is_finite_scalar():
    torch.manual_seed(0)
    vq = models.VectorQuantizer(num_embeddings=8, embedding_dim=4)
    z_e = torch.randn(16, 4, dtype=torch.float64)
    _, loss, _ = vq(z_e)

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert loss >= 0


def test_loss_is_zero_when_encoder_output_is_already_on_the_codebook():
    """Both terms of eq. 3 vanish when z_e equals its nearest codebook entry."""
    vq = _fixed_quantizer([[0.0, 0.0], [1.0, 1.0]])
    z_e = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    _, loss, _ = vq(z_e)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_commitment_cost_scales_only_the_commitment_term():
    """loss(beta) = codebook + beta * commitment, so the difference between
    two betas is exactly (beta2 - beta1) * commitment."""
    codebook = [[0.0, 0.0], [1.0, 1.0]]
    z_e = torch.tensor([[0.3, 0.4]], dtype=torch.float64)

    _, loss_a, _ = _fixed_quantizer(codebook, commitment_cost=0.25)(z_e)
    _, loss_b, _ = _fixed_quantizer(codebook, commitment_cost=1.25)(z_e)

    # Both terms are the same MSE here (||z_e - e||^2 either way), so the
    # commitment term equals the codebook term.
    _, loss_zero, _ = _fixed_quantizer(codebook, commitment_cost=0.0)(z_e)
    commitment = loss_zero  # == codebook term
    assert (loss_b - loss_a).item() == pytest.approx(commitment.item(), rel=1e-9)


# --------------------------------------------------------------------------
# Codebook perplexity
# --------------------------------------------------------------------------


def test_perplexity_bounds_and_extremes():
    k = 8
    vq = models.VectorQuantizer(num_embeddings=k, embedding_dim=2)

    collapsed = torch.zeros(64, dtype=torch.long)
    uniform = torch.arange(64, dtype=torch.long) % k

    p_collapsed = vq.perplexity(collapsed).item()
    p_uniform = vq.perplexity(uniform).item()

    # Everything routed to one entry -> perplexity 1.
    assert p_collapsed == pytest.approx(1.0, rel=1e-6)
    # Uniform usage over all K entries -> perplexity K.
    assert p_uniform == pytest.approx(float(k), rel=1e-6)
    # General bounds.
    for p in (p_collapsed, p_uniform):
        assert 1.0 - 1e-9 <= p <= k + 1e-9


# --------------------------------------------------------------------------
# VQVAE: interface compatibility with the AE backend
# --------------------------------------------------------------------------


def test_encode_decode_shapes_match_ae_contract():
    n_features, z_dim, batch = 24, 15, 7
    ae = models.AE(n_features=n_features, z_dim=z_dim)
    vq = models.VQVAE(n_features=n_features, z_dim=z_dim)
    x = torch.randn(batch, n_features, dtype=torch.float64)

    assert vq.encode(x).shape == ae.encode(x).shape == torch.Size([batch, z_dim])
    assert vq.decode(vq.encode(x)).shape == torch.Size([batch, n_features])
    assert vq(x).shape == x.shape


def test_encode_indices_returns_one_index_per_row():
    model = models.VQVAE(n_features=6, z_dim=3, num_embeddings=16)
    x = torch.randn(11, 6, dtype=torch.float64)
    indices = model.encode_indices(x)

    # One index per group per row; num_groups defaults to 1.
    assert indices.shape == torch.Size([11, 1])
    assert indices.dtype == torch.long
    assert int(indices.min()) >= 0 and int(indices.max()) < 16


def test_forward_populates_vq_loss():
    model = models.VQVAE(n_features=6, z_dim=3)
    # Present but zero before any forward pass, so training's getattr hook is
    # well defined from the start.
    assert torch.equal(model.vq_loss, torch.zeros((), dtype=torch.float64))

    model(torch.randn(8, 6, dtype=torch.float64))
    assert model.vq_loss.requires_grad
    assert torch.isfinite(model.vq_loss)


def test_ae_has_no_vq_loss_so_training_hook_is_inert():
    """The training hook is `getattr(model, "vq_loss", None) is not None`;
    non-quantizing backends must not trip it."""
    ae = models.AE(n_features=6, z_dim=3)
    assert getattr(ae, "vq_loss", None) is None


# --------------------------------------------------------------------------
# Coupling with helper.decompress's width inference
# --------------------------------------------------------------------------


def test_last_state_dict_entry_is_the_output_bias():
    """helper.decompress derives the reconstructed width from the LAST entry of
    the state dict:

        number_of_columns = len(model_dict[list(model_dict.keys())[-1]])

    so the codebook must not be the final registered parameter. If the
    quantizer were assigned after the decoder in VQVAE.__init__, this would
    read num_embeddings instead of n_features and decompression would build a
    wrongly shaped model.
    """
    n_features, z_dim, k = 24, 15, 512
    model = models.VQVAE(n_features=n_features, z_dim=z_dim, num_embeddings=k)

    keys = list(model.state_dict().keys())
    assert keys[-1] == "de4.bias"
    assert len(model.state_dict()[keys[-1]]) == n_features
    # Guard against the specific failure mode.
    assert len(model.state_dict()[keys[-1]]) != k


# --------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------


class _Config:
    vq_num_embeddings = 64
    vq_commitment_cost = 0.5


def test_model_kwargs_from_config_passes_vq_options_to_vqvae():
    kwargs = data_processing.model_kwargs_from_config(models.VQVAE, _Config)
    assert kwargs == {"num_embeddings": 64, "commitment_cost": 0.5}

    model = models.VQVAE(n_features=6, z_dim=3, **kwargs)
    assert model.quantizer.num_embeddings == 64
    assert model.quantizer.commitment_cost == 0.5


def test_model_kwargs_from_config_ignores_models_that_do_not_accept_them():
    """AE must still be constructed exactly as before, even if the config
    happens to define the VQ options."""
    assert data_processing.model_kwargs_from_config(models.AE, _Config) == {}


def test_model_kwargs_from_config_omits_options_absent_from_config():
    class Empty:
        pass

    assert data_processing.model_kwargs_from_config(models.VQVAE, Empty) == {}


def test_codebook_size_survives_a_save_load_round_trip(tmp_path):
    """A non-default codebook size must be reproducible at load time, which is
    what model_kwargs_from_config exists to guarantee."""
    model = models.VQVAE(n_features=24, z_dim=15, **{"num_embeddings": 64})
    path = tmp_path / "model.pt"
    data_processing.save_model(model, str(path))

    kwargs = data_processing.model_kwargs_from_config(models.VQVAE, _Config)
    reloaded = data_processing.load_model(
        models.VQVAE, model_path=str(path), n_features=24, z_dim=15, **kwargs
    )
    assert reloaded.quantizer.num_embeddings == 64
    torch.testing.assert_close(reloaded.quantizer.codebook, model.quantizer.codebook)


# --------------------------------------------------------------------------
# EMA codebook updates (appendix A.1) and dead-entry restarts
# --------------------------------------------------------------------------


def test_ema_codebook_receives_no_gradient_and_drops_the_codebook_term():
    """With EMA the codebook is maintained by moving averages, so it must take
    no gradient and the returned loss must be the commitment term alone."""
    torch.manual_seed(0)
    vq = models.VectorQuantizer(num_embeddings=4, embedding_dim=2, use_ema=True)
    assert vq.codebook.requires_grad is False

    z_e = torch.randn(16, 2, dtype=torch.float64, requires_grad=True)
    z_q, loss, _ = vq(z_e)
    loss.backward()
    assert vq.codebook.grad is None

    # loss == beta * mse(z_e, z_q.detach()) exactly
    vq2 = models.VectorQuantizer(num_embeddings=4, embedding_dim=2, use_ema=True)
    vq2.load_state_dict(vq.state_dict())
    vq2.eval()  # no EMA update, so the codebook is identical for the check
    z2 = z_e.detach()
    zq2, loss2, _ = vq2(z2)
    expected = 0.25 * torch.nn.functional.mse_loss(z2, zq2.detach())
    torch.testing.assert_close(loss2, expected)


def test_ema_moves_codebook_toward_the_mean_of_assigned_vectors():
    """One cluster of inputs -> the selected entry must migrate toward their
    mean, which is what the EMA update is for."""
    vq = models.VectorQuantizer(num_embeddings=2, embedding_dim=1, use_ema=True)
    with torch.no_grad():
        vq.codebook.copy_(torch.tensor([[[0.0], [10.0]]], dtype=torch.float64))
        # ema_w is the EMA's running numerator and must stay consistent with
        # the codebook. Real use loads both together via load_state_dict; a
        # manual override here has to do the same or the first update would
        # snap the codebook back toward the random init.
        vq.ema_w.copy_(vq.codebook.data)
    vq.train()

    target = 3.0
    z_e = torch.full((64, 1), target, dtype=torch.float64)
    start = float(vq.codebook[0, 0, 0])
    for _ in range(200):
        vq(z_e)
    end = float(vq.codebook[0, 0, 0])

    assert abs(end - target) < abs(start - target)  # moved toward it
    # EMA approaches asymptotically (decay 0.99, cluster_size starting at 1),
    # so after 200 steps it is close but not exact: ~2.993 here.
    assert end == pytest.approx(target, abs=1e-2)
    # The unused entry must STAND STILL rather than diverge. With
    # ema_cluster_size initialised to zero it ran away to ~2e4, because a
    # decaying ema_w was divided by a near-zero count.
    assert float(vq.codebook[0, 1, 0]) == pytest.approx(10.0, abs=1e-2)


def test_dead_entries_are_restarted_from_the_batch():
    """An entry that is never selected is never updated, so it stays dead
    unless explicitly re-seeded."""
    vq = models.VectorQuantizer(
        num_embeddings=3, embedding_dim=1, restart_dead=True, restart_after=0
    )
    with torch.no_grad():
        # Entry 0 sits on the data; 1 and 2 are far away and will never win.
        vq.codebook.copy_(torch.tensor([[[0.0], [50.0], [60.0]]], dtype=torch.float64))
    vq.train()
    z_e = torch.full((32, 1), 0.01, dtype=torch.float64)

    with torch.no_grad():
        vq(z_e)  # entries 1 and 2 go unused -> restarted onto the data
        vq(z_e)  # now they win the argmin, so entry 0 starves in turn

    final = vq.codebook[0, :, 0].detach()
    # The point of restarts: no entry is left stranded far from the data.
    # Entry 0 is itself restarted once the revived entries start winning,
    # which is the mechanism working, not a failure.
    assert torch.all((final - 0.01).abs() < 1.0), final


def test_restart_and_ema_buffers_do_not_break_state_dict_ordering():
    """The buffers added by EMA/restarts must not become the last state-dict
    entry, or helper.decompress's width inference breaks (see the test above)."""
    model = models.VQVAE(
        n_features=24, z_dim=15, num_groups=5, use_ema=True, restart_dead=True
    )
    keys = list(model.state_dict().keys())
    assert keys[-1] == "de4.bias"
    assert len(model.state_dict()[keys[-1]]) == 24
    assert "quantizer.ema_w" in keys and "quantizer.steps_unused" in keys


# --------------------------------------------------------------------------
# Index storage round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k,dtype", [(256, np.uint8), (512, np.uint16)])
def test_index_storage_round_trip_through_npz(tmp_path, k, dtype):
    """Narrowing indices to uint8/uint16 must be lossless, and the stored
    integers alone must rebuild the same reconstruction."""
    torch.manual_seed(0)
    model = models.VQVAE(n_features=24, z_dim=15, num_embeddings=k, num_groups=5)
    model.eval()
    x = torch.randn(200, 24, dtype=torch.float64)
    with torch.no_grad():
        idx = model.encode_indices(x)
        reference = model.decode_indices(idx)

    narrowed = idx.numpy().astype(dtype)
    assert np.array_equal(narrowed.astype(np.int64), idx.numpy())  # lossless

    path = tmp_path / "compressed.npz"
    np.savez(
        path,
        data=narrowed,
        latent_mode=np.array("indices"),
        z_dim=np.array(15),
        num_groups=np.array(5),
        num_embeddings=np.array(k),
    )
    loaded = np.load(path)
    assert str(loaded["latent_mode"]) == "indices"
    assert int(loaded["z_dim"]) == 15

    with torch.no_grad():
        out = model.decode_indices(torch.from_numpy(loaded["data"].astype(np.int64)))
    torch.testing.assert_close(out, reference)


def test_stored_indices_are_smaller_than_a_float_latent():
    """The reason the index path exists at all."""
    rows, z_dim, groups = 1000, 15, 5
    float_bytes = rows * z_dim * 8
    index_bytes = rows * groups * 2  # uint16
    assert index_bytes < float_bytes
    assert float_bytes / index_bytes == 12.0
