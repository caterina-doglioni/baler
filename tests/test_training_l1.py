import copy
import types

import pytest
import torch
from torch.utils.data import DataLoader

from baler.modules import helper, models, training, utils

N_FEATURES, Z_DIM, N_ROWS = 24, 15, 64

L1_MODELS = [models.AE, models.CFD_dense_AE, models.FPGA_prototype_model]


def make(model_cls, seed=0):
    """Model and data on helper.get_device(), as training.train() does (fit()
    moves each batch there, so a CPU model fails on a CUDA machine)."""
    torch.manual_seed(seed)
    device = helper.get_device()
    model = model_cls(n_features=N_FEATURES, z_dim=Z_DIM).to(device)
    dtype = next(model.parameters()).dtype
    data = torch.rand(N_ROWS, N_FEATURES, dtype=dtype).to(device)
    return model, data


def run_fit(model, data, l1, reg, config=None):
    """One fit() epoch over a single batch; returns (loss, recon, other part)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr 0: weights fixed
    loss, recon, l1_part, _ = training.fit(
        config=config or types.SimpleNamespace(),
        model=model,
        train_dl=DataLoader(data, batch_size=N_ROWS),
        model_children=list(model.children()),
        regular_param=reg,
        optimizer=optimizer,
        latent_dim=Z_DIM,
        RHO=0.05,
        l1=l1,
        n_dimensions=1,
    )
    return loss, recon, l1_part


@pytest.mark.parametrize("model_cls", L1_MODELS)
def test_forward_with_activations_matches_forward(model_cls):
    model, data = make(model_cls)
    recon, acts = model.forward_with_activations(data)
    assert torch.equal(recon, model(data))
    assert len(acts) >= 1
    assert all(a.shape[0] == N_ROWS for a in acts)


def test_ae_activations_are_hidden_layers_only():
    model, data = make(models.AE)
    _, acts = model.forward_with_activations(data)
    # en1-3 and de1-3; not the latent (z_dim wide) nor the output (n_features wide)
    assert [a.shape[1] for a in acts] == [200, 100, 50, 50, 100, 200]


def test_l1_activation_penalty_value():
    acts = [torch.tensor([[1.0, -3.0]]), torch.tensor([[2.0, 2.0, -2.0]])]
    assert utils.l1_activation_penalty(acts).item() == pytest.approx(2.0 + 2.0)


def test_fit_without_l1_is_reconstruction_only():
    model, data = make(models.AE)
    expected = utils.mse_sum_loss_l1([], data, model(data), 0.0, validate=True)[0]
    loss, recon, l1_part = run_fit(model, data, l1=False, reg=0.001)
    assert loss == pytest.approx(expected.item())
    assert recon == pytest.approx(loss)
    assert l1_part == 0.0


@pytest.mark.parametrize("model_cls", L1_MODELS)
def test_fit_with_l1_adds_weighted_penalty(model_cls):
    model, data = make(model_cls)
    recon_out, acts = model.forward_with_activations(data)
    expected_recon = utils.mse_sum_loss_l1([], data, recon_out, 0.0, True)[0].item()
    expected_l1 = 0.01 * utils.l1_activation_penalty(acts).item()
    loss, recon, l1_part = run_fit(model, data, l1=True, reg=0.01)
    assert recon == pytest.approx(expected_recon)
    assert l1_part == pytest.approx(expected_l1)
    assert l1_part > 0
    assert loss == pytest.approx(recon + l1_part)


def test_fit_swae_term_is_not_reconstruction():
    """With the SWAE loss, the reconstruction part is the MSE term only; the
    sliced-Wasserstein term goes to the other (regularization) part.
    CFD_dense_AE (float32): compute_swd draws float32 projections, so the SWAE
    loss fails for the float64 models, on main as well."""
    model, data = make(models.CFD_dense_AE)
    # The reconstruction part is deterministic; the SWD term uses random
    # projections, so only its consistency with the total is checked.
    expected_recon = utils.mse_sum_loss_l1([], data, model(data), 0.0, True)[0]
    cfg = types.SimpleNamespace(custom_loss_function="loss_function_swae")
    loss, recon, other = run_fit(model, data, l1=False, reg=0.0, config=cfg)
    assert recon == pytest.approx(expected_recon.item())
    assert other > 0
    assert recon + other == pytest.approx(loss)


def test_fit_with_l1_and_zero_reg_equals_no_l1():
    model, data = make(models.AE)
    loss_off = run_fit(copy.deepcopy(model), data, l1=False, reg=0.0)[0]
    loss_on = run_fit(copy.deepcopy(model), data, l1=True, reg=0.0)[0]
    assert loss_on == pytest.approx(loss_off)


def test_l1_penalty_reaches_the_gradients():
    """With a large reg_param the step must differ from the step without L1."""
    model, data = make(models.AE)
    grads = {}
    for l1 in (False, True):
        m = copy.deepcopy(model)
        recon, acts = m.forward_with_activations(data)
        loss = utils.mse_sum_loss_l1([], data, recon, 0.0, True)[0]
        if l1:
            loss = loss + 10.0 * utils.l1_activation_penalty(acts)
        loss.backward()
        grads[l1] = m.en1.weight.grad.clone()
    assert not torch.allclose(grads[False], grads[True])


def test_models_without_support_have_no_method():
    """l1_enabled() switches L1 off with a warning for these (e.g. the bundled
    CFD_project_still config uses Conv_AE)."""
    assert not hasattr(models.Conv_AE, "forward_with_activations")


def cfg(**kw):
    return types.SimpleNamespace(**kw)


def test_l1_off_by_default(capsys):
    assert training.l1_enabled(cfg(), models.AE(N_FEATURES, Z_DIM)) is False
    assert capsys.readouterr().out == ""


def test_l1_on_with_flag():
    c = cfg(l1_activation_penalty=True)
    assert training.l1_enabled(c, models.AE(N_FEATURES, Z_DIM)) is True


def test_old_l1_flag_is_ignored_with_note(capsys):
    assert training.l1_enabled(cfg(l1=True), models.AE(N_FEATURES, Z_DIM)) is False
    assert "l1_activation_penalty" in capsys.readouterr().out


def test_old_l1_flag_no_note_when_new_flag_set(capsys):
    c = cfg(l1=True, l1_activation_penalty=False)
    assert training.l1_enabled(c, models.AE(N_FEATURES, Z_DIM)) is False
    assert capsys.readouterr().out == ""


def test_l1_flag_on_unsupported_model_warns_and_disables(capsys):
    c = cfg(l1_activation_penalty=True)
    model = models.Conv_AE(n_features=N_FEATURES, z_dim=Z_DIM)
    assert training.l1_enabled(c, model) is False
    assert "WARNING" in capsys.readouterr().out


def test_template_has_flag_off():
    from baler.modules import helper

    text = helper.create_default_config("ws", "proj")
    assert "c.l1_activation_penalty        = False" in text
    assert "c.l1 " not in text
