"""CPU checks of the action heads with synthetic conditioning."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla.config import Config  # noqa: E402
from vla.data import KIN_DIM  # noqa: E402
from vla.heads import AuxHead, build_head  # noqa: E402
from vla.tasks import TASKS  # noqa: E402

H, B, L = 64, 3, 17


def cond():
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[1, 10:] = False  # padded row
    return dict(summary=torch.randn(B, H), memory=torch.randn(B, L, H), mask=mask, kin=torch.randn(B, KIN_DIM))


@pytest.mark.parametrize("head,denoiser,modes", [("regression", "mlp", 1), ("regression", "mlp", 3),
                                                  ("flow", "mlp", 1), ("flow", "dit", 1)])
def test_head_loss_and_prediction_shapes(head, denoiser, modes):
    cfg = Config(head=head, denoiser=denoiser, modes=modes, head_dim=32, head_layers=2, head_heads=4, pool_queries=2,
                 flow_steps=4)
    net = build_head(H, cfg)
    target = torch.randn(B, cfg.horizon, cfg.action_dim)
    loss = net.loss(cond(), target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    pred = net.predict(cond())
    assert pred.shape == (B, cfg.horizon, cfg.action_dim)


def test_multi_mode_regression_keeps_two_modes_apart():
    # Same conditioning, targets alternate between two paths: WTA must fit both, not their mean.
    torch.manual_seed(0)
    cfg = Config(head="regression", modes=2, head_dim=32, head_layers=2, head_heads=4, pool_queries=2)
    net = build_head(H, cfg)
    c = dict(summary=torch.zeros(2, H), memory=torch.zeros(2, L, H), mask=torch.ones(2, L, dtype=torch.bool),
             kin=torch.zeros(2, KIN_DIM))
    target = torch.zeros(2, cfg.horizon, cfg.action_dim)
    target[0, :, 1], target[1, :, 1] = 1.0, -1.0
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        net.loss(c, target).backward()
        opt.step()
    traj, _ = net._forward(c)
    best = (traj[0] - target[:1, None]).abs().mean((2, 3)).min().item()  # best mode vs the +1 path
    mean_path_error = 1.0  # what a single-mode regressor would end up with
    assert best < 0.2 < mean_path_error


def test_flow_medoid_selection():
    cfg = Config(head="flow", denoiser="mlp", head_dim=32, head_layers=2, head_heads=4, pool_queries=2, flow_steps=2,
                 flow_samples=3)
    net = build_head(H, cfg)
    assert net.predict(cond()).shape == (B, cfg.horizon, cfg.action_dim)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_aux_heads(name):
    cfg = Config(head_dim=32, head_layers=2, head_heads=4, pool_queries=2)
    head = AuxHead(H, cfg, TASKS[name])
    target = torch.rand(B, TASKS[name].out_dim)
    loss = head.loss(cond(), target)
    assert torch.isfinite(loss)
    pred = head.predict(cond()).numpy()
    assert pred.shape == (B, TASKS[name].out_dim)
    assert isinstance(TASKS[name].metrics(pred, target.numpy()), dict)
