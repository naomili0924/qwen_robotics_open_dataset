"""CPU checks of the RL machinery: log-probs, advantages, and a toy GRPO/PPO run on the heads alone."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla.config import Config  # noqa: E402
from vla.data import KIN_DIM  # noqa: E402
from vla.heads import ValueHead, build_head  # noqa: E402
from vla.rl import advantages, policy_objective  # noqa: E402

H, B, L = 64, 4, 9


def cond(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(summary=torch.randn(B, H, generator=g), memory=torch.randn(B, L, H, generator=g),
                mask=torch.ones(B, L, dtype=torch.bool), kin=torch.randn(B, KIN_DIM, generator=g))


def small(**kw):
    return Config(head_dim=32, head_layers=2, head_heads=4, pool_queries=2, flow_steps=4, **kw)


@pytest.mark.parametrize("head,denoiser,modes", [("regression", "mlp", 1), ("regression", "mlp", 3),
                                                  ("flow", "mlp", 1), ("flow", "dit", 1)])
def test_sampled_log_probs_match_recomputed(head, denoiser, modes):
    net = build_head(H, small(head=head, denoiser=denoiser, modes=modes))
    s = net.sample(cond(), 5)
    assert s["actions"].shape == (5, B, 10, 2) and s["logp"].shape == (5, B)
    assert torch.allclose(net.log_prob(cond(), s), s["logp"], atol=1e-4)


def test_advantages():
    r = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]])  # 3 samples x 2 scenarios
    a = advantages(r, "grpo")
    assert torch.allclose(a[:, 0], torch.tensor([-1.0, 1.0, 0.0]), atol=1e-3)
    assert torch.allclose(a[:, 1], torch.zeros(3))            # a constant group gives no signal
    a = advantages(r, "ppo", values=torch.tensor([2.0, 0.0]))
    assert abs(a.mean().item()) < 1e-6


def test_policy_objective_is_clipped_and_kl_nonnegative():
    old = torch.zeros(3, 2)
    new = torch.tensor([[1.0, -1.0], [0.0, 0.0], [0.1, -0.1]])  # ratios far outside the clip range in row 0
    adv = torch.ones(3, 2)
    loss, kl, frac = policy_objective(new, old, old, adv, 0.2, 0.0)
    assert kl.item() >= 0 and 0 < frac.item() < 1
    # with positive advantages the clipped surrogate cannot exceed 1 + eps per sample
    assert -loss.item() <= 1.2 + 1e-6


@pytest.mark.parametrize("algo,head,denoiser", [("grpo", "regression", "mlp"), ("grpo", "flow", "mlp"),
                                                 ("ppo", "regression", "mlp"), ("ppo", "flow", "dit")])
def test_toy_rl_improves_reward(algo, head, denoiser):
    """Reward = -|action - target|; a few updates on the heads alone must raise the mean reward."""
    torch.manual_seed(0)
    cfg = small(head=head, denoiser=denoiser, rl_std_init=0.5, rl_flow_noise=0.5, group_size=16, clip_eps=0.2, kl_beta=0.0)
    net = build_head(H, cfg)
    ref = copy.deepcopy(net)
    value = ValueHead(H, cfg) if algo == "ppo" else None
    params = list(net.parameters()) + (list(value.parameters()) if value else [])
    opt = torch.optim.Adam(params, lr=3e-3)
    target = torch.zeros(B, 10, 2)
    target[:, :, 0] = torch.linspace(0.1, 1.0, 10)

    def reward(actions):  # (G, B, T, D) -> (G, B)
        return -(actions - target[None]).abs().mean((2, 3))

    def mean_reward():
        with torch.no_grad():
            return reward(net.sample(cond(), 32)["actions"]).mean().item()

    before = mean_reward()
    for _ in range(40):
        c = cond()
        with torch.no_grad():
            sample = net.sample(c, cfg.group_size)
            r = reward(sample["actions"])
            ref_logp = ref.log_prob(c, sample)
            v = value(c) if value else None
        adv = advantages(r, algo, v)
        loss, _, _ = policy_objective(net.log_prob(c, sample), sample["logp"], ref_logp, adv, cfg.clip_eps, cfg.kl_beta)
        if value:
            loss = loss + 0.5 * ((value(c)[None] - r) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = mean_reward()
    assert after > before + 0.05, (before, after)
