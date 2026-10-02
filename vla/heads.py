"""Action heads on top of a frozen-or-adapted VLM.

Every head consumes the same conditioning: `summary` (B, H), the backbone's last
hidden state at the final prompt token, `memory` (B, L, H) with `mask` (B, L), all
backbone tokens, and `kin` (B, KIN_DIM), the robot's past positions, speed and goal
as numbers (the probe showed these are hard to read back out of pooled tokens).  Both are layer-normalised on entry: a few dimensions of
Qwen's hidden states are orders of magnitude larger than the rest.  Every head exposes
    loss(cond, target)  -> scalar
    predict(cond)       -> (B, horizon, action_dim)
with actions in normalised units (metres / action_scale).

Heads:
  RegressionHead  - MLP; with modes > 1 it proposes several trajectories and is
                    trained winner-takes-all, so it can keep two ways around an obstacle apart.
  FlowHead        - flow matching (rectified flow) with a denoiser that is either an
                    MLP on pooled features or a DiT with cross-attention to all tokens.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import KIN_DIM


def mlp(dims, dropout=0.0):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers[:-2])  # no activation after the last layer


class AttentionPool(nn.Module):
    """Learned queries that read the backbone tokens -> (B, queries, dim)."""

    def __init__(self, in_dim, dim, queries, heads):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, memory, mask):
        kv = self.proj(memory)
        q = self.queries.expand(memory.shape[0], -1, -1)
        out, _ = self.attn(q, kv, kv, key_padding_mask=~mask)
        return self.norm(out)


class Features(nn.Module):
    """Pooled backbone tokens + summary token + (optionally) the kinematic numbers -> one flat vector."""

    def __init__(self, in_dim, cfg):
        super().__init__()
        d = cfg.head_dim
        self.pool = AttentionPool(in_dim, d, cfg.pool_queries, cfg.head_heads)
        self.summary = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d))
        self.kin = mlp([KIN_DIM, d, d]) if cfg.kinematic_input else None
        self.width = d * (cfg.pool_queries + 1 + (1 if cfg.kinematic_input else 0))

    def forward(self, cond):
        parts = [self.pool(cond["memory"], cond["mask"]).flatten(1), self.summary(cond["summary"])]
        if self.kin is not None:
            parts.append(self.kin(cond["kin"]))
        return torch.cat(parts, 1)


class AuxHead(nn.Module):
    """MLP head for an auxiliary task (see vla.tasks)."""

    def __init__(self, in_dim, cfg, task):
        super().__init__()
        self.task = task
        self.features = Features(in_dim, cfg)
        self.net = mlp([self.features.width] + [cfg.head_dim] * max(1, cfg.head_layers // 2) + [task.out_dim])

    def forward(self, cond):
        return self.net(self.features(cond))

    def loss(self, cond, target):
        return self.task.loss(self(cond), target)

    @torch.no_grad()
    def predict(self, cond):
        return self(cond)


class RegressionHead(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.features = Features(in_dim, cfg)
        out = cfg.horizon * cfg.action_dim
        self.net = mlp([self.features.width] + [cfg.head_dim * 2] * cfg.head_layers + [cfg.modes * out + cfg.modes])

    def _forward(self, cond):
        out = self.net(self.features(cond))
        M, T, D = self.cfg.modes, self.cfg.horizon, self.cfg.action_dim
        traj = out[:, :M * T * D].view(-1, M, T, D)
        return traj, out[:, M * T * D:]

    def loss(self, cond, target):
        traj, logits = self._forward(cond)
        err = F.smooth_l1_loss(traj, target[:, None].expand_as(traj), reduction="none").mean((2, 3))  # (B, modes)
        best = err.argmin(1)
        loss = err.gather(1, best[:, None]).mean()
        if self.cfg.modes > 1:
            loss = loss + F.cross_entropy(logits, best)
        return loss

    @torch.no_grad()
    def predict(self, cond):
        traj, logits = self._forward(cond)
        return traj[torch.arange(len(traj)), logits.argmax(1)]


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


class MLPDenoiser(nn.Module):
    """v(x_t, t | pooled backbone features): residual MLP on the flattened trajectory.

    The noisy trajectory gets its own embedding of the same width as the conditioning;
    fed raw (20 numbers next to thousands of feature dimensions) the network learns to
    ignore it and regresses the mean instead of a velocity field.
    """

    def __init__(self, in_dim, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.head_dim
        self.cond_features = Features(in_dim, cfg)
        self.time = mlp([256, d, d])
        n = cfg.horizon * cfg.action_dim
        self.x_in = mlp([n, d, d])
        self.inp = nn.Linear(self.cond_features.width + 2 * d, d * 2)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, d * 4), nn.GELU(), nn.Linear(d * 4, d * 2))
            for _ in range(cfg.head_layers))
        self.out = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, n))

    def features(self, cond):
        return self.cond_features(cond)

    def forward(self, x, t, cond, feat=None):
        feat = self.features(cond) if feat is None else feat
        h = self.inp(torch.cat([self.x_in(x.flatten(1)), self.time(timestep_embedding(t, 256)), feat], 1))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h).view_as(x)


class DiTBlock(nn.Module):
    """Self-attention over action tokens, cross-attention to backbone tokens, MLP; adaLN-Zero on all three."""

    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim, elementwise_affine=False), nn.LayerNorm(dim, elementwise_affine=False), \
            nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = mlp([dim, dim * 4, dim])
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, memory, mask):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.ada(c)[:, None].chunk(9, dim=2)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.self_attn(h, h, h)[0]
        h = self.n2(x) * (1 + s2) + b2
        x = x + g2 * self.cross_attn(h, memory, memory, key_padding_mask=~mask)[0]
        h = self.n3(x) * (1 + s3) + b3
        return x + g3 * self.mlp(h)


class DiTDenoiser(nn.Module):
    """v(x_t, t | all backbone tokens): a small transformer over the horizon's action tokens."""

    def __init__(self, in_dim, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.head_dim
        self.x_in = nn.Linear(cfg.action_dim, d)
        self.pos = nn.Parameter(torch.randn(cfg.horizon, d) * 0.02)
        self.mem = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d))
        self.cond = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.kin = mlp([KIN_DIM, d, d]) if cfg.kinematic_input else None
        self.time = mlp([256, d, d])
        self.blocks = nn.ModuleList(DiTBlock(d, cfg.head_heads) for _ in range(cfg.head_layers))
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, cfg.action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def features(self, cond):
        summary = self.cond(cond["summary"])
        if self.kin is not None:
            summary = summary + self.kin(cond["kin"])
        return self.mem(cond["memory"]), summary

    def forward(self, x, t, cond, feat=None):
        memory, summary = self.features(cond) if feat is None else feat
        c = self.time(timestep_embedding(t, 256)) + summary
        h = self.x_in(x) + self.pos
        for blk in self.blocks:
            h = blk(h, c, memory, cond["mask"])
        s, b = self.out_ada(c)[:, None].chunk(2, dim=2)
        return self.out(self.out_norm(h) * (1 + s) + b)


class FlowHead(nn.Module):
    """Rectified-flow action head: x_t = (1 - t) noise + t action, learn v = action - noise."""

    def __init__(self, in_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.denoiser = {"mlp": MLPDenoiser, "dit": DiTDenoiser}[cfg.denoiser](in_dim, cfg)

    def loss(self, cond, target):
        B = target.shape[0]
        if self.cfg.time_sampling == "beta":  # emphasise the noisy end, as in pi0
            t = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(target.device)
        else:
            t = torch.rand(B, device=target.device)
        noise = torch.randn_like(target)
        x_t = (1 - t)[:, None, None] * noise + t[:, None, None] * target
        v = self.denoiser(x_t, t, cond)
        return F.mse_loss(v, target - noise)

    @torch.no_grad()
    def sample(self, cond, generator=None):
        B = cond["summary"].shape[0]
        feat = self.denoiser.features(cond)
        x = torch.randn(B, self.cfg.horizon, self.cfg.action_dim, device=cond["summary"].device, generator=generator)
        n = self.cfg.flow_steps
        for i in range(n):
            t = torch.full((B,), i / n, device=x.device)
            x = x + self.denoiser(x, t, cond, feat) / n
        return x

    @torch.no_grad()
    def predict(self, cond):
        samples = torch.stack([self.sample(cond) for _ in range(self.cfg.flow_samples)])  # (S, B, T, D)
        if self.cfg.flow_samples == 1:
            return samples[0]
        # medoid: the sample closest to all others, a point estimate that stays on one mode
        d = (samples[:, None] - samples[None]).flatten(3).norm(dim=3).sum(1)  # (S, B)
        return samples[d.argmin(0), torch.arange(samples.shape[1])]


def build_head(in_dim, cfg):
    return {"regression": RegressionHead, "flow": FlowHead}[cfg.head](in_dim, cfg)
