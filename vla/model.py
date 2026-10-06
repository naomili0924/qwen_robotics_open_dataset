"""Qwen-VL backbone + action head (+ auxiliary task heads), with frozen / LoRA / full backbone training."""
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from .heads import AuxHead, build_head
from .tasks import aux_tasks

INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
OPTIONAL_INPUT_KEYS = ("mm_token_type_ids",)  # Qwen3-VL's processor returns it and the model requires it


def _pad_linear(layer, n, dim):
    """The same linear layer with its output (dim 0) or input (dim 1) size zero-padded to n."""
    shape = list(layer.weight.shape)
    shape[dim] = n
    new = nn.Linear(shape[1], shape[0], bias=layer.bias is not None, device=layer.weight.device,
                    dtype=layer.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:layer.weight.shape[0], :layer.weight.shape[1]] = layer.weight
        if layer.bias is not None:
            new.bias.zero_()
            new.bias[:layer.bias.shape[0]] = layer.bias
    new.weight.requires_grad_(layer.weight.requires_grad)
    if layer.bias is not None:
        new.bias.requires_grad_(layer.bias.requires_grad)
    return new


def pad_vision_mlp(visual, multiple=64):
    """Zero-pad the vision MLPs' hidden width (3420 in Qwen2.5-VL) to a multiple of 64.

    3420 is not a multiple of 8, which sends cuBLAS to slow fallback kernels; the vision tower runs about
    1.6x faster padded.  Mathematically identical: padded gate/up rows are zero (silu(0) * 0 = 0) and the
    padded down-projection columns are zero.
    """
    for blk in getattr(visual, "blocks", []):
        m = blk.mlp
        if not hasattr(m, "gate_proj"):  # other vision towers (e.g. Qwen3-VL) have a plain two-layer MLP
            return
        n = -(-m.gate_proj.out_features // multiple) * multiple
        if n == m.gate_proj.out_features:
            continue
        m.gate_proj, m.up_proj = _pad_linear(m.gate_proj, n, 0), _pad_linear(m.up_proj, n, 0)
        m.down_proj = _pad_linear(m.down_proj, n, 1)


class NavPolicy(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.processor = AutoProcessor.from_pretrained(cfg.backbone)
        full = AutoModelForImageTextToText.from_pretrained(
            cfg.backbone, dtype=torch.float32 if cfg.backbone_mode == "full" else torch.bfloat16, device_map=device,
            **({"attn_implementation": cfg.attn_implementation} if cfg.attn_implementation else {}))
        self.backbone = full.model  # vision encoder + language model, without the vocabulary head
        del full.lm_head
        hidden = self.backbone.config.text_config.hidden_size
        if cfg.backbone_mode == "frozen":
            self.backbone.requires_grad_(False)
        elif cfg.backbone_mode == "lora":
            from peft import LoraConfig, get_peft_model
            targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
            if cfg.tune_vision:
                targets = (r".*(language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                           r"|visual\.blocks.*\.(qkv|proj|gate_proj|up_proj|down_proj))$")
            self.backbone = get_peft_model(self.backbone, LoraConfig(
                r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=targets))
        elif cfg.backbone_mode == "full":
            self.backbone.requires_grad_(True)
            if not cfg.tune_vision:
                self.backbone.visual.requires_grad_(False)
        else:
            raise ValueError(cfg.backbone_mode)
        if cfg.gradient_checkpointing and cfg.backbone_mode != "frozen":
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if not cfg.tune_vision:
            # A frozen vision tower still builds a 20+ GB autograd graph when called in grad mode
            # (its outputs feed trainable layers); run it under no_grad instead.
            visual = self.inner.visual
            visual.forward = torch.no_grad()(visual.forward)
        if cfg.pad_vision_mlp:
            pad_vision_mlp(self.inner.visual)
        self.hidden = hidden
        self.head = build_head(hidden, cfg).to(device)
        self.aux = nn.ModuleDict({t.name: AuxHead(hidden, cfg, t).to(device) for t in aux_tasks(cfg.tasks)})
        if cfg.init_from:
            self.load(cfg.init_from, strict=False)
        if cfg.freeze_backbone:
            self.backbone.requires_grad_(False)

    @property
    def inner(self):
        """The underlying Qwen2_5_VLModel, with or without the PEFT wrapper."""
        return self.backbone.base_model.model if self.cfg.backbone_mode == "lora" else self.backbone

    @property
    def backbone_trains(self):
        return any(p.requires_grad for p in self.backbone.parameters())

    def add_task(self, task):
        """Attach a new auxiliary head (see vla.tasks) to the shared embedding."""
        self.aux[task.name] = AuxHead(self.hidden, self.cfg, task).to(self.device)
        names = [s for s in self.cfg.tasks.split(",") if s]
        if task.name not in names:
            self.cfg.tasks = ",".join(names + [task.name])

    # ------------------------------------------------------------------ conditioning
    def encode(self, batch):
        """Backbone tokens for the heads: summary (B,H) at the last prompt token, memory (B,L,H), mask, kin."""
        inputs = {k: batch[k] for k in INPUT_KEYS}
        inputs.update({k: batch[k] for k in OPTIONAL_INPUT_KEYS if k in batch})
        inputs["use_cache"] = False
        # no graph under predict()'s no_grad (enable_grad here used to re-enable it and cost 10+ GB at inference)
        ctx = torch.enable_grad() if self.backbone_trains and torch.is_grad_enabled() else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.backbone(**inputs).last_hidden_state
        mask = batch["attention_mask"].bool()
        last = mask.sum(1) - 1
        return dict(summary=hs[torch.arange(len(hs)), last].float(), memory=hs.float(), mask=mask,
                    kin=batch["kin"].float())

    def losses(self, batch):
        cond = self.encode(batch)
        out = {}
        if "trajectory" in self.cfg.tasks:
            out["trajectory"] = self.head.loss(cond, batch["target"].float())
        for name, head in self.aux.items():
            out[name] = head.task.weight * head.loss(cond, batch["aux"][name].float())
        return out

    def loss(self, batch):
        return sum(self.losses(batch).values())

    @torch.no_grad()
    def predict(self, batch):
        """{'trajectory': (B, horizon, action_dim) in metres, <task>: raw head output (B, out_dim), ...}"""
        cond = self.encode(batch)
        out = {"trajectory": self.head.predict(cond) * self.cfg.action_scale}
        for name, head in self.aux.items():
            out[name] = head.predict(cond)
        return out

    # ------------------------------------------------------------------ checkpoints
    def trainable_parameters(self):
        heads = list(self.head.parameters()) + [p for h in self.aux.values() for p in h.parameters()]
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        return heads, backbone

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), d / "head.pt")
        torch.save({k: v.state_dict() for k, v in self.aux.items()}, d / "aux_heads.pt")
        if self.cfg.backbone_mode == "lora":
            self.backbone.save_pretrained(d / "lora", selected_adapters=["default"])
        elif self.cfg.backbone_mode == "full":
            # the trained weights only (a frozen vision tower is the base model's), in bf16: about 6 GB for the 3B model
            from safetensors.torch import save_file
            trained = {n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in self.backbone.named_parameters()
                       if p.requires_grad}
            save_file(trained, str(d / "backbone.safetensors"))
        self.cfg.save(d / "config.json")

    def load(self, directory, strict=True):
        """Load weights saved by `save`.  strict=False skips heads that do not exist in the checkpoint."""
        d = Path(directory)
        missing, unexpected = self.head.load_state_dict(torch.load(d / "head.pt", map_location=self.device), strict=False)
        if unexpected or [k for k in missing if k != "log_std"]:
            raise RuntimeError(f"action head mismatch for {d}: missing {missing}, unexpected {unexpected}")
        aux = torch.load(d / "aux_heads.pt", map_location=self.device) if (d / "aux_heads.pt").exists() else {}
        for name, head in self.aux.items():
            if name in aux:
                head.load_state_dict(aux[name])
            elif strict:
                raise KeyError(f"checkpoint {d} has no head for task {name!r}")
        if self.cfg.backbone_mode == "lora" and (d / "lora").exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(self.backbone, load_file(d / "lora" / "adapter_model.safetensors"))
        elif self.cfg.backbone_mode == "full" and (d / "backbone.safetensors").exists():
            from safetensors.torch import load_file
            state = load_file(str(d / "backbone.safetensors"), device=str(self.device))
            own = dict(self.backbone.named_parameters())
            unknown = [k for k in state if k not in own]
            if unknown:
                raise RuntimeError(f"backbone weights not in the model: {unknown[:5]}")
            with torch.no_grad():
                for k, v in state.items():
                    own[k].copy_(v.to(own[k].dtype))
        elif self.cfg.backbone_mode == "full" and (d / "backbone.pt").exists():
            self.backbone.load_state_dict(torch.load(d / "backbone.pt", map_location=self.device))
        return self

    def describe(self):
        heads, backbone = self.trainable_parameters()
        n_head = sum(p.numel() for p in heads)
        n_bb = sum(p.numel() for p in backbone)
        n_all = sum(p.numel() for p in self.backbone.parameters()) + n_head
        action = self.cfg.head + ("/" + self.cfg.denoiser if self.cfg.head == "flow" else "")
        return (f"backbone {self.cfg.backbone} ({self.cfg.backbone_mode}{', frozen' if self.cfg.freeze_backbone else ''}), "
                f"action head {action}, tasks {self.cfg.tasks}: trainable {n_head / 1e6:.1f}M heads + "
                f"{n_bb / 1e6:.1f}M backbone of {n_all / 1e9:.2f}B")
