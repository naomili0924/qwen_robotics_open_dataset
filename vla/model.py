"""Qwen-VL backbone + action head (+ auxiliary task heads), with frozen / LoRA / full backbone training."""
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from .heads import AuxHead, build_head
from .tasks import aux_tasks

INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")


class NavPolicy(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.processor = AutoProcessor.from_pretrained(cfg.backbone)
        full = AutoModelForImageTextToText.from_pretrained(
            cfg.backbone, dtype=torch.float32 if cfg.backbone_mode == "full" else torch.bfloat16, device_map=device)
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
        inputs["use_cache"] = False
        ctx = torch.enable_grad() if self.backbone_trains else torch.no_grad()
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
            self.backbone.save_pretrained(d / "lora")
        elif self.cfg.backbone_mode == "full":
            torch.save(self.backbone.state_dict(), d / "backbone.pt")
        self.cfg.save(d / "config.json")

    def load(self, directory, strict=True):
        """Load weights saved by `save`.  strict=False skips heads that do not exist in the checkpoint."""
        d = Path(directory)
        self.head.load_state_dict(torch.load(d / "head.pt", map_location=self.device))
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
