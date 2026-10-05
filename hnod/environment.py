"""Indoor / outdoor estimate per frame with CLIP zero-shot (environment_method "estimated:clip-vit-l14")."""
import numpy as np

MODEL = "openai/clip-vit-large-patch14"
METHOD = "estimated:clip-vit-l14"
INDOOR = ["a photo taken inside a building", "a photo of an indoor room", "a photo of a corridor inside a building",
          "a photo of a shop interior"]
OUTDOOR = ["a photo taken outdoors", "a photo of a street", "a photo of a park", "a photo of a sidewalk outside"]


class IndoorTagger:
    def __init__(self, device="cuda"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch, self.device = torch, device
        self.model = CLIPModel.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(MODEL)
        with torch.no_grad():
            t = self.proc(text=INDOOR + OUTDOOR, return_tensors="pt", padding=True).to(device)
            f = self.model.get_text_features(**t)
            f = getattr(f, "pooler_output", f)
            self.text = f / f.norm(dim=-1, keepdim=True)

    def __call__(self, images_rgb, batch=256):
        """P(indoor) for a list of HxWx3 uint8 RGB arrays."""
        torch, out = self.torch, []
        with torch.no_grad():
            for i in range(0, len(images_rgb), batch):
                x = self.proc(images=[np.ascontiguousarray(im) for im in images_rgb[i:i + batch]],
                              return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
                f = self.model.get_image_features(pixel_values=x)
                f = getattr(f, "pooler_output", f)
                f = f / f.norm(dim=-1, keepdim=True)
                p = (self.model.logit_scale.exp() * f @ self.text.T).float().softmax(-1)
                out.append(p[:, :len(INDOOR)].sum(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def per_frame(prob_sampled, sampled_idx, n, smooth=5):
    """Median-smooth sparse samples and interpolate to every frame."""
    p = np.asarray(prob_sampled, float)
    if len(p) >= smooth:
        pad = np.pad(p, smooth // 2, mode="edge")
        p = np.median(np.lib.stride_tricks.sliding_window_view(pad, smooth), axis=1)
    return np.interp(np.arange(n), sampled_idx, p).astype(np.float32)


def summarise(prob):
    frac = float((np.asarray(prob) > 0.5).mean()) if len(prob) else float("nan")
    env = "unknown" if not np.isfinite(frac) else "indoor" if frac >= 0.8 else "outdoor" if frac <= 0.2 else "mixed"
    return env, frac
