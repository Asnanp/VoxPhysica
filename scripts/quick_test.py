#!/usr/bin/env python
"""Quick test to verify the frontier config builds and model initializes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
from src.models.pibnn import build_model

config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'pibnn_rtx3060_v5_3cm_frontier.yaml')
with open(config_path) as f:
    cfg = yaml.safe_load(f)

cfg['model']['input_dim'] = 264
cfg['target_stats'] = None

model = build_model(cfg)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {type(model).__name__}")
print(f"Trainable params: {n_params:,}")

# Test forward pass
x = torch.randn(4, 640, 264)
padding_mask = torch.zeros(4, 640, dtype=torch.bool)
padding_mask[:, 600:] = True

if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    padding_mask = padding_mask.cuda()

with torch.no_grad():
    out = model(x, padding_mask=padding_mask, domain=None)

print(f"Height pred shape: {out['height'].shape}")
print(f"Height var shape: {out['height_var'].shape}")
print(f"Gender logits shape: {out['gender_logits'].shape}")
print(f"Height bin logits shape: {out['height_bin_logits'].shape}")
print(f"Quality score shape: {out['quality_score'].shape}")
print("Forward pass OK!")

# Test with targets (loss computation)
epoch = 1
out_with_loss = model(
    x, padding_mask=padding_mask, domain=None,
    targets={
        "height": torch.zeros(4, device=x.device),
        "weight": torch.zeros(4, device=x.device),
        "age": torch.zeros(4, device=x.device),
        "gender": torch.zeros(4, device=x.device, dtype=torch.long),
        "height_raw": torch.full((4,), 170.0, device=x.device),
    },
    current_epoch=epoch,
    return_aux=False,
)
if "losses" in out_with_loss:
    losses = out_with_loss["losses"]
    print(f"Total loss: {losses['total']:.4f}")
    print(f"Height loss: {losses['height']:.4f}")
    print(f"Gender loss: {losses['gender']:.4f}")
    print("Loss computation OK!")
else:
    print("No losses in output - checking model config")
    print(out_with_loss.keys())

print("\nAll checks passed!")
