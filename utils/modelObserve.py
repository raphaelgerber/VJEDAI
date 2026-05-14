#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import DataLoader, random_split
import torch
from pathlib import Path
import sys
import warnings
from huggingface_hub import hf_hub_download
from torchinfo import summary
warnings.filterwarnings("ignore")


TRAIN_DIR = '/cluster/courses/cil/monocular-depth-estimation/train/'
TEST_DIR = '/cluster/courses/cil/monocular-depth-estimation/test'
DEPTH_ANYTHING_MODEL = 'vitl' # vits, vitb, vitl, vitg
BATCH_SIZE = 16


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

PROJECT_ROOT = Path.home().resolve()
VJEPA_ROOT = PROJECT_ROOT / "external" / "vjepa2"
VJEPA_SRC = VJEPA_ROOT / "src"
DA_ROOT = PROJECT_ROOT / "external" / "Depth-Anything-V2"

for p in [str(DA_ROOT), str(PROJECT_ROOT / "external" / "Depth-Anything-V2")]:
    while p in sys.path:
        sys.path.remove(p)

sys.path.insert(0, str(VJEPA_SRC))
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

for module_name in list(sys.modules):
    if module_name == "app" or module_name.startswith("app."):
        del sys.modules[module_name]
vj_encoder, _ = torch.hub.load(
    str(VJEPA_ROOT),
    "vjepa2_1_vit_large_384",
    source="local",
)
vj_encoder = vj_encoder.to(device).eval()
print("VJepa loaded.")

summary(vj_encoder, input_size=(1, 3, 1, 384, 384), device=device)


sys.path.append(str(DA_ROOT))
from depth_anything_v2.dpt import DepthAnythingV2
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

da_encoder = DEPTH_ANYTHING_MODEL
da_checkpoint = hf_hub_download(repo_id="depth-anything/Depth-Anything-V2-Large", filename=f"depth_anything_v2_{da_encoder}.pth")
da_model = DepthAnythingV2(**model_configs[da_encoder])
da_model.load_state_dict(torch.load(da_checkpoint, map_location='cpu'))
da_model = da_model.to(device).eval()
print("DepthAnything loaded.")

summary(da_model, input_size=(1, 3, 392, 392), device=device)