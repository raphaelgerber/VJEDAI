#!/usr/bin/env python
# coding: utf-8
"""Train the JepaDepthAnything architecture end-to-end.

Mirrors the structure of Full_Pipeline_For_Cluster.py but swaps the
DA + cross-attention fusion model for the unified JepaDepthAnything model
(VJEPA 2.1 encoder + DA DPT decoder + Gaussian uncertainty head).

Two training modes are supported via ``LOSS_MODE`` (or the
``JDEPTH_LOSS_MODE`` env var):

* ``"si_mse"``: scale-invariant MSE on log depths. Ignores the predicted
  log-variance (the uncertainty head still runs forward but receives zero
  gradients, so it stays at its zero-init). Use this as a deterministic
  Stage 1 to get a clean depth-only baseline.
* ``"nll"`` (default): scale-invariant Gaussian NLL using the predicted
  per-pixel log-variance. Use this for a Stage 2 fine-tune that adds the
  uncertainty head on top of an already-good depth model.

Two-stage workflow (set env vars in the sbatch, no script edits needed):
    JDEPTH_LOSS_MODE=si_mse sbatch train_jdepth.sbatch
    JDEPTH_LOSS_MODE=nll \\
      JDEPTH_INIT_FROM=$SCRATCH/checkpoints/jepa_depth_large/best_si_mse.pth \\
      sbatch train_jdepth.sbatch

Validation SI-RMSE (the submission metric) is used for checkpoint
selection in both modes, so Stage 1 and Stage 2 'best' checkpoints are
directly comparable. Checkpoint filenames are mode-tagged
(``best_{LOSS_MODE}.pth`` / ``last_{LOSS_MODE}.pth``) to avoid clobbering.

Switch ``VARIANT`` between ``"base"`` and ``"large"`` to pick the
VJEPA+DA pairing; everything else (out_layers, embed dim, checkpoint
filename, repo id) is derived automatically from
``src.jepa_depth_anything.VARIANTS``.
"""

import os
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

TRAIN_DIR = "/cluster/courses/cil/monocular-depth-estimation/train/"
TEST_DIR = "/cluster/courses/cil/monocular-depth-estimation/test"

VARIANT = "large"       # "base" or "large"
BATCH_SIZE = 8          # tune per GPU; 8 is safe for vit-l on a single 5060 Ti
NUM_EPOCHS = 100
PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1
NUM_WORKERS = 4

# "si_mse": deterministic Stage 1, ignores the uncertainty head.
# "nll": heteroscedastic Stage 2, uses predicted log-variance.
LOSS_MODE = os.environ.get("JDEPTH_LOSS_MODE", "nll").lower()
if LOSS_MODE not in ("nll", "si_mse"):
    raise ValueError(
        f"Unknown LOSS_MODE={LOSS_MODE!r}; expected 'nll' or 'si_mse'."
    )

# Optional path to a Stage 1 checkpoint to warm-start from. Loaded with
# strict=False, so missing vjepa_encoder.* keys are fine (the encoder is
# loaded fresh from torch.hub). Set via env var to avoid editing this
# file each time you flip stages.
INIT_FROM = os.environ.get("JDEPTH_INIT_FROM", "").strip() or None
RESUME_FROM = os.environ.get("JDEPTH_RESUME_FROM", "").strip() or None
AUTO_RESUME = os.environ.get("JDEPTH_AUTO_RESUME", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Keep heavy checkpoints off $HOME: write them to $SCRATCH (the sbatch
# already points $HF_HOME/$TORCH_HOME under $SCRATCH/cache/, so we use a
# sibling $SCRATCH/checkpoints/ subtree to avoid colliding with caches).
# Falls back to a local ./checkpoints/ if $SCRATCH isn't set (e.g., laptop).
SCRATCH = Path(os.environ.get("SCRATCH", "."))
CHECKPOINT_DIR = SCRATCH / "checkpoints" / f"jepa_depth_{VARIANT}"
# Mode-tagged so Stage 1 (si_mse) and Stage 2 (nll) don't overwrite each
# other when run back-to-back from the same $SCRATCH.
BEST_CKPT = CHECKPOINT_DIR / f"best_{LOSS_MODE}.pth"
LAST_CKPT = CHECKPOINT_DIR / f"last_{LOSS_MODE}.pth"
SUBMISSION_CSV = Path("./submission.csv")


# -----------------------------------------------------------------------------
# Paths / sys.path setup
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path.home().resolve()
MONO_ROOT = Path(__file__).resolve().parent
SRC_DIR = MONO_ROOT / "src"
VJEPA_ROOT = PROJECT_ROOT / "external" / "vjepa2"
VJEPA_SRC = VJEPA_ROOT / "src"
DA_ROOT = PROJECT_ROOT / "external" / "Depth-Anything-V2"

for p in [str(DA_ROOT)]:
    while p in sys.path:
        sys.path.remove(p)

for p in [str(SRC_DIR), str(VJEPA_SRC), str(VJEPA_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

for module_name in list(sys.modules):
    if module_name == "app" or module_name.startswith("app."):
        del sys.modules[module_name]


# -----------------------------------------------------------------------------
# Imports that depend on sys.path
# -----------------------------------------------------------------------------

from dataset import TrainDataset, TestDataset           # noqa: E402
from preprocessing import vjepa_preprocessing           # noqa: E402
from create_submission import encode_depth, save_submission  # noqa: E402


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

print("loading data...")
train_dataset = TrainDataset(TRAIN_DIR)
val_size = int(len(train_dataset) * VAL_FRACTION)
train_size = len(train_dataset) - val_size
train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(0),
)
train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_dataset = TestDataset(TEST_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print(f"data loaded. train: {train_size} | val: {val_size} | test: {len(test_dataset)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# -----------------------------------------------------------------------------
# Build model
# -----------------------------------------------------------------------------

# IMPORTANT: DA_ROOT must NOT be on sys.path while VJEPA's hub.load runs.
# Inside the hub call, ``from app.vjepa_2_1.models import ...`` resolves
# ``app`` via sys.path. VJEPA ships ``vjepa2/app/`` WITHOUT __init__.py,
# making it only a PEP 420 namespace-package portion -- which never
# short-circuits the search. DA-V2 ships a regular ``app.py`` at its repo
# root, so Python prefers DA's app.py over VJEPA's namespace package
# regardless of sys.path order. The fix is to keep DA off sys.path until
# after the VJEPA hub.load returns.
for p in [str(DA_ROOT), str(MONO_ROOT / "external" / "Depth-Anything-V2")]:
    while p in sys.path:
        sys.path.remove(p)
for module_name in list(sys.modules):
    if module_name == "app" or module_name.startswith("app."):
        del sys.modules[module_name]

# jepa_depth_anything imports depth_anything_v2 lazily (inside
# JepaDepthAnything.__init__), so importing this module here does not yet
# require DA_ROOT on sys.path.
from jepa_depth_anything import VARIANTS, build_jepa_depth_anything  # noqa: E402

cfg = VARIANTS[VARIANT]
print(f"variant={VARIANT} | vjepa_arch={cfg['vjepa_arch']} | da_encoder={cfg['da_encoder']}")

vj_encoder, _ = torch.hub.load(
    str(VJEPA_ROOT),
    cfg["vjepa_arch"],
    source="local",
    out_layers=cfg["vjepa_out_layers"],
)
vj_encoder = vj_encoder.to(device).eval()
print(f"VJEPA ({cfg['vjepa_arch']}) loaded.")

# Now it's safe to expose DA so JepaDepthAnything can import its DPT head.
sys.path.append(str(DA_ROOT))

model = build_jepa_depth_anything(vj_encoder, variant=VARIANT, device=device)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"JepaDepthAnything built. trainable: {n_trainable/1e6:.2f}M | frozen: {n_frozen/1e6:.2f}M")
print(f"Loss mode: {LOSS_MODE}")

if INIT_FROM is not None:
    print(f"Warm-starting from {INIT_FROM}...")
    init_ckpt = torch.load(INIT_FROM, map_location=device)
    init_state = (
        init_ckpt["model_state_dict"]
        if isinstance(init_ckpt, dict) and "model_state_dict" in init_ckpt
        else init_ckpt
    )
    missing, unexpected = model.load_state_dict(init_state, strict=False)
    non_vjepa_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
    if non_vjepa_missing:
        sample = non_vjepa_missing[:5]
        suffix = "..." if len(non_vjepa_missing) > 5 else ""
        print(f"  WARN missing non-vjepa keys: {sample}{suffix}")
    if unexpected:
        sample = unexpected[:5]
        suffix = "..." if len(unexpected) > 5 else ""
        print(f"  WARN unexpected keys: {sample}{suffix}")
    print(
        f"  warm-start loaded. missing(non-vjepa)={len(non_vjepa_missing)} "
        f"unexpected={len(unexpected)}"
    )


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def _flatten_to_BHW(*tensors):
    out = []
    for t in tensors:
        if t.ndim == 4:
            t = t.squeeze(1)
        out.append(t)
    return out


def scale_invariant_rmse(pred, target, eps=1e-6):
    """Scale-invariant RMSE on log depths (validation metric)."""
    pred, target = _flatten_to_BHW(pred, target)
    mask = target > eps
    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    log_diff = (torch.log(pred) - torch.log(target))[mask]
    bias = -torch.mean(log_diff)
    return torch.sqrt(torch.mean((log_diff + bias) ** 2))


def scale_invariant_gaussian_nll(pred, target, log_var, eps=1e-6):
    """Scale-invariant Gaussian NLL using the predicted log-variance.

    The per-batch mean log-depth residual is removed to enforce scale
    invariance (matching the SI-RMSE objective). The predicted log-variance
    then calibrates the per-pixel noise.
    """
    pred, target, log_var = _flatten_to_BHW(pred, target, log_var)

    mask = target > eps
    if not mask.any():
        return log_var.sum() * 0.0  # safe zero with grad

    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    log_diff = torch.log(pred) - torch.log(target)

    bias = -torch.mean(log_diff[mask])
    residual = log_diff + bias  # zero-mean within mask

    # Tight clamp on log_var: previously [-10, 10] allowed variance up to
    # ~22000, which is an "infinity wins" attractor for the optimizer --
    # the loss saturated at 0.5 * 10 = 5.0 with all gradients dead. With
    # max=3, the collapsed plateau is only 0.5 * 3 = 1.5, so the model
    # actually has to fit depths to do better.
    log_var = torch.clamp(log_var, min=-7.0, max=3.0)
    inv_var = torch.exp(-log_var)
    nll = 0.5 * (log_var + residual ** 2 * inv_var)
    return torch.mean(nll[mask])


def scale_invariant_mse(pred, target, eps=1e-6):
    """Scale-invariant MSE on log depths (Stage 1 / no-uncertainty loss).

    Equivalent to ``scale_invariant_gaussian_nll`` with ``log_var = 0``
    (fixed variance 1, constants dropped): the per-batch mean log-depth
    residual is removed for scale invariance, then the squared residual is
    averaged over valid pixels. Use this to train a clean depth backbone
    without the heteroscedastic head; transition to NLL via a warm-start
    once depths are good.
    """
    pred, target = _flatten_to_BHW(pred, target)

    mask = target > eps
    if not mask.any():
        return pred.sum() * 0.0  # safe zero with grad

    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    log_diff = torch.log(pred) - torch.log(target)

    bias = -torch.mean(log_diff[mask])
    residual = log_diff + bias

    return torch.mean((residual ** 2)[mask])


def compute_loss(out, targets):
    """Dispatch to the active training loss based on ``LOSS_MODE``."""
    if LOSS_MODE == "nll":
        return scale_invariant_gaussian_nll(out["depth"], targets, out["log_var"])
    if LOSS_MODE == "si_mse":
        return scale_invariant_mse(out["depth"], targets)
    raise ValueError(f"Unknown LOSS_MODE={LOSS_MODE!r}")


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


def trainable_state_dict(m):
    """State dict without the frozen VJEPA encoder (saves a lot of disk)."""
    return {
        k: v for k, v in m.state_dict().items()
        if not k.startswith("vjepa_encoder.")
    }


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print("Starting training...")

best_val_rmse = float("inf")
epochs_without_improvement = 0
start_epoch = 0
last_epoch = -1
mean_train_loss = float("nan")
mean_val_loss = float("nan")
mean_val_rmse = float("nan")

resume_path = Path(RESUME_FROM) if RESUME_FROM else (LAST_CKPT if AUTO_RESUME else None)
if resume_path is not None and resume_path.exists():
    print(f"Resuming training from {resume_path}...")
    resume_ckpt = torch.load(resume_path, map_location=device)
    missing, unexpected = model.load_state_dict(
        resume_ckpt["model_state_dict"], strict=False
    )
    bad_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Unexpected/missing keys when resuming: "
            f"missing={bad_missing}, unexpected={unexpected}"
        )
    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    start_epoch = resume_ckpt["epoch"] + 1
    last_epoch = resume_ckpt["epoch"]
    mean_train_loss = resume_ckpt.get("train_loss", mean_train_loss)
    mean_val_loss = resume_ckpt.get("val_loss", mean_val_loss)
    mean_val_rmse = resume_ckpt.get("val_rmse", mean_val_rmse)
    best_val_rmse = resume_ckpt.get(
        "best_val_rmse", resume_ckpt.get("val_rmse", float("inf"))
    )
    epochs_without_improvement = resume_ckpt.get("epochs_without_improvement", 0)
    print(
        f"  resume loaded. next epoch={start_epoch + 1} | "
        f"best val si-rmse={best_val_rmse:.6f}"
    )
elif resume_path is not None:
    print(f"Resume checkpoint not found at {resume_path}; starting fresh.")

for epoch in range(start_epoch, NUM_EPOCHS):
    last_epoch = epoch
    model.train()

    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        targets = batch["depth"].to(device)

        optimizer.zero_grad()

        H, W = images.shape[-2:]
        out = model(vjepa_preprocessing(images), output_size=(H, W))

        loss = compute_loss(out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            total_rmse += scale_invariant_rmse(out["depth"], targets).item()
        num_batches += 1
        print(f"batch {num_batches}: {LOSS_MODE} = {loss.item():.6f}")

    mean_train_loss = total_loss / max(num_batches, 1)
    mean_train_rmse = total_rmse / max(num_batches, 1)

    model.eval()
    val_loss_sum = 0.0
    val_rmse_sum = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["depth"].to(device)

            H, W = images.shape[-2:]
            out = model(vjepa_preprocessing(images), output_size=(H, W))

            val_loss_sum += compute_loss(out, targets).item()
            val_rmse_sum += scale_invariant_rmse(out["depth"], targets).item()
            val_batches += 1

    mean_val_loss = val_loss_sum / max(val_batches, 1)
    mean_val_rmse = val_rmse_sum / max(val_batches, 1)

    print(
        f"Epoch {epoch + 1} | "
        f"train {LOSS_MODE}: {mean_train_loss:.6f} si-rmse: {mean_train_rmse:.6f} | "
        f"val {LOSS_MODE}: {mean_val_loss:.6f} si-rmse: {mean_val_rmse:.6f}"
    )

    # Select on val SI-RMSE (the submission metric) so Stage 1 (si_mse) and
    # Stage 2 (nll) checkpoints are picked by the same criterion and are
    # directly comparable.
    if mean_val_rmse < best_val_rmse:
        best_val_rmse = mean_val_rmse
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": trainable_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": mean_val_loss,
                "val_rmse": mean_val_rmse,
                "best_val_rmse": best_val_rmse,
                "config": {"variant": VARIANT, "loss_mode": LOSS_MODE},
            },
            BEST_CKPT,
        )
        print(f"Saved new best model: val si-rmse {mean_val_rmse:.6f}")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement}/{PATIENCE} epochs")
        if epochs_without_improvement >= PATIENCE:
            print("Early stopping")
            break


# -----------------------------------------------------------------------------
# Save last
# -----------------------------------------------------------------------------

torch.save(
    {
        "model_state_dict": trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": last_epoch,
        "train_loss": mean_train_loss,
        "val_loss": mean_val_loss,
        "val_rmse": mean_val_rmse,
        "best_val_rmse": best_val_rmse,
        "epochs_without_improvement": epochs_without_improvement,
        "config": {"variant": VARIANT, "loss_mode": LOSS_MODE},
    },
    LAST_CKPT,
)


# -----------------------------------------------------------------------------
# Inference for submission (uses the best checkpoint)
# -----------------------------------------------------------------------------

print(f"Loading best checkpoint from {BEST_CKPT}...")
checkpoint = torch.load(BEST_CKPT, map_location=device)
ckpt_variant = checkpoint["config"]["variant"]

# Reuse the in-memory VJEPA encoder; skip DA download since trained weights
# are about to overwrite depth_head anyway.
inference_model = build_jepa_depth_anything(
    vj_encoder,
    variant=ckpt_variant,
    device=device,
    load_da_pretrained=False,
)
missing, unexpected = inference_model.load_state_dict(
    checkpoint["model_state_dict"], strict=False,
)
bad_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
if bad_missing or unexpected:
    raise RuntimeError(
        f"Unexpected/missing keys when restoring trained model: "
        f"missing={bad_missing}, unexpected={unexpected}"
    )
inference_model.eval()

rows = []
with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)
        image_ids = batch["id"]
        H, W = images.shape[-2:]

        out = inference_model(vjepa_preprocessing(images), output_size=(H, W))
        pred_depths = out["depth"]
        if pred_depths.ndim == 4:
            pred_depths = pred_depths.squeeze(1)
        pred_depths = pred_depths.detach().cpu().numpy()

        for depth, image_id in zip(pred_depths, image_ids):
            rows.append({"id": f"{image_id}_depth", "Depths": encode_depth(depth)})

save_submission(rows, str(SUBMISSION_CSV))
print(f"Done. Submission written to {SUBMISSION_CSV} ({len(rows)} rows).")
