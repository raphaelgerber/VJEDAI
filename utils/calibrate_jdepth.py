#!/usr/bin/env python
# coding: utf-8
"""Evaluate uncertainty calibration for a trained JepaDepth checkpoint.

This script recreates the same validation split used by
Full_Pipeline_JepaDepth.py, loads a checkpoint, and measures whether the
predicted per-pixel log-variance is calibrated for scale-invariant log-depth
residuals.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")

TRAIN_DIR = "/cluster/courses/cil/monocular-depth-estimation/train/"
VAL_FRACTION = 0.1
BATCH_SIZE = 8
NUM_WORKERS = 4
VARIANT = "large"
LOG_VAR_MIN = -7.0
LOG_VAR_MAX = 3.0
CALIBRATION_LEVELS = (0.50, 0.68, 0.90, 0.95)
CALIBRATION_Z = {
    0.50: 0.67448975,
    0.68: 0.99445788,
    0.90: 1.64485363,
    0.95: 1.95996398,
}

MONO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path.home().resolve()
SRC_DIR = MONO_ROOT / "src"


def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_checkpoint():
    scratch = Path(os.environ.get("SCRATCH", "/work/scratch/msayfiddinov"))
    ckpt_dir = scratch / "checkpoints" / "jepa_depth_large"
    for name in ("best_nll.pth", "best.pth"):
        path = ckpt_dir / name
        if path.exists():
            return path
    return ckpt_dir / "best_nll.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibration study for trained JepaDepth uncertainty."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("JDEPTH_CKPT", default_checkpoint())),
        help="Path to a trained checkpoint. Defaults to $SCRATCH best_nll/best.",
    )
    parser.add_argument("--train-dir", type=Path, default=Path(TRAIN_DIR))
    parser.add_argument("--variant", default=None, choices=("base", "large"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--output", type=Path, default=Path("calibration_jdepth.csv"))
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def setup_paths():
    vjepa_root = Path(os.environ["VJEPA_ROOT"]) if "VJEPA_ROOT" in os.environ else first_existing(
        MONO_ROOT / "external" / "vjepa2",
        PROJECT_ROOT / "external" / "vjepa2",
    )
    da_root = Path(os.environ["DA_ROOT"]) if "DA_ROOT" in os.environ else first_existing(
        MONO_ROOT / "external" / "Depth-Anything-V2",
        PROJECT_ROOT / "external" / "Depth-Anything-V2",
    )

    for p in [str(da_root)]:
        while p in sys.path:
            sys.path.remove(p)

    for p in [str(SRC_DIR), str(vjepa_root / "src"), str(vjepa_root), str(PROJECT_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            del sys.modules[module_name]

    return vjepa_root, da_root


def flatten_to_bhw(*tensors):
    out = []
    for t in tensors:
        if t.ndim == 4:
            t = t.squeeze(1)
        out.append(t)
    return out


def load_validation_loader(train_dir, val_fraction, batch_size, num_workers):
    from dataset import TrainDataset

    dataset = TrainDataset(train_dir)
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    _, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, val_size


def build_model_from_checkpoint(checkpoint_path, requested_variant, device, vjepa_root, da_root):
    from jepa_depth_anything import VARIANTS, build_jepa_depth_anything

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    variant = requested_variant or config.get("variant", VARIANT)
    cfg = VARIANTS[variant]

    print(f"Loading VJEPA {cfg['vjepa_arch']} from {vjepa_root}...")
    vj_encoder, _ = torch.hub.load(
        str(vjepa_root),
        cfg["vjepa_arch"],
        source="local",
        out_layers=cfg["vjepa_out_layers"],
    )
    vj_encoder = vj_encoder.to(device).eval()

    sys.path.append(str(da_root))
    model = build_jepa_depth_anything(
        vj_encoder,
        variant=variant,
        device=device,
        load_da_pretrained=False,
    )
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Unexpected/missing keys when restoring model: "
            f"missing={bad_missing}, unexpected={unexpected}"
        )
    model.eval()
    return model, variant


def evaluate_calibration(model, loader, device, max_batches=None, eps=1e-6):
    from preprocessing import vjepa_preprocessing

    count = 0
    residual_sum_sq = 0.0
    sigma_sum = 0.0
    sigma_sq_sum = 0.0
    log_var_sum = 0.0
    nll_sum = 0.0
    z_sum = 0.0
    z_sum_sq = 0.0
    abs_z_sum = 0.0
    abs_res_sum = 0.0
    sigma_abs_res_sum = 0.0
    coverage_hits = {level: 0 for level in CALIBRATION_LEVELS}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            images = batch["image"].to(device)
            targets = batch["depth"].to(device)
            H, W = images.shape[-2:]
            out = model(vjepa_preprocessing(images), output_size=(H, W))

            pred, target, log_var = flatten_to_bhw(out["depth"], targets, out["log_var"])
            mask = target > eps
            if not mask.any():
                continue

            pred = torch.clamp(pred, min=eps)
            target = torch.clamp(target, min=eps)
            log_diff = torch.log(pred) - torch.log(target)
            residual = log_diff - torch.mean(log_diff[mask])
            log_var = torch.clamp(log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
            sigma = torch.exp(0.5 * log_var)

            residual = residual[mask]
            log_var = log_var[mask]
            sigma = sigma[mask]
            z = residual / torch.clamp(sigma, min=eps)
            abs_residual = torch.abs(residual)

            n = residual.numel()
            count += n
            residual_sum_sq += torch.sum(residual ** 2).item()
            sigma_sum += torch.sum(sigma).item()
            sigma_sq_sum += torch.sum(sigma ** 2).item()
            log_var_sum += torch.sum(log_var).item()
            nll_sum += torch.sum(0.5 * (log_var + z ** 2)).item()
            z_sum += torch.sum(z).item()
            z_sum_sq += torch.sum(z ** 2).item()
            abs_z_sum += torch.sum(torch.abs(z)).item()
            abs_res_sum += torch.sum(abs_residual).item()
            sigma_abs_res_sum += torch.sum(sigma * abs_residual).item()

            for level, z_value in CALIBRATION_Z.items():
                coverage_hits[level] += torch.sum(torch.abs(z) <= z_value).item()

    if count == 0:
        return {"num_pixels": 0}

    empirical_rmse = (residual_sum_sq / count) ** 0.5
    mean_sigma = sigma_sum / count
    rms_sigma = (sigma_sq_sum / count) ** 0.5
    mean_abs_res = abs_res_sum / count
    mean_z = z_sum / count
    z_std = max(z_sum_sq / count - mean_z ** 2, 0.0) ** 0.5
    mean_abs_z = abs_z_sum / count

    cov_sigma_abs_res = sigma_abs_res_sum / count - mean_sigma * mean_abs_res
    var_sigma = max(sigma_sq_sum / count - mean_sigma ** 2, 0.0)
    var_abs_res = max(residual_sum_sq / count - mean_abs_res ** 2, 0.0)
    corr = 0.0
    denom = (var_sigma * var_abs_res) ** 0.5
    if denom > 0:
        corr = cov_sigma_abs_res / denom

    stats = {
        "num_pixels": count,
        "si_log_residual_rmse": empirical_rmse,
        "mean_pred_sigma": mean_sigma,
        "rms_pred_sigma": rms_sigma,
        "rmse_over_rms_sigma": empirical_rmse / max(rms_sigma, eps),
        "mean_log_var": log_var_sum / count,
        "mean_gaussian_nll": nll_sum / count,
        "normalized_residual_mean": mean_z,
        "normalized_residual_std": z_std,
        "normalized_residual_mae": mean_abs_z,
        "abs_error_sigma_corr": corr,
        "mean_abs_si_log_residual": mean_abs_res,
    }
    ece = 0.0
    for level in CALIBRATION_LEVELS:
        coverage = coverage_hits[level] / count
        stats[f"coverage_{int(level * 100)}"] = coverage
        stats[f"coverage_gap_{int(level * 100)}"] = coverage - level
        ece += abs(coverage - level)
    stats["coverage_ece"] = ece / len(CALIBRATION_LEVELS)
    return stats


def save_stats(stats, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for key, value in stats.items():
            f.write(f"{key},{value}\n")


def main():
    args = parse_args()
    vjepa_root, da_root = setup_paths()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    val_loader, val_size = load_validation_loader(
        args.train_dir,
        args.val_fraction,
        args.batch_size,
        args.num_workers,
    )
    print(f"Validation split: {val_size} samples")

    model, variant = build_model_from_checkpoint(
        args.checkpoint,
        args.variant,
        device,
        vjepa_root,
        da_root,
    )
    print(f"Model restored. variant={variant}")

    stats = evaluate_calibration(model, val_loader, device, args.max_batches)
    save_stats(stats, args.output)

    if stats.get("num_pixels", 0) == 0:
        print("No valid depth pixels found.")
        return

    print(
        "Calibration | "
        f"norm-std: {stats['normalized_residual_std']:.4f} | "
        f"rmse/rms-sigma: {stats['rmse_over_rms_sigma']:.4f} | "
        f"ece: {stats['coverage_ece']:.4f} | "
        f"corr(abs err, sigma): {stats['abs_error_sigma_corr']:.4f}"
    )
    print(
        "Coverage | "
        f"50: {stats['coverage_50']:.4f} | "
        f"68: {stats['coverage_68']:.4f} | "
        f"90: {stats['coverage_90']:.4f} | "
        f"95: {stats['coverage_95']:.4f}"
    )
    print(f"Metrics written to {args.output}")


if __name__ == "__main__":
    main()
