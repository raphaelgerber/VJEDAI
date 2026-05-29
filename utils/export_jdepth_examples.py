#!/usr/bin/env python
# coding: utf-8
"""Export JepaDepth prediction examples.

Writes one PNG per example into an output folder. Each PNG contains three
side-by-side panels: original RGB image, depth prediction, uncertainty.
Depth uses red for closer and blue for farther. Uncertainty is grayscale where
black means most uncertain within that example.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from infer_jdepth import (
    apply_uncertainty_temperature,
    build_model,
    load_calibration_temperature,
    resolve_ckpt,
    setup_syspath,
)

TEST_DIR = "/cluster/courses/cil/monocular-depth-estimation/test"
DEFAULT_CKPT = "/work/scratch/msayfiddinov/checkpoints/jepa_depth_large/best20_nll.pth"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="path to trained checkpoint (.pth)")
    p.add_argument("--variant", default="large", choices=["base", "large"])
    p.add_argument("--test-dir", default=TEST_DIR)
    p.add_argument("--out-dir", default="jdepth_examples", help="folder for example PNGs")
    p.add_argument("--num-examples", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--panel-width", type=int, default=320)
    p.add_argument(
        "--uncertainty-temperature",
        type=float,
        default=1.0,
        help="Scale predicted uncertainty as sigma *= T.",
    )
    p.add_argument(
        "--calibration-csv",
        default=None,
        help="Optional calibration_jdepth.csv; uses temperature_scale from it if present.",
    )
    return p.parse_args()


def robust_normalize(values, lo=2.0, hi=98.0):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(values[finite], [lo, hi])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(values[finite].min())
        high = float(values[finite].max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    norm = (values - low) / (high - low)
    return np.clip(np.nan_to_num(norm), 0.0, 1.0)


def colorize_depth(depth):
    x = 1.0 - robust_normalize(depth)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return ((np.stack([r, g, b], axis=-1)) * 255.0).astype(np.uint8)


def colorize_uncertainty(log_var):
    sigma = np.exp(0.5 * np.asarray(log_var, dtype=np.float32))
    uncertainty = robust_normalize(sigma)
    gray = ((1.0 - uncertainty) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def image_from_tensor(image):
    image = image.detach().float().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_panel(array, width):
    img = Image.fromarray(array)
    height = max(1, round(img.height * width / img.width))
    return img.resize((width, height), Image.BILINEAR)


def load_checkpoint(model, ckpt_path, variant, device):
    print(f"Loading checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if isinstance(ckpt, dict) and "config" in ckpt:
        ck_variant = ckpt["config"].get("variant")
        if ck_variant and ck_variant != variant:
            sys.exit(
                f"Checkpoint variant={ck_variant} != --variant={variant}; "
                f"rerun with --variant {ck_variant}."
            )
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
    if bad_missing or unexpected:
        sys.exit(
            f"Unexpected/missing keys restoring model: "
            f"missing={bad_missing}, unexpected={unexpected}"
        )
    model.eval()


def main():
    args = parse_args()
    ckpt_path = resolve_ckpt(args)
    csv_temperature = load_calibration_temperature(args.calibration_csv)
    uncertainty_temperature = csv_temperature or args.uncertainty_temperature
    if uncertainty_temperature <= 0:
        sys.exit("--uncertainty-temperature must be positive")
    print(f"Uncertainty temperature: {uncertainty_temperature:.6g}")

    setup_syspath()
    from dataset import TestDataset  # noqa: E402
    from preprocessing import vjepa_preprocessing  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = TestDataset(args.test_dir)
    n = min(args.num_examples, len(dataset))
    if n == 0:
        sys.exit(f"No test images found in {args.test_dir}")
    loader = DataLoader(Subset(dataset, range(n)), batch_size=args.batch_size)

    model = build_model(args.variant, device)
    load_checkpoint(model, ckpt_path, args.variant, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            image_ids = batch.get("id", [f"example_{i:02d}" for i in range(written, written + images.shape[0])])
            h, w = images.shape[-2:]
            out = model(vjepa_preprocessing(images), output_size=(h, w))
            depth = out["depth"]
            log_var = apply_uncertainty_temperature(out["log_var"], uncertainty_temperature)
            if depth.ndim == 4:
                depth = depth.squeeze(1)
            if log_var.ndim == 4:
                log_var = log_var.squeeze(1)

            for image_t, depth_t, log_var_t, image_id in zip(images, depth, log_var, image_ids):
                original = resize_panel(image_from_tensor(image_t), args.panel_width)
                depth_img = resize_panel(
                    colorize_depth(depth_t.detach().float().cpu().numpy()),
                    args.panel_width,
                )
                uncertainty_img = resize_panel(
                    colorize_uncertainty(log_var_t.detach().float().cpu().numpy()),
                    args.panel_width,
                )
                row = Image.new("RGB", (original.width * 3, original.height))
                row.paste(original, (0, 0))
                row.paste(depth_img, (original.width, 0))
                row.paste(uncertainty_img, (original.width * 2, 0))

                safe_id = str(image_id).replace("/", "_").replace(" ", "_")
                out_path = out_dir / f"example_{written:02d}_{safe_id}.png"
                row.save(out_path)
                written += 1

    print(f"Wrote {written} examples to {out_dir}")


if __name__ == "__main__":
    main()
