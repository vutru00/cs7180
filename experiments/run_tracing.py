"""Phase 1 — Causal Tracing CLI.

Usage:
    python experiments/run_tracing.py --bias-dim gender --target unet --output results/nie_gender_unet.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so internal packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Causal Tracing")
    parser.add_argument("--prompts", type=str, default="data/occupation_prompts.json",
                        help="Path to prompt dataset JSON.")
    parser.add_argument("--bias-dim", type=str, choices=["gender", "age"], default="gender")
    parser.add_argument("--target", type=str, choices=["unet", "textenc", "both"], default="unet")
    parser.add_argument("--n-images", type=int, default=5,
                        help="Number of images (seeds) per prompt.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for output JSON.")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load prompts
    with open(args.prompts) as f:
        data = json.load(f)

    prompts = [
        {"prompt": p, "subject": s}
        for p, s in zip(data["prompts"], data["subject_tokens"])
    ]

    # Load models
    from models.load_model import load_sd_pipeline, load_mivolo_predictor
    from classifiers.mivolo_classifier import MiVOLOClassifier
    from tracing.nie import run_full_tracing, save_results

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = load_sd_pipeline(device=args.device, dtype=dtype)
    predictor = load_mivolo_predictor(device=args.device)
    classifier = MiVOLOClassifier(predictor)

    # Run tracing
    results = run_full_tracing(
        pipe=pipe,
        classifier=classifier,
        prompts=prompts,
        target=args.target,
        bias_dim=args.bias_dim,
        n_images=args.n_images,
        device=args.device,
    )

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
