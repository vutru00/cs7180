"""Phase 2 — Dynamic (Time-Resolved) Mediation CLI.

Usage:
    python experiments/run_dynamic.py --window early --layers results/top_causal_layers.json \
        --output results/dynamic_early.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Timestep windows for 50-step DDIM (counting down from 999 to 0)
TIMESTEP_WINDOWS = {
    "early": (999, 700),
    "mid": (700, 300),
    "late": (300, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Dynamic Mediation")
    parser.add_argument("--prompts", type=str, default="data/occupation_prompts.json")
    parser.add_argument("--bias-dim", type=str, choices=["gender", "age"], default="gender")
    parser.add_argument("--target", type=str, choices=["unet", "textenc", "both"], default="unet")
    parser.add_argument("--window", type=str, choices=["early", "mid", "late"], required=True,
                        help="Timestep window for restoration.")
    parser.add_argument("--layers", type=str, required=True,
                        help="Path to JSON list of layer names to probe.")
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--output", type=str, required=True)
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

    # Load candidate layers
    with open(args.layers) as f:
        layer_names = json.load(f)

    timestep_window = TIMESTEP_WINDOWS[args.window]

    # Load models
    from models.load_model import load_sd_pipeline, load_mivolo_predictor
    from classifiers.mivolo_classifier import MiVOLOClassifier
    from tracing.nie import run_full_tracing, save_results

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = load_sd_pipeline(device=args.device, dtype=dtype)
    predictor = load_mivolo_predictor(device=args.device)
    classifier = MiVOLOClassifier(predictor)

    results = run_full_tracing(
        pipe=pipe,
        classifier=classifier,
        prompts=prompts,
        target=args.target,
        bias_dim=args.bias_dim,
        n_images=args.n_images,
        timestep_window=timestep_window,
        layer_names=layer_names,
        device=args.device,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
