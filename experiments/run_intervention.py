"""Phase 3 — Intervention CLI.

Usage:
    python experiments/run_intervention.py --method hard_block \
        --causal-layers results/top_causal_layers.json --output results/intervention.csv \
        --visualize
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TIMESTEP_WINDOWS = {
    "early": (999, 900),
    "mid": (700, 300),
    "late": (300, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Interventions")
    parser.add_argument("--prompts", type=str, default="data/occupation_prompts.json")
    parser.add_argument("--steering-prompts", type=str, default="data/steering_prompts.json")
    parser.add_argument("--causal-layers", type=str, default=None)
    parser.add_argument("--method", type=str, required=True,
                        choices=["hard_block", "soft_steer", "prompt_aug",
                                 "random_block", "patching", "pca_projection"])
    parser.add_argument("--target", type=str, choices=["unet", "textenc"], default="unet",
                        help="Which model component to intervene on.")
    parser.add_argument("--window", type=str, choices=["early", "mid", "late"], default=None,
                        help="Restrict intervention to a timestep window (UNet only).")
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true",
                        help="Save images and generate per-prompt grids + summary chart.")
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Visualization output directory (default: results/<method>/).")
    # PCA projection options
    parser.add_argument("--gender-pairs", type=str, default="data/gender_pairs.json",
                        help="Path to gender contrastive pairs JSON.")
    parser.add_argument("--pair-type", type=str, choices=["definitional", "contextual"],
                        default="contextual", help="Which pair category for PCA extraction.")
    parser.add_argument("--token-position", type=str,
                        choices=["eos", "last_subject", "occupation"], default="eos",
                        help="Token position for PCA extraction and projection.")
    parser.add_argument("--n-components", type=int, default=1,
                        help="Number of PCA components for gender subspace.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.prompts) as f:
        prompt_data = json.load(f)
    prompts = prompt_data["prompts"]
    subjects = prompt_data["subject_tokens"]

    causal_layers = None
    if args.causal_layers:
        with open(args.causal_layers) as f:
            causal_layers = json.load(f)

    from models.load_model import load_sd_pipeline, load_mivolo_predictor
    from classifiers.mivolo_classifier import MiVOLOClassifier
    from eval.image_quality import CLIPScorer

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = load_sd_pipeline(device=args.device, dtype=dtype)
    predictor = load_mivolo_predictor(device=args.device)
    classifier = MiVOLOClassifier(predictor)
    clip_scorer = CLIPScorer(device=args.device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0

    # Set up visualization directory
    viz_dir = None
    if args.visualize:
        viz_dir = Path(args.viz_dir or f"results/{args.method}")
        (viz_dir / "images").mkdir(parents=True, exist_ok=True)
        (viz_dir / "grids").mkdir(parents=True, exist_ok=True)

    # Accumulators
    bias_summary = {p: {"baseline": [], "intervention": []} for p in prompts}
    # {prompt: [(baseline_img, intervention_img), ...]}  — all images kept when visualizing
    grid_data = {p: [] for p in prompts} if args.visualize else {}

    with open(args.output, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "condition", "method", "prompt", "image_idx",
            "gender_score", "age", "clip_score", "detected",
        ])
        if write_header:
            writer.writeheader()

        for prompt, subject in zip(prompts, subjects):
            occupation = _short_label(prompt)

            for img_idx in range(args.n_images):
                seed = 2026+ img_idx

                # ── Baseline ─────────────────────────────────────────────────
                baseline_img = _generate_baseline(pipe, prompt, seed)
                baseline_result = classifier.predict_single(baseline_img)
                baseline_clip = clip_scorer.score([baseline_img], [prompt])
                baseline_detected = baseline_result is not None
                baseline_male_prob = (
                    _to_male_prob(baseline_result) if baseline_detected else float("nan")
                )

                writer.writerow({
                    "condition": "baseline",
                    "method": args.method,
                    "prompt": prompt,
                    "image_idx": img_idx,
                    "gender_score": baseline_result["gender_score"] if baseline_detected else "",
                    "age": baseline_result["age"] if baseline_detected else "",
                    "clip_score": round(baseline_clip, 4),
                    "detected": baseline_detected,
                })

                # ── Intervention ─────────────────────────────────────────────
                interv_img = _generate(
                    args.method, pipe, prompt, subject, seed, causal_layers, args,
                    baseline_score=baseline_male_prob,
                )
                interv_result = classifier.predict_single(interv_img)
                interv_clip = clip_scorer.score([interv_img], [prompt])
                interv_detected = interv_result is not None
                interv_male_prob = (
                    _to_male_prob(interv_result) if interv_detected else float("nan")
                )

                writer.writerow({
                    "condition": "intervention",
                    "method": args.method,
                    "prompt": prompt,
                    "image_idx": img_idx,
                    "gender_score": interv_result["gender_score"] if interv_detected else "",
                    "age": interv_result["age"] if interv_detected else "",
                    "clip_score": round(interv_clip, 4),
                    "detected": interv_detected,
                })
                csvfile.flush()

                # Accumulate scores
                if not math.isnan(baseline_male_prob):
                    bias_summary[prompt]["baseline"].append(baseline_male_prob)
                if not math.isnan(interv_male_prob):
                    bias_summary[prompt]["intervention"].append(interv_male_prob)

                # Save individual images and collect for grids
                if args.visualize:
                    img_dir = viz_dir / "images" / occupation
                    img_dir.mkdir(parents=True, exist_ok=True)
                    baseline_img.save(img_dir / f"baseline_{img_idx}.png")
                    interv_img.save(img_dir / f"{args.method}_{img_idx}.png")
                    grid_data[prompt].append((baseline_img, interv_img))

    logger.info(f"Results saved to {args.output}")

    if args.visualize:
        _save_per_prompt_grids(grid_data, args.method, viz_dir / "grids")
        _save_summary_chart(bias_summary, args.method, viz_dir / "summary.png")
        _print_summary_table(bias_summary, args.method)
        logger.info(f"Visualization saved to {viz_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _generate_baseline(pipe, prompt, seed):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(prompt, generator=generator).images[0]


def _generate(method, pipe, prompt, subject, seed, causal_layers, args, baseline_score=None):
    target = args.target
    timestep_window = TIMESTEP_WINDOWS.get(args.window) if args.window else None

    if method == "hard_block":
        from intervention.hard_block import generate_with_hard_block
        if not causal_layers:
            raise ValueError("--causal-layers required for hard_block")
        return generate_with_hard_block(
            pipe, prompt, causal_layers, seed, target=target,
            timestep_window=timestep_window,
        )

    elif method == "soft_steer":
        from intervention.soft_steer import compute_steering_vector, generate_with_steering
        if not causal_layers:
            raise ValueError("--causal-layers required for soft_steer")
        with open(args.steering_prompts) as f:
            steer_data = json.load(f)
        layer = causal_layers[0]
        sv = compute_steering_vector(
            pipe, steer_data["male"], steer_data["female"], layer, target=target,
        )
        return generate_with_steering(
            pipe, prompt, layer, sv, seed=seed, target=target,
            baseline_score=baseline_score, timestep_window=timestep_window,
        )

    elif method == "prompt_aug":
        augmented = prompt.replace("A photo of a ", "A photo of a person who is a ")
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        return pipe(augmented, generator=generator).images[0]

    elif method == "random_block":
        from intervention.hard_block import generate_with_hard_block
        from tracing.hooks import enumerate_unet_layers, enumerate_text_encoder_layers
        if target == "textenc":
            all_layers = enumerate_text_encoder_layers()
        else:
            all_layers = enumerate_unet_layers(pipe.unet)
        non_causal = [l for l in all_layers if l not in (causal_layers or [])]
        n = min(len(causal_layers or []) or 5, len(non_causal))
        return generate_with_hard_block(
            pipe, prompt, random.sample(non_causal, n), seed, target=target,
            timestep_window=timestep_window,
        )

    elif method == "patching":
        from intervention.patching import generate_with_patching
        if not causal_layers:
            raise ValueError("--causal-layers required for patching")
        return generate_with_patching(
            pipe, prompt, subject, causal_layers, seed, target=target,
            timestep_window=timestep_window,
        )

    elif method == "pca_projection":
        from intervention.pca_gender import compute_gender_directions, generate_with_pca_projection
        if not causal_layers:
            raise ValueError("--causal-layers required for pca_projection")

        # Compute per-layer gender directions once and cache across calls
        if not hasattr(_generate, "_pca_cache"):
            _generate._pca_cache = {}
        cache_key = (tuple(causal_layers), args.pair_type, args.token_position)
        if cache_key not in _generate._pca_cache:
            with open(args.gender_pairs) as f:
                pairs_data = json.load(f)
            pair_set = pairs_data[args.pair_type]
            subjects = pair_set.get("subjects")

            gender_dirs = compute_gender_directions(
                pipe, pair_set["male"], pair_set["female"], causal_layers,
                token_position_type=args.token_position,
                male_subjects=subjects, female_subjects=subjects,
                n_components=args.n_components,
            )
            _generate._pca_cache[cache_key] = gender_dirs

        gender_dirs = _generate._pca_cache[cache_key]
        return generate_with_pca_projection(
            pipe, prompt, subject, gender_dirs, causal_layers,
            token_position_type=args.token_position,
            seed=seed, target=target, timestep_window=timestep_window,
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def _to_male_prob(result):
    score = result["gender_score"]
    return score if result["gender"] == "male" else 1.0 - score


def _short_label(prompt):
    return prompt.replace("A photo of a ", "").replace("A photo of an ", "").strip()


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def _save_per_prompt_grids(grid_data, method, grids_dir):
    """Save one grid PNG per prompt: 2 rows (original, intervention) x n_images columns."""
    import matplotlib.pyplot as plt

    grids_dir = Path(grids_dir)
    grids_dir.mkdir(parents=True, exist_ok=True)

    for prompt, pairs in grid_data.items():
        if not pairs:
            continue

        occupation = _short_label(prompt)
        n_imgs = len(pairs)
        fig, axes = plt.subplots(2, n_imgs, figsize=(3 * n_imgs, 6.5))

        # Handle single-image edge case
        if n_imgs == 1:
            axes = axes.reshape(2, 1)

        for col, (baseline_img, interv_img) in enumerate(pairs):
            axes[0, col].imshow(baseline_img)
            axes[0, col].axis("off")
            axes[0, col].set_title(f"seed {2026 + col}", fontsize=8)

            axes[1, col].imshow(interv_img)
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("Original", fontsize=10, rotation=90, labelpad=8)
        axes[1, 0].set_ylabel(method, fontsize=10, rotation=90, labelpad=8)

        fig.suptitle(f"{occupation}", fontsize=12, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(grids_dir / f"{occupation}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"  Per-prompt grids saved to {grids_dir}/")


def _save_summary_chart(bias_summary, method, out_path):
    """Save a bar chart comparing baseline vs intervention male-probability per prompt."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_prompts = list(bias_summary.keys())
    labels = [_short_label(p) for p in all_prompts]

    baseline_means = [
        np.mean(bias_summary[p]["baseline"]) if bias_summary[p]["baseline"] else float("nan")
        for p in all_prompts
    ]
    interv_means = [
        np.mean(bias_summary[p]["intervention"]) if bias_summary[p]["intervention"] else float("nan")
        for p in all_prompts
    ]

    x = np.arange(len(all_prompts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(all_prompts) * 1.2), 5))
    ax.bar(x - width / 2, baseline_means, width, label="Original", color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, interv_means, width, label=method, color="#DD8452", alpha=0.85)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="Parity (0.5)")

    ax.set_ylabel("Mean male probability")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.legend(fontsize=9)

    overall_before = np.nanmean(baseline_means)
    overall_after = np.nanmean(interv_means)
    gap_before = abs(overall_before - 0.5)
    gap_after = abs(overall_after - 0.5)
    delta = gap_before - gap_after
    arrow = "\u2193" if delta >= 0 else "\u2191"
    ax.set_title(
        f"Gender bias: parity gap {gap_before:.3f} \u2192 {gap_after:.3f}  "
        f"({arrow}{abs(delta):.3f})",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Summary chart saved to {out_path}")


def _print_summary_table(bias_summary, method):
    """Print a text summary table to stdout."""
    all_prompts = list(bias_summary.keys())

    baseline_means = [
        np.mean(bias_summary[p]["baseline"]) if bias_summary[p]["baseline"] else float("nan")
        for p in all_prompts
    ]
    interv_means = [
        np.mean(bias_summary[p]["intervention"]) if bias_summary[p]["intervention"] else float("nan")
        for p in all_prompts
    ]

    overall_before = np.nanmean(baseline_means)
    overall_after = np.nanmean(interv_means)
    gap_before = abs(overall_before - 0.5)
    gap_after = abs(overall_after - 0.5)
    delta = gap_before - gap_after

    print(f"\n{'='*60}")
    print(f"  Intervention: {method}")
    print(f"{'='*60}")
    print(f"  {'Prompt':<25} {'Gap before':>10} {'Gap after':>10} {'Change':>10}")
    print(f"  {'-'*55}")
    for p, b, a in zip(all_prompts, baseline_means, interv_means):
        if math.isnan(b) or math.isnan(a):
            continue
        gap_b = abs(b - 0.5)
        gap_a = abs(a - 0.5)
        print(f"  {_short_label(p):<25} {gap_b:>10.3f} {gap_a:>10.3f} {gap_b - gap_a:>+10.3f}")
    print(f"  {'-'*55}")
    print(f"  {'OVERALL':<25} {gap_before:>10.3f} {gap_after:>10.3f} {delta:>+10.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
