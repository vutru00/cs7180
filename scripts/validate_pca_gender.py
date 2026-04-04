"""Validate PCA gender directions across all text encoder layers.

For each layer, computes the gender direction from one set of pairs
(train), then validates on a held-out set. Reports separation accuracy,
occupation entanglement, and singular value spectrum.

Usage:
    python scripts/validate_pca_gender.py \
        --train-pairs contextual \
        --holdout-pairs definitional \
        --token-position eos \
        --n-components 1 \
        --output results/pca_validation.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate PCA gender direction")
    parser.add_argument("--gender-pairs", type=str, default="data/gender_pairs.json")
    parser.add_argument("--occupation-prompts", type=str, default="data/occupation_prompts.json")
    parser.add_argument("--train-pairs", type=str, choices=["definitional", "contextual"],
                        default="contextual", help="Pair set to compute PCA from.")
    parser.add_argument("--holdout-pairs", type=str, choices=["definitional", "contextual"],
                        default="definitional", help="Pair set for separation check.")
    parser.add_argument("--token-position", type=str,
                        choices=["eos", "last_subject", "occupation"], default="eos")
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--layers", type=str, default=None,
                        help="JSON file with layer names. Default: all 24 text encoder layers.")
    parser.add_argument("--output", type=str, default="results/pca_validation.png")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    with open(args.gender_pairs) as f:
        pairs = json.load(f)
    with open(args.occupation_prompts) as f:
        occ_data = json.load(f)

    train_set = pairs[args.train_pairs]
    holdout_set = pairs[args.holdout_pairs]

    # Load models
    from models.load_model import load_sd_pipeline
    from tracing.hooks import enumerate_text_encoder_layers
    from intervention.pca_gender import (
        _compute_single_layer_direction,
        validate_gender_direction,
        extract_hidden_states,
    )

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = load_sd_pipeline(device=args.device, dtype=dtype)

    # Determine layers
    if args.layers:
        with open(args.layers) as f:
            layer_names = json.load(f)
    else:
        layer_names = enumerate_text_encoder_layers()

    # Resolve subjects
    train_subjects = train_set.get("subjects")
    holdout_subjects = holdout_set.get("subjects")

    # ── Compute and validate for each layer ──────────────────────────────────
    results = []

    for layer_name in layer_names:
        print(f"\n{'─'*70}")
        print(f"Layer: {layer_name}")
        print(f"{'─'*70}")

        # Compute gender direction from train set
        gender_dir = _compute_single_layer_direction(
            pipe, train_set["male"], train_set["female"], layer_name,
            args.token_position, train_subjects, train_subjects,
            args.n_components,
        )

        # Singular value analysis
        sv = gender_dir.singular_values
        total_var = (sv ** 2).sum()
        top_k_var = (sv[:args.n_components] ** 2).sum() / total_var
        print(f"  Singular values (top 5): {sv[:5].tolist()}")
        print(f"  Top-{args.n_components} explain: {top_k_var:.1%}")
        if len(sv) > 1:
            print(f"  SV ratio (1st/2nd): {sv[0]/sv[1]:.2f}x")

        # Validate: separation on held-out pairs
        holdout_subjects = holdout_set.get("subjects")
        val = validate_gender_direction(
            pipe, gender_dir, layer_name,
            holdout_set["male"], holdout_set["female"],
            occ_data["prompts"], occ_data["subject_tokens"],
            args.token_position,
            held_out_male_subjects=holdout_subjects,
            held_out_female_subjects=holdout_subjects,
        )

        print(f"  Separation accuracy: {val['separation_accuracy']:.1%}")
        print(f"  Occupation variance along gender dir: {val['occupation_gender_variance']:.4f}")
        print(f"  Occupation variance along random dir: {val['occupation_random_variance']:.4f}")
        ratio = (val["occupation_gender_variance"] / val["occupation_random_variance"]
                 if val["occupation_random_variance"] > 0 else float("inf"))
        print(f"  Entanglement ratio (lower=cleaner):   {ratio:.2f}")

        results.append({
            "layer": layer_name,
            "separation_accuracy": val["separation_accuracy"],
            "occ_gender_var": val["occupation_gender_variance"],
            "occ_random_var": val["occupation_random_variance"],
            "entanglement_ratio": ratio,
            "top_k_explained": float(top_k_var),
            "sv_ratio": float(sv[0] / sv[1]) if len(sv) > 1 else float("inf"),
            "singular_values": sv.tolist(),
        })

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Summary: train={args.train_pairs}, holdout={args.holdout_pairs}, "
          f"position={args.token_position}, k={args.n_components}")
    print(f"{'='*80}")
    print(f"  {'Layer':<45} {'Sep%':>5} {'Var%':>6} {'Entgl':>6} {'SV1/2':>6}")
    print(f"  {'-'*75}")

    for r in results:
        short = r["layer"].replace("text_model.encoder.layers.", "L").replace(".self_attn", ".sa").replace(".mlp", ".mlp")
        print(f"  {short:<45} {r['separation_accuracy']:>5.1%} "
              f"{r['top_k_explained']:>5.1%} {r['entanglement_ratio']:>6.2f} "
              f"{r['sv_ratio']:>6.1f}")

    # Best layer
    best = max(results, key=lambda r: r["separation_accuracy"])
    print(f"\n  Best separation: {best['layer']} ({best['separation_accuracy']:.1%})")

    cleanest = min(results, key=lambda r: r["entanglement_ratio"])
    print(f"  Cleanest (low entanglement): {cleanest['layer']} ({cleanest['entanglement_ratio']:.2f})")
    print(f"{'='*80}\n")

    # ── Visualization ────────────────────────────────────────────────────────
    _plot_results(results, args, Path(args.output))

    # Save JSON results alongside the plot
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w") as f:
        # Convert singular_values to plain lists for JSON
        json_results = [{k: v for k, v in r.items()} for r in results]
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {json_path}")


def _plot_results(results, args, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    short_names = [
        r["layer"].replace("text_model.encoder.layers.", "L")
        .replace(".self_attn", ".sa").replace(".mlp", ".mlp")
        for r in results
    ]
    x = np.arange(n)

    # ── Panel 1: Separation accuracy per layer ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    accs = [r["separation_accuracy"] for r in results]
    colors = ["#2ecc71" if a >= 0.8 else "#e67e22" if a >= 0.6 else "#e74c3c" for a in accs]
    ax1.bar(x, accs, color=colors)
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance")
    ax1.set_ylabel("Separation Accuracy")
    ax1.set_title("Gender Separation on Held-out Pairs")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=55, ha="right", fontsize=6)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7)

    # ── Panel 2: Entanglement ratio per layer ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    entgl = [min(r["entanglement_ratio"], 5.0) for r in results]  # cap for display
    colors2 = ["#2ecc71" if e < 1.0 else "#e67e22" if e < 2.0 else "#e74c3c" for e in entgl]
    ax2.bar(x, entgl, color=colors2)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8, label="Random baseline")
    ax2.set_ylabel("Entanglement Ratio")
    ax2.set_title("Occupation-Gender Entanglement (lower = cleaner)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=55, ha="right", fontsize=6)
    ax2.legend(fontsize=7)

    # ── Panel 3: Explained variance (top-k) per layer ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    expvar = [r["top_k_explained"] for r in results]
    ax3.bar(x, expvar, color="#3498db")
    ax3.set_ylabel(f"Variance Explained (top-{args.n_components})")
    ax3.set_title("PCA Dominance per Layer")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=55, ha="right", fontsize=6)
    ax3.set_ylim(0, 1.05)

    # ── Panel 4: Singular value spectrum for top layers ──────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    # Show spectrum for best 3 layers by separation
    top3 = sorted(results, key=lambda r: r["separation_accuracy"], reverse=True)[:3]
    for r in top3:
        svs = r["singular_values"][:10]
        short = (r["layer"].replace("text_model.encoder.layers.", "L")
                 .replace(".self_attn", ".sa").replace(".mlp", ".mlp"))
        ax4.plot(range(1, len(svs) + 1), svs, "o-", markersize=4, label=short)
    ax4.set_xlabel("Component")
    ax4.set_ylabel("Singular Value")
    ax4.set_title("Singular Value Spectrum (top 3 layers)")
    ax4.legend(fontsize=7)

    fig.suptitle(
        f"PCA Gender Direction Validation\n"
        f"train={args.train_pairs}, holdout={args.holdout_pairs}, "
        f"position={args.token_position}, k={args.n_components}",
        fontsize=11, y=1.02,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
