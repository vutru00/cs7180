"""Experiment: swap male/female activations layer-by-layer for "nurse".

For both the text encoder and UNet, this script:
1. Runs "A photo of a male nurse" and "A photo of a female nurse" and
   records every layer's activations.
2. For each layer, generates images where that layer's activation is
   swapped from the opposite gender's run (all other layers unchanged).
3. Scores each generated image with MiVOLO and reports the gender result.

This reveals which layers carry the gender signal: if swapping layer L from
the male run into the female run flips the output gender, that layer is a
strong gender mediator.

Usage:
    python scripts/experiment_swap_activations.py --target textenc --n-images 5
    python scripts/experiment_swap_activations.py --target unet --n-images 3
    python scripts/experiment_swap_activations.py --target both --n-images 5
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MALE_PROMPT = "A photo of a male nurse"
FEMALE_PROMPT = "A photo of a female nurse"


def parse_args():
    parser = argparse.ArgumentParser(description="Swap activation experiment")
    parser.add_argument("--target", type=str, choices=["textenc", "unet", "both"],
                        default="both")
    parser.add_argument("--n-images", type=int, default=5,
                        help="Number of seeds to average over.")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/swap_experiment")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Text encoder experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_textenc_experiment(pipe, classifier, layer_names, n_images, base_seed, device):
    """Swap text encoder activations layer-by-layer.

    The text encoder runs once per image. For each layer, we:
    - Encode the male prompt with a hook that replaces that layer's output
      with the female run's activation (and vice versa).
    - Generate images with the swapped conditioning and score them.
    """
    from tracing.corrupt import encode_prompt_clean, get_uncond_embeddings
    from tracing.hooks import ActivationCache
    from tracing.restore import custom_denoise

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler

    uncond = get_uncond_embeddings(text_encoder, tokenizer)

    results = []

    for seed_offset in range(n_images):
        seed = base_seed + seed_offset
        logger.info(f"[TextEnc] seed={seed}")

        # ── Record all layer activations for both prompts ────────────────
        male_cache = ActivationCache()
        male_cache.register_record_hooks(text_encoder, layer_names)
        try:
            encode_prompt_clean(text_encoder, tokenizer, MALE_PROMPT)
        finally:
            male_cache.remove_all_hooks()

        female_cache = ActivationCache()
        female_cache.register_record_hooks(text_encoder, layer_names)
        try:
            encode_prompt_clean(text_encoder, tokenizer, FEMALE_PROMPT)
        finally:
            female_cache.remove_all_hooks()

        # ── Baselines (no swap) ──────────────────────────────────────────
        for label, prompt in [("male_baseline", MALE_PROMPT), ("female_baseline", FEMALE_PROMPT)]:
            cond = encode_prompt_clean(text_encoder, tokenizer, prompt)
            embeds = torch.cat([uncond, cond])
            th = [0]
            img = custom_denoise(unet, scheduler, vae, embeds, th, seed, device)
            res = classifier.predict_single(img)
            results.append({
                "seed": seed, "target": "textenc", "layer": label,
                "source_prompt": prompt,
                "gender": res["gender"] if res else None,
                "gender_score": res["gender_score"] if res else float("nan"),
                "male_prob": _to_male_prob(res) if res else float("nan"),
            })

        # ── Swap each layer ──────────────────────────────────────────────
        for layer_name in tqdm(layer_names, desc=f"  TextEnc layers (seed={seed})", leave=False):
            # Male prompt with layer swapped from female
            swap_cache = ActivationCache()
            swap_cache.register_patch_hook(
                text_encoder, layer_name, female_cache.cache,
                conditional_only=False,
            )
            try:
                cond_m2f = encode_prompt_clean(text_encoder, tokenizer, MALE_PROMPT)
            finally:
                swap_cache.remove_all_hooks()

            embeds = torch.cat([uncond, cond_m2f])
            th = [0]
            img = custom_denoise(unet, scheduler, vae, embeds, th, seed, device)
            res = classifier.predict_single(img)
            results.append({
                "seed": seed, "target": "textenc", "layer": layer_name,
                "source_prompt": "male+female_layer",
                "gender": res["gender"] if res else None,
                "gender_score": res["gender_score"] if res else float("nan"),
                "male_prob": _to_male_prob(res) if res else float("nan"),
            })

            # Female prompt with layer swapped from male
            swap_cache2 = ActivationCache()
            swap_cache2.register_patch_hook(
                text_encoder, layer_name, male_cache.cache,
                conditional_only=False,
            )
            try:
                cond_f2m = encode_prompt_clean(text_encoder, tokenizer, FEMALE_PROMPT)
            finally:
                swap_cache2.remove_all_hooks()

            embeds = torch.cat([uncond, cond_f2m])
            th = [0]
            img = custom_denoise(unet, scheduler, vae, embeds, th, seed, device)
            res = classifier.predict_single(img)
            results.append({
                "seed": seed, "target": "textenc", "layer": layer_name,
                "source_prompt": "female+male_layer",
                "gender": res["gender"] if res else None,
                "gender_score": res["gender_score"] if res else float("nan"),
                "male_prob": _to_male_prob(res) if res else float("nan"),
            })

        male_cache.clear_cache()
        female_cache.clear_cache()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# UNet experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_unet_experiment(pipe, classifier, layer_names, n_images, base_seed, device):
    """Swap UNet activations layer-by-layer.

    The UNet runs 50x per image (once per timestep). For each layer, we:
    - Run a clean male denoising pass and record that layer at every timestep.
    - Run a clean female denoising pass and record that layer at every timestep.
    - Generate with the male conditioning but swap the target layer's activations
      from the female run at every timestep (and vice versa).
    """
    from tracing.corrupt import encode_prompt_clean, get_uncond_embeddings
    from tracing.hooks import ActivationCache
    from tracing.restore import custom_denoise

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler

    uncond = get_uncond_embeddings(text_encoder, tokenizer)
    cond_male = encode_prompt_clean(text_encoder, tokenizer, MALE_PROMPT)
    cond_female = encode_prompt_clean(text_encoder, tokenizer, FEMALE_PROMPT)
    embeds_male = torch.cat([uncond, cond_male])
    embeds_female = torch.cat([uncond, cond_female])

    results = []

    for seed_offset in range(n_images):
        seed = base_seed + seed_offset
        logger.info(f"[UNet] seed={seed}")

        # ── Record ALL layers for both prompts in one pass each ──────────
        th = [0]

        male_cache = ActivationCache()
        male_cache.register_record_hooks(unet, layer_names, timestep_holder=th)
        try:
            img_m = custom_denoise(unet, scheduler, vae, embeds_male, th, seed, device)
        finally:
            male_cache.remove_all_hooks()
        male_all = {k: v.cpu() for k, v in male_cache.cache.items()}
        male_cache.clear_cache()

        female_cache = ActivationCache()
        female_cache.register_record_hooks(unet, layer_names, timestep_holder=th)
        try:
            img_f = custom_denoise(unet, scheduler, vae, embeds_female, th, seed, device)
        finally:
            female_cache.remove_all_hooks()
        female_all = {k: v.cpu() for k, v in female_cache.cache.items()}
        female_cache.clear_cache()

        # ── Baselines ────────────────────────────────────────────────────
        for label, img in [("male_baseline", img_m), ("female_baseline", img_f)]:
            res = classifier.predict_single(img)
            results.append({
                "seed": seed, "target": "unet", "layer": label,
                "source_prompt": MALE_PROMPT if "male" in label else FEMALE_PROMPT,
                "gender": res["gender"] if res else None,
                "gender_score": res["gender_score"] if res else float("nan"),
                "male_prob": _to_male_prob(res) if res else float("nan"),
            })

        # ── Swap each layer one at a time ────────────────────────────────
        for layer_name in tqdm(layer_names, desc=f"  UNet layers (seed={seed})", leave=False):
            # Male run with one layer swapped from female
            swap = ActivationCache()
            swap.register_patch_hook(
                unet, layer_name, female_all,
                timestep_holder=th, conditional_only=True,
            )
            try:
                img = custom_denoise(unet, scheduler, vae, embeds_male, th, seed, device)
            finally:
                swap.remove_all_hooks()

            res = classifier.predict_single(img)
            results.append({
                "seed": seed, "target": "unet", "layer": layer_name,
                "source_prompt": "male+female_layer",
                "gender": res["gender"] if res else None,
                "gender_score": res["gender_score"] if res else float("nan"),
                "male_prob": _to_male_prob(res) if res else float("nan"),
            })

            # Female run with one layer swapped from male
            swap2 = ActivationCache()
            swap2.register_patch_hook(
                unet, layer_name, male_all,
                timestep_holder=th, conditional_only=True,
            )
            try:
                img = custom_denoise(unet, scheduler, vae, embeds_female, th, seed, device)
            finally:
                swap2.remove_all_hooks()

            res = classifier.predict_single(img)
            results.append({
                "seed": seed, "target": "unet", "layer": layer_name,
                "source_prompt": "female+male_layer",
                "gender": res["gender"] if res else None,
                "gender_score": res["gender_score"] if res else float("nan"),
                "male_prob": _to_male_prob(res) if res else float("nan"),
            })

        del male_all, female_all

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_male_prob(result):
    if result is None:
        return float("nan")
    score = result["gender_score"]
    return score if result["gender"] == "male" else 1.0 - score


def _summarize_and_print(results, target_name):
    """Aggregate results across seeds and print a table."""
    import pandas as pd

    df = pd.DataFrame(results)
    df = df[df["target"] == target_name]

    if df.empty:
        return df

    # ── Baselines ────────────────────────────────────────────────────────
    baselines = df[df["layer"].str.contains("baseline")]
    for label in ["male_baseline", "female_baseline"]:
        sub = baselines[baselines["layer"] == label]
        mp = sub["male_prob"].dropna()
        frac_male = (mp > 0.5).mean() if len(mp) > 0 else float("nan")
        mean_mp = mp.mean() if len(mp) > 0 else float("nan")
        print(f"  {label:<30} male_frac={frac_male:.0%}  mean_male_prob={mean_mp:.3f}  (n={len(mp)})")

    print()

    # ── Per-layer swap results ───────────────────────────────────────────
    swap_df = df[~df["layer"].str.contains("baseline")]

    # Group: for each layer, split by direction
    agg = (
        swap_df.groupby(["layer", "source_prompt"])["male_prob"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Pivot: one row per layer, columns for each swap direction
    m2f = agg[agg["source_prompt"] == "male+female_layer"].set_index("layer")
    f2m = agg[agg["source_prompt"] == "female+male_layer"].set_index("layer")

    # Compute male fraction from male_prob
    swap_frac = (
        swap_df.groupby(["layer", "source_prompt"])["male_prob"]
        .apply(lambda s: (s.dropna() > 0.5).mean())
        .reset_index()
        .rename(columns={"male_prob": "male_frac"})
    )
    frac_m2f = swap_frac[swap_frac["source_prompt"] == "male+female_layer"].set_index("layer")
    frac_f2m = swap_frac[swap_frac["source_prompt"] == "female+male_layer"].set_index("layer")

    print(f"  {'Layer':<45} {'M→swap_F':>9} {'F→swap_M':>9} {'M→male%':>8} {'F→male%':>8}")
    print(f"  {'-'*82}")

    all_layers = list(swap_df["layer"].unique())
    for layer in all_layers:
        m2f_val = m2f.loc[layer, "mean"] if layer in m2f.index else float("nan")
        f2m_val = f2m.loc[layer, "mean"] if layer in f2m.index else float("nan")
        m2f_frac = frac_m2f.loc[layer, "male_frac"] if layer in frac_m2f.index else float("nan")
        f2m_frac = frac_f2m.loc[layer, "male_frac"] if layer in frac_f2m.index else float("nan")

        short = (layer.replace("text_model.encoder.layers.", "L")
                 .replace(".self_attn", ".sa").replace(".mlp", ".mlp")
                 .replace("down_blocks.", "d").replace("up_blocks.", "u")
                 .replace("mid_block.", "m").replace(".transformer_blocks.0.", ".tb0.")
                 .replace(".resnets.", ".r"))
        print(f"  {short:<45} {m2f_val:>9.3f} {f2m_val:>9.3f} {m2f_frac:>7.0%} {f2m_frac:>7.0%}")

    return df


def _plot_results(all_results, output_dir):
    """Create a visualization of swap results."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(all_results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for target_name in df["target"].unique():
        tdf = df[(df["target"] == target_name) & ~df["layer"].str.contains("baseline")]
        if tdf.empty:
            continue

        # Compute per-layer mean male_prob for each swap direction
        agg = tdf.groupby(["layer", "source_prompt"])["male_prob"].mean().reset_index()
        m2f = agg[agg["source_prompt"] == "male+female_layer"].set_index("layer")["male_prob"]
        f2m = agg[agg["source_prompt"] == "female+male_layer"].set_index("layer")["male_prob"]

        layers = list(m2f.index)
        n = len(layers)
        x = np.arange(n)

        short_names = [
            l.replace("text_model.encoder.layers.", "L")
            .replace(".self_attn", ".sa").replace(".mlp", ".mlp")
            .replace("down_blocks.", "d").replace("up_blocks.", "u")
            .replace("mid_block.", "m").replace(".transformer_blocks.0.", ".tb0.")
            .replace(".resnets.", ".r")
            for l in layers
        ]

        fig, ax = plt.subplots(figsize=(max(12, n * 0.4), 5))
        width = 0.35
        ax.bar(x - width / 2, m2f.values, width, label="Male + female layer swap",
               color="#4C72B0", alpha=0.85)
        ax.bar(x + width / 2, f2m.values, width, label="Female + male layer swap",
               color="#DD8452", alpha=0.85)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Parity")

        # Baseline reference lines
        baselines = df[(df["target"] == target_name) & df["layer"].str.contains("baseline")]
        male_bl = baselines[baselines["layer"] == "male_baseline"]["male_prob"].mean()
        female_bl = baselines[baselines["layer"] == "female_baseline"]["male_prob"].mean()
        ax.axhline(male_bl, color="#4C72B0", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(female_bl, color="#DD8452", ls=":", lw=0.8, alpha=0.5)

        ax.set_ylabel("Male Probability after Swap")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        fontsize = 6 if n > 30 else 7 if n > 15 else 8
        ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=fontsize)
        ax.legend(fontsize=8)
        ax.set_title(f"Activation Swap Experiment — {target_name} (nurse)")

        fig.tight_layout()
        fig.savefig(output_dir / f"swap_{target_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved: {output_dir / f'swap_{target_name}.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    from models.load_model import load_sd_pipeline, load_mivolo_predictor
    from classifiers.mivolo_classifier import MiVOLOClassifier
    from tracing.hooks import enumerate_text_encoder_layers, enumerate_unet_layers

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = load_sd_pipeline(device=args.device, dtype=dtype)
    predictor = load_mivolo_predictor(device=args.device)
    classifier = MiVOLOClassifier(predictor)

    targets = [args.target] if args.target != "both" else ["textenc", "unet"]
    all_results = []

    for target in targets:
        if target == "textenc":
            layers = enumerate_text_encoder_layers()
            print(f"\n{'='*85}")
            print(f"  TEXT ENCODER — {len(layers)} layers, {args.n_images} seeds")
            print(f"{'='*85}\n")
            results = run_textenc_experiment(
                pipe, classifier, layers, args.n_images, args.base_seed, args.device,
            )
            all_results.extend(results)
            _summarize_and_print(results, "textenc")

        elif target == "unet":
            layers = enumerate_unet_layers(pipe.unet)
            print(f"\n{'='*85}")
            print(f"  UNET — {len(layers)} layers, {args.n_images} seeds")
            print(f"{'='*85}\n")
            results = run_unet_experiment(
                pipe, classifier, layers, args.n_images, args.base_seed, args.device,
            )
            all_results.extend(results)
            _summarize_and_print(results, "unet")

    # Save raw results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "swap_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {json_path}")

    # Plot
    _plot_results(all_results, output_dir)


if __name__ == "__main__":
    main()
