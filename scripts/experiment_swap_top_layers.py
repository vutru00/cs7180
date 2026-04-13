"""Swap experiment with image output for the top-N layers from a previous run.

Reads swap_results.json, ranks layers by flip score, takes the top-N,
then re-runs the swap experiment saving all generated images and producing
a visual comparison grid per layer.  Supports timestep-windowed swaps
(early/mid/late) to reveal *when* during denoising each layer carries
the gender signal.

Usage:
    # All timesteps (default)
    python scripts/experiment_swap_top_layers.py \
        --swap-results results/swap_experiment/swap_results.json \
        --top-k 5 --n-images 5 \
        --output-dir results/swap_top_layers

    # With timestep windows
    python scripts/experiment_swap_top_layers.py \
        --top-k 5 --n-images 3 --windows \
        --output-dir results/swap_top_layers_windows
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MALE_PROMPT = "A photo of a male nurse"
FEMALE_PROMPT = "A photo of a female nurse"

TIMESTEP_WINDOWS = {
    "all":   None,
    "early": (999, 700),
    "mid":   (700, 300),
    "late":  (300, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Swap top layers with image output")
    parser.add_argument("--swap-results", type=str,
                        default="results/swap_experiment/swap_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/swap_top_layers")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--windows", action="store_true",
                        help="Run swaps for early/mid/late timestep windows in addition to all.")
    return parser.parse_args()


def rank_layers(swap_results_path, top_k):
    with open(swap_results_path) as f:
        data = json.load(f)
    swap = [r for r in data if "baseline" not in r["layer"]]
    stats = defaultdict(lambda: {"m2f": [], "f2m": []})
    for r in swap:
        mp = r["male_prob"]
        if mp != mp:
            continue
        if r["source_prompt"] == "male+female_layer":
            stats[r["layer"]]["m2f"].append(mp)
        else:
            stats[r["layer"]]["f2m"].append(mp)
    ranked = []
    for layer, vals in stats.items():
        m2f = np.mean(vals["m2f"]) if vals["m2f"] else float("nan")
        f2m = np.mean(vals["f2m"]) if vals["f2m"] else float("nan")
        if np.isnan(m2f) or np.isnan(f2m):
            continue
        ranked.append({"layer": layer, "m2f": m2f, "f2m": f2m, "flip": (1 - m2f) + f2m})
    ranked.sort(key=lambda x: x["flip"], reverse=True)
    return ranked[:top_k]


def main():
    args = parse_args()

    top_layers = rank_layers(args.swap_results, args.top_k)
    layer_names = [l["layer"] for l in top_layers]

    # Decide which windows to run
    if args.windows:
        window_keys = ["all", "early", "mid", "late"]
    else:
        window_keys = ["all"]

    print(f"Top {args.top_k} layers by flip score:")
    for i, l in enumerate(top_layers):
        print(f"  {i+1}. {l['layer']}  (flip={l['flip']:.3f})")
    print(f"Windows: {window_keys}\n")

    # Load models
    from models.load_model import load_sd_pipeline, load_mivolo_predictor
    from classifiers.mivolo_classifier import MiVOLOClassifier
    from tracing.corrupt import encode_prompt_clean, get_uncond_embeddings
    from tracing.hooks import ActivationCache
    from tracing.restore import custom_denoise

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = load_sd_pipeline(device=args.device, dtype=dtype)
    predictor = load_mivolo_predictor(device=args.device)
    classifier = MiVOLOClassifier(predictor)

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    device = args.device

    uncond = get_uncond_embeddings(text_encoder, tokenizer)
    cond_male = encode_prompt_clean(text_encoder, tokenizer, MALE_PROMPT)
    cond_female = encode_prompt_clean(text_encoder, tokenizer, FEMALE_PROMPT)
    embeds_male = torch.cat([uncond, cond_male])
    embeds_female = torch.cat([uncond, cond_female])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    for seed_offset in range(args.n_images):
        seed = args.base_seed + seed_offset
        logger.info(f"Seed {seed}")
        th = [0]

        seed_images = {}
        seed_scores = {}

        # ── Baselines ────────────────────────────────────────────────────
        gen_m = torch.Generator(device=device).manual_seed(seed)
        img_m_bl = pipe(MALE_PROMPT, generator=gen_m).images[0]
        seed_images["male_baseline"] = img_m_bl
        seed_scores["male_baseline"] = classifier.predict_single(img_m_bl)

        gen_f = torch.Generator(device=device).manual_seed(seed)
        img_f_bl = pipe(FEMALE_PROMPT, generator=gen_f).images[0]
        seed_images["female_baseline"] = img_f_bl
        seed_scores["female_baseline"] = classifier.predict_single(img_f_bl)

        # ── Record activations at top layers ─────────────────────────────
        male_cache = ActivationCache()
        male_cache.register_record_hooks(unet, layer_names, timestep_holder=th)
        try:
            custom_denoise(unet, scheduler, vae, embeds_male, th, seed, device)
        finally:
            male_cache.remove_all_hooks()
        male_acts = {k: v.cpu() for k, v in male_cache.cache.items()}
        male_cache.clear_cache()

        female_cache = ActivationCache()
        female_cache.register_record_hooks(unet, layer_names, timestep_holder=th)
        try:
            custom_denoise(unet, scheduler, vae, embeds_female, th, seed, device)
        finally:
            female_cache.remove_all_hooks()
        female_acts = {k: v.cpu() for k, v in female_cache.cache.items()}
        female_cache.clear_cache()

        # ── Swap each layer × each window ────────────────────────────────
        for layer_name in layer_names:
            for wk in window_keys:
                tw = TIMESTEP_WINDOWS[wk]
                suffix = f"{_short(layer_name)}|{wk}"

                # Male prompt, layer swapped from female
                swap = ActivationCache()
                swap.register_patch_hook(
                    unet, layer_name, female_acts,
                    timestep_holder=th, timestep_window=tw, conditional_only=True,
                )
                try:
                    img = custom_denoise(unet, scheduler, vae, embeds_male, th, seed, device)
                finally:
                    swap.remove_all_hooks()
                key = f"male+swap({suffix})"
                seed_images[key] = img
                seed_scores[key] = classifier.predict_single(img)

                # Female prompt, layer swapped from male
                swap2 = ActivationCache()
                swap2.register_patch_hook(
                    unet, layer_name, male_acts,
                    timestep_holder=th, timestep_window=tw, conditional_only=True,
                )
                try:
                    img2 = custom_denoise(unet, scheduler, vae, embeds_female, th, seed, device)
                finally:
                    swap2.remove_all_hooks()
                key2 = f"female+swap({suffix})"
                seed_images[key2] = img2
                seed_scores[key2] = classifier.predict_single(img2)

        del male_acts, female_acts
        all_data.append({"seed": seed, "images": seed_images, "scores": seed_scores})

    # ── Save individual images ───────────────────────────────────────────
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for entry in all_data:
        seed = entry["seed"]
        for key, img in entry["images"].items():
            safe_key = key.replace("/", "_").replace("(", "").replace(")", "").replace("|", "_")
            img.save(img_dir / f"seed{seed}_{safe_key}.png")
    logger.info(f"Individual images saved to {img_dir}/")

    # ── Save scores JSON ─────────────────────────────────────────────────
    scores_out = []
    for entry in all_data:
        for key, res in entry["scores"].items():
            scores_out.append({
                "seed": entry["seed"],
                "condition": key,
                "gender": res["gender"] if res else None,
                "gender_score": res["gender_score"] if res else None,
                "male_prob": _to_male_prob(res),
                "age": res["age"] if res else None,
            })
    with open(out_dir / "scores.json", "w") as f:
        json.dump(scores_out, f, indent=2)

    # ── Generate grids ───────────────────────────────────────────────────
    _make_per_seed_grids(all_data, layer_names, window_keys, out_dir)
    _make_window_comparison(all_data, layer_names, window_keys, out_dir)
    _print_summary(all_data, layer_names, window_keys)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _short(layer_name):
    return (layer_name
            .replace("down_blocks.", "d").replace("up_blocks.", "u")
            .replace("mid_block.", "m").replace(".transformer_blocks.0.", ".tb0.")
            .replace(".resnets.", ".r").replace(".attentions.", ".a"))


def _to_male_prob(result):
    if result is None:
        return float("nan")
    s = result["gender_score"]
    return s if result["gender"] == "male" else 1.0 - s


def _gender_label(result):
    if result is None:
        return "?"
    mp = _to_male_prob(result)
    g = result["gender"][0].upper()
    return f"{g} ({mp:.2f})"


# ──────────────────────────────────────────────────────────────────────────────
# Per-seed grids: rows = [male, female], cols = [baseline, layer1×windows, ...]
# ──────────────────────────────────────────────────────────────────────────────

def _make_per_seed_grids(all_data, layer_names, window_keys, out_dir):
    import matplotlib.pyplot as plt

    grid_dir = out_dir / "grids"
    grid_dir.mkdir(exist_ok=True)

    n_windows = len(window_keys)
    n_layers = len(layer_names)
    # Columns: baseline + (layers × windows)
    n_cols = 1 + n_layers * n_windows

    for entry in all_data:
        seed = entry["seed"]
        images = entry["images"]
        scores = entry["scores"]

        fig, axes = plt.subplots(2, n_cols, figsize=(2.5 * n_cols, 6))

        for gender_row, gender in enumerate(["male", "female"]):
            bl_key = f"{gender}_baseline"
            axes[gender_row, 0].imshow(images[bl_key])
            axes[gender_row, 0].set_title(
                f"Baseline\n{_gender_label(scores[bl_key])}", fontsize=6)
            axes[gender_row, 0].set_ylabel(f"{gender.title()} nurse", fontsize=8)
            axes[gender_row, 0].axis("off")

            col = 1
            for layer_name in layer_names:
                for wk in window_keys:
                    suffix = f"{_short(layer_name)}|{wk}"
                    key = f"{gender}+swap({suffix})"
                    axes[gender_row, col].imshow(images[key])
                    title = f"{_short(layer_name)}\n{wk}\n{_gender_label(scores[key])}"
                    axes[gender_row, col].set_title(title, fontsize=5)
                    axes[gender_row, col].axis("off")
                    col += 1

        fig.suptitle(f"Activation Swap — seed {seed}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(grid_dir / f"seed_{seed}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Per-seed grids saved to {grid_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Window comparison: one figure per layer, rows = [male, female],
# cols = [baseline, all, early, mid, late], rows averaged across seeds
# ──────────────────────────────────────────────────────────────────────────────

def _make_window_comparison(all_data, layer_names, window_keys, out_dir):
    import matplotlib.pyplot as plt

    comp_dir = out_dir / "window_comparison"
    comp_dir.mkdir(exist_ok=True)

    n_windows = len(window_keys)
    n_cols = 1 + n_windows  # baseline + each window

    for layer_name in layer_names:
        short = _short(layer_name)

        # Use first seed for representative images
        entry = all_data[0]
        images = entry["images"]
        scores = entry["scores"]

        fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

        for gender_row, gender in enumerate(["male", "female"]):
            # Baseline
            bl_key = f"{gender}_baseline"
            axes[gender_row, 0].imshow(images[bl_key])
            axes[gender_row, 0].set_title(
                f"Baseline\n{_gender_label(scores[bl_key])}", fontsize=8)
            axes[gender_row, 0].set_ylabel(f"{gender.title()} nurse", fontsize=10)
            axes[gender_row, 0].axis("off")

            # Each window
            for j, wk in enumerate(window_keys):
                suffix = f"{short}|{wk}"
                key = f"{gender}+swap({suffix})"
                axes[gender_row, j + 1].imshow(images[key])

                # Compute mean male_prob across all seeds for annotation
                all_mp = []
                for e in all_data:
                    r = e["scores"].get(key)
                    if r:
                        all_mp.append(_to_male_prob(r))
                mean_mp = np.nanmean(all_mp) if all_mp else float("nan")
                window_desc = wk if wk == "all" else f"{wk}\n({TIMESTEP_WINDOWS[wk][0]}→{TIMESTEP_WINDOWS[wk][1]})"
                axes[gender_row, j + 1].set_title(
                    f"{window_desc}\n{_gender_label(scores[key])}\navg={mean_mp:.2f}",
                    fontsize=7)
                axes[gender_row, j + 1].axis("off")

        fig.suptitle(f"Timestep Window Comparison — {layer_name}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(comp_dir / f"{short}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Bar chart: all layers × all windows ──────────────────────────────
    if len(window_keys) > 1:
        _make_window_bar_chart(all_data, layer_names, window_keys, comp_dir)

    logger.info(f"Window comparisons saved to {comp_dir}/")


def _make_window_bar_chart(all_data, layer_names, window_keys, comp_dir):
    """Bar chart: x = layer, grouped bars = windows, y = male_prob for each direction."""
    import matplotlib.pyplot as plt

    # Aggregate scores
    agg = defaultdict(list)
    for entry in all_data:
        for key, res in entry["scores"].items():
            agg[key].append(_to_male_prob(res))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(layer_names) * 3), 8))

    x = np.arange(len(layer_names))
    n_w = len(window_keys)
    total_width = 0.8
    bar_width = total_width / n_w
    colors = {"all": "#3498db", "early": "#e74c3c", "mid": "#f39c12", "late": "#2ecc71"}

    for ax, direction, title in [
        (ax1, "male", "Male nurse + female-layer swap"),
        (ax2, "female", "Female nurse + male-layer swap"),
    ]:
        for i, wk in enumerate(window_keys):
            means = []
            for layer_name in layer_names:
                suffix = f"{_short(layer_name)}|{wk}"
                key = f"{direction}+swap({suffix})"
                vals = [v for v in agg.get(key, []) if v == v]
                means.append(np.mean(vals) if vals else float("nan"))

            offset = (i - n_w / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width, label=wk,
                   color=colors.get(wk, "#999"), alpha=0.85)

        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("Male Probability")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(l) for l in layer_names], rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Gender Flip by Timestep Window", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(comp_dir / "window_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(all_data, layer_names, window_keys):
    agg = defaultdict(list)
    for entry in all_data:
        for key, res in entry["scores"].items():
            agg[key].append(_to_male_prob(res))

    print(f"\n{'='*80}")
    print(f"  Summary across {len(all_data)} seeds, windows: {window_keys}")
    print(f"{'='*80}")

    # Baselines
    for key in ["male_baseline", "female_baseline"]:
        vals = [v for v in agg[key] if v == v]
        mp = np.mean(vals) if vals else float("nan")
        mf = np.mean([v > 0.5 for v in vals]) if vals else float("nan")
        print(f"  {key:<45} MP={mp:.3f}  male%={mf:.0%}")
    print()

    # Per layer × window
    header = f"  {'Layer':<25}"
    for wk in window_keys:
        header += f" │ {wk:^17}"
    print(header)
    print(f"  {'-' * (25 + 20 * len(window_keys))}")

    for layer_name in layer_names:
        short = _short(layer_name)
        for direction in ["male", "female"]:
            row = f"  {direction[0].upper()}+{short:<22}"
            for wk in window_keys:
                suffix = f"{short}|{wk}"
                key = f"{direction}+swap({suffix})"
                vals = [v for v in agg.get(key, []) if v == v]
                mp = np.mean(vals) if vals else float("nan")
                mf = np.mean([v > 0.5 for v in vals]) if vals else float("nan")
                row += f" │ {mp:>6.3f} ({mf:>4.0%})  "
            print(row)
        print()

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
