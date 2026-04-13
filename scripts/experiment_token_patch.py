"""Experiment: patch a single token's activation between prompts.

Generates images from "A photo of a nurse" but with the activation at the
"nurse" token in text_model.encoder.layers.0.self_attn replaced by the
activation from "A photo of a male nurse" at that same token position.

Also runs the reverse direction (male nurse with neutral nurse's activation)
and a female nurse variant for comparison.

Usage:
    python scripts/experiment_token_patch.py --n-images 5
    python scripts/experiment_token_patch.py --n-images 10 --layer text_model.encoder.layers.0.self_attn
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

NEUTRAL_PROMPT = "A photo of a nurse"
MALE_PROMPT = "A photo of a male nurse"
FEMALE_PROMPT = "A photo of a female nurse"
SUBJECT = "nurse"
LAYER = "text_model.encoder.layers.0.self_attn"


def parse_args():
    parser = argparse.ArgumentParser(description="Token-level activation patching experiment")
    parser.add_argument("--layer", type=str, default=LAYER,
                        help="Text encoder layer to patch.")
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/token_patch_experiment")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    layer_name = args.layer

    from models.load_model import load_sd_pipeline, load_mivolo_predictor
    from classifiers.mivolo_classifier import MiVOLOClassifier
    from tracing.corrupt import encode_prompt_clean, get_uncond_embeddings, find_subject_token_positions
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

    # Find "nurse" token positions in each prompt
    pos_neutral = find_subject_token_positions(tokenizer, NEUTRAL_PROMPT, SUBJECT)
    pos_male = find_subject_token_positions(tokenizer, MALE_PROMPT, SUBJECT)
    pos_female = find_subject_token_positions(tokenizer, FEMALE_PROMPT, SUBJECT)

    print(f"Token positions for '{SUBJECT}':")
    print(f"  '{NEUTRAL_PROMPT}': {pos_neutral}")
    print(f"  '{MALE_PROMPT}': {pos_male}")
    print(f"  '{FEMALE_PROMPT}': {pos_female}")
    print(f"Layer: {layer_name}\n")

    # ── Record activations for all three prompts ─────────────────────────
    def record_layer(prompt):
        cache = ActivationCache()
        cache.register_record_hooks(text_encoder, [layer_name])
        try:
            cond = encode_prompt_clean(text_encoder, tokenizer, prompt)
        finally:
            cache.remove_all_hooks()
        return cache.cache[layer_name].clone(), cond  # (1, 77, 768), (1, 77, 768)

    act_neutral, cond_neutral = record_layer(NEUTRAL_PROMPT)
    act_male, cond_male = record_layer(MALE_PROMPT)
    act_female, cond_female = record_layer(FEMALE_PROMPT)

    # ── Define patch conditions ──────────────────────────────────────────
    # Each condition: (base_prompt, base_cond, source_activation, source_positions, target_positions, label)
    conditions = [
        # Baselines
        ("neutral_baseline", NEUTRAL_PROMPT, cond_neutral, None, None, None),
        ("male_baseline", MALE_PROMPT, cond_male, None, None, None),
        ("female_baseline", FEMALE_PROMPT, cond_female, None, None, None),
        # Main experiment: neutral nurse with male nurse's activation at "nurse" token
        ("neutral+male_nurse_token", NEUTRAL_PROMPT, None,
         act_male, pos_male, pos_neutral),
        # Reverse: male nurse with neutral nurse's activation at "nurse" token
        ("male+neutral_nurse_token", MALE_PROMPT, None,
         act_neutral, pos_neutral, pos_male),
        # Also try: neutral nurse with female nurse's activation
        ("neutral+female_nurse_token", NEUTRAL_PROMPT, None,
         act_female, pos_female, pos_neutral),
        # And: female nurse with neutral nurse's activation
        ("female+neutral_nurse_token", FEMALE_PROMPT, None,
         act_neutral, pos_neutral, pos_female),
    ]

    out_dir = Path(args.output_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    all_data = []

    for seed_offset in range(args.n_images):
        seed = args.base_seed + seed_offset
        logger.info(f"Seed {seed}")

        seed_images = {}
        seed_scores = {}

        for label, base_prompt, base_cond, src_act, src_pos, tgt_pos in conditions:
            if base_cond is not None and src_act is None:
                # Baseline — just generate normally
                cond = base_cond
            else:
                # Patched — encode with hook that swaps specific token positions
                cond = _encode_with_token_patch(
                    text_encoder, tokenizer, base_prompt, layer_name,
                    src_act, src_pos, tgt_pos,
                )

            embeds = torch.cat([uncond, cond])
            th = [0]
            img = custom_denoise(unet, scheduler, vae, embeds, th, seed, device)

            res = classifier.predict_single(img)
            seed_images[label] = img
            seed_scores[label] = res

            img.save(out_dir / "images" / f"seed{seed}_{label}.png")

        all_data.append({"seed": seed, "images": seed_images, "scores": seed_scores})

    # ── Save scores ──────────────────────────────────────────────────────
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

    # ── Visualization ────────────────────────────────────────────────────
    _make_grids(all_data, conditions, out_dir)
    _print_summary(all_data, conditions)


def _encode_with_token_patch(
    text_encoder, tokenizer, base_prompt, layer_name,
    source_activation, source_positions, target_positions,
):
    """Encode a prompt but replace specific token positions in a layer's output.

    Args:
        text_encoder: CLIP text encoder.
        tokenizer: CLIP tokenizer.
        base_prompt: The prompt to encode.
        layer_name: Layer to hook.
        source_activation: (1, 77, 768) tensor from the source prompt's layer output.
        source_positions: Token positions in the source activation to copy from.
        target_positions: Token positions in the base prompt to paste into.
    """
    device = text_encoder.device
    src = source_activation.to(device)

    def patch_hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        tensor = (output[0] if is_tuple else output).clone()

        for src_pos, tgt_pos in zip(source_positions, target_positions):
            tensor[:, tgt_pos, :] = src[:, src_pos, :]

        return (tensor,) + output[1:] if is_tuple else tensor

    handle = text_encoder.text_model.encoder.modules
    # Register hook on the target layer
    name_to_module = dict(text_encoder.named_modules())
    module = name_to_module[layer_name]
    handle = module.register_forward_hook(patch_hook)

    try:
        from tracing.corrupt import encode_prompt_clean
        cond = encode_prompt_clean(text_encoder, tokenizer, base_prompt)
    finally:
        handle.remove()

    return cond


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


def _make_grids(all_data, conditions, out_dir):
    import matplotlib.pyplot as plt

    cond_labels = [c[0] for c in conditions]
    n_cols = len(cond_labels)
    n_rows = len(all_data)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, entry in enumerate(all_data):
        seed = entry["seed"]
        for col, label in enumerate(cond_labels):
            ax = axes[row, col]
            ax.imshow(entry["images"][label])
            ax.axis("off")

            score_str = _gender_label(entry["scores"][label])
            if row == 0:
                # Wrap long labels
                display = label.replace("_", "\n", 1).replace("_", " ")
                ax.set_title(f"{display}\n{score_str}", fontsize=6)
            else:
                ax.set_title(score_str, fontsize=7)

            if col == 0:
                ax.set_ylabel(f"seed {seed}", fontsize=8)

    fig.suptitle(
        'Token-Level Activation Patching at "nurse"\n'
        f"Layer: {LAYER}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grid saved to {out_dir / 'grid.png'}")


def _print_summary(all_data, conditions):
    cond_labels = [c[0] for c in conditions]

    agg = defaultdict(list)
    for entry in all_data:
        for key, res in entry["scores"].items():
            agg[key].append(_to_male_prob(res))

    print(f"\n{'='*75}")
    print(f"  Token-Level Patching Summary ({len(all_data)} seeds)")
    print(f"  Layer: {LAYER}")
    print(f"{'='*75}")
    print(f"  {'Condition':<35} {'Mean MP':>8} {'Male%':>7} {'n':>4}")
    print(f"  {'-'*58}")

    for label in cond_labels:
        vals = [v for v in agg[label] if v == v]
        mp = np.mean(vals) if vals else float("nan")
        mf = np.mean([v > 0.5 for v in vals]) if vals else float("nan")
        print(f"  {label:<35} {mp:>8.3f} {mf:>6.0%} {len(vals):>4}")

    print(f"{'='*75}")
    print()
    print("  Key comparisons:")
    bl = [v for v in agg["neutral_baseline"] if v == v]
    patched = [v for v in agg["neutral+male_nurse_token"] if v == v]
    if bl and patched:
        print(f"    neutral baseline → +male token:  "
              f"{np.mean(bl):.3f} → {np.mean(patched):.3f}  "
              f"(Δ={np.mean(patched)-np.mean(bl):+.3f})")

    patched_f = [v for v in agg["neutral+female_nurse_token"] if v == v]
    if bl and patched_f:
        print(f"    neutral baseline → +female token: "
              f"{np.mean(bl):.3f} → {np.mean(patched_f):.3f}  "
              f"(Δ={np.mean(patched_f)-np.mean(bl):+.3f})")

    m_bl = [v for v in agg["male_baseline"] if v == v]
    m_patched = [v for v in agg["male+neutral_nurse_token"] if v == v]
    if m_bl and m_patched:
        print(f"    male baseline → +neutral token:   "
              f"{np.mean(m_bl):.3f} → {np.mean(m_patched):.3f}  "
              f"(Δ={np.mean(m_patched)-np.mean(m_bl):+.3f})")

    f_bl = [v for v in agg["female_baseline"] if v == v]
    f_patched = [v for v in agg["female+neutral_nurse_token"] if v == v]
    if f_bl and f_patched:
        print(f"    female baseline → +neutral token: "
              f"{np.mean(f_bl):.3f} → {np.mean(f_patched):.3f}  "
              f"(Δ={np.mean(f_patched)-np.mean(f_bl):+.3f})")
    print()


if __name__ == "__main__":
    main()
