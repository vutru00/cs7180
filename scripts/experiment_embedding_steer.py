"""Experiment: replace the gender token's activation at layer 0 self_attn.

Two modes:
  A) "A photo of a female nurse" — replace "female" token's activation with w * delta
  B) "A photo of a <|endoftext|> nurse" — replace the [UNK] token's activation with w * delta

where delta = act_male[nurse_layer] - act_female[nurse_layer] recorded at the
"female"/"male" token positions in "A photo of a female/male nurse".

Binary search finds w in [0, 1] for ~50% male/female across 20 images.

Usage:
    python scripts/experiment_embedding_steer.py
    python scripts/experiment_embedding_steer.py --n-images 20 --iterations 5
    python scripts/experiment_embedding_steer.py --layer text_model.encoder.layers.0.mlp
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

LAYER = "text_model.encoder.layers.0.self_attn"
MALE_PROMPT = "A photo of a male nurse"
FEMALE_PROMPT = "A photo of a female nurse"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace gender token activation at a layer with w*delta")
    parser.add_argument("--layer", type=str, default=LAYER)
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/embedding_steer")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def find_token_pos(tokenizer, prompt, word):
    """Find the position of a single-token word in the padded sequence."""
    input_ids = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    ).input_ids[0].tolist()
    word_id = tokenizer(word, add_special_tokens=False).input_ids[0]
    return input_ids.index(word_id)


def compute_layer_delta(pipe, layer_name):
    """Record layer activations for male/female prompts at the gender token.

    delta = act_male_at_"male" - act_female_at_"female" in layer output space.
    """
    from tracing.corrupt import encode_prompt_clean
    from tracing.hooks import ActivationCache

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    male_pos = find_token_pos(tokenizer, MALE_PROMPT, "male")
    female_pos = find_token_pos(tokenizer, FEMALE_PROMPT, "female")

    cache_m = ActivationCache()
    cache_m.register_record_hooks(text_encoder, [layer_name])
    try:
        encode_prompt_clean(text_encoder, tokenizer, MALE_PROMPT)
    finally:
        cache_m.remove_all_hooks()

    cache_f = ActivationCache()
    cache_f.register_record_hooks(text_encoder, [layer_name])
    try:
        encode_prompt_clean(text_encoder, tokenizer, FEMALE_PROMPT)
    finally:
        cache_f.remove_all_hooks()

    act_male = cache_m.cache[layer_name][0, male_pos, :]    # (768,)
    act_female = cache_f.cache[layer_name][0, female_pos, :]  # (768,)

    return act_male, male_pos, female_pos


def encode_with_replacement(pipe, prompt, replace_pos, layer_name, delta, weight):
    """Encode a prompt, hooking the layer to replace the activation at replace_pos with w*delta."""
    from tracing.corrupt import encode_prompt_clean

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    device = text_encoder.device
    d = delta.to(device)

    def replace_hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        tensor = (output[0] if is_tuple else output).clone()
        tensor[:, replace_pos, :] = (weight * d).to(tensor.dtype)
        return (tensor,) + output[1:] if is_tuple else tensor

    name_to_module = dict(text_encoder.named_modules())
    handle = name_to_module[layer_name].register_forward_hook(replace_hook)
    try:
        cond = encode_prompt_clean(text_encoder, tokenizer, prompt)
    finally:
        handle.remove()
    return cond


def evaluate(pipe, classifier, prompt, replace_pos, layer_name, delta, weight, n_images, base_seed):
    """Generate n_images and return (male_frac, mean_mp, images)."""
    from tracing.corrupt import get_uncond_embeddings
    from tracing.restore import custom_denoise

    device = pipe.device
    uncond = get_uncond_embeddings(pipe.text_encoder, pipe.tokenizer)
    cond = encode_with_replacement(pipe, prompt, replace_pos, layer_name, delta, weight)
    embeds = torch.cat([uncond, cond])

    images, male_probs = [], []
    male_count, total = 0, 0

    for i in range(n_images):
        th = [0]
        img = custom_denoise(pipe.unet, pipe.scheduler, pipe.vae, embeds, th,
                             base_seed + i, device)
        res = classifier.predict_single(img)
        images.append(img)
        if res is None:
            male_probs.append(float("nan"))
            continue
        mp = res["gender_score"] if res["gender"] == "male" else 1.0 - res["gender_score"]
        male_probs.append(mp)
        total += 1
        if mp > 0.5:
            male_count += 1

    male_frac = male_count / total if total > 0 else float("nan")
    return male_frac, float(np.nanmean(male_probs)), images


def binary_search(pipe, classifier, mode_label, prompt, replace_pos, layer_name, delta,
                  n_images, iterations, base_seed, out_dir):
    """Run binary search for one mode. Returns search log."""
    mode_dir = out_dir / mode_label
    (mode_dir / "images" / "baseline").mkdir(parents=True, exist_ok=True)

    # Baseline (w=0 means the token activation is zeroed — effectively removed)
    # Actually w=0 removes gender signal; let's also show the unmodified baseline
    print(f"\n  [{mode_label}] Baseline (no hook)...")
    from tracing.corrupt import encode_prompt_clean, get_uncond_embeddings
    from tracing.restore import custom_denoise
    uncond = get_uncond_embeddings(pipe.text_encoder, pipe.tokenizer)
    cond_raw = encode_prompt_clean(pipe.text_encoder, pipe.tokenizer, prompt)
    embeds_raw = torch.cat([uncond, cond_raw])

    bl_images, bl_male_probs = [], []
    bl_male_count, bl_total = 0, 0
    for i in range(n_images):
        th = [0]
        img = custom_denoise(pipe.unet, pipe.scheduler, pipe.vae, embeds_raw, th,
                             base_seed + i, pipe.device)
        res = classifier.predict_single(img)
        bl_images.append(img)
        img.save(mode_dir / "images" / "baseline" / f"{i}.png")
        if res is None:
            bl_male_probs.append(float("nan"))
            continue
        mp = res["gender_score"] if res["gender"] == "male" else 1.0 - res["gender_score"]
        bl_male_probs.append(mp)
        bl_total += 1
        if mp > 0.5:
            bl_male_count += 1

    bl_frac = bl_male_count / bl_total if bl_total > 0 else float("nan")
    bl_mp = float(np.nanmean(bl_male_probs))
    print(f"  [{mode_label}] baseline: male_frac={bl_frac:.0%}  mean_mp={bl_mp:.3f}")

    log = [{"iter": 0, "w": "baseline", "male_frac": bl_frac, "mean_mp": bl_mp}]

    lo, hi = 0.0, 1.0
    best_w, best_diff, best_frac, best_images = 0.0, abs(bl_frac - 0.5), bl_frac, bl_images

    for it in range(1, iterations + 1):
        w = (lo + hi) / 2
        print(f"  [{mode_label}] iter {it}/{iterations}: w={w:.4f}  [{lo:.4f}, {hi:.4f}]")

        frac, mp, images = evaluate(
            pipe, classifier, prompt, replace_pos, layer_name, delta,
            w, n_images, base_seed,
        )
        print(f"    male_frac={frac:.0%}  mean_mp={mp:.3f}")

        log.append({"iter": it, "w": round(w, 6), "lo": round(lo, 6), "hi": round(hi, 6),
                     "male_frac": frac, "mean_mp": round(mp, 4)})

        idir = mode_dir / "images" / f"w_{w:.4f}"
        idir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(idir / f"{i}.png")

        diff = abs(frac - 0.5)
        if diff < best_diff:
            best_diff, best_w, best_frac, best_images = diff, w, frac, images

        if frac < 0.5:
            lo = w
            print("    -> too female, increase w")
        elif frac > 0.5:
            hi = w
            print("    -> too male, decrease w")
        else:
            print("    -> parity!")
            break

    print(f"  [{mode_label}] best w={best_w:.4f}  male_frac={best_frac:.0%}\n")
    return log, bl_images, best_images, best_w, bl_frac


def main():
    args = parse_args()

    from models.load_model import load_sd_pipeline, load_mivolo_predictor
    from classifiers.mivolo_classifier import MiVOLOClassifier

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = load_sd_pipeline(device=args.device, dtype=dtype)
    predictor = load_mivolo_predictor(device=args.device)
    classifier = MiVOLOClassifier(predictor)

    tokenizer = pipe.tokenizer

    # Compute delta at the layer: act_male["male"] - act_female["female"]
    delta, male_pos, female_pos = compute_layer_delta(pipe, args.layer)
    print(f"Layer: {args.layer}")
    print(f"'male' at pos {male_pos} in \"{MALE_PROMPT}\"")
    print(f"'female' at pos {female_pos} in \"{FEMALE_PROMPT}\"")
    print(f"Delta norm: {delta.norm():.4f}")

    out_dir = Path(args.output_dir)

    # ── Mode A: "A photo of a female nurse" — replace "female" activation ─
    prompt_a = FEMALE_PROMPT
    replace_pos_a = female_pos  # position of "female" token
    print(f"\n{'='*65}")
    print(f"  Mode A: \"{prompt_a}\"")
    print(f"  Replace activation at pos {replace_pos_a} ('female') with w * delta")
    print(f"{'='*65}")

    log_a, bl_a, best_a, bw_a, blfrac_a = binary_search(
        pipe, classifier, "mode_A_female", prompt_a, replace_pos_a, args.layer, delta,
        args.n_images, args.iterations, args.base_seed, out_dir,
    )

    # ── Mode B: "A photo of a <|endoftext|> nurse" — replace [UNK] activation ─
    unk_token = tokenizer.unk_token  # <|endoftext|>
    prompt_b = f"A photo of a {unk_token} nurse"
    # The UNK token lands at the same position as "female" (pos 5)
    replace_pos_b = female_pos
    print(f"{'='*65}")
    print(f"  Mode B: \"{prompt_b}\"")
    print(f"  Replace activation at pos {replace_pos_b} ('{unk_token}') with w * delta")
    print(f"{'='*65}")

    log_b, bl_b, best_b, bw_b, blfrac_b = binary_search(
        pipe, classifier, "mode_B_unk", prompt_b, replace_pos_b, args.layer, delta,
        args.n_images, args.iterations, args.base_seed, out_dir,
    )

    # ── Save logs ────────────────────────────────────────────────────────
    with open(out_dir / "search_log.json", "w") as f:
        json.dump({"mode_A_female": log_a, "mode_B_unk": log_b}, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Mode A (\"{FEMALE_PROMPT}\", replace 'female'):")
    print(f"    baseline male_frac={blfrac_a:.0%}  best w={bw_a:.4f}")
    print(f"  Mode B (\"A photo of a {unk_token} nurse\", replace '{unk_token}'):")
    print(f"    baseline male_frac={blfrac_b:.0%}  best w={bw_b:.4f}")
    print(f"{'='*65}\n")

    _visualize(log_a, log_b, bl_a, best_a, bw_a, bl_b, best_b, bw_b, args, out_dir)


def _visualize(log_a, log_b, bl_a, best_a, bw_a, bl_b, best_b, bw_b, args, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── Row 0: convergence curves ────────────────────────────────────────
    for col, (log, label, color) in enumerate([
        (log_a, "Mode A: replace 'female'", "#e74c3c"),
        (log_b, "Mode B: replace [UNK]", "#2980b9"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        iters = [e["iter"] for e in log]
        fracs = [e["male_frac"] for e in log]
        ax.plot(iters, fracs, "o-", color=color, label="Male fraction")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Target")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Male fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(label)
        ax.legend(fontsize=8)

    # ── Row 1: Mode A images ─────────────────────────────────────────────
    n_show = min(5, len(bl_a))
    inner_a1 = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[1, 0], wspace=0.05)
    for i in range(n_show):
        ax = fig.add_subplot(inner_a1[0, i])
        ax.imshow(bl_a[i])
        ax.axis("off")
        if i == 0:
            ax.set_title("A: baseline", fontsize=8)

    n_show = min(5, len(best_a))
    inner_a2 = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[1, 1], wspace=0.05)
    for i in range(n_show):
        ax = fig.add_subplot(inner_a2[0, i])
        ax.imshow(best_a[i])
        ax.axis("off")
        if i == 0:
            ax.set_title(f"A: best w={bw_a:.4f}", fontsize=8)

    # ── Row 2: Mode B images ─────────────────────────────────────────────
    n_show = min(5, len(bl_b))
    inner_b1 = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[2, 0], wspace=0.05)
    for i in range(n_show):
        ax = fig.add_subplot(inner_b1[0, i])
        ax.imshow(bl_b[i])
        ax.axis("off")
        if i == 0:
            ax.set_title("B: baseline", fontsize=8)

    n_show = min(5, len(best_b))
    inner_b2 = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[2, 1], wspace=0.05)
    for i in range(n_show):
        ax = fig.add_subplot(inner_b2[0, i])
        ax.imshow(best_b[i])
        ax.axis("off")
        if i == 0:
            ax.set_title(f"B: best w={bw_b:.4f}", fontsize=8)

    fig.suptitle(
        f"Gender Token Replacement at {args.layer}\n"
        f"delta = act_male['male'] - act_female['female']  |  replace token activation with w * delta",
        fontsize=10,
    )
    fig.savefig(out_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Summary saved to {out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
