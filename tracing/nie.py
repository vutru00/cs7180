"""Natural Indirect Effect computation and full causal-tracing orchestrator."""

import json
import logging
from collections import defaultdict

import torch
from tqdm import tqdm

from tracing.corrupt import generate_corrupted_image
from tracing.hooks import enumerate_text_encoder_layers, enumerate_unet_layers
from tracing.restore import generate_clean_and_cache, generate_restored_image

logger = logging.getLogger(__name__)


def compute_nie(score_restored, score_corrupted):
    """NIE = score(restored) - score(corrupted)."""
    return score_restored - score_corrupted


def run_full_tracing(
    pipe,
    classifier,
    prompts,
    target,
    bias_dim,
    n_images=5,
    base_seed=42,
    timestep_window=None,
    layer_names=None,
    device="cuda",
):
    """Orchestrate the three-pass causal tracing protocol.

    For each prompt and seed:
      1. Clean pass – generate image and record all layer activations.
      2. Corrupted pass – generate image with subject → "person".
      3. Restored passes – for each layer, corrupted + that layer patched from
         clean cache. Compute NIE = score(restored) - score(corrupted).

    Args:
        pipe: StableDiffusionPipeline (already on device).
        classifier: MiVOLOClassifier instance.
        prompts: List of dicts ``{"prompt": str, "subject": str}``.
        target: "unet", "textenc", or "both".
        bias_dim: "gender" or "age".
        n_images: Number of images (seeds) per prompt.
        base_seed: Starting seed.
        timestep_window: Optional ``(t_high, t_low)`` for Phase 2.
        layer_names: Optional list of layer names to probe. If None, enumerate
            all layers for the given target.
        device: Torch device string.

    Returns:
        Dict ``{prompt_text: {layer_name: nie_value}}``.
    """
    targets = [target] if target != "both" else ["textenc", "unet"]

    # Determine layers to probe
    all_layers = {}
    for tgt in targets:
        if layer_names is not None:
            all_layers[tgt] = layer_names
        elif tgt == "textenc":
            all_layers[tgt] = enumerate_text_encoder_layers()
        elif tgt == "unet":
            all_layers[tgt] = enumerate_unet_layers(pipe.unet)

    results = {}
    timestep_holder = [0]

    for prompt_info in tqdm(prompts, desc="Prompts"):
        prompt_text = prompt_info["prompt"]
        subject = prompt_info["subject"]

        # Accumulate NIE across seeds: {(target, layer): [nie_values]}
        nie_accum = defaultdict(list)

        for seed_offset in range(n_images):
            seed = base_seed + seed_offset

            for tgt in targets:
                layers = all_layers[tgt]

                # 1. Clean pass + record activations
                logger.info(f"Clean pass: prompt='{prompt_text}', seed={seed}, target={tgt}")
                clean_image, clean_cache = generate_clean_and_cache(
                    pipe, prompt_text, seed, layers, tgt, timestep_holder,
                )

                # Move clean cache to CPU to free GPU memory
                clean_cache = {k: v.cpu() for k, v in clean_cache.items()}

                # 2. Corrupted pass
                logger.info(f"Corrupted pass: prompt='{prompt_text}', seed={seed}")
                corrupted_image = generate_corrupted_image(pipe, prompt_text, subject, seed)

                # Score corrupted image
                corrupted_scores = classifier.extract_bias_score([corrupted_image], dim=bias_dim)
                score_corrupted = corrupted_scores[0]

                if score_corrupted != score_corrupted:  # NaN check
                    logger.warning(f"No detection in corrupted image for '{prompt_text}' seed={seed}")
                    continue

                # 3. Restored passes — one per layer
                for layer in tqdm(layers, desc=f"Layers ({tgt})", leave=False):
                    restored_image = generate_restored_image(
                        pipe, prompt_text, subject, layer, clean_cache,
                        tgt, seed, timestep_holder, timestep_window,
                    )

                    restored_scores = classifier.extract_bias_score([restored_image], dim=bias_dim)
                    score_restored = restored_scores[0]

                    if score_restored != score_restored:  # NaN
                        continue

                    nie = compute_nie(score_restored, score_corrupted)
                    nie_accum[(tgt, layer)].append(nie)

                # Free clean cache for this seed
                del clean_cache

        # Average NIE across seeds
        prompt_results = {}
        for (tgt, layer), values in nie_accum.items():
            key = f"{tgt}/{layer}" if len(targets) > 1 else layer
            prompt_results[key] = sum(values) / len(values) if values else 0.0

        results[prompt_text] = prompt_results

    return results


def save_results(results, output_path):
    """Save tracing results as JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
