"""PCA-based gender direction extraction and projection intervention.

Identifies the gender subspace in CLIP text encoder hidden states via PCA
on contrastive male/female prompt pairs, then projects it out during encoding
to remove gender signal while preserving occupation semantics.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from tracing.corrupt import encode_prompt_clean, get_uncond_embeddings
from tracing.hooks import ActivationCache
from tracing.restore import custom_denoise

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenderDirection:
    components: torch.Tensor             # (k, 768) orthonormal rows
    singular_values: torch.Tensor        # full spectrum for inspection
    explained_variance_ratio: torch.Tensor  # (k,) fraction of variance per component
    mean_diff: torch.Tensor              # (768,) mean of difference vectors


# ──────────────────────────────────────────────────────────────────────────────
# Token position helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_token_position(tokenizer, prompt, subject=None, position_type="eos"):
    """Find a token position index in the 77-token padded sequence.

    Args:
        tokenizer: CLIP tokenizer.
        prompt: Text prompt.
        subject: Subject word (required for last_subject/occupation).
        position_type: "eos", "last_subject", or "occupation".

    Returns:
        int position index.
    """
    if position_type == "eos":
        input_ids = tokenizer(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids[0].tolist()
        try:
            return input_ids.index(49407)
        except ValueError:
            raise ValueError(f"EOS token (49407) not found in: {prompt}")

    elif position_type in ("last_subject", "occupation"):
        if subject is None:
            raise ValueError(f"subject required for position_type='{position_type}'")
        from tracing.corrupt import find_subject_token_positions
        positions = find_subject_token_positions(tokenizer, prompt, subject)
        return positions[-1]

    else:
        raise ValueError(f"Unknown position_type: {position_type}")


# ──────────────────────────────────────────────────────────────────────────────
# Hidden state extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_hidden_states(
    pipe, prompts, layer_name, token_position_type="eos", subjects=None,
):
    """Extract hidden-state vectors at a specific layer and token position.

    Args:
        pipe: StableDiffusionPipeline.
        prompts: List of prompt strings.
        layer_name: Text encoder layer (e.g. "text_model.encoder.layers.0.self_attn").
        token_position_type: "eos", "last_subject", or "occupation".
        subjects: Parallel list of subject strings (needed for last_subject/occupation).

    Returns:
        Tensor (N, 768).
    """
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    activations = []

    for i, prompt in enumerate(prompts):
        subject = subjects[i] if subjects is not None else None
        pos = find_token_position(tokenizer, prompt, subject, token_position_type)

        cache = ActivationCache()
        cache.register_record_hooks(text_encoder, [layer_name])
        try:
            encode_prompt_clean(text_encoder, tokenizer, prompt)
        finally:
            cache.remove_all_hooks()

        tensor = cache.cache[layer_name]  # (1, 77, 768)
        activations.append(tensor[0, pos, :].cpu())
        cache.clear_cache()

    return torch.stack(activations)  # (N, 768)


# ──────────────────────────────────────────────────────────────────────────────
# PCA gender direction — single layer
# ──────────────────────────────────────────────────────────────────────────────

def _compute_single_layer_direction(
    pipe, male_prompts, female_prompts, layer_name,
    token_position_type, male_subjects, female_subjects, n_components,
):
    """Compute gender direction for one layer."""
    h_male = extract_hidden_states(
        pipe, male_prompts, layer_name, token_position_type, male_subjects,
    )
    h_female = extract_hidden_states(
        pipe, female_prompts, layer_name, token_position_type, female_subjects,
    )

    D = (h_male - h_female).float()
    mean_diff = D.mean(dim=0)
    D_centered = D - mean_diff

    U, S, Vh = torch.linalg.svd(D_centered, full_matrices=False)

    components = Vh[:n_components]
    total_var = (S ** 2).sum()
    explained = (S[:n_components] ** 2) / total_var

    logger.info(
        f"  {layer_name}: top-{n_components} explain {explained.sum():.1%}, "
        f"S={S[:3].tolist()}"
    )

    return GenderDirection(
        components=components,
        singular_values=S,
        explained_variance_ratio=explained,
        mean_diff=mean_diff,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PCA gender direction — multiple layers
# ──────────────────────────────────────────────────────────────────────────────

def compute_gender_directions(
    pipe,
    male_prompts,
    female_prompts,
    layer_names,
    token_position_type="eos",
    male_subjects=None,
    female_subjects=None,
    n_components=1,
):
    """Compute per-layer gender directions via PCA for each causal layer.

    Args:
        pipe: StableDiffusionPipeline.
        male_prompts: List of male prompts (paired with female_prompts).
        female_prompts: List of female prompts.
        layer_names: List of text encoder layer names to compute directions for.
        token_position_type: "eos", "last_subject", or "occupation".
        male_subjects: Subject tokens for male prompts (if needed).
        female_subjects: Subject tokens for female prompts (if needed).
        n_components: Number of PCA components (k) per layer.

    Returns:
        Dict mapping layer_name -> GenderDirection.
    """
    logger.info(
        f"Computing gender directions for {len(layer_names)} layers, "
        f"position={token_position_type}, k={n_components}"
    )

    directions = {}
    for layer_name in layer_names:
        directions[layer_name] = _compute_single_layer_direction(
            pipe, male_prompts, female_prompts, layer_name,
            token_position_type, male_subjects, female_subjects, n_components,
        )

    return directions


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_gender_direction(
    pipe,
    gender_dir,
    layer_name,
    held_out_male_prompts,
    held_out_female_prompts,
    occupation_prompts,
    occupation_subjects,
    token_position_type="eos",
    held_out_male_subjects=None,
    held_out_female_subjects=None,
):
    """Validate the gender direction with separation and orthogonality checks.

    Args:
        held_out_male_subjects: Subjects for held-out male prompts. If None
            and token_position_type requires subjects, falls back to EOS for
            the separation check.
        held_out_female_subjects: Same for female.

    Returns:
        Dict with separation_accuracy, occupation_gender_variance,
        occupation_random_variance, singular_value_spectrum.
    """
    g = gender_dir.components[1].float()  # primary direction (768,)

    # Check 1: separation on held-out gendered prompts
    # Fall back to EOS if subjects are unavailable for the requested position type
    sep_position = token_position_type
    if token_position_type in ("last_subject", "occupation"):
        if held_out_male_subjects is None or held_out_female_subjects is None:
            sep_position = "eos"

    h_male = extract_hidden_states(
        pipe, held_out_male_prompts, layer_name, sep_position,
        held_out_male_subjects,
    ).float()
    h_female = extract_hidden_states(
        pipe, held_out_female_prompts, layer_name, sep_position,
        held_out_female_subjects,
    ).float()

    male_proj = h_male @ g
    female_proj = h_female @ g

    n_correct = (male_proj > 0).sum() + (female_proj < 0).sum()
    n_total = len(male_proj) + len(female_proj)
    separation_accuracy = float(n_correct / n_total)

    # Check 2: occupation variance along gender direction vs random
    h_occ = extract_hidden_states(
        pipe, occupation_prompts, layer_name, token_position_type, occupation_subjects,
    ).float()

    occ_gender_var = float((h_occ @ g).var())

    random_dir = torch.randn_like(g)
    random_dir = random_dir / random_dir.norm()
    occ_random_var = float((h_occ @ random_dir).var())

    results = {
        "separation_accuracy": separation_accuracy,
        "occupation_gender_variance": occ_gender_var,
        "occupation_random_variance": occ_random_var,
        "singular_value_spectrum": gender_dir.singular_values[:10].tolist(),
    }

    logger.info(f"Validation [{layer_name}]: separation={separation_accuracy:.2%}, "
                f"occ_gender_var={occ_gender_var:.4f}, "
                f"occ_random_var={occ_random_var:.4f}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Image generation with PCA projection
# ──────────────────────────────────────────────────────────────────────────────

def generate_with_pca_projection(
    pipe,
    prompt,
    subject,
    gender_dirs,
    layer_names,
    token_position_type="eos",
    seed=0,
    target="textenc",
    timestep_window=None,
    num_steps=50,
    guidance_scale=7.5,
):
    """Generate an image with per-layer gender subspace projected out.

    Each layer in *layer_names* gets its own projection from *gender_dirs*.

    Args:
        pipe: StableDiffusionPipeline.
        prompt: Text prompt.
        subject: Subject token for position finding.
        gender_dirs: Dict mapping layer_name -> GenderDirection.
        layer_names: Layers to apply projection at.
        token_position_type: "eos", "last_subject", or "occupation".
        seed: Random seed.
        target: "textenc" (primary) or "unet".
        timestep_window: Optional (t_high, t_low) for UNet.

    Returns:
        PIL Image.
    """
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    device = pipe.device

    pos = find_token_position(tokenizer, prompt, subject, token_position_type)

    if target == "textenc":
        cache = ActivationCache()
        for layer_name in layer_names:
            components = gender_dirs[layer_name].components
            cache.register_projection_hook(
                text_encoder, layer_name, components,
                token_positions=[pos], conditional_only=False,
            )
        try:
            cond = encode_prompt_clean(text_encoder, tokenizer, prompt)
        finally:
            cache.remove_all_hooks()

        uncond = get_uncond_embeddings(text_encoder, tokenizer)
        prompt_embeds = torch.cat([uncond, cond])
        timestep_holder = [0]
        return custom_denoise(
            unet, scheduler, vae, prompt_embeds, timestep_holder,
            seed, device, guidance_scale, num_steps,
        )

    else:  # unet
        cond = encode_prompt_clean(text_encoder, tokenizer, prompt)
        uncond = get_uncond_embeddings(text_encoder, tokenizer)
        prompt_embeds = torch.cat([uncond, cond])

        timestep_holder = [0]
        cache = ActivationCache()
        for layer_name in layer_names:
            components = gender_dirs[layer_name].components
            cache.register_projection_hook(
                unet, layer_name, components,
                token_positions=[pos],
                timestep_holder=timestep_holder,
                timestep_window=timestep_window,
                conditional_only=True,
            )
        try:
            image = custom_denoise(
                unet, scheduler, vae, prompt_embeds, timestep_holder,
                seed, device, guidance_scale, num_steps,
            )
        finally:
            cache.remove_all_hooks()
        return image
