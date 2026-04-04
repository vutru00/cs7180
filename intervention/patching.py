"""Patching intervention: selectively de-bias causal layers.

Runs a clean forward pass (original prompt) but patches the *causal* layers
with activations from a corrupted forward pass (subject → "person").  The
causal layers — identified by tracing as the ones that mediate demographic
bias — receive the de-biased "person" activations, while every other layer
keeps the original occupation signal.  This preserves image quality and
occupation relevance while reducing gender/age bias.
"""

import torch

from tracing.corrupt import encode_corrupted_prompt, encode_prompt_clean, get_uncond_embeddings
from tracing.hooks import ActivationCache
from tracing.restore import custom_denoise


def generate_with_patching(
    pipe,
    prompt,
    subject,
    layer_names,
    seed,
    target="unet",
    timestep_window=None,
    num_steps=50,
    guidance_scale=7.5,
):
    """Generate an image with causal layers patched to de-biased activations.

    1. Run a *corrupted* forward pass (subject → "person") to collect
       de-biased activations at the causal layers.
    2. Run a *clean* forward pass (original prompt) but hook the causal
       layers to replace their outputs with the de-biased activations.

    Args:
        pipe: StableDiffusionPipeline.
        prompt: Original text prompt (e.g. "A photo of a nurse").
        subject: Subject token to corrupt (e.g. "nurse").
        layer_names: Causal layer names to patch with de-biased activations.
        seed: Random seed (same for both passes).
        target: "unet" or "textenc".
        timestep_window: Optional ``(t_high, t_low)`` (UNet only).
        num_steps: Denoising steps.
        guidance_scale: CFG scale.

    Returns:
        PIL Image.
    """
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    device = pipe.device

    if target == "textenc":
        return _patching_textenc(
            text_encoder, tokenizer, unet, vae, scheduler,
            prompt, subject, layer_names, seed, device,
            num_steps, guidance_scale,
        )
    else:
        return _patching_unet(
            text_encoder, tokenizer, unet, vae, scheduler,
            prompt, subject, layer_names, seed, device,
            timestep_window, num_steps, guidance_scale,
        )


def _patching_textenc(
    text_encoder, tokenizer, unet, vae, scheduler,
    prompt, subject, layer_names, seed, device,
    num_steps, guidance_scale,
):
    """Text encoder patching: collect de-biased activations, patch into clean run."""
    # Step 1: corrupted encoding — record de-biased activations at causal layers
    record_cache = ActivationCache()
    record_cache.register_record_hooks(text_encoder, layer_names)
    try:
        encode_corrupted_prompt(text_encoder, tokenizer, prompt, subject)
    finally:
        record_cache.remove_all_hooks()

    debiased_cache = dict(record_cache.cache)  # {layer_name: tensor}
    record_cache.clear_cache()

    # Step 2: clean encoding with causal layers patched from de-biased cache
    patch_cache = ActivationCache()
    for layer_name in layer_names:
        patch_cache.register_patch_hook(
            text_encoder, layer_name, debiased_cache,
            conditional_only=False,
        )
    try:
        cond_patched = encode_prompt_clean(text_encoder, tokenizer, prompt)
    finally:
        patch_cache.remove_all_hooks()

    # Step 3: normal denoising with patched conditioning
    uncond = get_uncond_embeddings(text_encoder, tokenizer)
    prompt_embeds = torch.cat([uncond, cond_patched])
    timestep_holder = [0]
    return custom_denoise(
        unet, scheduler, vae, prompt_embeds, timestep_holder,
        seed, device, guidance_scale, num_steps,
    )


def _patching_unet(
    text_encoder, tokenizer, unet, vae, scheduler,
    prompt, subject, layer_names, seed, device,
    timestep_window, num_steps, guidance_scale,
):
    """UNet patching: collect de-biased activations, patch into clean run."""
    timestep_holder = [0]

    # Step 1: corrupted forward pass — record de-biased activations at causal layers
    cond_corrupted = encode_corrupted_prompt(text_encoder, tokenizer, prompt, subject)
    uncond = get_uncond_embeddings(text_encoder, tokenizer)
    prompt_embeds_corrupted = torch.cat([uncond, cond_corrupted])

    record_cache = ActivationCache()
    record_cache.register_record_hooks(
        unet, layer_names, timestep_holder=timestep_holder,
    )
    try:
        custom_denoise(
            unet, scheduler, vae, prompt_embeds_corrupted, timestep_holder,
            seed, device, guidance_scale, num_steps,
        )
    finally:
        record_cache.remove_all_hooks()

    debiased_cache = {k: v.cpu() for k, v in record_cache.cache.items()}
    record_cache.clear_cache()

    # Step 2: clean pass with causal layers patched from de-biased cache
    cond_clean = encode_prompt_clean(text_encoder, tokenizer, prompt)
    prompt_embeds_clean = torch.cat([uncond, cond_clean])

    patch_cache = ActivationCache()
    for layer_name in layer_names:
        patch_cache.register_patch_hook(
            unet, layer_name, debiased_cache,
            timestep_holder=timestep_holder,
            timestep_window=timestep_window,
            conditional_only=True,
        )
    try:
        image = custom_denoise(
            unet, scheduler, vae, prompt_embeds_clean, timestep_holder,
            seed, device, guidance_scale, num_steps,
        )
    finally:
        patch_cache.remove_all_hooks()

    return image
