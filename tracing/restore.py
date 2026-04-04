"""Restoration passes and custom denoising loop for causal tracing.

The custom denoising loop replicates the behaviour of
``StableDiffusionPipeline.__call__`` but exposes per-step control over
a ``timestep_holder`` so that activation hooks can be timestep-gated.
"""

import torch
from diffusers import DDIMScheduler

from tracing.corrupt import encode_corrupted_prompt, encode_prompt_clean, get_uncond_embeddings
from tracing.hooks import ActivationCache, hook_context


# ======================================================================
# Custom denoising loop
# ======================================================================

def custom_denoise(
    unet,
    scheduler,
    vae,
    prompt_embeds,
    timestep_holder,
    seed,
    device,
    guidance_scale=7.5,
    num_steps=50,
    latent_shape=(1, 4, 64, 64),
):
    """Run the full denoising loop with explicit timestep tracking.

    Args:
        unet: The UNet model.
        scheduler: Diffusion scheduler (e.g. DDIMScheduler).
        vae: VAE decoder.
        prompt_embeds: Concatenated ``[uncond_embeds, cond_embeds]`` of shape
            ``(2, 77, 768)``.
        timestep_holder: Mutable ``[current_t]`` list updated each step.
        seed: Random seed for reproducible initial latents.
        device: Torch device.
        guidance_scale: Classifier-free guidance scale.
        num_steps: Number of denoising steps.
        latent_shape: Shape of the initial noise latents.

    Returns:
        PIL Image.
    """
    scheduler.set_timesteps(num_steps, device=device)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=unet.dtype)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        timestep_holder[0] = t.item()

        latent_input = torch.cat([latents, latents])  # CFG: [uncond, cond]
        latent_input = scheduler.scale_model_input(latent_input, t)

        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=prompt_embeds).sample

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to image
    with torch.no_grad():
        latents_scaled = latents / vae.config.scaling_factor
        image = vae.decode(latents_scaled).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    from diffusers.image_processor import VaeImageProcessor
    processor = VaeImageProcessor()
    pil_images = processor.numpy_to_pil(image)
    return pil_images[0]


# ======================================================================
# Clean generation + activation recording
# ======================================================================

def generate_clean_and_cache(
    pipe,
    prompt,
    seed,
    layer_names,
    target,
    timestep_holder,
    num_steps=50,
    guidance_scale=7.5,
):
    """Run a clean forward pass and record activations at all specified layers.

    Args:
        pipe: StableDiffusionPipeline.
        prompt: Text prompt.
        seed: Random seed.
        layer_names: Layer names to record.
        target: "unet" or "textenc".
        timestep_holder: Mutable ``[current_t]`` list.
        num_steps: Number of denoising steps.
        guidance_scale: CFG scale.

    Returns:
        (PIL Image, dict of cached activations)
    """
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    device = pipe.device

    cache = ActivationCache()

    try:
        if target == "textenc":
            # Record text encoder layers during encoding
            cache.register_record_hooks(text_encoder, layer_names)
            cond_embeds = encode_prompt_clean(text_encoder, tokenizer, prompt)
            cache.remove_all_hooks()

            # Generate image normally (no hooks during denoising)
            uncond = get_uncond_embeddings(text_encoder, tokenizer)
            prompt_embeds = torch.cat([uncond, cond_embeds])
            image = custom_denoise(
                unet, scheduler, vae, prompt_embeds, timestep_holder,
                seed, device, guidance_scale, num_steps,
            )

        elif target == "unet":
            # Encode prompt normally
            cond_embeds = encode_prompt_clean(text_encoder, tokenizer, prompt)
            uncond = get_uncond_embeddings(text_encoder, tokenizer)
            prompt_embeds = torch.cat([uncond, cond_embeds])

            # Record UNet layers during denoising (keyed by timestep)
            cache.register_record_hooks(unet, layer_names, timestep_holder=timestep_holder)
            image = custom_denoise(
                unet, scheduler, vae, prompt_embeds, timestep_holder,
                seed, device, guidance_scale, num_steps,
            )
            cache.remove_all_hooks()

        else:
            raise ValueError(f"Unknown target: {target}")

    except Exception:
        cache.remove_all_hooks()
        raise

    return image, cache.cache


# ======================================================================
# Restored generation
# ======================================================================

def generate_restored_image(
    pipe,
    prompt,
    subject,
    restore_layer,
    clean_cache,
    target,
    seed,
    timestep_holder,
    timestep_window=None,
    num_steps=50,
    guidance_scale=7.5,
):
    """Corrupted generation with one layer's activations restored from the clean cache.

    Args:
        pipe: StableDiffusionPipeline.
        prompt: Text prompt.
        subject: Subject token to corrupt.
        restore_layer: Layer name to restore.
        clean_cache: Dict of clean activations from generate_clean_and_cache.
        target: "unet" or "textenc".
        seed: Same seed used for clean pass.
        timestep_holder: Mutable ``[current_t]`` list.
        timestep_window: Optional ``(t_high, t_low)`` for Phase 2.
        num_steps: Number of denoising steps.
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

    cache = ActivationCache()

    try:
        if target == "textenc":
            # Patch one text encoder layer: run corrupted encoding but hook
            # the target layer to output its clean activation instead
            cache.register_patch_hook(
                text_encoder, restore_layer, clean_cache,
                timestep_holder=None, timestep_window=None, conditional_only=False,
            )
            cond_embeds = encode_corrupted_prompt(text_encoder, tokenizer, prompt, subject)
            cache.remove_all_hooks()

            uncond = get_uncond_embeddings(text_encoder, tokenizer)
            prompt_embeds = torch.cat([uncond, cond_embeds])
            image = custom_denoise(
                unet, scheduler, vae, prompt_embeds, timestep_holder,
                seed, device, guidance_scale, num_steps,
            )

        elif target == "unet":
            # Encode with corrupted conditioning
            cond_embeds = encode_corrupted_prompt(text_encoder, tokenizer, prompt, subject)
            uncond = get_uncond_embeddings(text_encoder, tokenizer)
            prompt_embeds = torch.cat([uncond, cond_embeds])

            # Hook the target UNet layer to patch from clean cache
            cache.register_patch_hook(
                unet, restore_layer, clean_cache,
                timestep_holder=timestep_holder,
                timestep_window=timestep_window,
                conditional_only=True,
            )
            image = custom_denoise(
                unet, scheduler, vae, prompt_embeds, timestep_holder,
                seed, device, guidance_scale, num_steps,
            )
            cache.remove_all_hooks()

        else:
            raise ValueError(f"Unknown target: {target}")

    except Exception:
        cache.remove_all_hooks()
        raise

    return image
