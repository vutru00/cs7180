"""Hard-block intervention: zero out causal layer outputs during generation."""

import torch

from tracing.corrupt import encode_prompt_clean, get_uncond_embeddings
from tracing.hooks import ActivationCache
from tracing.restore import custom_denoise


def generate_with_hard_block(
    pipe,
    prompt,
    layer_names,
    seed,
    target="unet",
    timestep_window=None,
    num_steps=50,
    guidance_scale=7.5,
):
    """Generate an image with specified layers zeroed out.

    Args:
        pipe: StableDiffusionPipeline.
        prompt: Text prompt.
        layer_names: List of layer names to zero out.
        seed: Random seed.
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
        # Zero text encoder layers during encoding, then generate normally
        cache = ActivationCache()
        for layer_name in layer_names:
            cache.register_zero_hook(
                text_encoder, layer_name, conditional_only=False,
            )
        try:
            cond_embeds = encode_prompt_clean(text_encoder, tokenizer, prompt)
        finally:
            cache.remove_all_hooks()

        uncond = get_uncond_embeddings(text_encoder, tokenizer)
        prompt_embeds = torch.cat([uncond, cond_embeds])
        timestep_holder = [0]
        return custom_denoise(
            unet, scheduler, vae, prompt_embeds, timestep_holder,
            seed, device, guidance_scale, num_steps,
        )

    else:  # unet
        cond_embeds = encode_prompt_clean(text_encoder, tokenizer, prompt)
        uncond = get_uncond_embeddings(text_encoder, tokenizer)
        prompt_embeds = torch.cat([uncond, cond_embeds])

        timestep_holder = [0]
        cache = ActivationCache()
        try:
            for layer_name in layer_names:
                cache.register_zero_hook(
                    unet, layer_name,
                    timestep_holder=timestep_holder,
                    timestep_window=timestep_window,
                    conditional_only=True,
                )
            image = custom_denoise(
                unet, scheduler, vae, prompt_embeds, timestep_holder,
                seed, device, guidance_scale, num_steps,
            )
        finally:
            cache.remove_all_hooks()
        return image
