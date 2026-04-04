"""Soft-steering intervention: subtract a gender steering vector from layer outputs."""

import torch
from tqdm import tqdm

from tracing.corrupt import encode_prompt_clean, get_uncond_embeddings
from tracing.hooks import ActivationCache
from tracing.restore import custom_denoise


def compute_steering_vector(
    pipe,
    male_prompts,
    female_prompts,
    layer_name,
    target="unet",
    n_images=5,
    base_seed=42,
):
    """Compute a steering vector as mean(male activations) - mean(female activations).

    Args:
        pipe: StableDiffusionPipeline.
        male_prompts: List of male-gendered prompts.
        female_prompts: List of female-gendered prompts.
        layer_name: Layer name to compute the vector for.
        target: "unet" or "textenc".
        n_images: Seeds per prompt.
        base_seed: Starting seed.

    Returns:
        Steering vector tensor.
    """
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    device = pipe.device

    timestep_holder = [0]

    def collect_mean_activation(prompts_list):
        activations = []
        for prompt in tqdm(prompts_list, desc="Collecting activations", leave=False):
            for seed_offset in range(n_images):
                seed = base_seed + seed_offset

                if target == "textenc":
                    cache = ActivationCache()
                    cache.register_record_hooks(text_encoder, [layer_name])
                    try:
                        encode_prompt_clean(text_encoder, tokenizer, prompt)
                    finally:
                        cache.remove_all_hooks()

                    tensor = cache.cache.get(layer_name)
                    if tensor is not None:
                        activations.append(tensor.cpu())
                    cache.clear_cache()

                else:  # unet
                    cond = encode_prompt_clean(text_encoder, tokenizer, prompt)
                    uncond = get_uncond_embeddings(text_encoder, tokenizer)
                    prompt_embeds = torch.cat([uncond, cond])

                    cache = ActivationCache()
                    cache.register_record_hooks(
                        unet, [layer_name], timestep_holder=timestep_holder,
                    )
                    try:
                        custom_denoise(
                            unet, scheduler, vae, prompt_embeds, timestep_holder,
                            seed, device,
                        )
                    finally:
                        cache.remove_all_hooks()

                    layer_tensors = [
                        v for (n, t), v in cache.cache.items() if n == layer_name
                    ]
                    if layer_tensors:
                        mean_across_steps = torch.stack(layer_tensors).mean(dim=0)
                        activations.append(mean_across_steps.cpu())
                    cache.clear_cache()

        return torch.stack(activations).mean(dim=0) if activations else None

    male_mean = collect_mean_activation(male_prompts)
    female_mean = collect_mean_activation(female_prompts)

    if male_mean is None or female_mean is None:
        raise RuntimeError("Failed to collect activations for steering vector")

    return male_mean - female_mean


def generate_with_steering(
    pipe,
    prompt,
    layer_name,
    steering_vector,
    alpha=1.0,
    seed=0,
    target="unet",
    baseline_score=None,
    timestep_window=None,
    num_steps=50,
    guidance_scale=7.5,
):
    """Generate an image with adaptive gender steering toward parity.

    The steering vector points in the male direction (mean_male - mean_female).
    Subtracting it pushes toward female; adding it pushes toward male.

    When *baseline_score* is provided (male probability from baseline generation),
    the direction is chosen automatically:
      - baseline_score > 0.5 (male-biased): subtract sv (push toward female)
      - baseline_score < 0.5 (female-biased): add sv (push toward male)

    Args:
        pipe: StableDiffusionPipeline.
        prompt: Text prompt.
        layer_name: Layer to steer.
        steering_vector: Tensor from compute_steering_vector (male direction).
        alpha: Steering strength (sign may be flipped based on baseline_score).
        seed: Random seed.
        target: "unet" or "textenc".
        baseline_score: Optional male probability [0,1] from baseline generation.
            If provided, steering direction is chosen to push toward 0.5.
        timestep_window: Optional ``(t_high, t_low)`` (UNet only).
        num_steps: Denoising steps.
        guidance_scale: CFG scale.

    Returns:
        PIL Image.
    """
    # Adapt direction: if baseline is female-biased, flip alpha to push toward male
    if baseline_score is not None and baseline_score < 0.5:
        alpha = -alpha
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    device = pipe.device

    if target == "textenc":
        # Steer text encoder layer during encoding, then generate normally
        cache = ActivationCache()
        cache.register_steer_hook(
            text_encoder, layer_name, steering_vector, alpha=alpha,
            conditional_only=False,
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
        try:
            cache.register_steer_hook(
                unet, layer_name, steering_vector, alpha=alpha,
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
