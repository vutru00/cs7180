"""Subject-token corruption for causal tracing.

Replaces the subject token embedding (e.g. "nurse") with the embedding of
"person" so that occupation-specific demographic associations are removed
while preserving the human anchor needed for MiVOLO detection.
"""

import torch


def find_subject_token_positions(tokenizer, prompt, subject):
    """Find the contiguous span of *subject* tokens within *prompt*.

    Returns a list of integer positions within the padded token sequence.
    Raises ValueError if the subject tokens are not found.
    """
    prompt_ids = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    ).input_ids[0].tolist()

    subject_ids = tokenizer(
        subject, add_special_tokens=False, return_tensors="pt",
    ).input_ids[0].tolist()

    n = len(subject_ids)
    for start in range(len(prompt_ids) - n + 1):
        if prompt_ids[start : start + n] == subject_ids:
            return list(range(start, start + n))

    raise ValueError(
        f"Subject '{subject}' (tokens {subject_ids}) not found in prompt '{prompt}' "
        f"(tokens {prompt_ids[:20]}...)"
    )


def create_corrupted_embeddings(text_encoder, tokenizer, prompt, subject, replacement="person"):
    """Create token-level embeddings with *subject* replaced by *replacement*.

    Returns a tensor of shape ``(1, 77, 768)`` suitable for passing as
    ``inputs_embeds`` to the text encoder.
    """
    device = text_encoder.device

    # Full prompt token embeddings (token + position)
    prompt_ids = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        clean_embeds = text_encoder.text_model.embeddings(prompt_ids)  # (1, 77, 768)

    # Replacement token embedding (token embedding only, no position)
    replacement_ids = tokenizer(
        replacement, add_special_tokens=False, return_tensors="pt",
    ).input_ids[0].to(device)

    with torch.no_grad():
        replacement_token_embeds = text_encoder.text_model.embeddings.token_embedding(
            replacement_ids
        )  # (N_rep, 768)

    # Find subject positions and replace
    positions = find_subject_token_positions(tokenizer, prompt, subject)

    corrupted = clean_embeds.clone()
    for i, pos in enumerate(positions):
        rep_idx = min(i, len(replacement_token_embeds) - 1)
        corrupted[0, pos] = replacement_token_embeds[rep_idx]

    return corrupted


def encode_corrupted_prompt(text_encoder, tokenizer, prompt, subject, replacement="person"):
    """Run the text encoder with corrupted token embeddings.

    CLIPTextModel.forward() always requires input_ids and calls
    self.embeddings(input_ids=...) internally — it does not accept
    inputs_embeds at the top level.  We therefore hook the embeddings
    sub-module to replace its output with our pre-computed corrupted
    embeddings before the encoder layers see it.

    Returns the final hidden-state conditioning tensor (1, 77, 768).
    """
    device = text_encoder.device

    corrupted_embeds = create_corrupted_embeddings(
        text_encoder, tokenizer, prompt, subject, replacement,
    )

    # input_ids are still needed for the attention mask / EOS pooling logic
    input_ids = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    ).input_ids.to(device)

    # Hook the embeddings layer to return corrupted embeddings instead
    def replace_embeddings(module, inputs, output):
        return corrupted_embeds

    handle = text_encoder.text_model.embeddings.register_forward_hook(replace_embeddings)
    try:
        with torch.no_grad():
            encoder_output = text_encoder(input_ids=input_ids)
    finally:
        handle.remove()

    return encoder_output[0]  # last_hidden_state


def encode_prompt_clean(text_encoder, tokenizer, prompt):
    """Encode a prompt through the text encoder normally (no corruption)."""
    device = text_encoder.device
    input_ids = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        encoder_output = text_encoder(input_ids)
    return encoder_output[0]


def get_uncond_embeddings(text_encoder, tokenizer):
    """Encode the empty (unconditional) prompt for classifier-free guidance."""
    return encode_prompt_clean(text_encoder, tokenizer, "")


def generate_corrupted_image(
    pipe, prompt, subject, seed, num_steps=50, guidance_scale=7.5,
):
    """Generate a full image using corrupted conditioning.

    Uses the diffusers ``prompt_embeds`` API to bypass internal text encoding.
    """
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    device = pipe.device

    corrupted_cond = encode_corrupted_prompt(text_encoder, tokenizer, prompt, subject)
    uncond = get_uncond_embeddings(text_encoder, tokenizer)

    generator = torch.Generator(device=device).manual_seed(seed)
    output = pipe(
        prompt_embeds=corrupted_cond,
        negative_prompt_embeds=uncond,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return output.images[0]
