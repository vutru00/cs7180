"""Activation caching and patching infrastructure for causal tracing."""

from contextlib import contextmanager

import torch
import torch.nn as nn


class ActivationCache:
    """Records and patches activations via PyTorch forward hooks.

    Two modes:
        record – passthrough but save the output tensor.
        patch  – replace the module output with a previously stored tensor.

    For UNet hooks that run once per timestep, activations are keyed by
    (layer_name, timestep) when a timestep_holder is provided.
    """

    def __init__(self):
        self.cache = {}
        self._handles = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def register_record_hooks(self, model, layer_names, timestep_holder=None):
        """Attach hooks that save output tensors to *self.cache*.

        If *timestep_holder* is provided (a mutable list whose element [0] is
        updated externally with the current diffusion timestep), keys become
        ``(layer_name, int(timestep))``.  Otherwise keys are just the layer
        name (suitable for the text encoder which runs once).
        """
        name_to_module = dict(model.named_modules())
        for name in layer_names:
            module = name_to_module.get(name)
            if module is None:
                raise ValueError(f"Module '{name}' not found in model")

            hook_fn = self._make_record_hook(name, timestep_holder)
            handle = module.register_forward_hook(hook_fn)
            self._handles.append(handle)

    def _make_record_hook(self, name, timestep_holder):
        cache = self.cache

        def hook_fn(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            key = (name, int(timestep_holder[0])) if timestep_holder is not None else name
            cache[key] = tensor.detach().clone()

        return hook_fn

    # ------------------------------------------------------------------
    # Patching
    # ------------------------------------------------------------------

    def register_patch_hook(
        self,
        model,
        layer_name,
        clean_cache,
        timestep_holder=None,
        timestep_window=None,
        conditional_only=True,
    ):
        """Attach a hook that replaces the output with a cached clean tensor.

        Args:
            model: The model (UNet or text encoder).
            layer_name: Dot-path of the module to hook.
            clean_cache: Dict mapping keys to tensors (same key scheme as recording).
            timestep_holder: Mutable ``[current_t]`` list (needed for UNet).
            timestep_window: ``(t_high, t_low)`` – only patch when
                ``t_high >= current_t >= t_low``.  Timesteps count down.
            conditional_only: If True only patch the conditional (second) half
                of the CFG batch.
        """
        name_to_module = dict(model.named_modules())
        module = name_to_module.get(layer_name)
        if module is None:
            raise ValueError(f"Module '{layer_name}' not found in model")

        hook_fn = self._make_patch_hook(
            layer_name, clean_cache, timestep_holder, timestep_window, conditional_only,
        )
        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    def _make_patch_hook(self, name, clean_cache, timestep_holder, timestep_window, conditional_only):
        def hook_fn(module, input, output):
            # Timestep gate
            if timestep_holder is not None and timestep_window is not None:
                t = int(timestep_holder[0])
                t_high, t_low = timestep_window
                if not (t_high >= t >= t_low):
                    return output

            # Look up the clean activation
            key = (name, int(timestep_holder[0])) if timestep_holder is not None else name
            clean_tensor = clean_cache.get(key)
            if clean_tensor is None:
                return output

            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output

            # Ensure device match
            if clean_tensor.device != tensor.device:
                clean_tensor = clean_tensor.to(tensor.device)

            if conditional_only and tensor.shape[0] >= 2:
                # CFG batch: [unconditional, conditional]
                half = tensor.shape[0] // 2
                patched = tensor.clone()
                patched[half:] = clean_tensor[half:]
                result = patched
            else:
                result = clean_tensor.clone()

            return (result,) + output[1:] if is_tuple else result

        return hook_fn

    # ------------------------------------------------------------------
    # Zeroing (for hard-block intervention)
    # ------------------------------------------------------------------

    def register_zero_hook(
        self,
        model,
        layer_name,
        timestep_holder=None,
        timestep_window=None,
        conditional_only=True,
    ):
        """Attach a hook that zeros out the layer output (conditional pass only)."""
        name_to_module = dict(model.named_modules())
        module = name_to_module.get(layer_name)
        if module is None:
            raise ValueError(f"Module '{layer_name}' not found in model")

        def hook_fn(module, input, output):
            if timestep_holder is not None and timestep_window is not None:
                t = int(timestep_holder[0])
                t_high, t_low = timestep_window
                if not (t_high >= t >= t_low):
                    return output

            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output

            if conditional_only and tensor.shape[0] >= 2:
                half = tensor.shape[0] // 2
                patched = tensor.clone()
                patched[half:] = 0
                result = patched
            else:
                result = torch.zeros_like(tensor)

            return (result,) + output[1:] if is_tuple else result

        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    # ------------------------------------------------------------------
    # Steering (for soft-steer intervention)
    # ------------------------------------------------------------------

    def register_steer_hook(
        self,
        model,
        layer_name,
        steering_vector,
        alpha=1.0,
        timestep_holder=None,
        timestep_window=None,
        conditional_only=True,
    ):
        """Subtract alpha * steering_vector from the layer output."""
        name_to_module = dict(model.named_modules())
        module = name_to_module.get(layer_name)
        if module is None:
            raise ValueError(f"Module '{layer_name}' not found in model")

        def hook_fn(module, input, output):
            if timestep_holder is not None and timestep_window is not None:
                t = int(timestep_holder[0])
                t_high, t_low = timestep_window
                if not (t_high >= t >= t_low):
                    return output

            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output
            sv = steering_vector.to(tensor.device, tensor.dtype)

            if conditional_only and tensor.shape[0] >= 2:
                half = tensor.shape[0] // 2
                patched = tensor.clone()
                patched[half:] = patched[half:] - alpha * sv
                result = patched
            else:
                result = tensor - alpha * sv

            return (result,) + output[1:] if is_tuple else result

        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    # ------------------------------------------------------------------
    # Projection (for PCA gender subspace removal)
    # ------------------------------------------------------------------

    def register_projection_hook(
        self,
        model,
        layer_name,
        projection_components,
        token_positions=None,
        timestep_holder=None,
        timestep_window=None,
        conditional_only=True,
    ):
        """Project out a subspace from the layer output at specified token positions.

        Removes the component of the hidden state lying in the subspace spanned
        by *projection_components*:  ``h = h - G.T @ (G @ h)``

        Args:
            projection_components: Tensor ``(k, hidden_dim)`` with orthonormal rows
                defining the subspace to remove (e.g. gender direction from PCA).
            token_positions: List of token indices to apply projection at.
                ``None`` applies to all positions.
        """
        name_to_module = dict(model.named_modules())
        module = name_to_module.get(layer_name)
        if module is None:
            raise ValueError(f"Module '{layer_name}' not found in model")

        def hook_fn(module, input, output):
            if timestep_holder is not None and timestep_window is not None:
                t = int(timestep_holder[0])
                t_high, t_low = timestep_window
                if not (t_high >= t >= t_low):
                    return output

            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output
            G = projection_components.to(tensor.device, tensor.dtype)  # (k, d)

            if conditional_only and tensor.shape[0] >= 2:
                half = tensor.shape[0] // 2
                patched = tensor.clone()
                target = patched[half:]
            else:
                patched = tensor.clone()
                target = patched

            if token_positions is not None:
                for pos in token_positions:
                    h = target[:, pos, :]                   # (batch, d)
                    target[:, pos, :] = h - (h @ G.T) @ G  # project out
            else:
                # All positions: target is (batch, seq, d)
                target[:] = target - (target @ G.T) @ G

            return (patched,) + output[1:] if is_tuple else patched

        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def remove_all_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear_cache(self):
        self.cache.clear()

    def to_device(self, device):
        """Move all cached tensors to *device* (e.g. 'cpu' to free GPU)."""
        self.cache = {k: v.to(device) for k, v in self.cache.items()}

    def get_layer_cache(self, layer_name):
        """Return {timestep: tensor} for a single layer (UNet keyed cache)."""
        return {
            ts: tensor
            for (name, ts), tensor in self.cache.items()
            if isinstance((name, ts), tuple) and name == layer_name
        }


@contextmanager
def hook_context():
    """Context manager ensuring hooks are cleaned up on exit."""
    cache = ActivationCache()
    try:
        yield cache
    finally:
        cache.remove_all_hooks()


# ======================================================================
# Layer enumeration helpers
# ======================================================================

def enumerate_text_encoder_layers():
    """Return all hookable layer names in the CLIP text encoder (24 total)."""
    layers = []
    for i in range(12):
        prefix = f"text_model.encoder.layers.{i}"
        layers.append(f"{prefix}.self_attn")
        layers.append(f"{prefix}.mlp")
    return layers


def enumerate_unet_layers(unet):
    """Walk the UNet and return all hookable layer names.

    Collects ResNet blocks, self-attention, cross-attention, and feed-forward
    sub-blocks inside every transformer block in down_blocks, mid_block,
    and up_blocks.
    """
    layers = []
    for name, module in unet.named_modules():
        # ResNet blocks
        if name.endswith((".resnets.0", ".resnets.1", ".resnets.2", ".resnets.3")):
            # Only top-level resnets, not sub-modules
            if "resnets" in name.split(".")[-2:]:
                layers.append(name)
        # Transformer sub-blocks (attention and FF)
        if ".transformer_blocks." in name:
            parts = name.split(".")
            if parts[-1] in ("attn1", "attn2", "ff"):
                layers.append(name)
    return layers
