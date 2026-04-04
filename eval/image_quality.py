"""Image quality metrics: FID and CLIP-Score."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def compute_fid(generated_images, reference_images, device="cuda"):
    """Compute Frechet Inception Distance between two image sets.

    Args:
        generated_images: List of PIL Images.
        reference_images: List of PIL Images.
        device: Torch device.

    Returns:
        FID score (float, lower is better).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Add real images
    for img in reference_images:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
        fid.update(tensor, real=True)

    # Add generated images
    for img in generated_images:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
        fid.update(tensor, real=False)

    return float(fid.compute())


class CLIPScorer:
    """CLIP model loaded once, reused across calls."""

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cuda"):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def score(self, images, prompts):
        """Mean CLIP cosine similarity between images and their prompts.

        Args:
            images: List of PIL Images.
            prompts: List of text prompts or a single string.

        Returns:
            Mean CLIP-Score (float, higher is better).
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        img_tensors = torch.stack([
            self.preprocess(img.convert("RGB")) for img in images
        ]).to(self.device)
        text_tokens = self.tokenizer(prompts).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_tensors)
            text_features = self.model.encode_text(text_tokens)

        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = (img_features * text_features).sum(dim=-1)
        return float(similarities.mean())


_default_scorer: CLIPScorer | None = None


def compute_clip_score(images, prompts, device="cuda"):
    """Mean CLIP cosine similarity between images and their prompts.

    Loads the CLIP model on first call and reuses it for subsequent calls
    on the same device.

    Args:
        images: List of PIL Images.
        prompts: List of text prompts or a single string.
        device: Torch device (only used on first call).

    Returns:
        Mean CLIP-Score (float, higher is better).
    """
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = CLIPScorer(device=device)
    return _default_scorer.score(images, prompts)
