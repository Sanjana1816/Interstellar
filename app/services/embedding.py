"""
Embedding service using OpenCLIP / RemoteCLIP.

Provides text and image encoding for semantic search over satellite imagery tiles.
Model is loaded once at startup and cached in memory.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level cache for the loaded model
_model = None
_preprocess = None
_tokenizer = None
_device = None


def _get_device():
    """Determine best available device."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model():
    """
    Load the CLIP model (OpenAI ViT-B-32 by default, or RemoteCLIP if weights are available).
    Called once at application startup.
    """
    global _model, _preprocess, _tokenizer, _device

    if _model is not None:
        logger.info("Model already loaded, skipping.")
        return

    import torch
    import open_clip

    _device = _get_device()
    logger.info(f"Loading CLIP model on device: {_device}")

    # Check for RemoteCLIP weights
    remoteclip_path = settings.remoteclip_weights
    if remoteclip_path and Path(remoteclip_path).exists():
        logger.info(f"Loading RemoteCLIP from: {remoteclip_path}")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name,
            pretrained=remoteclip_path,
        )
    else:
        # Default: OpenAI ViT-B-32
        logger.info(
            f"Loading default CLIP: {settings.clip_model_name} / {settings.clip_pretrained}"
        )
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name,
            pretrained=settings.clip_pretrained,
        )

    _tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
    _model = _model.to(_device)
    _model.eval()

    logger.info("CLIP model loaded successfully.")


def encode_image(image_input) -> np.ndarray:
    """
    Encode an image into a normalized embedding vector.

    Args:
        image_input: PIL Image, file path (str/Path), or numpy array (C, H, W)

    Returns:
        Normalized embedding vector (np.ndarray of shape [embedding_dim])
    """
    import torch

    if _model is None:
        load_model()

    # Convert input to PIL Image
    if isinstance(image_input, np.ndarray):
        image = _numpy_to_pil(image_input)
    elif isinstance(image_input, (str, Path)):
        image = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Preprocess and encode
    image_tensor = _preprocess(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = _model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def encode_text(query: str) -> np.ndarray:
    """
    Encode a text query into a normalized embedding vector.

    Args:
        query: Natural-language search query string

    Returns:
        Normalized embedding vector (np.ndarray of shape [embedding_dim])
    """
    import torch

    if _model is None:
        load_model()

    text_tokens = _tokenizer([query]).to(_device)

    with torch.no_grad():
        features = _model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def encode_images_batch(images: list, batch_size: int = 32) -> np.ndarray:
    """
    Encode a batch of images into normalized embedding vectors.

    Args:
        images: List of PIL Images, file paths, or numpy arrays
        batch_size: Number of images to process at once

    Returns:
        Embedding matrix (np.ndarray of shape [N, embedding_dim])
    """
    import torch

    if _model is None:
        load_model()

    all_features = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pil_images = []

        for img_input in batch:
            if isinstance(img_input, np.ndarray):
                pil_images.append(_numpy_to_pil(img_input))
            elif isinstance(img_input, (str, Path)):
                pil_images.append(Image.open(str(img_input)).convert("RGB"))
            elif isinstance(img_input, Image.Image):
                pil_images.append(img_input.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image input type: {type(img_input)}")

        tensors = torch.stack([_preprocess(img) for img in pil_images]).to(_device)

        with torch.no_grad():
            features = _model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)

        all_features.append(features.cpu().numpy())

    return np.vstack(all_features)


def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert a numpy array to a PIL RGB image.
    Handles (C, H, W) and (H, W, C) layouts, and multi-band satellite data.
    """
    if arr.ndim == 2:
        # Single band — convert to grayscale RGB
        arr = np.nan_to_num(arr, nan=0.0)
        normalized = _percentile_normalize(arr)
        return Image.fromarray(np.stack([normalized] * 3, axis=-1))

    if arr.ndim == 3:
        if arr.shape[0] <= arr.shape[2]:
            # (C, H, W) → (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))

        arr = np.nan_to_num(arr, nan=0.0)

        if arr.shape[2] >= 3:
            # Use first 3 channels as RGB (for satellite: typically B04, B03, B02 or similar)
            # Reorder if we have B02=0, B03=1, B04=2 → RGB = B04, B03, B02
            rgb = arr[:, :, :3]
            if rgb.shape[2] == 3:
                rgb = rgb[:, :, ::-1]  # BGR → RGB for S2 band ordering
        else:
            rgb = np.stack([arr[:, :, 0]] * 3, axis=-1)

        normalized = _percentile_normalize(rgb)
        return Image.fromarray(normalized)

    raise ValueError(f"Unexpected array shape: {arr.shape}")


def _percentile_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0-255 using percentile stretch."""
    valid = arr[arr > 0]
    if len(valid) == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    p2, p98 = np.percentile(valid, [2, 98])
    stretched = np.clip((arr - p2) / (p98 - p2 + 1e-10) * 255, 0, 255)
    return stretched.astype(np.uint8)


def get_embedding_dim() -> int:
    """Return the embedding dimension of the loaded model."""
    return settings.embedding_dim
