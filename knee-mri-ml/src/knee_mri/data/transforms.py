"""Per-slice preprocessing for MRI volumes.

A volume arrives as ``(num_slices, H, W)``. Each slice is resized to a square,
optionally augmented, normalized, and stacked into 3 channels so it can feed an
ImageNet-pretrained 2D backbone. The result is a float tensor of shape
``(num_slices, 3, image_size, image_size)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# ImageNet statistics (backbones are pretrained on these).
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def volume_to_tensor(
    volume: np.ndarray,
    image_size: int = 224,
    max_slices: int = 0,
    train: bool = False,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Convert a raw ``(S, H, W)`` volume into a normalized ``(S, 3, D, D)`` tensor."""
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume (S, H, W), got shape {volume.shape}")

    vol = torch.as_tensor(np.ascontiguousarray(volume), dtype=torch.float32)

    # Optionally subsample slices (keeps memory bounded on thick series).
    if max_slices and vol.shape[0] > max_slices:
        vol = _select_slices(vol, max_slices, train, rng)

    # Scale pixel intensities to [0, 1]. Real MRNet arrays are 0-255.
    vmax = float(vol.max())
    if vmax > 0:
        vol = vol / vmax

    # Resize every slice to a square via bilinear interpolation.
    vol = F.interpolate(
        vol.unsqueeze(1),  # (S, 1, H, W)
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )  # (S, 1, D, D)

    if train:
        vol = _augment(vol, rng)

    vol = vol.repeat(1, 3, 1, 1)  # grayscale -> 3 channels
    vol = (vol - _MEAN) / _STD
    return vol


def _select_slices(
    vol: torch.Tensor, k: int, train: bool, rng: np.random.Generator | None
) -> torch.Tensor:
    """Keep ``k`` slices: a random contiguous window when training, else centred."""
    s = vol.shape[0]
    if train and rng is not None:
        start = int(rng.integers(0, s - k + 1))
    else:
        start = (s - k) // 2
    return vol[start : start + k]


def _augment(vol: torch.Tensor, rng: np.random.Generator | None) -> torch.Tensor:
    """Light, label-preserving augmentation: horizontal flip + small shift."""
    if rng is None:
        return vol
    if rng.random() < 0.5:
        vol = torch.flip(vol, dims=[-1])  # left-right flip
    shift = int(rng.integers(-8, 9))
    if shift:
        vol = torch.roll(vol, shifts=shift, dims=-2)
    return vol
