"""MRNet-style classifier.

Pipeline for a single exam:

    for each plane:
        (S, 3, D, D) --backbone--> (S, C, h, w)
                      --avgpool--> (S, C)
                   --slice pool--> (C,)          # max or avg over slices
    concat planes  --> (C * num_planes,)
    dropout + linear --> (num_tasks,)  logits

Separate backbone weights per plane (as in the paper) capture plane-specific
appearance. Batch size is fixed at 1 because slice counts vary per exam.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn


def _build_backbone(name: str, pretrained: bool):
    """Return (feature_extractor, feature_dim) for a 2D CNN backbone."""
    import torchvision.models as tvm

    if name == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        net = tvm.resnet18(weights=weights)
        # Drop avgpool + fc; keep conv stack -> (N, 512, h, w).
        extractor = nn.Sequential(*list(net.children())[:-2])
        return extractor, 512
    if name == "alexnet":
        weights = tvm.AlexNet_Weights.DEFAULT if pretrained else None
        net = tvm.alexnet(weights=weights)
        return net.features, 256  # -> (N, 256, h, w)
    raise ValueError(f"unsupported backbone {name!r}")


class PlaneNet(nn.Module):
    """Encode one plane's volume into a single feature vector."""

    def __init__(self, backbone: str, pretrained: bool, pool: str) -> None:
        super().__init__()
        self.backbone, self.feature_dim = _build_backbone(backbone, pretrained)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (S, 3, D, D) for one exam.
        if x.dim() != 4:
            raise ValueError(f"expected (S, 3, D, D), got {tuple(x.shape)}")
        feats = self.backbone(x)              # (S, C, h, w)
        feats = self.spatial_pool(feats)      # (S, C, 1, 1)
        feats = feats.flatten(1)              # (S, C)
        if self.pool == "max":
            pooled = feats.max(dim=0).values  # (C,)
        else:
            pooled = feats.mean(dim=0)        # (C,)
        return pooled


class KneeMRIModel(nn.Module):
    def __init__(
        self,
        planes: Sequence[str] = ("axial", "coronal", "sagittal"),
        num_tasks: int = 3,
        backbone: str = "resnet18",
        pretrained: bool = False,
        pool: str = "max",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.planes: List[str] = list(planes)
        self.encoders = nn.ModuleDict(
            {plane: PlaneNet(backbone, pretrained, pool) for plane in self.planes}
        )
        feat_dim = sum(self.encoders[p].feature_dim for p in self.planes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_tasks)

    def forward(self, planes: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = set(self.planes) - set(planes)
        if missing:
            raise KeyError(f"missing planes in input: {sorted(missing)}")
        features = [self.encoders[p](planes[p]) for p in self.planes]
        fused = torch.cat(features, dim=0)   # (C * num_planes,)
        fused = self.dropout(fused)
        return self.classifier(fused)        # (num_tasks,)


def build_model(config, num_tasks: int | None = None) -> KneeMRIModel:
    """Construct a model from a :class:`knee_mri.config.Config`."""
    return KneeMRIModel(
        planes=config.planes,
        num_tasks=num_tasks if num_tasks is not None else len(config.tasks),
        backbone=config.backbone,
        pretrained=config.pretrained,
        pool=config.pool,
        dropout=config.dropout,
    )
