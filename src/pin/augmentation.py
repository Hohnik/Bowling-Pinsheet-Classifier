"""Shared training augmentations for sheet photographs and field crops."""

from __future__ import annotations

from typing import Any

import cv2


def photo_transforms() -> list[Any]:
    """Return camera and print-quality transforms shared by both models."""
    import albumentations as A

    return [
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.Defocus(radius=(2, 5)),
            ],
            p=0.15,
        ),
        A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.12),
        A.ImageCompression(quality_range=(45, 95), p=0.20),
        A.OneOf([A.CLAHE(), A.RandomGamma()], p=0.15),
        A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), p=0.12),
        A.Downscale(scale_range=(0.65, 0.95), p=0.10),
    ]


def serialized_photo_transforms() -> list[dict[str, object]]:
    """Serialize the shared transforms for the Ultralytics trainer."""
    import albumentations as A

    return [A.to_dict(transform) for transform in photo_transforms()]


def crop_augmenter() -> Any:
    """Add label-preserving geometry to the shared photo transforms."""
    import albumentations as A

    return A.Compose(
        [
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.06, 0.06),
                rotate=(-12, 12),
                shear=(-2, 2),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.80,
            ),
            A.Perspective(
                scale=(0.001, 0.01),
                keep_size=True,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.15,
            ),
            *photo_transforms(),
        ]
    )
