from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class SamplePaths:
    sample_id: str
    image_path: Path
    hed_path: Path
    gt_path: Path


@dataclass(slots=True)
class Sample:
    sample_id: str
    image: np.ndarray
    hed: np.ndarray
    gt: np.ndarray


def load_image(path: Path, grayscale: bool) -> np.ndarray:
    image = Image.open(path)
    image = image.convert("L" if grayscale else "RGB")
    array = np.asarray(image, dtype=np.float32)
    if grayscale:
        return array / 255.0
    return array


def discover_split(inputs_dir: Path, split: str) -> list[SamplePaths]:
    image_dir = inputs_dir / "images" / split
    hed_dir = inputs_dir / "HED_map" / split
    gt_dir = inputs_dir / "GT_map" / split
    samples: list[SamplePaths] = []

    for image_path in sorted(image_dir.glob("*.jpg")):
        sample_id = image_path.stem
        hed_path = hed_dir / f"{sample_id}_edge.png"
        gt_path = gt_dir / f"{sample_id}_gt.png"
        if not hed_path.exists() or not gt_path.exists():
            continue
        samples.append(
            SamplePaths(
                sample_id=sample_id,
                image_path=image_path,
                hed_path=hed_path,
                gt_path=gt_path,
            )
        )

    if not samples:
        raise FileNotFoundError(f"No paired samples found for split '{split}' in {inputs_dir}")
    return samples


class DatasetIndex:
    def __init__(self, inputs_dir: Path, split: str):
        self.inputs_dir = inputs_dir
        self.split = split
        self.samples = discover_split(inputs_dir, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        paths = self.samples[index]
        return Sample(
            sample_id=paths.sample_id,
            image=load_image(paths.image_path, grayscale=False),
            hed=load_image(paths.hed_path, grayscale=True),
            gt=load_image(paths.gt_path, grayscale=True),
        )