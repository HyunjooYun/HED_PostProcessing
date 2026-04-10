from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image, ImageFilter

from .config import ParameterState, PostProcessConfig
from .metrics import binarize


ACTIONS: tuple[str, ...] = (
    "threshold_up",
    "threshold_down",
    "blur_up",
    "blur_down",
    "morphology_up",
    "morphology_down",
    "nms_up",
    "nms_down",
    "noop",
)


def apply_action(state: ParameterState, action_name: str, config: PostProcessConfig) -> ParameterState:
    next_state = replace(state)
    if action_name == "threshold_up":
        next_state.threshold += config.threshold_step
    elif action_name == "threshold_down":
        next_state.threshold -= config.threshold_step
    elif action_name == "blur_up":
        next_state.blur_sigma += config.blur_step
    elif action_name == "blur_down":
        next_state.blur_sigma -= config.blur_step
    elif action_name == "morphology_up":
        next_state.morphology_radius += config.morphology_step
    elif action_name == "morphology_down":
        next_state.morphology_radius -= config.morphology_step
    elif action_name == "nms_up":
        next_state.nms_strength += config.nms_step
    elif action_name == "nms_down":
        next_state.nms_strength -= config.nms_step
    return next_state.clip()


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L")


def _from_pil(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def _non_max_suppression_like(binary_map: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.2:
        return binary_map
    radius = max(1, int(round(strength * 2)))
    padded = np.pad(binary_map, radius, mode="edge")
    height, width = binary_map.shape
    local_sum = np.zeros_like(binary_map)
    for row_offset in range(2 * radius + 1):
        for col_offset in range(2 * radius + 1):
            local_sum += padded[row_offset : row_offset + height, col_offset : col_offset + width]
    keep_threshold = max(1.0, local_sum.max() * 0.15)
    return np.where(local_sum >= keep_threshold, binary_map, 0.0).astype(np.float32)


def run_postprocess(hed_map: np.ndarray, params: ParameterState) -> np.ndarray:
    blurred = _from_pil(_to_pil(hed_map).filter(ImageFilter.GaussianBlur(radius=params.blur_sigma)))
    binary = binarize(blurred, params.threshold).astype(np.float32)
    kernel_size = max(3, params.morphology_radius * 2 + 1)
    dilated = _from_pil(_to_pil(binary).filter(ImageFilter.MaxFilter(size=kernel_size)))
    closed = _from_pil(_to_pil(dilated).filter(ImageFilter.MinFilter(size=kernel_size)))
    thinned = _non_max_suppression_like(closed, params.nms_strength)
    return binarize(thinned, 0.5).astype(np.float32)