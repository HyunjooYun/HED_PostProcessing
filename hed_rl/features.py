from __future__ import annotations

import numpy as np

from .config import ParameterState


def compute_observation_features(hed_map: np.ndarray, processed: np.ndarray, params: ParameterState, step_ratio: float) -> np.ndarray:
    edge_density = float(processed.mean())
    hed_density = float(hed_map.mean())
    continuity_horizontal = float(np.mean(np.abs(np.diff(processed, axis=1)) < 0.5))
    continuity_vertical = float(np.mean(np.abs(np.diff(processed, axis=0)) < 0.5))
    noise_level = float(np.mean(np.abs(processed - hed_map)))
    normalized_params = np.array(
        [
            params.threshold,
            (params.blur_sigma - 0.2) / 2.8,
            (params.morphology_radius - 1.0) / 6.0,
            (params.nms_strength - 0.2) / 0.8,
        ],
        dtype=np.float32,
    )
    image_stats = np.array(
        [
            hed_density,
            edge_density,
            continuity_horizontal,
            continuity_vertical,
            noise_level,
            step_ratio,
        ],
        dtype=np.float32,
    )
    return np.concatenate([image_stats, normalized_params]).astype(np.float32)