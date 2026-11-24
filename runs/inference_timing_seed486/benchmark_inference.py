from __future__ import annotations

import csv
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
import stable_baselines3
import torch
from PIL import Image
from stable_baselines3 import DQN, PPO

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from hed_rl.config import PostProcessConfig
from hed_rl.data import InferenceDatasetIndex, load_gt_map
from hed_rl.env import HEDInferenceEnv
from hed_rl.metrics import binarize, iou_score, precision_recall_f1
from hed_rl.otsu import compute_otsu_threshold

OUTPUT = ROOT / "runs" / "inference_timing_seed486"
METHODS = ("HED-ODS", "HED-Otsu", "DQN", "PPO")
REPEATS = 5


def policy_run(model, env):
    observation, info = env.reset()
    history = [dict(info["params"])]
    truncated = False
    steps = 0
    while not truncated:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, _, truncated, info = env.step(int(action))
        history.append(dict(info["params"]))
        steps += 1
    if env.processed_map is None or steps != 30:
        raise RuntimeError("Invalid policy inference output")
    return env.processed_map.copy(), steps, history


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dataset = InferenceDatasetIndex(ROOT / "inputs", "test")
    if len(dataset) != 200:
        raise RuntimeError(f"Expected 200 test images, found {len(dataset)}")
    samples = [dataset[i] for i in range(len(dataset))]
    threshold = json.loads((ROOT / "runs" / "ods_binary_maps" / "ods_threshold.json").read_text())["best_threshold"]
    models = {
        "DQN": DQN.load(str(ROOT / "runs" / "dqn_seed486_sqrt8" / "dqn_final.zip"), device="cpu"),
        "PPO": PPO.load(str(ROOT / "runs" / "ppo_seed486_sqrt8" / "ppo_final.zip"), device="cpu"),
    }

    def run(method, sample, envs):
        if method == "HED-ODS":
            return binarize(sample.hed, threshold).astype(np.float32)
        if method == "HED-Otsu":
            return binarize(sample.hed, compute_otsu_threshold(sample.hed)).astype(np.float32)
        return policy_run(models[method], envs[method])[0]

    warmup_envs = {name: HEDInferenceEnv(samples[:5], PostProcessConfig()) for name in models}
    for sample in samples[:5]:
        for method in METHODS:
            run(method, sample, warmup_envs)
    envs = {name: HEDInferenceEnv(samples, PostProcessConfig()) for name in models}
    durations = {name: [[] for _ in range(REPEATS)] for name in METHODS}
    outputs = {name: [] for name in METHODS}
    for repeat in range(REPEATS):
        for env in envs.values():
            env.restart_sequence()
        for index, sample in enumerate(samples):
            for offset in range(len(METHODS)):
                method = METHODS[(repeat + index + offset) % len(METHODS)]
                start = time.perf_counter_ns()
                binary = run(method, sample, envs)
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                durations[method][repeat].append(elapsed_ms)
                binary_bool = binary.astype(bool)
                if repeat == 0:
                    outputs[method].append(binary_bool)
                elif not np.array_equal(binary_bool, outputs[method][index]):
                    raise RuntimeError(f"{method} output changed in repeat {repeat + 1}: {sample.sample_id}")
            if (index + 1) % 50 == 0:
                print(f"Repeat {repeat + 1}/{REPEATS}: {index + 1}/200", flush=True)

    # Validate outside the timed region against the existing test results.
    for method, suffix, folder in (
        ("HED-ODS", "ods", "ods_binary_maps"),
        ("HED-Otsu", "otsu", "otsu_binary_maps"),
    ):
        for sample, output in zip(samples, outputs[method]):
            path = ROOT / "runs" / folder / "test" / "maps" / f"{sample.sample_id}_{suffix}_binary.png"
            with Image.open(path) as image:
                reference = np.asarray(image.convert("L")) > 0
            if not np.array_equal(output, reference):
                raise RuntimeError(f"{method} output mismatch for {sample.sample_id}")
    for method in ("DQN", "PPO"):
        ref_path = ROOT / "runs" / f"{method.lower()}_seed486_sqrt8" / "test_eval" / "test_rows.csv"
        with ref_path.open(newline="", encoding="utf-8") as handle:
            references = {row["sample_id"]: row for row in csv.DictReader(handle)}
        if len(references) != len(samples):
            raise RuntimeError(f"{method} reference count mismatch")
        for sample, output in zip(samples, outputs[method]):
            gt = load_gt_map(ROOT / "inputs", "test", sample.sample_id)
            _, _, f1 = precision_recall_f1(output, gt)
            iou = iou_score(output, gt)
            ref = references[sample.sample_id]
            if int(ref["steps"]) != 30 or not np.isclose(f1, float(ref["f1"]), atol=1e-10, rtol=0) or not np.isclose(iou, float(ref["iou"]), atol=1e-10, rtol=0):
                raise RuntimeError(f"{method} reference metric mismatch for {sample.sample_id}")
    with (OUTPUT / "inference_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat", "sample_id", *[f"{method}_ms" for method in METHODS]])
        for repeat in range(REPEATS):
            for i, sample in enumerate(samples):
                writer.writerow([repeat + 1, sample.sample_id, *[durations[method][repeat][i] for method in METHODS]])
    with (OUTPUT / "per_image_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *[f"{method}_mean_ms" for method in METHODS]])
        for i, sample in enumerate(samples):
            writer.writerow([sample.sample_id, *[np.mean([durations[method][r][i] for r in range(REPEATS)]) for method in METHODS]])
    summary = {
        "scope": "In-memory HED score map to binary map; excludes HED network, input/output disk I/O, GT evaluation, checkpoint loading and offline ODS threshold selection.",
        "dataset": "inputs test split",
        "sample_count": len(samples),
        "repeats": REPEATS,
        "image_shapes_height_width": {str(k): v for k, v in Counter(tuple(s.hed.shape) for s in samples).items()},
        "warmup_images_per_method": 5,
        "method_order": "rotated per image and repeat",
        "device": "CPU",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "ods_threshold": threshold,
        "checkpoints": {method: f"runs\\{method.lower()}_seed486_sqrt8\\{method.lower()}_final.zip" for method in models},
        "validation": "Outputs match across all repeats; 200 baseline maps exactly match existing maps; policy F1 and IoU match existing seed-486 final-model test rows, 30 steps each.",
        "results_ms": {
            method: {
                "mean": float(np.mean(np.mean(values, axis=0))),
                "image_std_population_ddof_0": float(np.std(np.mean(values, axis=0))),
                "round_means": [float(x) for x in np.mean(values, axis=1)],
                "round_mean_std_population_ddof_0": float(np.std(np.mean(values, axis=1))),
                "image_mean_min": float(np.min(np.mean(values, axis=0))),
                "image_mean_max": float(np.max(np.mean(values, axis=0))),
            } for method, values in durations.items()
        },
    }
    (OUTPUT / "inference_timing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
