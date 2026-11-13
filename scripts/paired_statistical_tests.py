from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ("boundary_f1", "iou")
COMPARISONS = (("HED-ODS", "DQN"), ("HED-ODS", "PPO"), ("DQN", "PPO"))

# (dqn_rows, ppo_rows) per seed, relative to the runs directory.
SEED_SOURCES = {
    24: ("dqn_seed24_sqrt8/test_policy_rows.csv", "ppo_seed24_sqrt8/test_policy_rows.csv"),
    42: ("dqn_sqrt8/d_test_policy_rows.csv", "ppo_sqrt8/test_policy_rows.csv"),
    123: ("dqn_seed123_sqrt8_fresh/test_policy_rows.csv", "ppo_seed123_sqrt8_fresh/test_policy_rows.csv"),
    486: ("dqn_seed486_sqrt8/test_eval/test_rows.csv", "ppo_seed486_sqrt8/test_eval/test_rows.csv"),
    999: ("dqn_seed999_sqrt8/test_policy_rows.csv", "ppo_seed999_sqrt8/test_policy_rows.csv"),
}
ODS_ROWS = "ods_binary_maps/test/rows.csv"


def load_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    return frame.set_index("sample_id")[list(METRICS)].sort_index()


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    diff = b - a
    std = diff.std(ddof=1)
    nonzero = diff[diff != 0]
    if len(nonzero) > 0:
        wilcoxon_p = float(stats.wilcoxon(nonzero).pvalue)
        ranks = stats.rankdata(np.abs(nonzero))
        pos, neg = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
        rank_biserial = float((pos - neg) / (pos + neg))
    else:
        wilcoxon_p, rank_biserial = 1.0, 0.0
    t_p = float(stats.ttest_rel(b, a).pvalue) if std > 0 else 1.0
    ci_low, ci_high = bootstrap_ci(diff, rng, n_boot)
    return {
        "n": len(diff),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohen_dz": float(diff.mean() / std) if std > 0 else 0.0,
        "rank_biserial": rank_biserial,
        "wilcoxon_p": wilcoxon_p,
        "paired_t_p": t_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired tests on per-image test metrics (HED-ODS, DQN, PPO).")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paired_stats"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    ods = load_rows(args.runs_dir / ODS_ROWS)
    reported = pd.read_csv(args.runs_dir / "all_seed_test_summary.csv").set_index(["seed", "algorithm"])
    results: list[dict[str, object]] = []

    for seed, (dqn_path, ppo_path) in SEED_SOURCES.items():
        methods = {
            "HED-ODS": ods,
            "DQN": load_rows(args.runs_dir / dqn_path),
            "PPO": load_rows(args.runs_dir / ppo_path),
        }
        for name, frame in methods.items():
            if len(frame) != 200 or not frame.index.equals(ods.index):
                raise ValueError(f"seed {seed} {name}: expected the same 200 test images as HED-ODS")
            if name != "HED-ODS":
                for metric in METRICS:
                    expected = reported.loc[(str(seed), name), f"{metric}_mean"]
                    if not np.isclose(frame[metric].mean(), expected, atol=1e-5):
                        raise ValueError(
                            f"seed {seed} {name} {metric}: rows mean {frame[metric].mean():.6f} "
                            f"!= reported {expected:.6f}"
                        )

        for metric in METRICS:
            block = []
            for a_name, b_name in COMPARISONS:
                a = methods[a_name][metric].to_numpy(dtype=float)
                b = methods[b_name][metric].to_numpy(dtype=float)
                row = {"seed": seed, "metric": metric, "comparison": f"{b_name} - {a_name}"}
                row.update(paired_stats(a, b, rng, args.n_boot))
                block.append(row)
            for row, adjusted in zip(block, holm([r["wilcoxon_p"] for r in block])):
                row["wilcoxon_p_holm"] = adjusted
            results.extend(block)

    table = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "paired_tests.csv", index=False)

    lines = [
        "Paired tests on 200 test images (difference = second method - first method). "
        "CI: bootstrap 95% on the mean difference; Holm correction over the 3 comparisons per seed and metric.",
        "",
    ]
    for metric in METRICS:
        lines += [
            f"### {metric}",
            "",
            "| seed | comparison | mean diff [95% CI] | Cohen's d_z | rank-biserial | Wilcoxon p | Holm p | paired-t p |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, r in table[table["metric"] == metric].iterrows():
            lines.append(
                f"| {r.seed} | {r.comparison} | {r.mean_diff:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] "
                f"| {r.cohen_dz:+.3f} | {r.rank_biserial:+.3f} | {r.wilcoxon_p:.3g} "
                f"| {r.wilcoxon_p_holm:.3g} | {r.paired_t_p:.3g} |"
            )
        lines.append("")
    (args.output_dir / "paired_tests.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
