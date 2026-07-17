"""
Compare GPT counseling responses: heart disease vs depression/bipolar.

Both runs used the same fixed WHO Zone IV AUDIT scenario (score 25).
Differences therefore reflect comorbidity profiles in the patient data.

Usage:
  .venv/bin/python plot_heart_vs_depression.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"

HEART_CSV = (
    OUTPUT_DIR
    / "heartdisease_real_patients_20260713T043120Z_responses.csv"
)
DEPRESSION_CSV = (
    OUTPUT_DIR
    / "depressionbipolar_real_patients_20260713T045603Z_responses.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "examination results"
    / "comparison_heart_vs_depression"
    / "gpt_response_comparison.png"
)


def load_group(csv_path: Path, label: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results file: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.copy()
    dataframe["disease_group_label"] = label
    return dataframe


def keyword_rate(series: pd.Series, pattern: str) -> float:
    return float(
        series.fillna("")
        .str.lower()
        .str.contains(pattern, regex=True)
        .mean()
    )


def main() -> None:
    heart = load_group(HEART_CSV, "Heart disease")
    depression = load_group(DEPRESSION_CSV, "Depression / bipolar")
    combined = pd.concat([heart, depression], ignore_index=True)

    groups = ["Heart disease", "Depression / bipolar"]
    colors = {
        "Heart disease": "#1f4e79",
        "Depression / bipolar": "#8b3a3a",
    }

    semantic_means = [
        float(
            combined.loc[
                combined["disease_group_label"] == group,
                "semantic_cosine_similarity",
            ].mean()
        )
        for group in groups
    ]
    word_means = [
        float(
            combined.loc[
                combined["disease_group_label"] == group,
                "word_structural_similarity",
            ].mean()
        )
        for group in groups
    ]
    length_means = [
        float(
            combined.loc[
                combined["disease_group_label"] == group,
                "response_word_count",
            ].mean()
        )
        for group in groups
    ]

    mention_labels = [
        "Mentions heart\nterms",
        "Mentions depression\n/ mood terms",
        "Mentions referral\n/ specialist terms",
    ]
    heart_mentions = [
        keyword_rate(
            heart["model_response"],
            r"\bheart\b|cardiac|cardiovascular",
        ),
        keyword_rate(
            heart["model_response"],
            r"depression|bipolar|\bmood\b",
        ),
        keyword_rate(
            heart["model_response"],
            r"specialist|referral|treatment|support",
        ),
    ]
    depression_mentions = [
        keyword_rate(
            depression["model_response"],
            r"\bheart\b|cardiac|cardiovascular",
        ),
        keyword_rate(
            depression["model_response"],
            r"depression|bipolar|\bmood\b",
        ),
        keyword_rate(
            depression["model_response"],
            r"specialist|referral|treatment|support",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "GPT counseling responses under fixed WHO Zone IV AUDIT (score 25)\n"
        "Heart disease (n=530) vs Depression/bipolar (n=1,058)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: semantic similarity distributions
    ax = axes[0, 0]
    for group in groups:
        values = combined.loc[
            combined["disease_group_label"] == group,
            "semantic_cosine_similarity",
        ].dropna()
        ax.hist(
            values,
            bins=20,
            range=(0.35, 0.95),
            alpha=0.55,
            label=f"{group} (mean={values.mean():.3f})",
            color=colors[group],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_title("A. Semantic similarity to benchmark")
    ax.set_xlabel("Semantic cosine similarity (0–1)")
    ax.set_ylabel("Number of GPT responses")
    ax.legend(fontsize=8)
    ax.set_xlim(0.35, 0.95)

    # Panel B: mean metric bars
    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        semantic_means,
        width,
        label="Semantic similarity",
        color="#2f6b4f",
    )
    bars2 = ax.bar(
        x + width / 2,
        word_means,
        width,
        label="Word similarity",
        color="#a67c2a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Mean similarity (0–1)")
    ax.set_title("B. Mean similarity by disease group")
    ax.legend(fontsize=8)
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Panel C: response length
    ax = axes[1, 0]
    data = [
        combined.loc[
            combined["disease_group_label"] == group,
            "response_word_count",
        ].dropna()
        for group in groups
    ]
    box = ax.boxplot(
        data,
        tick_labels=groups,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.7)
    ax.set_title("C. GPT response length")
    ax.set_ylabel("Word count")
    for index, mean_value in enumerate(length_means, start=1):
        ax.scatter(
            index,
            mean_value,
            color="white",
            edgecolor="black",
            zorder=3,
            s=35,
            label="Mean" if index == 1 else None,
        )
    ax.legend(fontsize=8)

    # Panel D: disease-content mentions in GPT text
    ax = axes[1, 1]
    x = np.arange(len(mention_labels))
    width = 0.35
    ax.bar(
        x - width / 2,
        [value * 100 for value in heart_mentions],
        width,
        label="Heart disease patients",
        color=colors["Heart disease"],
    )
    ax.bar(
        x + width / 2,
        [value * 100 for value in depression_mentions],
        width,
        label="Depression/bipolar patients",
        color=colors["Depression / bipolar"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(mention_labels, fontsize=8)
    ax.set_ylabel("Percent of GPT responses (%)")
    ax.set_ylim(0, 100)
    ax.set_title("D. What GPT actually talks about")
    ax.legend(fontsize=8)

    fig.text(
        0.5,
        0.01,
        "Source: output/heartdisease_real_patients_*.csv and "
        "output/depressionbipolar_real_patients_*.csv · fixed Zone IV AUDIT",
        ha="center",
        fontsize=8,
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEFAULT_OUTPUT, dpi=160)
    plt.close(fig)

    print(f"Saved plot: {DEFAULT_OUTPUT}")
    print(
        "Heart disease: "
        f"n={len(heart)}, semantic={semantic_means[0]:.3f}, "
        f"word={word_means[0]:.3f}, words={length_means[0]:.1f}"
    )
    print(
        "Depression/bipolar: "
        f"n={len(depression)}, semantic={semantic_means[1]:.3f}, "
        f"word={word_means[1]:.3f}, words={length_means[1]:.1f}"
    )


if __name__ == "__main__":
    main()
