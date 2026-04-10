import os
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
OUTDIR = "figures"

TARGET = "Diabetes_binary"
POSITIVE_LABEL = "Diabetes"
NEGATIVE_LABEL = "No diabetes"

AGE_LABELS = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+",
}


def setup_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = df[TARGET].astype(int)
    df["Diabetes_label"] = df[TARGET].map({0: NEGATIVE_LABEL, 1: POSITIVE_LABEL})
    return df


def save_class_balance(df):
    counts = df["Diabetes_label"].value_counts().reindex([NEGATIVE_LABEL, POSITIVE_LABEL])
    percentages = counts / counts.sum() * 100
    plot_df = counts.rename_axis("Diabetes status").reset_index(name="Count")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=plot_df,
        x="Diabetes status",
        y="Count",
        hue="Diabetes status",
        palette={NEGATIVE_LABEL: "#4C78A8", POSITIVE_LABEL: "#E45756"},
        legend=False,
        ax=ax,
    )

    for idx, (count, pct) in enumerate(zip(counts.values, percentages.values)):
        ax.text(idx, count + counts.max() * 0.015, f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom")

    ax.set_title("Balanced Class Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of respondents")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "01_class_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def save_target_correlations(df):
    corr = (
        df.drop(columns=["Diabetes_label"])
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#72B7B2" if val > 0 else "#F58518" for val in corr.values]
    ax.barh(corr.index, corr.values, color=colors)

    for y_pos, value in enumerate(corr.values):
        offset = 0.01 if value >= 0 else -0.01
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, y_pos, f"{value:.2f}", va="center", ha=ha, fontsize=10)

    ax.set_title("Top Correlations With Diabetes Outcome")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("")
    ax.axvline(0, color="black", linewidth=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "02_top_correlations.png"), bbox_inches="tight")
    plt.close(fig)


def save_bmi_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.violinplot(
        data=df,
        x="Diabetes_label",
        y="BMI",
        palette={NEGATIVE_LABEL: "#4C78A8", POSITIVE_LABEL: "#E45756"},
        hue="Diabetes_label",
        legend=False,
        inner="box",
        cut=0,
        ax=ax,
    )
    ax.set_title("BMI Distribution by Diabetes Status")
    ax.set_xlabel("")
    ax.set_ylabel("BMI")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "03_bmi_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def save_age_profile(df):
    age_dist = (
        df.groupby(["Age", "Diabetes_label"])
        .size()
        .reset_index(name="count")
        .sort_values("Age")
    )
    age_dist["share"] = age_dist.groupby("Diabetes_label")["count"].transform(lambda s: s / s.sum())
    age_dist["Age group"] = age_dist["Age"].astype(int).map(AGE_LABELS)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(
        data=age_dist,
        x="Age group",
        y="share",
        hue="Diabetes_label",
        style="Diabetes_label",
        markers=True,
        linewidth=2.5,
        palette={NEGATIVE_LABEL: "#4C78A8", POSITIVE_LABEL: "#E45756"},
        ax=ax,
    )
    ax.set_title("Age-Group Profile Within Each Class")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Share within class")
    ax.tick_params(axis="x", rotation=35)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "04_age_profile.png"), bbox_inches="tight")
    plt.close(fig)


def save_risk_factor_comparison(df):
    feature_labels = {
        "HighBP": "High blood pressure",
        "HighChol": "High cholesterol",
        "DiffWalk": "Difficulty walking",
        "HeartDiseaseorAttack": "Heart disease/attack",
        "Stroke": "Stroke",
        "PhysActivity": "Physical activity",
    }

    rows = []
    for feature, label in feature_labels.items():
        rates = df.groupby("Diabetes_label")[feature].mean().reindex([NEGATIVE_LABEL, POSITIVE_LABEL]) * 100
        for diabetes_label, rate in rates.items():
            rows.append({"Feature": label, "Diabetes status": diabetes_label, "Rate": rate})

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=plot_df,
        x="Rate",
        y="Feature",
        hue="Diabetes status",
        palette={NEGATIVE_LABEL: "#4C78A8", POSITIVE_LABEL: "#E45756"},
        ax=ax,
    )
    ax.set_title("Risk Factor Prevalence by Diabetes Status")
    ax.set_xlabel("Respondents with indicator (%)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "05_risk_factor_prevalence.png"), bbox_inches="tight")
    plt.close(fig)


def save_summary(df):
    corr = (
        df.drop(columns=["Diabetes_label"])
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(5)
    )

    bmi_means = df.groupby("Diabetes_label")["BMI"].mean().reindex([NEGATIVE_LABEL, POSITIVE_LABEL])
    genhlth_means = df.groupby("Diabetes_label")["GenHlth"].mean().reindex([NEGATIVE_LABEL, POSITIVE_LABEL])
    highbp_rates = df.groupby("Diabetes_label")["HighBP"].mean().reindex([NEGATIVE_LABEL, POSITIVE_LABEL]) * 100

    summary = dedent(
        f"""
        Dataset used: {DATA_PATH}
        Rows: {len(df):,}
        Columns: {df.shape[1] - 1}

        Quick report notes
        - The dataset is perfectly balanced: {NEGATIVE_LABEL} = {int((df[TARGET] == 0).sum()):,}, {POSITIVE_LABEL} = {int((df[TARGET] == 1).sum()):,}.
        - The strongest linear associations with diabetes are: {", ".join(f"{col} ({val:.2f})" for col, val in corr.items())}.
        - Mean BMI rises from {bmi_means[NEGATIVE_LABEL]:.2f} in the non-diabetes group to {bmi_means[POSITIVE_LABEL]:.2f} in the diabetes group.
        - Average general health score rises from {genhlth_means[NEGATIVE_LABEL]:.2f} to {genhlth_means[POSITIVE_LABEL]:.2f}; higher values indicate worse self-rated health.
        - High blood pressure is reported by {highbp_rates[NEGATIVE_LABEL]:.1f}% of the non-diabetes group versus {highbp_rates[POSITIVE_LABEL]:.1f}% of the diabetes group.
        """
    ).strip()

    with open(os.path.join(OUTDIR, "report_notes.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary + "\n")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    setup_style()
    df = load_data()

    save_class_balance(df)
    save_target_correlations(df)
    save_bmi_distribution(df)
    save_age_profile(df)
    save_risk_factor_comparison(df)
    save_summary(df)

    print(f"Saved report-ready figures to: {OUTDIR}")


if __name__ == "__main__":
    main()
