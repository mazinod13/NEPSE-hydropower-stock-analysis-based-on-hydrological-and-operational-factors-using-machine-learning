"""
Rainfall and Production Preprocessing
=====================================

Reads Datasets/rainfall&production.csv, engineers annual hydrology and
production features, saves feature CSVs, and creates charts.

Outputs:
    Features/Production/production_features.csv
    Features/Production/{COMPANY}_production_features.csv
    Features/Rainfall/rainfall_features.csv
    Features/Rainfall/{COMPANY}_rainfall_features.csv
    Output/RP_*.png

Run:
    python rainfall_production_preprocessing.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("Datasets") / "rainfall&production.csv"
PRODUCTION_DIR = Path("Features") / "Production"
RAINFALL_DIR = Path("Features") / "Rainfall"
OUTPUT_DIR = Path("Output")

COMPANY_COLORS = {
    "BARUN": "#1E8449",
    "CHCL": "#922B21",
    "RADHI": "#1B4F72",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    }
)


def ensure_dirs() -> None:
    for directory in (PRODUCTION_DIR, RAINFALL_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    station_split_columns = ["Rainfall Station (District", " Elevation)"]
    if all(column in df.columns for column in station_split_columns):
        df["Rainfall Station (District, Elevation)"] = (
            df[station_split_columns[0]].astype(str).str.strip()
            + ", "
            + df[station_split_columns[1]].astype(str).str.strip()
        )
        df = df.drop(columns=station_split_columns)

    required_columns = {
        "Company",
        "Station Location",
        "Rainfall Station (District, Elevation)",
        "Year",
        "Energy Generation (GWh)",
        "Annual Rainfall (mm)",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Company"] = df["Company"].astype(str).str.strip().str.upper()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Energy Generation (GWh)"] = pd.to_numeric(
        df["Energy Generation (GWh)"], errors="coerce"
    )
    df["Annual Rainfall (mm)"] = pd.to_numeric(
        df["Annual Rainfall (mm)"], errors="coerce"
    )
    df = df.dropna(subset=["Company", "Year"]).sort_values(["Company", "Year"])
    return df.reset_index(drop=True)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _add_common_time_features(
    df: pd.DataFrame, value_column: str, feature_prefix: str
) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("Company", group_keys=False)

    out[f"{feature_prefix}_lag_1y"] = grouped[value_column].shift(1)
    out[f"{feature_prefix}_lag_2y"] = grouped[value_column].shift(2)
    out[f"{feature_prefix}_yoy_change"] = grouped[value_column].diff(1)
    out[f"{feature_prefix}_yoy_pct_change"] = grouped[value_column].pct_change(1)
    out[f"{feature_prefix}_rolling_3y_mean"] = grouped[value_column].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    out[f"{feature_prefix}_rolling_3y_std"] = grouped[value_column].transform(
        lambda s: s.rolling(window=3, min_periods=2).std()
    )
    out[f"{feature_prefix}_expanding_mean"] = grouped[value_column].transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    out[f"{feature_prefix}_vs_company_mean"] = (
        out[value_column] - grouped[value_column].transform("mean")
    )
    out[f"{feature_prefix}_company_zscore"] = grouped[value_column].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else 0
    )
    return out


def build_production_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = _add_common_time_features(
        df,
        value_column="Energy Generation (GWh)",
        feature_prefix="generation_gwh",
    )
    grouped = feature_df.groupby("Company", group_keys=False)

    feature_df["generation_cumulative_gwh"] = grouped[
        "Energy Generation (GWh)"
    ].cumsum()
    feature_df["generation_per_mm_rainfall"] = _safe_divide(
        feature_df["Energy Generation (GWh)"], feature_df["Annual Rainfall (mm)"]
    )
    feature_df["generation_to_rainfall_index"] = grouped[
        "generation_per_mm_rainfall"
    ].transform(lambda s: s / s.mean() if s.mean() else np.nan)

    columns = [
        "Company",
        "Year",
        "Station Location",
        "Rainfall Station (District, Elevation)",
        "Energy Generation (GWh)",
        "generation_gwh_lag_1y",
        "generation_gwh_lag_2y",
        "generation_gwh_yoy_change",
        "generation_gwh_yoy_pct_change",
        "generation_gwh_rolling_3y_mean",
        "generation_gwh_rolling_3y_std",
        "generation_gwh_expanding_mean",
        "generation_gwh_vs_company_mean",
        "generation_gwh_company_zscore",
        "generation_cumulative_gwh",
        "generation_per_mm_rainfall",
        "generation_to_rainfall_index",
    ]
    return feature_df[columns]


def build_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = _add_common_time_features(
        df,
        value_column="Annual Rainfall (mm)",
        feature_prefix="rainfall_mm",
    )
    grouped = feature_df.groupby("Company", group_keys=False)

    feature_df["rainfall_cumulative_mm"] = grouped["Annual Rainfall (mm)"].cumsum()
    feature_df["rainfall_intensity_category"] = grouped[
        "Annual Rainfall (mm)"
    ].transform(
        lambda s: pd.qcut(
            s.rank(method="first"),
            q=3,
            labels=["low", "normal", "high"],
        )
    )
    feature_df["rainfall_to_generation_index"] = grouped[
        "Annual Rainfall (mm)"
    ].transform(lambda s: s / s.mean() if s.mean() else np.nan)

    columns = [
        "Company",
        "Year",
        "Station Location",
        "Rainfall Station (District, Elevation)",
        "Annual Rainfall (mm)",
        "rainfall_mm_lag_1y",
        "rainfall_mm_lag_2y",
        "rainfall_mm_yoy_change",
        "rainfall_mm_yoy_pct_change",
        "rainfall_mm_rolling_3y_mean",
        "rainfall_mm_rolling_3y_std",
        "rainfall_mm_expanding_mean",
        "rainfall_mm_vs_company_mean",
        "rainfall_mm_company_zscore",
        "rainfall_cumulative_mm",
        "rainfall_intensity_category",
        "rainfall_to_generation_index",
    ]
    return feature_df[columns]


def save_features(production_df: pd.DataFrame, rainfall_df: pd.DataFrame) -> None:
    production_df.to_csv(PRODUCTION_DIR / "production_features.csv", index=False)
    rainfall_df.to_csv(RAINFALL_DIR / "rainfall_features.csv", index=False)

    for company, company_df in production_df.groupby("Company"):
        company_df.to_csv(
            PRODUCTION_DIR / f"{company}_production_features.csv", index=False
        )

    for company, company_df in rainfall_df.groupby("Company"):
        company_df.to_csv(RAINFALL_DIR / f"{company}_rainfall_features.csv", index=False)


def plot_production_trends(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for company, company_df in df.groupby("Company"):
        ax.plot(
            company_df["Year"],
            company_df["Energy Generation (GWh)"],
            marker="o",
            linewidth=2,
            label=company,
            color=COMPANY_COLORS.get(company),
        )
    ax.set_title("Annual Energy Generation by Company")
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy Generation (GWh)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Company")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "RP_01_production_trends.png", bbox_inches="tight")
    plt.close(fig)


def plot_rainfall_trends(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for company, company_df in df.groupby("Company"):
        ax.plot(
            company_df["Year"],
            company_df["Annual Rainfall (mm)"],
            marker="o",
            linewidth=2,
            label=company,
            color=COMPANY_COLORS.get(company),
        )
    ax.set_title("Annual Rainfall by Company Station")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Rainfall (mm)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Company")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "RP_02_rainfall_trends.png", bbox_inches="tight")
    plt.close(fig)


def plot_rainfall_vs_production(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for company, company_df in df.groupby("Company"):
        ax.scatter(
            company_df["Annual Rainfall (mm)"],
            company_df["Energy Generation (GWh)"],
            s=70,
            alpha=0.85,
            label=company,
            color=COMPANY_COLORS.get(company),
        )
        for _, row in company_df.iterrows():
            ax.annotate(
                str(int(row["Year"])),
                (row["Annual Rainfall (mm)"], row["Energy Generation (GWh)"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_title("Rainfall vs Energy Generation")
    ax.set_xlabel("Annual Rainfall (mm)")
    ax.set_ylabel("Energy Generation (GWh)")
    ax.grid(alpha=0.25)
    ax.legend(title="Company")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "RP_03_rainfall_vs_production.png", bbox_inches="tight")
    plt.close(fig)


def plot_company_dual_axis(df: pd.DataFrame) -> None:
    companies = sorted(df["Company"].unique())
    fig, axes = plt.subplots(len(companies), 1, figsize=(10, 4 * len(companies)))
    if len(companies) == 1:
        axes = [axes]

    for ax, company in zip(axes, companies):
        company_df = df[df["Company"] == company]
        color = COMPANY_COLORS.get(company, "#333333")
        ax2 = ax.twinx()

        ax.plot(
            company_df["Year"],
            company_df["Energy Generation (GWh)"],
            marker="o",
            color=color,
            linewidth=2,
            label="Generation",
        )
        ax2.bar(
            company_df["Year"],
            company_df["Annual Rainfall (mm)"],
            alpha=0.22,
            color="#4D96FF",
            label="Rainfall",
        )

        ax.set_title(f"{company}: Generation and Rainfall")
        ax.set_xlabel("Year")
        ax.set_ylabel("Generation (GWh)")
        ax2.set_ylabel("Rainfall (mm)")
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "RP_04_company_generation_rainfall.png", bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric = df[
        [
            "Energy Generation (GWh)",
            "Annual Rainfall (mm)",
            "generation_per_mm_rainfall",
        ]
    ].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(numeric, cmap="RdYlBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric.columns)))
    ax.set_yticks(range(len(numeric.columns)))
    ax.set_xticklabels(numeric.columns, rotation=30, ha="right")
    ax.set_yticklabels(numeric.columns)

    for i in range(len(numeric.index)):
        for j in range(len(numeric.columns)):
            ax.text(j, i, f"{numeric.iloc[i, j]:.2f}", ha="center", va="center")

    ax.set_title("Production and Rainfall Feature Correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "RP_05_feature_correlation.png", bbox_inches="tight")
    plt.close(fig)


def create_charts(df: pd.DataFrame, production_df: pd.DataFrame) -> None:
    merged = df.merge(
        production_df[["Company", "Year", "generation_per_mm_rainfall"]],
        on=["Company", "Year"],
        how="left",
    )
    plot_production_trends(df)
    plot_rainfall_trends(df)
    plot_rainfall_vs_production(df)
    plot_company_dual_axis(df)
    plot_correlation_heatmap(merged)


def main() -> None:
    ensure_dirs()
    raw_df = load_dataset()
    production_df = build_production_features(raw_df)
    rainfall_df = build_rainfall_features(raw_df)

    save_features(production_df, rainfall_df)
    create_charts(raw_df, production_df)

    print(f"Loaded rows: {len(raw_df)}")
    print(f"Production features: {PRODUCTION_DIR / 'production_features.csv'}")
    print(f"Rainfall features: {RAINFALL_DIR / 'rainfall_features.csv'}")
    print(f"Charts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
