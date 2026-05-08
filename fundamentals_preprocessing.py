import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURATION — Edit these to add/remove features
# ─────────────────────────────────────────────

BS_FEATURES = [
    "Paid up Capital",
    "Share Premium",
    "Reserves",
    "Long Term Liabilities",
    "Total Sources of Funds",
    "Net Fixed Assets",
    "Investments",
    "Cash at Hand",
    "Receivables",
    "Advances, Payments, Loans and Deposits",
    "Inventory",
    "Total Current Assets",
    "Short Term Liabilities",
    "Deferred Liabilities",
    "Total Short Term Liabilities",
    "Application of Funds",
]

IS_FEATURES = [
    "Income from Sale of Energy",
    "Cost of Production",
    "Gross Profit",
    "Income from Dividend",
    "Forex Gain",
    "Income from Other Sources",
    "Administrative Expenses",
    "Operating Profit",
    "Net Interest Income",
    "Depreciation",
    "Profit Before Taxes",
    "Taxes",
    "Bonuses",
    "Net Profit",
]

KS_FEATURES = [
    "Total Revenue (Rs. 000)",
    "Gross Profit (Rs. 000)",
    "Net Income (Rs. 000)",
    "Return on Asset (TTM) %",
    "Return on Equity (TTM) %",
    "EPS (Anualized)",
    "Reported PE (Annualized)",
    "Book Value per Share",
    "Market Value Per Share",
    "Dividend Per Share (Rs.)",
]


class FundamentalsPreprocessor:
    """
    Loads one or more company fundamentals CSVs and returns
    a clean, wide-format feature DataFrame indexed by
    (Company, Period).
    """

    def __init__(self):
        self.raw_dfs = []          # raw loaded frames
        self.features_df = None    # final output

    # ──────────────────────────────────────────
    #  LOADING
    # ──────────────────────────────────────────

    def load(self, filepath: str):
        """Load a single CSV file."""
        df = pd.read_csv(filepath)
        df["__source_file__"] = os.path.basename(filepath)
        self.raw_dfs.append(df)
        print(f"✓ Loaded  {os.path.basename(filepath)}  |  {len(df)} rows")

    def load_multiple(self, filepaths: list):
        """Load multiple CSV files (one per company)."""
        for fp in filepaths:
            self.load(fp)

    # ──────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────

    @staticmethod
    def _detect_period_columns(df: pd.DataFrame) -> list:
        """Return all quarter/period columns (everything that isn't metadata)."""
        meta_cols = {"Sector", "Company", "Data Version", "Statement Type",
                     "Particulars", "__source_file__"}
        return [c for c in df.columns if c not in meta_cols]

    @staticmethod
    def _melt_statement(df: pd.DataFrame,
                        statement_type: str,
                        features: list,
                        period_cols: list) -> pd.DataFrame:
        """
        Filter to one statement type, keep only desired rows,
        pivot wide → long, return tidy frame.
        """
        sub = df[df["Statement Type"] == statement_type].copy()
        # Only rows whose Particular is in our feature list
        sub = sub[sub["Particulars"].isin(features)]
        if sub.empty:
            return pd.DataFrame()

        # Melt periods to rows
        melted = sub.melt(
            id_vars=["Company", "Particulars"],
            value_vars=period_cols,
            var_name="Period",
            value_name="Value"
        )
        melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
        # Pivot so each Particular becomes a column
        pivoted = melted.pivot_table(
            index=["Company", "Period"],
            columns="Particulars",
            values="Value",
            aggfunc="first"
        ).reset_index()
        pivoted.columns.name = None
        return pivoted

    @staticmethod
    def _parse_period(period_str: str):
        """
        Parse a period string like '2024/25 (2081/82) Q3'
        into (fiscal_year, quarter, sort_key).
        """
        try:
            parts = period_str.strip().split()
            # parts[0] = '2024/25', parts[-1] = 'Q3'
            year_str = parts[0]          # e.g. '2024/25'
            quarter_str = parts[-1]      # e.g. 'Q3'
            start_year = int(year_str.split("/")[0])
            quarter_num = int(quarter_str[1])
            sort_key = start_year * 10 + quarter_num
            return year_str, quarter_str, sort_key
        except Exception:
            return period_str, "?", 0

    # ──────────────────────────────────────────
    #  FEATURE ENGINEERING
    # ──────────────────────────────────────────

    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ratio / growth features from raw columns.
        All divisions are NaN-safe.
        """
        eps = 1e-9  # avoid divide-by-zero

        # ── Balance Sheet Ratios ──────────────────
        if {"Total Current Assets", "Short Term Liabilities"}.issubset(df.columns):
            df["Current Ratio"] = (
                df["Total Current Assets"] / (df["Short Term Liabilities"] + eps)
            )

        if {"Total Sources of Funds", "Long Term Liabilities"}.issubset(df.columns):
            df["Equity"] = df["Total Sources of Funds"] - df["Long Term Liabilities"]
            df["Debt to Equity"] = (
                df["Long Term Liabilities"] / (df["Equity"] + eps)
            )

        if {"Net Fixed Assets", "Total Sources of Funds"}.issubset(df.columns):
            df["Asset Intensity"] = (
                df["Net Fixed Assets"] / (df["Total Sources of Funds"] + eps)
            )

        if {"Investments", "Total Sources of Funds"}.issubset(df.columns):
            df["Investment Ratio"] = (
                df["Investments"] / (df["Total Sources of Funds"] + eps)
            )

        # ── Income Statement Ratios ───────────────
        if {"Gross Profit", "Income from Sale of Energy"}.issubset(df.columns):
            df["Gross Margin"] = (
                df["Gross Profit"] / (df["Income from Sale of Energy"] + eps)
            )

        if {"Net Profit", "Income from Sale of Energy"}.issubset(df.columns):
            df["Net Margin"] = (
                df["Net Profit"] / (df["Income from Sale of Energy"] + eps)
            )

        if {"Operating Profit", "Income from Sale of Energy"}.issubset(df.columns):
            df["Operating Margin"] = (
                df["Operating Profit"] / (df["Income from Sale of Energy"] + eps)
            )

        if {"Taxes", "Profit Before Taxes"}.issubset(df.columns):
            df["Effective Tax Rate"] = (
                df["Taxes"] / (df["Profit Before Taxes"] + eps)
            )

        if {"Administrative Expenses", "Income from Sale of Energy"}.issubset(df.columns):
            df["OpEx Ratio"] = (
                df["Administrative Expenses"] / (df["Income from Sale of Energy"] + eps)
            )

        if {"Depreciation", "Net Fixed Assets"}.issubset(df.columns):
            df["Depreciation Rate"] = (
                df["Depreciation"] / (df["Net Fixed Assets"] + eps)
            )

        # ── Cross-statement Ratios ────────────────
        if {"Net Profit", "Total Sources of Funds"}.issubset(df.columns):
            df["ROA_manual"] = (
                df["Net Profit"] / (df["Total Sources of Funds"] + eps)
            )

        if {"Net Profit", "Equity"}.issubset(df.columns):
            df["ROE_manual"] = df["Net Profit"] / (df["Equity"] + eps)

        # ── QoQ Growth Rates (per company) ────────
        growth_cols = [
            "Income from Sale of Energy",
            "Net Profit",
            "Gross Profit",
            "Operating Profit",
            "Total Sources of Funds",
            "Total Current Assets",
        ]
        for col in growth_cols:
            if col in df.columns:
                new_col = f"QoQ_{col.replace(' ', '_').replace(',', '')}"
                df[new_col] = (
                    df.groupby("Company")[col]
                    .pct_change()
                    .replace([np.inf, -np.inf], np.nan)
                )

        # ── YoY Growth (Q4 to Q4) ─────────────────
        if "Income from Sale of Energy" in df.columns:
            df["YoY_Revenue"] = (
                df.groupby(["Company", "Quarter"])["Income from Sale of Energy"]
                .pct_change()
                .replace([np.inf, -np.inf], np.nan)
            )

        return df

    # ──────────────────────────────────────────
    #  MAIN PIPELINE
    # ──────────────────────────────────────────

    def get_features(self) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.
        Returns a wide DataFrame: one row per (Company, Period).
        """
        if not self.raw_dfs:
            raise ValueError("No data loaded. Call load() or load_multiple() first.")

        all_company_frames = []

        for raw in self.raw_dfs:
            period_cols = self._detect_period_columns(raw)

            # Extract each statement
            bs = self._melt_statement(raw, "BS", BS_FEATURES, period_cols)
            is_ = self._melt_statement(raw, "IS", IS_FEATURES, period_cols)
            ks = self._melt_statement(raw, "KS", KS_FEATURES, period_cols)

            frames = [f for f in [bs, is_, ks] if not f.empty]
            if not frames:
                continue

            # Merge on Company + Period
            merged = frames[0]
            for f in frames[1:]:
                merged = pd.merge(merged, f, on=["Company", "Period"], how="outer")

            all_company_frames.append(merged)

        # Combine all companies
        combined = pd.concat(all_company_frames, ignore_index=True)

        # ── Parse period into structured columns ──
        parsed = combined["Period"].apply(self._parse_period)
        combined["Fiscal Year"] = parsed.apply(lambda x: x[0])
        combined["Quarter"]     = parsed.apply(lambda x: x[1])
        combined["Sort Key"]    = parsed.apply(lambda x: x[2])

        # ── Sort chronologically ──────────────────
        combined = combined.sort_values(
            ["Company", "Sort Key"]
        ).reset_index(drop=True)

        # ── Drop all-NaN periods (no data) ────────
        feature_cols = [c for c in combined.columns
                        if c not in {"Company", "Period", "Fiscal Year",
                                     "Quarter", "Sort Key"}]
        combined = combined.dropna(subset=feature_cols, how="all")

        # ── Derived / engineered features ─────────
        combined = self._add_derived_features(combined)

        # ── Clean up column order ─────────────────
        id_cols = ["Company", "Fiscal Year", "Quarter", "Period"]
        other_cols = [c for c in combined.columns
                      if c not in id_cols + ["Sort Key"]]
        combined = combined[id_cols + other_cols]

        self.features_df = combined
        print(f"\n✓ Feature extraction complete!")
        print(f"  Shape : {combined.shape}")
        print(f"  Companies : {combined['Company'].unique().tolist()}")
        print(f"  Periods   : {combined['Quarter'].unique().tolist()}")
        print(f"  Features  : {len(other_cols)} columns")
        return combined

    def save(self, output_path: str = "features_output.csv"):
        """Save the feature DataFrame to CSV."""
        if self.features_df is None:
            self.get_features()
        self.features_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved → {output_path}")

    def summary(self):
        """Print a quick summary of coverage and missing values."""
        if self.features_df is None:
            self.get_features()
        df = self.features_df
        print("\n" + "═" * 60)
        print("  FEATURE SUMMARY")
        print("═" * 60)
        print(f"  Rows        : {len(df)}")
        print(f"  Columns     : {df.shape[1]}")
        print(f"  Companies   : {df['Company'].nunique()}")
        print(f"  Date range  : {df['Period'].iloc[0]} → {df['Period'].iloc[-1]}")
        print("\n  Missing value % per feature (top 10 most missing):")
        miss = (df.isnull().mean() * 100).sort_values(ascending=False).head(10)
        for col, pct in miss.items():
            if pct > 0:
                print(f"    {col:<45}  {pct:5.1f}%")
        print("═" * 60)


# ─────────────────────────────────────────────
#  QUICK-RUN DEMO  (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Accept file paths as CLI args, else default to RADHI.csv
    files = sys.argv[1:] if len(sys.argv) > 1 else ["RADHI.csv"]

    fp = FundamentalsPreprocessor()
    fp.load_multiple(files)
    df = fp.get_features()
    fp.summary()
    fp.save("features_output.csv")

    print("\nFirst 3 rows preview:")
    print(df.head(3).T.to_string())