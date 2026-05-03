from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")

BASE_DIR   = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "merged_df_cleaned.csv"
REPORT_DIR = BASE_DIR / "reports"
LOG_PATH   = BASE_DIR / "logs" / "EDA.log"
FIG_DIR    = REPORT_DIR / "figures"

for d in [REPORT_DIR, FIG_DIR, BASE_DIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_STATUSES = {
    "charged off", "default",
    "late (31-120 days)", "late (16-30 days)",
    "does not meet the credit policy. status:charged off",
}
NON_DEFAULT_STATUSES = {
    "fully paid",
    #   "current",
    "does not meet the credit policy. status:fully paid",
}
MACRO_COLS = ["CPI", "Unemployment Rate", "Federal Funds Rate"]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("EDA")


def load_data(logger):
    logger.info("Loading %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)
    logger.info("Loaded %s rows, %s cols", len(df), len(df.columns))
    return df


def build_target(df, logger):
    if "loan_status" not in df.columns:
        logger.warning("loan_status missing — skipping target build")
        return df

    status = df["loan_status"].astype(str).str.strip().str.lower()

    # drop current loans 
    current_mask = status == "current"
    n_dropped = int(current_mask.sum())
    df = df[~current_mask].copy()
    status = status[~current_mask]
    logger.info(f"Dropped {n_dropped} current loans (unresolved outcome)")

    df["target_default"] = pd.NA
    df.loc[status.isin(DEFAULT_STATUSES),     "target_default"] = 1
    df.loc[status.isin(NON_DEFAULT_STATUSES), "target_default"] = 0
    df["target_default"] = pd.to_numeric(df["target_default"], errors="coerce")

    n1 = int((df["target_default"] == 1).sum())
    n0 = int((df["target_default"] == 0).sum())
    logger.info(f"Target built — default: {n1}, non-default: {n0}, unknown: {df['target_default'].isna().sum()}")
    return df

def savefig(name):
    p = FIG_DIR / f"{name}.png"
    plt.savefig(p, bbox_inches="tight", dpi=110)
    plt.close("all")
    return p


def overview(df, logger):
    logger.info("Data overview")
    meta = {
        "n_rows":    len(df),
        "n_cols":    len(df.columns),
        "n_numeric": int(df.select_dtypes(include="number").shape[1]),
        "n_categ":   int(df.select_dtypes(include=["object", "category"]).shape[1]),
        "total_nulls":    int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    logger.info("Meta: %s", meta)
    return {
        "stats_numeric":     df.describe(include="number").T,
        "stats_categorical": df.describe(include=["object"]).T,
        "meta": meta,
    }


def plot_missing(df, logger):
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("No missing values")
        return None

    logger.info("%s columns have missing values", len(missing))
    fig, ax = plt.subplots(figsize=(12, max(4, len(missing) * 0.4)))
    colors = ["#d73027" if v > 0.5 else "#fc8d59" if v > 0.2 else "#fee090"
              for v in missing.values]
    bars = ax.barh(missing.index, missing.values * 100, color=colors)
    ax.axvline(50, color="red",    linestyle="--", linewidth=1, label="50%")
    ax.axvline(20, color="orange", linestyle="--", linewidth=1, label="20%")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values per Column", fontweight="bold")
    ax.legend()
    for bar, val in zip(bars, missing.values):
        ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    return savefig("01_missing_values")


def check_target_distribution(df, logger):
    out = {}

    if "loan_status" in df.columns:
        vc = df["loan_status"].value_counts()
        logger.info("Loan status counts:\n%s", vc.to_string())
        out["loan_status_counts"] = vc

    if "target_default" not in df.columns:
        return out

    td    = df["target_default"].dropna()
    n1    = int((td == 1).sum())
    n0    = int((td == 0).sum())
    total = n1 + n0
    ratio = round(n0 / n1, 2) if n1 else float("inf")

    out.update({
        "n_default":           n1,
        "n_non_default":       n0,
        "imbalance_ratio":     ratio,
        "majority_baseline_acc": round(n0 / total, 4),
    })
    logger.info(f"Default: {n1} ({100*n1/total:.1f}%), Non-default: {n0} ({100*n0/total:.1f}%), ratio {ratio}:1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["Non-Default", "Default"], [n0, n1], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Target Class Counts")
    axes[0].set_ylabel("Count")
    for bar, val in zip(axes[0].patches, [n0, n1]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total * 0.005,
                     f"{val:,}\n({100*val/total:.1f}%)", ha="center", fontsize=10)

    axes[1].pie([n0, n1], labels=["Non-Default", "Default"],
                colors=["#4C72B0", "#DD8452"], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Class Split")
    plt.suptitle("Class Imbalance", fontweight="bold")
    plt.tight_layout()
    out["fig"] = savefig("02_target_distribution")
    return out

def plot_correlation(df, logger):
    logger.info("Correlation heatmap")
    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                annot=False, linewidths=0.3, vmin=-1, vmax=1,
                ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title("Pearson Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return savefig("03_correlation_heatmap")


def plot_univariate(df, logger):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info("Univariate — %s numeric, %s categorical", len(num_cols), len(cat_cols))

    figs = []
    chunk = 4
    for i, start in enumerate(range(0, len(num_cols), chunk)):
        cols = num_cols[start: start + chunk]
        fig, axes = plt.subplots(len(cols), 2, figsize=(14, 3.5 * len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for row, col in enumerate(cols):
            data = df[col].dropna()
            axes[row][0].hist(data, bins=40, color="#4C72B0", alpha=0.75, edgecolor="white")
            axes[row][0].set_title(f"{col} — histogram")
            axes[row][0].set_ylabel("Count")
            axes[row][1].boxplot(data, vert=False, patch_artist=True,
                                 boxprops=dict(facecolor="#55A868", alpha=0.6))
            axes[row][1].set_title(f"{col} — boxplot")
        plt.suptitle(f"Numeric distributions (batch {i+1})", fontweight="bold")
        plt.tight_layout()
        figs.append(savefig(f"04_univariate_numeric_{i+1:02d}"))

    for col in cat_cols:
        counts = df[col].astype(str).value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        counts.plot(kind="bar", ax=ax, color="#C44E52", alpha=0.8, edgecolor="white")
        ax.set_title(f"{col} — top categories", fontweight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        figs.append(savefig(f"04_univariate_cat_{col}"))

    return figs

def default_by_category(df, logger):
    if "target_default" not in df.columns:
        return []

    skip = {"id", "emp_title", "issue_d", "earliest_cr_line"}
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if c not in skip]
    figs = []

    for col in cat_cols:
        top = df[col].astype(str).value_counts().head(8).index
        sub = df[df[col].astype(str).isin(top)].copy()
        sub["target_default"] = pd.to_numeric(sub["target_default"], errors="coerce")
        stats = (sub.groupby(col)["target_default"]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "default_rate", "count": "n"})
                    .sort_values("default_rate", ascending=False)
                    .reset_index())

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].bar(stats[col].astype(str), stats["default_rate"] * 100,
                    color="#E74C3C", alpha=0.8)
        axes[0].set_title(f"Default Rate by {col}", fontweight="bold")
        axes[0].set_ylabel("Default Rate (%)")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(stats[col].astype(str), stats["n"], color="#3498DB", alpha=0.8)
        axes[1].set_title(f"Loan Count by {col}", fontweight="bold")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=45)

        plt.suptitle(f"{col}", fontweight="bold")
        plt.tight_layout()
        figs.append(savefig(f"07_default_rate_{col}"))
        logger.info(f"{col} — highest default rate: {stats['default_rate'].max()*100:.1f}% ({stats.iloc[0][col]})")

    return figs


def numeric_vs_default(df, logger):
    if "target_default" not in df.columns:
        return []

    key = ["loan_amnt", "annual_inc", "installment", "dti",
           "int_rate", "revol_util", "fico_range_high", "avg_cur_bal"]
    present = [c for c in key if c in df.columns]
    dfc = df.dropna(subset=["target_default"]).copy()
    dfc["target_default"] = dfc["target_default"].astype(int)

    figs = []
    for start in range(0, len(present), 4):
        cols = present[start: start + 4]
        fig, axes = plt.subplots(len(cols), 2, figsize=(14, 4 * len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for i, col in enumerate(cols):
            d0 = dfc[dfc["target_default"] == 0][col].dropna()
            d1 = dfc[dfc["target_default"] == 1][col].dropna()
            d0.plot.kde(ax=axes[i][0], color="#4C72B0", label="Non-Default")
            d1.plot.kde(ax=axes[i][0], color="#DD8452", label="Default")
            axes[i][0].set_title(f"{col} — KDE")
            axes[i][0].legend()
            axes[i][1].boxplot([d0, d1], labels=["Non-Default", "Default"],
                               patch_artist=True,
                               boxprops=dict(facecolor="#AED6F1", alpha=0.6))
            axes[i][1].set_title(f"{col} — boxplot")
            logger.info(f"{col} — mean diff (default - non-default): {d1.mean() - d0.mean():.4f}")
        plt.tight_layout()
        figs.append(savefig(f"08_numeric_vs_default_{start//4+1:02d}"))
    return figs


def plot_interest_rate(df, logger):
    if "int_rate" not in df.columns or "target_default" not in df.columns:
        return None

    dfp = df.dropna(subset=["int_rate", "target_default"]).copy()
    dfp["target_default"] = dfp["target_default"].astype(int)
    if dfp["int_rate"].dtype == object:
        dfp["int_rate"] = dfp["int_rate"].str.replace("%", "").astype(float)

    nd = dfp[dfp["target_default"] == 0]["int_rate"]
    d  = dfp[dfp["target_default"] == 1]["int_rate"]
    logger.info(f"Interest rate — non-default: {nd.mean():.2f}%, default: {d.mean():.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    nd.plot.kde(ax=axes[0], color="#4C72B0", label="Non-Default")
    d.plot.kde(ax=axes[0],  color="#DD8452", label="Default")
    axes[0].set_title("Interest Rate KDE", fontweight="bold")
    axes[0].set_xlabel("Interest Rate (%)")
    axes[0].legend()
    axes[1].boxplot([nd, d], labels=["Non-Default", "Default"],
                   patch_artist=True, boxprops=dict(facecolor="#AED6F1", alpha=0.7))
    axes[1].set_title("Interest Rate by Default Status", fontweight="bold")
    axes[1].set_ylabel("Rate (%)")
    plt.tight_layout()
    return savefig("09_interest_rate")


def plot_dti(df, logger):
    if "dti" not in df.columns or "target_default" not in df.columns:
        return None

    dfd = df.dropna(subset=["dti", "target_default"]).copy()
    dfd = dfd[dfd["dti"] <= dfd["dti"].quantile(0.99)]
    dfd["target_default"] = dfd["target_default"].astype(int)

    nd = dfd[dfd["target_default"] == 0]["dti"]
    d  = dfd[dfd["target_default"] == 1]["dti"]
    logger.info(f"DTI — non-default: {nd.mean():.2f}, default: {d.mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    nd.plot.kde(ax=axes[0], color="#4C72B0", label="Non-Default")
    d.plot.kde(ax=axes[0],  color="#DD8452", label="Default")
    axes[0].set_title("DTI Distribution", fontweight="bold")
    axes[0].set_xlabel("DTI (%)")
    axes[0].legend()
    axes[1].boxplot([nd, d], labels=["Non-Default", "Default"], patch_artist=True,
                   boxprops=dict(facecolor="#A9DFBF", alpha=0.7))
    axes[1].set_title("DTI by Default Status", fontweight="bold")
    axes[1].set_ylabel("DTI (%)")
    plt.tight_layout()
    return savefig("10_dti_analysis")


def plot_loan_grade(df, logger):
    if "sub_grade" not in df.columns or "target_default" not in df.columns:
        return None

    grade_df = (df.dropna(subset=["target_default"])
                  .groupby("sub_grade")["target_default"]
                  .agg(["mean", "count"])
                  .reset_index()
                  .rename(columns={"mean": "default_rate", "count": "n_loans"})
                  .sort_values("sub_grade"))

    best  = grade_df.loc[grade_df["default_rate"].idxmin()]
    worst = grade_df.loc[grade_df["default_rate"].idxmax()]
    logger.info(f"Sub-grade — best: {best['sub_grade']} ({best['default_rate']*100:.1f}%), "
                f"worst: {worst['sub_grade']} ({worst['default_rate']*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(grade_df["sub_grade"], grade_df["default_rate"] * 100,
                color="#E74C3C", alpha=0.8)
    axes[0].set_title("Default Rate by Sub-Grade", fontweight="bold")
    axes[0].set_ylabel("Default Rate (%)")
    axes[0].tick_params(axis="x", rotation=75)

    axes[1].bar(grade_df["sub_grade"], grade_df["n_loans"], color="#3498DB", alpha=0.8)
    axes[1].set_title("Loan Volume by Sub-Grade", fontweight="bold")
    axes[1].set_ylabel("Loans")
    axes[1].tick_params(axis="x", rotation=75)
    plt.tight_layout()
    return savefig("11_loan_grade")


def plot_emp_length(df, logger):
    if "emp_length" not in df.columns or "target_default" not in df.columns:
        return None

    dfe = df.dropna(subset=["emp_length", "target_default"]).copy()
    dfe["emp_yrs"] = pd.to_numeric(
        dfe["emp_length"].astype(str).str.replace(r"\D", "", regex=True), errors="coerce"
    )
    dfe = dfe.dropna(subset=["emp_yrs"])
    dfe["target_default"] = dfe["target_default"].astype(int)

    stats = (dfe.groupby("emp_yrs")["target_default"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "default_rate", "count": "n"})
                .reset_index().sort_values("emp_yrs"))
    logger.info("Emp length default rates:\n%s", stats.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(stats["emp_yrs"], stats["n"], color="#5DADE2", alpha=0.8)
    axes[0].set_title("Loan Volume by Employment Length", fontweight="bold")
    axes[0].set_xlabel("Years")
    axes[0].set_ylabel("Count")

    axes[1].plot(stats["emp_yrs"], stats["default_rate"] * 100,
                 color="#E74C3C", marker="o")
    axes[1].set_title("Default Rate by Employment Length", fontweight="bold")
    axes[1].set_xlabel("Years")
    axes[1].set_ylabel("Default Rate (%)")
    axes[1].grid(True, alpha=0.4)
    plt.tight_layout()
    return savefig("12_emp_length")


def plot_annual_income(df, logger):
    if "annual_inc" not in df.columns or "target_default" not in df.columns:
        return None

    dfi = df.dropna(subset=["annual_inc", "target_default"]).copy()
    dfi = dfi[(dfi["annual_inc"] > 0) & (dfi["annual_inc"] <= dfi["annual_inc"].quantile(0.99))]
    dfi["log_inc"] = np.log10(dfi["annual_inc"])
    dfi["target_default"] = dfi["target_default"].astype(int)

    nd = dfi[dfi["target_default"] == 0]["log_inc"]
    d  = dfi[dfi["target_default"] == 1]["log_inc"]
    logger.info(f"log10(income) — non-default: {nd.mean():.3f}, default: {d.mean():.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    nd.plot.kde(ax=ax, color="#4C72B0", label="Non-Default")
    d.plot.kde(ax=ax,  color="#DD8452", label="Default")
    ax.set_title("Annual Income (log scale) by Default Status", fontweight="bold")
    ax.set_xlabel("log10(Annual Income)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return savefig("13_annual_income")


def plot_purpose(df, logger):
    if "purpose" not in df.columns or "target_default" not in df.columns:
        return None

    dfp = df.dropna(subset=["target_default"]).copy()
    dfp["target_default"] = dfp["target_default"].astype(int)
    stats = (dfp.groupby("purpose")
                 .agg(n=("purpose", "count"), default_rate=("target_default", "mean"))
                 .reset_index().sort_values("default_rate", ascending=False))

    logger.info(f"Highest-risk purpose: {stats.iloc[0]['purpose']} ({stats.iloc[0]['default_rate']*100:.1f}%)")
    logger.info(f"Lowest-risk purpose:  {stats.iloc[-1]['purpose']} ({stats.iloc[-1]['default_rate']*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(stats["purpose"], stats["default_rate"] * 100, color="#E74C3C", alpha=0.8)
    axes[0].set_title("Default Rate by Purpose", fontweight="bold")
    axes[0].set_xlabel("Default Rate (%)")

    sv = stats.sort_values("n")
    axes[1].barh(sv["purpose"], sv["n"], color="#3498DB", alpha=0.8)
    axes[1].set_title("Loan Volume by Purpose", fontweight="bold")
    axes[1].set_xlabel("Loans")
    plt.tight_layout()
    return savefig("14_loan_purpose")

def plot_fico(df, logger):
    fico_col = next((c for c in ["fico_range_high", "fico_range_low", "fico"] if c in df.columns), None)
    if fico_col is None or "target_default" not in df.columns:
        logger.warning("FICO column not found")
        return None

    dff = df.dropna(subset=[fico_col, "target_default"]).copy()
    dff["target_default"] = dff["target_default"].astype(int)

    bins   = [580, 620, 660, 700, 740, 780, 850]
    labels = ["580-620", "620-660", "660-700", "700-740", "740-780", "780-850"]
    dff["fico_band"] = pd.cut(dff[fico_col], bins=bins, labels=labels)
    band_stats = (dff.groupby("fico_band", observed=True)["target_default"]
                     .agg(["mean", "count"])
                     .rename(columns={"mean": "default_rate", "count": "n"})
                     .reset_index())

    nd = dff[dff["target_default"] == 0][fico_col]
    d  = dff[dff["target_default"] == 1][fico_col]
    logger.info(f"FICO — non-default: {nd.mean():.1f}, default: {d.mean():.1f}, diff: {nd.mean()-d.mean():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    nd.plot.kde(ax=axes[0], color="#4C72B0", label="Non-Default")
    d.plot.kde(ax=axes[0],  color="#DD8452", label="Default")
    axes[0].set_title("FICO Score KDE", fontweight="bold")
    axes[0].set_xlabel("FICO Score")
    axes[0].legend()

    axes[1].bar(band_stats["fico_band"].astype(str), band_stats["default_rate"] * 100,
                color="#E74C3C", alpha=0.8)
    axes[1].set_title("Default Rate by FICO Band", fontweight="bold")
    axes[1].set_ylabel("Default Rate (%)")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(band_stats["fico_band"].astype(str), band_stats["n"],
                color="#3498DB", alpha=0.8)
    axes[2].set_title("Volume by FICO Band", fontweight="bold")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return savefig("15_fico_analysis")


def plot_home_ownership(df, logger):
    if "home_ownership" not in df.columns or "target_default" not in df.columns:
        return None

    dfh = df.dropna(subset=["home_ownership", "target_default"]).copy()
    dfh["target_default"] = dfh["target_default"].astype(int)
    stats = (dfh.groupby("home_ownership")["target_default"]
                 .agg(["mean", "count"])
                 .rename(columns={"mean": "default_rate", "count": "n"})
                 .sort_values("default_rate", ascending=False)
                 .reset_index())
    logger.info("Home ownership rates:\n%s", stats.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(stats["home_ownership"], stats["default_rate"] * 100,
                color="#8E44AD", alpha=0.8)
    axes[0].set_title("Default Rate by Home Ownership", fontweight="bold")
    axes[0].set_ylabel("Default Rate (%)")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(stats["home_ownership"], stats["n"], color="#2ECC71", alpha=0.8)
    axes[1].set_title("Loan Count by Home Ownership", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return savefig("16_home_ownership")



def plot_verification(df, logger):
    if "verification_status" not in df.columns or "target_default" not in df.columns:
        return None

    dfv = df.dropna(subset=["verification_status", "target_default"]).copy()
    dfv["target_default"] = dfv["target_default"].astype(int)
    stats = (dfv.groupby("verification_status")["target_default"]
                 .agg(["mean", "count"])
                 .rename(columns={"mean": "default_rate", "count": "n"})
                 .sort_values("default_rate", ascending=False)
                 .reset_index())
    logger.info("Verification status rates:\n%s", stats.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(stats))
    width = 0.35
    ax.bar([i - width/2 for i in x], stats["default_rate"] * 100,
           width, label="Default Rate (%)", color="#E74C3C", alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar([i + width/2 for i in x], stats["n"],
            width, label="Loan Count", color="#3498DB", alpha=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stats["verification_status"], rotation=20)
    ax.set_ylabel("Default Rate (%)", color="#E74C3C")
    ax2.set_ylabel("Loan Count", color="#3498DB")
    ax.set_title("Verification Status — Default Rate vs Volume", fontweight="bold")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, loc="upper right")
    plt.tight_layout()
    return savefig("17_verification_status")



def plot_temporal(df, logger):
    date_col = next((c for c in ["issue_d", "earliest_cr_line"] if c in df.columns), None)
    if date_col is None or "target_default" not in df.columns:
        logger.warning("Date column not found, skipping temporal")
        return None

    dft = df[[date_col, "target_default", "loan_amnt"]].copy()
    dft[date_col] = pd.to_datetime(dft[date_col], errors="coerce")
    dft = dft.dropna(subset=[date_col])
    dft["ym"] = dft[date_col].dt.to_period("M")

    monthly = (dft.groupby("ym")
                  .agg(n_loans=("loan_amnt", "count"),
                       default_rate=("target_default", "mean"),
                       avg_loan=("loan_amnt", "mean"))
                  .reset_index())
    monthly["ym"] = monthly["ym"].astype(str)
    logger.info(f"Temporal: {monthly['ym'].iloc[0]} to {monthly['ym'].iloc[-1]} ({len(monthly)} months)")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(monthly["ym"], monthly["n_loans"], color="#3498DB", linewidth=1.5)
    axes[0].set_title("Monthly Loan Originations", fontweight="bold")
    axes[0].set_ylabel("# Loans")

    axes[1].plot(monthly["ym"], monthly["default_rate"] * 100, color="#E74C3C", linewidth=1.5)
    axes[1].set_title("Monthly Default Rate (%)", fontweight="bold")
    axes[1].set_ylabel("Default Rate (%)")

    axes[2].plot(monthly["ym"], monthly["avg_loan"], color="#27AE60", linewidth=1.5)
    axes[2].set_title("Average Loan Amount Over Time", fontweight="bold")
    axes[2].set_ylabel("Avg Loan ($)")

    step = max(1, len(monthly) // 20)
    ticks = list(range(0, len(monthly), step))
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels([monthly["ym"].iloc[i] for i in ticks], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Temporal Trends", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return savefig("19_temporal_trends")



def top_corr_pairs(df, logger):
    logger.info("Top correlated pairs")
    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr().abs()
    pairs  = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                  .stack().sort_values(ascending=False))
    top = pairs.head(15).reset_index()
    top.columns = ["feature_1", "feature_2", "abs_corr"]
    logger.info("Top pairs:\n%s", top.to_string(index=False))

    top["label"] = top["feature_1"] + " vs " + top["feature_2"]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(top["label"], top["abs_corr"], color="#5DADE2", alpha=0.8)
    ax.axhline(0.8, color="red", linestyle="--", label="0.8 threshold")
    ax.set_title("Top Feature Pairs by Absolute Correlation", fontweight="bold")
    ax.set_ylabel("|Pearson r|")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    return {"top_pairs": top, "fig": savefig("20_top_corr_pairs")}



def feature_importance(df, logger):
    if "target_default" not in df.columns:
        return {}

    dfc = df.dropna(subset=["target_default"]).copy()
    num_df = dfc.select_dtypes(include="number").drop(columns=["target_default"], errors="ignore")
    target = dfc["target_default"]
    X = num_df.fillna(num_df.median())
    y = target.loc[X.index]

    pearson = (num_df.join(target).corr()["target_default"]
               .drop("target_default").abs().rename("pearson_abs"))

    logger.info("Computing mutual information …")
    mi = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi, index=X.columns, name="mutual_info")

    logger.info("Fitting random forest …")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns, name="random_forest")

    summary = (pd.DataFrame({"pearson_abs": pearson, "mutual_info": mi_scores, "random_forest": rf_imp})
                 .dropna(subset=["pearson_abs"])
                 .sort_values("random_forest", ascending=False)
                 .round(4))
    summary.index.name = "Feature"
    logger.info("Feature importance (top 15):\n%s", summary.head(15).to_string())

    top15  = summary.head(15).copy()
    normed = top15.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    x      = np.arange(len(normed))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - w, normed["pearson_abs"],   w, label="Pearson |r|",   color="#4C72B0", alpha=0.8)
    ax.bar(x,     normed["mutual_info"],   w, label="Mutual Info",   color="#55A868", alpha=0.8)
    ax.bar(x + w, normed["random_forest"], w, label="Random Forest", color="#C44E52", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(normed.index, rotation=45, ha="right")
    ax.set_title("Feature Importance (normalised)", fontweight="bold")
    ax.set_ylabel("Normalised Score")
    ax.legend()
    plt.tight_layout()
    return {"summary": summary, "fig": savefig("21_feature_importance")}

def make_report(meta, target_info, top_pairs, imp, logger):
    logger.info("Building HTML report")

    def tbl(df, max_rows=20):
        if df is None or df.empty:
            return "<p>No data.</p>"
        return df.head(max_rows).to_html(
            classes="table table-sm table-striped table-bordered", border=0, index=True)

    all_figs = sorted(FIG_DIR.glob("*.png"))
    gallery  = ""
    for fp in all_figs:
        rel = fp.relative_to(REPORT_DIR)
        gallery += (
            f'<div class="col-12 col-md-6 mb-4">'
            f'<div class="card h-100 shadow-sm">'
            f'<img src="{rel}" class="card-img-top" loading="lazy" '
            f'style="max-height:420px;object-fit:contain;padding:8px">'
            f'<div class="card-footer text-muted small">{fp.stem.replace("_"," ").title()}</div>'
            f'</div></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Default EDA Report</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#f8f9fa; }}
  .hero {{ background: linear-gradient(135deg,#1a3a5c 0%,#2980b9 100%);
           color:#fff; padding:3rem 2rem; border-radius:0 0 12px 12px; }}
  h2 {{ color:#1a3a5c; border-bottom:2px solid #2980b9; padding-bottom:.4rem; }}
  .kpi-card {{ border-radius:10px; border:none; }}
  .kpi-value {{ font-size:2rem; font-weight:700; }}
  .table {{ font-size:.85rem; }}
  .section {{ margin-top:3rem; }}
  footer {{ margin-top:4rem; padding:1.5rem; background:#1a3a5c; color:#adb5bd;
            text-align:center; border-radius:12px 12px 0 0; }}
</style>
</head>
<body>

<div class="hero text-center mb-4">
  <h1 class="display-5 fw-bold">Loan Default — Exploratory Data Analysis</h1>
  <p class="lead">Statistical report for dashboard preparation</p>
</div>

<div class="container-fluid px-4">

<div class="row g-3 mb-4">
  <div class="col-6 col-md-3">
    <div class="card kpi-card bg-primary text-white">
      <div class="card-body"><div class="kpi-value">{meta.get("n_rows",0):,}</div>
        <div>Total Records</div></div></div></div>
  <div class="col-6 col-md-3">
    <div class="card kpi-card bg-secondary text-white">
      <div class="card-body"><div class="kpi-value">{meta.get("n_cols",0)}</div>
        <div>Features</div></div></div></div>
  <div class="col-6 col-md-3">
    <div class="card kpi-card bg-danger text-white">
      <div class="card-body"><div class="kpi-value">{target_info.get("n_default",0):,}</div>
        <div>Defaults</div></div></div></div>
  <div class="col-6 col-md-3">
    <div class="card kpi-card bg-success text-white">
      <div class="card-body"><div class="kpi-value">{target_info.get("n_non_default",0):,}</div>
        <div>Non-Defaults</div></div></div></div>
</div>

<div class="alert alert-warning">
  <strong>Class Imbalance:</strong>
  ratio <strong>{target_info.get("imbalance_ratio","N/A")}:1</strong> (Non-Default : Default).
  Majority-class baseline accuracy: <strong>{100*target_info.get("majority_baseline_acc",0):.2f}%</strong>.
  Consider SMOTE or class weighting during modelling.
</div>

<div class="section">
  <h2>Top Correlated Feature Pairs</h2>
  {tbl(top_pairs)}
  <p class="text-muted small">Pairs with |r| &gt; 0.8 are candidates for removal.</p>
</div>

<div class="section">
  <h2>Feature Importance</h2>
  {tbl(imp)}
</div>

<div class="section">
  <h2>Figures</h2>
  <div class="row">{gallery}</div>
</div>

</div>

<footer>
  EDA Pipeline — Loan Default Analysis — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    out = REPORT_DIR / "EDA_Report.html"
    out.write_text(html, encoding="utf-8")
    logger.info("Report saved: %s", out)
    return out

def main():
    logger = setup_logging()
    logger.info("Starting EDA pipeline")

    df = load_data(logger)
    df = build_target(df, logger)

    meta        = overview(df, logger)
    plot_missing(df, logger)
    target_info = check_target_distribution(df, logger)
    plot_correlation(df, logger)
    plot_univariate(df, logger)
    default_by_category(df, logger)
    numeric_vs_default(df, logger)
    plot_interest_rate(df, logger)
    plot_dti(df, logger)
    plot_loan_grade(df, logger)
    plot_emp_length(df, logger)
    plot_annual_income(df, logger)
    plot_purpose(df, logger)
    plot_fico(df, logger)
    plot_home_ownership(df, logger)
    plot_verification(df, logger)
    plot_temporal(df, logger)
    pairs = top_corr_pairs(df, logger)
    imp   = feature_importance(df, logger)

    report = make_report(
        meta          = meta.get("meta", {}),
        target_info   = target_info,
        top_pairs     = pairs.get("top_pairs", pd.DataFrame()),
        imp           = imp.get("summary", pd.DataFrame()),
        logger        = logger,
    )

    logger.info("Done. Report: %s | Figures: %s", report, FIG_DIR)


if __name__ == "__main__":
    main()