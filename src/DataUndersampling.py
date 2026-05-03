import argparse
from pathlib import Path

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Undersample majority class in a training dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "train.csv",
        help="Input CSV path (default: data/train.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "train_undersampled.csv",
        help="Output CSV path (default: data/train_undersampled.csv)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="loan_status",
        help="Target column name (default: loan_status)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help=(
            "Desired minority/majority ratio after undersampling. "
            "1.0 means fully balanced classes."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.input}")

    if args.ratio <= 0 or args.ratio > 1:
        raise ValueError("--ratio must be in the range (0, 1].")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    print("Class distribution before undersampling:")
    print(y.value_counts(dropna=False).to_string())

    sampler = RandomUnderSampler(
        sampling_strategy=args.ratio,
        random_state=args.seed,
    )
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    out_df = X_resampled.copy()
    out_df[args.target] = y_resampled

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print("\nClass distribution after undersampling:")
    print(out_df[args.target].value_counts(dropna=False).to_string())
    print(f"\nSaved undersampled data to: {args.output}")
    print(f"Rows before: {len(df):,}")
    print(f"Rows after:  {len(out_df):,}")


if __name__ == "__main__":
    main()
