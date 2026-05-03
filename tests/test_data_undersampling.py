
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.DataUndersampling import main


def _make_csv(tmp_path: Path, n_majority: int = 80, n_minority: int = 20,
              target: str = "loan_status") -> Path:
    majority = pd.DataFrame({"feature_a": range(n_majority),
                              "feature_b": range(n_majority, n_majority * 2),
                              target: [0] * n_majority})
    minority = pd.DataFrame({"feature_a": range(n_minority),
                              "feature_b": range(n_minority, n_minority * 2),
                              target: [1] * n_minority})
    df = pd.concat([majority, minority], ignore_index=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _run_main(tmp_path: Path, extra_args: list[str] | None = None,
              n_majority: int = 80, n_minority: int = 20,
              target: str = "loan_status") -> Path:
    """Create a CSV, call main() with sensible defaults, return output path."""
    input_path = _make_csv(tmp_path, n_majority, n_minority, target)
    output_path = tmp_path / "out.csv"
    argv = [
        "--input", str(input_path),
        "--output", str(output_path),
        "--target", target,
    ] + (extra_args or [])
    with patch("sys.argv", ["DataUndersampling.py"] + argv):
        main()
    return output_path


class TestMainErrorHandling:
    def test_raises_if_input_missing(self, tmp_path):
        missing = tmp_path / "no_such_file.csv"
        output = tmp_path / "out.csv"
        with patch("sys.argv", [
            "DataUndersampling.py",
            "--input", str(missing),
            "--output", str(output),
        ]):
            with pytest.raises(FileNotFoundError, match="Input file not found"):
                main()

    def test_raises_if_target_column_missing(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_path = tmp_path / "no_target.csv"
        df.to_csv(csv_path, index=False)
        output = tmp_path / "out.csv"
        with patch("sys.argv", [
            "DataUndersampling.py",
            "--input", str(csv_path),
            "--output", str(output),
            "--target", "loan_status",
        ]):
            with pytest.raises(ValueError, match="Target column"):
                main()

    def test_raises_if_ratio_zero(self, tmp_path):
        csv_path = _make_csv(tmp_path)
        output = tmp_path / "out.csv"
        with patch("sys.argv", [
            "DataUndersampling.py",
            "--input", str(csv_path),
            "--output", str(output),
            "--ratio", "0.0",
        ]):
            with pytest.raises(ValueError, match="--ratio must be"):
                main()

    def test_raises_if_ratio_negative(self, tmp_path):
        csv_path = _make_csv(tmp_path)
        output = tmp_path / "out.csv"
        with patch("sys.argv", [
            "DataUndersampling.py",
            "--input", str(csv_path),
            "--output", str(output),
            "--ratio", "-0.5",
        ]):
            with pytest.raises(ValueError, match="--ratio must be"):
                main()

    def test_raises_if_ratio_greater_than_one(self, tmp_path):
        csv_path = _make_csv(tmp_path)
        output = tmp_path / "out.csv"
        with patch("sys.argv", [
            "DataUndersampling.py",
            "--input", str(csv_path),
            "--output", str(output),
            "--ratio", "1.1",
        ]):
            with pytest.raises(ValueError, match="--ratio must be"):
                main()

    def test_ratio_exactly_one_does_not_raise(self, tmp_path):
        """Boundary value: ratio=1.0 is valid."""
        output = _run_main(tmp_path, extra_args=["--ratio", "1.0"])
        assert output.exists()


class TestMainOutputFile:
    def test_output_file_is_created(self, tmp_path):
        output = _run_main(tmp_path)
        assert output.exists()

    def test_output_is_valid_csv(self, tmp_path):
        output = _run_main(tmp_path)
        df = pd.read_csv(output)
        assert isinstance(df, pd.DataFrame)

    def test_output_contains_target_column(self, tmp_path):
        output = _run_main(tmp_path)
        df = pd.read_csv(output)
        assert "loan_status" in df.columns

    def test_output_contains_feature_columns(self, tmp_path):
        output = _run_main(tmp_path)
        df = pd.read_csv(output)
        assert "feature_a" in df.columns
        assert "feature_b" in df.columns

    def test_output_parent_dirs_created(self, tmp_path):
        """Output path with a non-existent parent should be created."""
        input_path = _make_csv(tmp_path)
        nested_output = tmp_path / "a" / "b" / "out.csv"
        with patch("sys.argv", [
            "DataUndersampling.py",
            "--input", str(input_path),
            "--output", str(nested_output),
        ]):
            main()
        assert nested_output.exists()


class TestMainClassBalance:
    def test_balanced_output_with_ratio_one(self, tmp_path):
        output = _run_main(tmp_path, extra_args=["--ratio", "1.0"],
                           n_majority=80, n_minority=20)
        df = pd.read_csv(output)
        counts = df["loan_status"].value_counts()
        assert counts[0] == counts[1]

    def test_majority_class_is_reduced(self, tmp_path):
        output = _run_main(tmp_path, n_majority=80, n_minority=20)
        df = pd.read_csv(output)
        assert len(df) < 100  # original total

    def test_minority_class_is_unchanged(self, tmp_path):
        """RandomUnderSampler never adds minority samples."""
        output = _run_main(tmp_path, n_majority=80, n_minority=20)
        df = pd.read_csv(output)
        assert df["loan_status"].value_counts()[1] == 20

    def test_partial_ratio_reduces_majority(self, tmp_path):
        """ratio=0.5 → majority count = minority / 0.5 = 40."""
        output = _run_main(tmp_path, extra_args=["--ratio", "0.5"],
                           n_majority=80, n_minority=20)
        df = pd.read_csv(output)
        counts = df["loan_status"].value_counts()
        # minority stays 20, majority becomes 20/0.5 = 40
        assert counts[1] == 20
        assert counts[0] == 40

    def test_already_balanced_data_unchanged(self, tmp_path):
        """If data is already balanced, ratio=1.0 keeps it as-is."""
        output = _run_main(tmp_path, extra_args=["--ratio", "1.0"],
                           n_majority=50, n_minority=50)
        df = pd.read_csv(output)
        counts = df["loan_status"].value_counts()
        assert counts[0] == 50
        assert counts[1] == 50

    def test_output_only_contains_original_labels(self, tmp_path):
        output = _run_main(tmp_path)
        df = pd.read_csv(output)
        assert set(df["loan_status"].unique()).issubset({0, 1})


class TestMainCustomTarget:
    def test_custom_target_column_works(self, tmp_path):
        output = _run_main(tmp_path, target="status")
        df = pd.read_csv(output)
        assert "status" in df.columns

    def test_default_target_absent_after_custom(self, tmp_path):
        """When a custom target is used, 'loan_status' should not appear."""
        output = _run_main(tmp_path, target="status")
        df = pd.read_csv(output)
        assert "loan_status" not in df.columns


class TestMainReproducibility:
    def test_same_seed_produces_same_output(self, tmp_path):
        out1 = _run_main(tmp_path / "run1", extra_args=["--seed", "7"])
        out2 = _run_main(tmp_path / "run2", extra_args=["--seed", "7"])
        df1 = pd.read_csv(out1).sort_values(by=["feature_a", "feature_b"]).reset_index(drop=True)
        df2 = pd.read_csv(out2).sort_values(by=["feature_a", "feature_b"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_may_differ(self, tmp_path):
        """With a large-enough dataset two seeds should yield different rows."""
        out1 = _run_main(tmp_path / "run1", extra_args=["--seed", "1"],
                         n_majority=200, n_minority=50)
        out2 = _run_main(tmp_path / "run2", extra_args=["--seed", "99"],
                         n_majority=200, n_minority=50)
        df1 = pd.read_csv(out1)
        df2 = pd.read_csv(out2)
        # Same shape, but at least one row should differ
        assert df1.shape == df2.shape
        assert not df1.equals(df2)


class TestMainStdout:
    def test_prints_before_distribution(self, tmp_path, capsys):
        _run_main(tmp_path)
        captured = capsys.readouterr()
        assert "before undersampling" in captured.out.lower()

    def test_prints_after_distribution(self, tmp_path, capsys):
        _run_main(tmp_path)
        captured = capsys.readouterr()
        assert "after undersampling" in captured.out.lower()

    def test_prints_saved_path(self, tmp_path, capsys):
        output = _run_main(tmp_path)
        captured = capsys.readouterr()
        assert str(output) in captured.out

    def test_prints_rows_before(self, tmp_path, capsys):
        _run_main(tmp_path)
        captured = capsys.readouterr()
        assert "Rows before:" in captured.out

    def test_prints_rows_after(self, tmp_path, capsys):
        _run_main(tmp_path)
        captured = capsys.readouterr()
        assert "Rows after:" in captured.out

    def test_row_counts_are_accurate(self, tmp_path, capsys):
        _run_main(tmp_path, n_majority=80, n_minority=20)
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "40" in captured.out