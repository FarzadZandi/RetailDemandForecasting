from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast import (
    calculate_series_profile,
    load_and_prepare_data,
    score_cross_validation,
    validate_user_inputs,
)


def write_input(path: Path, rows: list[tuple[str, str, float]]) -> Path:
    pd.DataFrame(rows, columns=["product_class", "Month", "sales_volume"]).to_csv(
        path, index=False
    )
    return path


def test_validate_user_inputs_rejects_bad_metric_and_more_than_five_ids(
    tmp_path: Path,
) -> None:
    source = write_input(tmp_path / "input.csv", [("A", "2024-01-01", 1.0)])
    with pytest.raises(ValueError, match="metric"):
        validate_user_inputs(source, ["A"], "mae")
    with pytest.raises(ValueError, match="at most five"):
        validate_user_inputs(source, list("ABCDEF"), "mse")


def test_data_preparation_inserts_zero_for_missing_month(tmp_path: Path) -> None:
    source = write_input(
        tmp_path / "input.csv",
        [("A", "2024-01-01", 10.0), ("A", "2024-03-01", 12.0)],
    )
    prepared, source_counts = load_and_prepare_data(source, ["A"])
    assert source_counts == {"A": 2}
    assert prepared["ds"].tolist() == list(pd.date_range("2024-01-01", periods=3, freq="MS"))
    assert prepared["y"].tolist() == [10.0, 0.0, 12.0]


def test_short_series_is_category_one() -> None:
    series = pd.DataFrame(
        {
            "unique_id": ["A"] * 23,
            "ds": pd.date_range("2023-01-01", periods=23, freq="MS"),
            "y": np.arange(1, 24, dtype=float),
        }
    )
    profile = calculate_series_profile(series, source_observations=23)
    assert profile.category == 1
    assert profile.seasonality_strength is None


def test_mape_excludes_zero_actuals_and_mse_keeps_them() -> None:
    cv = pd.DataFrame({"y": [0.0, 10.0], "Model": [5.0, 8.0]})
    assert score_cross_validation(cv, ["Model"], "mape")["Model"] == pytest.approx(0.2)
    assert score_cross_validation(cv, ["Model"], "mse")["Model"] == pytest.approx(14.5)
