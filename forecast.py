"""Category-aware retail demand forecasting with StatsForecast.

Edit the three assignment inputs immediately below or pass command-line
overrides. The rest of the module is intentionally functional so it can be
tested and reused without relying on notebook state.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    HistoricAverage,
    Holt,
    HoltWinters,
    Naive,
    RandomWalkWithDrift,
    SeasonalNaive,
    SimpleExponentialSmoothing,
    WindowAverage,
)
from statsmodels.tsa.seasonal import STL

# Required assignment inputs. Command-line arguments can override them.
file_name = "proj1_exampleinput.csv"
ids_to_forecast = ["C2208", "C1030", "C1036", "C1126", "C1042"]
metric = "mape"

SEASON_LENGTH = 12
FORECAST_HORIZON = 12
CV_STEP_SIZE = 6
CV_WINDOWS = 3
MIN_CV_OBSERVATIONS = 48
HIGH_SEASONALITY_THRESHOLD = 0.60


@dataclass(frozen=True)
class SeriesProfile:
    """Properties used to choose the candidate model family."""

    unique_id: str
    source_observations: int
    prepared_observations: int
    inserted_zero_months: int
    has_zero: bool
    all_positive: bool
    seasonality_strength: float | None
    residual_variability: float | None
    category: int


def validate_user_inputs(
    input_file: str | Path, selected_ids: list[str], accuracy_metric: str
) -> None:
    """Validate the assignment inputs before any model work begins."""

    if accuracy_metric not in {"mape", "mse"}:
        raise ValueError("metric must be either 'mape' or 'mse'.")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError("ids_to_forecast must be a non-empty list of strings.")
    if len(selected_ids) > 5:
        raise ValueError("ids_to_forecast may contain at most five series.")
    if any(not isinstance(value, str) or not value.strip() for value in selected_ids):
        raise ValueError("Every value in ids_to_forecast must be a non-empty string.")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("ids_to_forecast must not contain duplicates.")
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")


def load_and_prepare_data(
    input_file: str | Path, selected_ids: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, validate, filter, and regularize monthly sales data.

    Missing months between a series' first and last observation are interpreted
    as zero sales. This makes the monthly frequency explicit and ensures that
    zero-demand periods affect categorization and model evaluation.
    """

    raw = pd.read_csv(input_file)
    required = {"product_class", "Month", "sales_volume"}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = raw.loc[:, ["product_class", "Month", "sales_volume"]].rename(
        columns={"product_class": "unique_id", "Month": "ds", "sales_volume": "y"}
    )
    data["ds"] = pd.to_datetime(data["ds"], errors="raise")
    data["y"] = pd.to_numeric(data["y"], errors="raise")

    if data[["unique_id", "ds", "y"]].isna().any().any():
        raise ValueError("Input data contains missing identifiers, dates, or sales values.")
    if data.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Input data contains duplicate product_class and Month rows.")
    if (data["ds"].dt.day != 1).any():
        raise ValueError("Month values must represent the first day of each month.")
    if (data["y"] < 0).any():
        raise ValueError("sales_volume must not contain negative values.")

    available_ids = set(data["unique_id"].astype(str))
    missing_ids = [uid for uid in selected_ids if uid not in available_ids]
    if missing_ids:
        raise ValueError(f"Unknown time-series IDs: {missing_ids}")

    filtered = data[data["unique_id"].isin(selected_ids)].copy()
    source_counts = filtered.groupby("unique_id", observed=True).size().to_dict()
    prepared_parts: list[pd.DataFrame] = []

    for uid in selected_ids:
        series = filtered[filtered["unique_id"] == uid].sort_values("ds")
        complete_months = pd.date_range(series["ds"].min(), series["ds"].max(), freq="MS")
        regularized = (
            series.set_index("ds")[["y"]]
            .reindex(complete_months, fill_value=0.0)
            .rename_axis("ds")
            .reset_index()
        )
        regularized.insert(0, "unique_id", uid)
        prepared_parts.append(regularized)

    prepared = pd.concat(prepared_parts, ignore_index=True)
    prepared["unique_id"] = prepared["unique_id"].astype(str)
    return prepared, {str(key): int(value) for key, value in source_counts.items()}


def calculate_series_profile(
    series: pd.DataFrame, source_observations: int
) -> SeriesProfile:
    """Measure series properties and assign one of the five project categories."""

    values = series.sort_values("ds")["y"].to_numpy(dtype=float)
    prepared_observations = len(values)
    inserted_zero_months = prepared_observations - source_observations
    has_zero = bool(np.any(values == 0))
    all_positive = bool(np.all(values > 0))
    seasonality_strength: float | None = None
    residual_variability: float | None = None

    if prepared_observations < 2 * SEASON_LENGTH:
        category = 1
    else:
        decomposition = STL(values, period=SEASON_LENGTH, robust=True).fit()
        residual = np.asarray(decomposition.resid, dtype=float)
        seasonal = np.asarray(decomposition.seasonal, dtype=float)
        denominator = float(np.var(residual + seasonal))
        seasonality_strength = (
            max(0.0, min(1.0, 1.0 - float(np.var(residual)) / denominator))
            if denominator > 0
            else 0.0
        )
        mean_level = float(np.mean(values))
        residual_variability = (
            float(np.std(residual) / abs(mean_level)) if mean_level != 0 else math.inf
        )
        high_seasonality = seasonality_strength >= HIGH_SEASONALITY_THRESHOLD

        if has_zero and high_seasonality:
            category = 2
        elif high_seasonality and all_positive:
            category = 3
        elif has_zero:
            category = 4
        else:
            category = 5

    return SeriesProfile(
        unique_id=str(series["unique_id"].iloc[0]),
        source_observations=source_observations,
        prepared_observations=prepared_observations,
        inserted_zero_months=inserted_zero_months,
        has_zero=has_zero,
        all_positive=all_positive,
        seasonality_strength=seasonality_strength,
        residual_variability=residual_variability,
        category=category,
    )


def model_factories_for_category(category: int) -> dict[str, Callable[[], object]]:
    """Return category-appropriate models, including Naive as the benchmark."""

    pools: dict[int, dict[str, Callable[[], object]]] = {
        1: {
            "SES": lambda: SimpleExponentialSmoothing(alpha=0.4, alias="SES"),
            "Naive": lambda: Naive(alias="Naive"),
        },
        2: {
            "SeasonalNaive": lambda: SeasonalNaive(
                season_length=SEASON_LENGTH, alias="SeasonalNaive"
            ),
            "AutoETS_Additive": lambda: AutoETS(
                season_length=SEASON_LENGTH, model="AAA", alias="AutoETS_Additive"
            ),
            "HoltWinters_Additive": lambda: HoltWinters(
                season_length=SEASON_LENGTH,
                error_type="A",
                alias="HoltWinters_Additive",
            ),
            "HistoricAverage": lambda: HistoricAverage(alias="HistoricAverage"),
            "Naive": lambda: Naive(alias="Naive"),
        },
        3: {
            "SeasonalNaive": lambda: SeasonalNaive(
                season_length=SEASON_LENGTH, alias="SeasonalNaive"
            ),
            "AutoETS": lambda: AutoETS(season_length=SEASON_LENGTH, alias="AutoETS"),
            "HoltWinters_Multiplicative": lambda: HoltWinters(
                season_length=SEASON_LENGTH,
                error_type="M",
                alias="HoltWinters_Multiplicative",
            ),
            "AutoARIMA": lambda: AutoARIMA(
                season_length=SEASON_LENGTH, alias="AutoARIMA"
            ),
            "Naive": lambda: Naive(alias="Naive"),
        },
        4: {
            "WindowAverage_12": lambda: WindowAverage(
                window_size=12, alias="WindowAverage_12"
            ),
            "RandomWalkWithDrift": lambda: RandomWalkWithDrift(
                alias="RandomWalkWithDrift"
            ),
            "Holt": lambda: Holt(alias="Holt"),
            "HistoricAverage": lambda: HistoricAverage(alias="HistoricAverage"),
            "Naive": lambda: Naive(alias="Naive"),
        },
        5: {
            "AutoETS": lambda: AutoETS(season_length=SEASON_LENGTH, alias="AutoETS"),
            "AutoARIMA": lambda: AutoARIMA(
                season_length=SEASON_LENGTH, alias="AutoARIMA"
            ),
            "Holt": lambda: Holt(alias="Holt"),
            "HistoricAverage": lambda: HistoricAverage(alias="HistoricAverage"),
            "Naive": lambda: Naive(alias="Naive"),
        },
    }
    return pools[category]


def score_cross_validation(
    cv_frame: pd.DataFrame, model_names: list[str], accuracy_metric: str
) -> dict[str, float]:
    """Calculate MSE or MAPE for each model over every CV error.

    MAPE is undefined where the actual value is zero. Those observations are
    excluded from MAPE after the required warning is emitted by ``run_forecast``.
    """

    scores: dict[str, float] = {}
    actual = cv_frame["y"].to_numpy(dtype=float)
    for model_name in model_names:
        predicted = cv_frame[model_name].to_numpy(dtype=float)
        finite = np.isfinite(actual) & np.isfinite(predicted)
        if accuracy_metric == "mape":
            finite &= actual != 0
            error = np.abs((actual[finite] - predicted[finite]) / actual[finite])
        else:
            error = np.square(actual[finite] - predicted[finite])
        scores[model_name] = float(np.mean(error)) if error.size else math.nan
    return scores


def select_model(
    series: pd.DataFrame, profile: SeriesProfile, accuracy_metric: str
) -> tuple[str, dict[str, float]]:
    """Select the lowest-error candidate, or the rule-based model for short series."""

    factories = model_factories_for_category(profile.category)
    if profile.prepared_observations < MIN_CV_OBSERVATIONS:
        return "SES", {"SES": math.nan, "Naive": math.nan}

    models = [factory() for factory in factories.values()]
    forecaster = StatsForecast(
        models=models,
        freq="MS",
        n_jobs=-1,
        fallback_model=Naive(alias="FallbackNaive"),
    )
    cv_frame = forecaster.cross_validation(
        df=series,
        h=FORECAST_HORIZON,
        step_size=CV_STEP_SIZE,
        n_windows=CV_WINDOWS,
    )
    scores = score_cross_validation(cv_frame, list(factories), accuracy_metric)
    finite_scores = {name: value for name, value in scores.items() if np.isfinite(value)}
    if not finite_scores:
        raise RuntimeError(f"No candidate model produced a valid score for {profile.unique_id}.")
    return min(finite_scores, key=finite_scores.get), scores


def forecast_selected_model(
    series: pd.DataFrame, category: int, chosen_model: str
) -> pd.DataFrame:
    """Fit the selected model to all observations and forecast the next year."""

    model = model_factories_for_category(category)[chosen_model]()
    forecaster = StatsForecast(models=[model], freq="MS", n_jobs=1)
    forecast = forecaster.forecast(df=series, h=FORECAST_HORIZON)
    return forecast.rename(columns={chosen_model: "y_hat"})


def run_forecast(
    input_file: str | Path,
    selected_ids: list[str],
    accuracy_metric: str,
    output_directory: str | Path = "outputs",
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run the complete assignment workflow and save forecasts as CSV."""

    validate_user_inputs(input_file, selected_ids, accuracy_metric)
    data, source_counts = load_and_prepare_data(input_file, selected_ids)

    if accuracy_metric == "mape" and (data["y"] == 0).any():
        warnings.warn(
            "At least one selected series contains zero sales. MAPE is undefined "
            "for those observations, so zero-actual CV rows are excluded from MAPE.",
            UserWarning,
            stacklevel=2,
        )

    profiles: list[SeriesProfile] = []
    forecasts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for uid in selected_ids:
        series = data[data["unique_id"] == uid].copy()
        profile = calculate_series_profile(series, source_counts[uid])
        profiles.append(profile)
        chosen_model, scores = select_model(series, profile, accuracy_metric)
        chosen_accuracy = scores.get(chosen_model, math.nan)
        naive_accuracy = scores.get("Naive", math.nan)
        forecast = forecast_selected_model(series, profile.category, chosen_model)
        forecast["chosen_model"] = chosen_model
        forecast["category"] = profile.category
        forecast["metric"] = accuracy_metric
        forecast["chosen_accuracy"] = chosen_accuracy
        forecast["naive_accuracy"] = naive_accuracy
        forecasts.append(forecast)
        summaries.append(
            {
                "unique_id": uid,
                "category": profile.category,
                "chosen_model": chosen_model,
                "metric": accuracy_metric,
                "chosen_accuracy": chosen_accuracy,
                "naive_accuracy": naive_accuracy,
                "cv_status": (
                    "completed"
                    if profile.prepared_observations >= MIN_CV_OBSERVATIONS
                    else "skipped: fewer than 48 observations"
                ),
            }
        )

    forecast_frame = pd.concat(forecasts, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    profile_frame = pd.DataFrame(asdict(profile) for profile in profiles)
    summary_frame = summary_frame.merge(profile_frame, on=["unique_id", "category"])

    safe_ids = "_".join(re.sub(r"[^A-Za-z0-9_-]", "", uid) for uid in selected_ids)
    output_path = Path(output_directory) / f"forecasts_{safe_ids}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_frame.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return forecast_frame, summary_frame, output_path


def format_score(value: float) -> str:
    """Format an accuracy score without representing unavailable CV as zero."""

    return "n/a" if pd.isna(value) else f"{value:.4f}"


def parse_arguments() -> argparse.Namespace:
    """Parse optional command-line overrides for the three assignment inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", dest="input_file", default=file_name)
    parser.add_argument("--ids", nargs="+", default=ids_to_forecast)
    parser.add_argument("--metric", choices=["mape", "mse"], default=metric)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    """Execute the forecasting workflow and print the required summary."""

    args = parse_arguments()
    script_directory = Path(__file__).resolve().parent
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = script_directory / input_path
    output_directory = Path(args.output_dir)
    if not output_directory.is_absolute():
        output_directory = script_directory / output_directory

    _, summary, output_path = run_forecast(
        input_path,
        list(args.ids),
        args.metric,
        output_directory,
    )
    for row in summary.itertuples(index=False):
        print(
            f"Time series: {row.unique_id} | Category: {row.category} | "
            f"Chosen model: {row.chosen_model} | Metric: {row.metric.upper()} | "
            f"Chosen accuracy: {format_score(row.chosen_accuracy)} | "
            f"Naive benchmark: {format_score(row.naive_accuracy)} | "
            f"CV: {row.cv_status}"
        )
    print(f"Forecasts saved to: {output_path}")


if __name__ == "__main__":
    main()
