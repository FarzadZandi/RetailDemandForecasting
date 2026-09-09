# Retail Demand Forecasting

This project forecasts up to five monthly product-class sales series and selects a suitable statistical model for each series. It implements Task 1 of the TUM Business Forecasting Project 1 with the current StatsForecast API.

## What the project does

The pipeline:

1. validates the required `file_name`, `ids_to_forecast`, and `metric` inputs;
2. loads the supplied long-format CSV and checks its schema, dates, duplicates, and values;
3. fills missing months between a series' first and last observation with zero sales;
4. measures seasonal strength with robust STL decomposition;
5. assigns each series to one of the five project categories;
6. compares category-appropriate models with three rolling forecast origins;
7. selects the lowest-error model for each eligible series;
8. forecasts the next 12 months and saves one tidy CSV.

The script always includes Naive as the cross-validation benchmark. Series with fewer than 48 prepared monthly observations skip cross-validation as required by the assignment. They use Simple Exponential Smoothing by rule, and their chosen-model and Naive accuracy values are reported as unavailable rather than as zero.

## Project structure

```text
.
|-- forecast.py                    Main forecasting script
|-- proj1_exampleinput.csv         Original project input
|-- proj1_exampleinput_with_zeros.csv
|                                  Two explicit zero-sales observations for testing
|-- tests/test_forecast.py         Validation and scoring tests
|-- requirements.txt               Runtime dependencies
|-- requirements-dev.txt           Test and lint dependencies
|-- pyproject.toml                 Project and tool configuration
`-- outputs/                       Generated forecasts, ignored by Git
```

## Setup

Python 3.11-3.13 is supported. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Run the forecast

The assignment variables are at the top of `forecast.py`:

```python
file_name = "proj1_exampleinput.csv"
ids_to_forecast = ["C2208", "C1030", "C1036", "C1126", "C1042"]
metric = "mape"
```

Run with those values:

```powershell
python forecast.py
```

Or override them without editing the file:

```powershell
python forecast.py --file proj1_exampleinput.csv --ids C1096 C3029 C6810 --metric mse
```

Forecasts are written to `outputs/forecasts_<ids>.csv`. The console prints the required series name, category, chosen model, metric, chosen-model score, and Naive benchmark.

## Model-selection logic

Seasonality is considered high when STL seasonal strength is at least 0.60. The formula is:

```text
max(0, 1 - variance(residual) / variance(residual + seasonal))
```

| Category | Rule | Candidate models |
| --- | --- | --- |
| 1 | Fewer than 24 monthly observations | Simple Exponential Smoothing; Naive benchmark |
| 2 | Zero sales and high seasonality | Seasonal Naive, additive AutoETS, additive Holt-Winters, Historic Average; Naive benchmark |
| 3 | High seasonality and strictly positive sales | Seasonal Naive, AutoETS, multiplicative Holt-Winters, AutoARIMA; Naive benchmark |
| 4 | Zero sales and low or no seasonality | 12-month Window Average, Random Walk with Drift, Holt, Historic Average; Naive benchmark |
| 5 | All remaining series | AutoETS, AutoARIMA, Holt, Historic Average; Naive benchmark |

Cross-validation uses a 12-month horizon, advances every 6 months, and uses three origins. This covers the assignment's two-year evaluation period: the earliest and latest forecast origins are 12 months apart, and their forecast horizons span 24 months in total.

## Data assumptions

- The input columns must be `product_class`, `Month`, and `sales_volume`.
- `Month` must be the first day of a month.
- Product-month pairs must be unique.
- Sales values may be fractional but cannot be negative.
- A missing month inside the observed life of a product class means zero sales. This assumption makes irregular series monthly and prevents missing sales periods from being mistaken for uninterrupted positive demand.
- MAPE is undefined when actual sales are zero. The script emits a warning and excludes zero-actual cross-validation rows from MAPE. MSE includes all observations.

## Verification

```powershell
python -m ruff check .
python -m pytest
python forecast.py --ids C2208 C1030 C1036 C1126 C1042 --metric mape
```

The final command deliberately includes `C2208`, a short series, to verify that the no-cross-validation path still produces a forecast.

## Historical artifacts

Course documents, reports, earlier notebook iterations, prompt notes, generated forecast variants, copied library code, and Python cache files are intentionally excluded from the repository. They are not required to run the final project.
