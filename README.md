# Realized-volatility forecasting models

A library for comparing volatility forecasting models across assets. It builds
daily realized variance (RV) from tick quotes, forecasts it walk-forward with
competing models, and compares them with QLIKE, log-RV MSE and Diebold–Mariano
tests.

**Status:** the full pipeline works for one asset (EUR/USD, below). Running the
same comparison over several assets is the next step.

## Pipeline

```python
from datetime import date
from pathlib import Path

from rvol import experiment
from rvol.data.storage import ParquetDatasetRepository
from rvol.data.volatility import audit_daily_dataset, build_daily_dataset
from rvol.walkforward import WalkForwardPlanner, WalkForwardRunner

# 1. ticks -> one RV per FX session (17:00 New York close), from 5-minute returns
daily = build_daily_dataset(sorted(Path("data/ticks/EURUSD").glob("EURUSD_*.parquet")))
assert audit_daily_dataset(daily).passed
ParquetDatasetRepository("data/derived/EURUSD_daily_rv_5min.parquet").save(daily)

# 2. walk-forward forecasts of next-session log RV, every model refitted each session
runner = WalkForwardRunner(
    WalkForwardPlanner(
        min_train_size=252,
        forecast_start=date(2024, 10, 1),  # Oct–Dec 2024 only calibrates smearing
        forecast_end=date(2026, 9, 30),
    )
)
forecasts = experiment.forecast(
    experiment.prepare(daily), experiment.benchmark_models(), runner, asset="EURUSD"
)

# 3. QLIKE on smeared variance forecasts, MSE on log forecasts, DM tests vs HAR
comparison = experiment.compare(forecasts, evaluation_start=date(2025, 1, 1))
experiment.results_table(comparison)
```

Tick quotes are downloaded beforehand, one quarter per call:

```sh
python -m rvol.data.sources.dukascopy EURUSD 2023-01-01 2023-03-31
```

## Example: EUR/USD

453 out-of-sample sessions, 2 January 2025 – 30 September 2026, produced by the
code above. Lower loss is better.

| model | QLIKE | HAR lower by | DM p-value | log-RV MSE | DM p-value | HAR wins (QLIKE) |
|---|---|---|---|---|---|---|
| **HAR** | **0.152** | — | — | **0.265** | — | — |
| AR(1) | 0.182 | 16.3% | 2×10⁻⁵ | 0.310 | 1×10⁻⁴ | 61% of days |
| EWMA | 0.184 | 17.2% | 8×10⁻⁴ | 0.305 | 0.009 | 63% of days |
| naive (today = tomorrow) | 0.192 | 20.6% | 2×10⁻⁴ | 0.359 | 6×10⁻⁶ | 51% of days |
| historical mean | 0.326 | 53.2% | 2×10⁻¹¹ | 0.541 | 1×10⁻⁷ | 76% of days |

HAR is best under both metrics and beats every benchmark at the 1% level. Its
advantage over the naive forecast comes from avoiding large misses: it wins only
half of the sessions but has a 21% lower mean QLIKE.

## Models and evaluation

All models forecast next-session log RV:

- **historical mean** of past log RV;
- **naive**: tomorrow equals today;
- **EWMA** of past log RV (decay 0.94);
- **AR(1)** on today's log RV;
- **HAR** (Corsi, 2009): regression on today's log RV and its 5- and
  22-session means.

Evaluation:

- **QLIKE is the headline metric.** RV is a noisy estimate of the true variance.
  QLIKE still ranks forecasts correctly under that noise, while MSE on
  logarithms does not (Patton, 2011).
- **Back to variance.** exp(log forecast) is the median of RV, not its mean, so
  it forecasts too low. Each variance forecast is multiplied by a Duan smearing
  factor estimated from the model's earlier errors observable at the forecast
  date.
- **Tests.** One-sided Diebold–Mariano tests of each benchmark against the
  candidate, with a Newey–West variance. Robustness tables cover HAC lag
  choice, win rates, monthly losses and the largest errors.

Limitations:

- AR(1) is nested in HAR, so its DM test is approximate and conservative
  (Clark & West, 2007).
- Models are fitted by least squares on logarithms but scored by QLIKE on
  variance.
- Sessions follow the FX convention (17:00 New York close), which suits
  currencies and metals but not exchange-traded assets with trading hours.

## Data layout

Data files live under `data/`, one folder per stage (not in git):

- `raw/dukascopy/<SYMBOL>/`: downloaded hourly tick files, a cache that is never
  edited;
- `ticks/<SYMBOL>/`: decoded bid/ask quotes, one Parquet file per quarter;
- `derived/`: daily RV datasets.

## Project layout

```text
src/rvol/
├── data/                    acquire, persist and prepare observations
│   ├── sources/dukascopy/   tick download client, raw-hour cache, decoding, CLI
│   ├── market/              FX sessions and timestamp sampling
│   ├── volatility/          daily RV dataset, coverage audit, log-RV features
│   ├── preparation/         target construction and feature/target alignment
│   └── storage/             atomic Parquet storage
├── estimators/              realized variance on price arrays and per session
├── models/                  historical mean, naive, EWMA, AR(1), HAR, OLS
├── walkforward/             walk-forward planning and execution
├── evaluation/              losses, metrics, pairwise tests (DM)
│   ├── robustness/          diagnostics: HAC lags, win rates, monthly losses, errors
│   └── volatility/          QLIKE, log-RV MSE and variance conversion
└── experiment.py            prepare, forecast, compare, results table
tests/                       mirrors src/rvol/, plus integration/
```

Dependencies point one way: `models` ← `walkforward` ← `evaluation` ←
`experiment`. Data acquisition and preparation are independent of the
forecasting packages, and the daily dataset builds on `estimators`.
`tests/integration/test_architecture.py` enforces these directions.

Code follows Java-style layout rules:

- one public class per module, named after the class in snake_case
  (`HARModel` → `har_model.py`); free functions live next to the class they
  serve, or in a module named after their topic (`smearing.py`);
- each package re-exports its public API in `__init__.py`, and other packages
  import only from there (`from rvol.models import HARModel`); `data` and
  `data.sources` only group subpackages;
- classes name the interface they implement (`class HARModel(ForecastModel)`),
  like Java's `implements`;
- modules starting with `_` are internal to their package;
- tests mirror the source tree: `models/har_model.py` → `tests/models/test_har_model.py`.

- **Models** receive numerical arrays only; they never choose columns, build
  targets or transform units. A configured model fits arrays and returns a
  fitted predictor.
- **Walk-forward is split in two.** `WalkForwardPlanner` decides when to refit
  and which rows are observable. `WalkForwardRunner` executes that plan.
- **Evaluation** consumes forecast tables; it never sees models or market data.

## Using the library

### Walk-forward on a prepared table

A prepared table has one row per forecast origin:

| column | meaning |
|---|---|
| origin_date | when the inputs are known |
| target_date | when the outcome is observed |
| target | the outcome, in the caller's scale |
| other columns | numerical inputs, selected by name |

Origins and targets must be unique, every origin must precede its target, and
inputs must use only information available at the origin.

```python
from rvol.evaluation import MSEMetric, evaluate
from rvol.models import HARModel
from rvol.walkforward import WalkForwardPlanner, WalkForwardRunner

runner = WalkForwardRunner(
    WalkForwardPlanner(window="expanding", min_train_size=252, refit_every=1)
)
forecasts = runner.run(
    prepared,
    model=HARModel(),
    feature_columns=("log_rv_daily", "log_rv_weekly", "log_rv_monthly"),
)
evaluation = evaluate(forecasts, metrics=(MSEMetric(),))
```

Planner settings:

- `window="rolling"` with `window_size` uses a rolling window instead of an
  expanding one.
- `forecast_start` and `forecast_end` are inclusive target-date bounds. They
  keep earlier history for training and keep the refit schedule.
- Only labels with `target_date <= origin_date` enter training.

`runner.plan(prepared).execution` returns the settings and the actual refit
anchor, so you can record exactly how forecasts were produced.

Forecast tables have these columns: model, origin_date, target_date, actual,
predicted, n_train, fit_date, train_start_date, train_end_date and asset.
`evaluate` requires one asset and identical dates and actual values across the
models it compares.

### Adding a model

A model needs a `name` and `fit(X, y)` returning an object with `predict(X)`. No
base class or registration is required:

```python
import numpy as np

from rvol.models import ConstantPredictor


class MedianModel:
    name = "median"

    def fit(self, features, target):
        return ConstantPredictor(float(np.median(target)), n_features=features.shape[1])
```

A predictor may also implement `update(X, y)`, returning new state after newly
observable outcomes between refits; EWMA uses this.

### Adding a metric, test or diagnostic

Evaluation is extended the same way, by a class with a name and one method:

| kind | method | used by |
|---|---|---|
| metric | `losses(actual, predicted)` → one non-negative loss per forecast | `evaluate(metrics=...)` |
| pairwise test | `compare(baseline_loss, candidate_loss)` → `ComparisonOutcome` | `evaluate(comparison_tests=...)` |
| diagnostic | `run(forecasts, evaluation)` → `DataFrame` | `analyze_robustness(diagnostics=...)` |

`experiment.compare` passes the same choices through (`log_metrics`,
`variance_metrics`, `comparison_tests`, `diagnostics`); its defaults are
log-RV MSE, QLIKE, the DM test and `standard_diagnostics(candidate)`.

```python
from rvol.evaluation import DieboldMarianoTest, MAEMetric, evaluate
from rvol.evaluation.robustness import HacSensitivity, WinRates, analyze_robustness

evaluation = evaluate(
    forecasts,
    metrics=(MAEMetric(),),
    comparison_pairs=[("EWMA", "HAR")],
    comparison_tests=(DieboldMarianoTest(), MyOtherTest()),
)
tables = analyze_robustness(
    forecasts, evaluation, (WinRates("HAR"), HacSensitivity("HAR", lags=range(10)))
)
```

## Install and check

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not slow"
ruff check src tests
mypy src tests
```
