# Realized-volatility forecasting

Compares volatility forecasting models. It builds daily realized variance (RV)
from tick quotes, forecasts it walk-forward with several models, and scores them
with QLIKE and Diebold–Mariano tests.

**Status:** works end to end for EUR/USD. Next step: the same comparison across
several assets.

## Results: EUR/USD

453 out-of-sample sessions, January 2025 – September 2026. Lower loss is better.

| model | QLIKE | HAR lower by | DM p-value | HAR wins |
|---|---|---|---|---|
| **HAR** | **0.152** | — | — | — |
| AR(1) | 0.182 | 16.3% | 2×10⁻⁵ | 61% of days |
| EWMA | 0.184 | 17.2% | 8×10⁻⁴ | 63% of days |
| naive | 0.192 | 20.6% | 2×10⁻⁴ | 51% of days |
| historical mean | 0.326 | 53.2% | 2×10⁻¹¹ | 76% of days |

HAR beats every benchmark at the 1% level. Against the naive forecast it wins
only half the days, but avoids the large misses.

## How it works

1. **Data.** Download Dukascopy ticks and build one RV per FX session (17:00
   New York close) from 5-minute returns.
2. **Forecasts.** Every model predicts next-session log RV and is refitted each
   session on past data only.
3. **Evaluation.** Log forecasts are converted back to variance with a Duan
   smearing correction, then scored with QLIKE, which stays reliable when RV is
   a noisy measure of true variance (Patton, 2011). Each model is tested against
   HAR with a one-sided Diebold–Mariano test.

Models: historical mean, naive, EWMA, AR(1), HAR (Corsi, 2009).

## Usage

Download ticks, one quarter per call:

```sh
python -m rvol.data.sources.dukascopy EURUSD 2023-01-01 2023-03-31
```

Run the comparison:

```python
from datetime import date
from pathlib import Path

from rvol import experiment
from rvol.data.volatility import build_daily_dataset
from rvol.walkforward import WalkForwardPlanner, WalkForwardRunner

daily = build_daily_dataset(sorted(Path("data/ticks/EURUSD").glob("*.parquet")))
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
comparison = experiment.compare(forecasts, evaluation_start=date(2025, 1, 1))
print(experiment.results_table(comparison))
```

The full study, with figures and explanations, is one notebook per asset:
[`notebooks/01_eurusd.ipynb`](notebooks/01_eurusd.ipynb). Install its extras with
`pip install -e ".[notebooks]"`.

## Extending

Each piece is a small class; no registration is needed.

| add a | implement | pass to |
|---|---|---|
| model | `name`, `fit(X, y)` → predictor with `predict(X)` | `experiment.forecast` |
| metric | `name`, `losses(actual, predicted)` | `evaluate(metrics=...)` |
| test | `name`, `compare(baseline_loss, candidate_loss)` | `evaluate(comparison_tests=...)` |

## Project layout

```text
src/rvol/
├── data/          download, sessions, daily RV, storage
├── estimators/    realized variance
├── models/        forecasting models
├── walkforward/   walk-forward fitting and forecasting
├── evaluation/    metrics, tests, robustness checks
└── experiment.py  the full comparison
notebooks/         one study per asset
tests/             mirrors src/rvol/
```

One public class per file (`HARModel` → `har_model.py`); each package exports
its API from `__init__.py`.

`models`, `walkforward`, `evaluation`, `data.preparation` and `data.storage` know
nothing about volatility or markets; a test checks their imports and vocabulary.
Domain code lives in `data.sources`, `data.market`, `data.volatility` and
`estimators`, and `experiment.py` combines the two.

## Limitations

- AR(1) is nested in HAR, so its DM test is conservative (Clark & West, 2007).
- Sessions follow the FX convention, so exchange-traded assets need their own
  session rules.

## Development

```sh
pip install -e ".[dev]"
pytest -m "not slow"
ruff check src tests
mypy src tests
```
