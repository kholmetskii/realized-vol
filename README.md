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
| SHAR | 0.151 | −1.0% | 0.96 | 48% of days |
| **HAR** | **0.152** | — | — | — |
| HAR-ridge | 0.164 | 6.7% | 7×10⁻⁴ | 64% of days |
| AR(1) | 0.182 | 16.3% | 2×10⁻⁵ | 61% of days |
| EWMA | 0.184 | 17.2% | 8×10⁻⁴ | 63% of days |
| naive | 0.192 | 20.6% | 2×10⁻⁴ | 51% of days |
| historical mean | 0.326 | 53.2% | 2×10⁻¹¹ | 76% of days |

HAR, the model every other is tested against, beats five of the six
benchmarks at the 1% level. Against the naive forecast it wins only half the
days, but avoids the large misses.

SHAR's mean loss is 1% below HAR's. The reverse test, whether SHAR is better,
gives p between 0.02 and 0.06 depending on the HAC lag, and nearly all of
SHAR's edge comes from ten sessions: suggestive, not decisive. Its slopes say
that variance from rising EUR/USD prices predicts more than variance from
falling ones, at every quarterly refit. HAR-ridge loses because its penalty,
chosen on the October–December 2024 forecasts, was the largest in the grid;
small penalties match HAR.

## How it works

1. **Data.** Download Dukascopy ticks and build one RV per FX session (17:00
   New York close) from 5-minute returns, split into the semivariances of
   rising and falling returns.
2. **Forecasts.** Every model predicts next-session log RV and is refitted each
   session on past data only.
3. **Evaluation.** Log forecasts are converted back to variance with a Duan
   smearing correction, then scored with QLIKE, which stays reliable when RV is
   a noisy measure of true variance (Patton, 2011). Each model is tested against
   HAR with a one-sided Diebold–Mariano test.

Models: two forecasting rules and their special cases. Exponential smoothing
with fixed weights gives the naive forecast (decay 0) and EWMA (decay 0.94);
least-squares linear regression gives the historical mean (no regressors),
AR(1), HAR (Corsi, 2009) and SHAR, which splits HAR's daily term into the two
log semivariances (Patton & Sheppard, 2015). HAR-ridge is HAR with a ridge
penalty chosen on the calibration window that precedes the evaluation.

## Usage

Download ticks, one quarter per call:

```sh
python -m rvol.data.sources.dukascopy EURUSD 2023-01-01 2023-03-31
```

Run the comparison:

```python
from rvol import experiment
from rvol.data.layout import DataLayout
from rvol.data.volatility import build_daily_dataset

daily = build_daily_dataset(sorted(DataLayout().ticks("EURUSD").glob("*.parquet")))
prepared = experiment.prepare(daily)
forecasts = experiment.forecast(prepared, experiment.study_models(prepared), asset="EURUSD")
print(experiment.results_table(experiment.compare(forecasts)))
```

The study design (forecast and evaluation dates, the minimum training sample,
the candidate model, the ridge grid) is a set of named constants at the top of
`rvol/experiment.py`; every function uses them as defaults.

The full study, with figures and explanations, is one notebook per asset:
[`notebooks/01_eurusd.ipynb`](notebooks/01_eurusd.ipynb). Install its extras with
`pip install -e ".[notebooks]"`, then rerun it from the terminal with

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eurusd.ipynb
```

Its figures come from `notebooks/_plots.py`, and every number in its text is
printed by a code cell, so a rerun on new data leaves no stale figure. For
another pair, copy it and change `ASSET`; only the commentary is written by hand.

## Extending

| add a | implement | pass to |
|---|---|---|
| forecasting rule | a class with `fit(X, y)` returning itself and `predict(X)` | `ForecastModel(name, rule, regressors)` |
| feature | `TrailingMeanFeatures(column, windows)`, columns named `<column>_mean_<window>` | `experiment.FEATURE_TABLES` |
| model | `ForecastModel(name, rule, regressors)`, e.g. `ForecastModel("HAR-ridge-1", LinearRegression(ridge=1.0), HAR_FEATURES)` | `experiment.forecast` or `walk_forward` |
| loss | a function `loss(actual, predicted)` returning one loss per forecast | `forecast_losses` |

## Project layout

```text
src/rvol/
├── data/          download, sessions, daily RV, forecasting tables
├── estimators/    realized variance and semivariances
├── models/        forecasting models
├── walkforward/   walk-forward forecasts, smearing
├── evaluation/    losses, Diebold–Mariano test, robustness checks
└── experiment.py  the full comparison
notebooks/         one study per asset
tests/             unit and integration tests, laid out like src/rvol/
```

One public class per file (`LinearRegression` → `linear_regression.py`); each package exports
its API from `__init__.py`.

`models`, `walkforward`, `evaluation` and `data.preparation` know nothing about
volatility or markets; a test checks their imports and vocabulary.
Domain code lives in `data.sources`, `data.market`, `data.volatility` and
`estimators`, and `experiment.py` combines the two.

## Limitations

- AR(1) and the historical mean are nested in HAR, so their DM tests are
  conservative (Clark & West, 2007).
- Seven models compared pairwise invite chance findings; a model confidence set
  across assets is planned.
- The ridge penalty is tuned on about three months of forecasts, too few to
  choose it reliably.
- Each session's RV sums the returns between its own 5-minute prices, so the one
  5-minute return spanning 17:00 New York, the boundary between two sessions, is
  left out (on average 0.2% of RV).
- Sessions follow the FX convention, so exchange-traded assets need their own
  session rules.

## Development

```sh
pip install -e ".[dev]"
pytest -m "not slow"
ruff check src tests
mypy src tests
```
