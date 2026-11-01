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

Run a comparison from the `notebooks` folder:

```python
from design import DESIGN, HAR, LAYOUT, LOG_RV, LOG_RV_MEANS
from rvol.data.preparation import build_forecast_data
from rvol.data.volatility import daily_dataset
from rvol.evaluation import results_table
from rvol.models import ExponentialSmoothing, ForecastModel

naive = ForecastModel("naive", ExponentialSmoothing(decay=0.0))
data = build_forecast_data(daily_dataset("EURUSD", LAYOUT), [LOG_RV_MEANS], value_col=LOG_RV)
run = DESIGN.run(data, [HAR, naive], series="EURUSD")
print(results_table(run.evaluation))
```

The study design (forecast and evaluation dates, the minimum training sample,
the candidate model, the loss) is a `StudyDesign` in `notebooks/design.py`,
next to the assets and HAR, the model every other is tested against. Each
notebook defines the other models its question is about.

The study, with figures and explanations, is one notebook per research
question; each covers every asset in `ASSETS`:

| notebook | question |
|---|---|
| [`01_data`](notebooks/01_data.ipynb) | Is our volatility measure sound? |
| [`02_har`](notebooks/02_har.ipynb) | Does HAR beat simple benchmarks? Also explains the methods. |
| [`03_asymmetry`](notebooks/03_asymmetry.ipynb) | Do rises and falls predict volatility differently? |
| [`04_shrinkage`](notebooks/04_shrinkage.ipynb) | Does a ridge penalty help HAR? |

Install the extras with `pip install -e ".[notebooks]"`, then rerun them all
from the terminal with

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Each notebook states its question, defines its models, and gives its answer
per asset before the evidence. The notebooks share only `notebooks/design.py`
(the study's choices) and `notebooks/_display.py` (figures and tables, one
panel per asset); every computation they share is library code with tests, and
code only one notebook needs sits in a collapsed helpers cell inside it. Every number in the text is printed by a code cell, so a rerun on new
data or more assets leaves nothing stale.

## Extending

| add a | implement | pass to |
|---|---|---|
| forecasting rule | a class with `fit(X, y)` returning itself and `predict(X)` | `ForecastModel(name, rule, regressors)` |
| feature | `TrailingMeanFeatures(column, windows)`, columns named `<column>_mean_<window>` | `build_forecast_data(data, [...], value_col=...)` |
| model | `ForecastModel(name, rule, regressors)`, e.g. `ForecastModel("HAR-ridge-1", LinearRegression(ridge=1.0), HAR_FEATURES)` | `StudyDesign.run` or `walk_forward` |
| loss | a function `loss(actual, predicted)` returning one loss per forecast | `forecast_losses` |

## Project layout

```text
src/rvol/
├── data/          download, sessions, daily RV, forecasting tables
├── estimators/    realized variance and semivariances
├── models/        forecasting models
├── walkforward/   walk-forward forecasts, smearing
└── evaluation/    losses, Diebold–Mariano test, robustness checks
notebooks/
├── design.py      the study's choices: assets, dates, loss, HAR
├── _display.py    figures and tables, one panel per asset
└── 0*.ipynb       one notebook per research question
tests/             unit and integration tests of the library, laid out like src/rvol/
```

One public class per file (`LinearRegression` → `linear_regression.py`); each package exports
its API from `__init__.py`.

`models`, `walkforward`, `evaluation` and `data.preparation` know nothing about
volatility or markets; a test checks their imports and vocabulary.
Domain code lives in `data.sources`, `data.market`, `data.volatility` and
`estimators`. The study is the application built on the library: its choices
live in `notebooks/design.py` and its models in the notebooks, which combine the
two kinds of library code (`04_shrinkage` stops if its tuning used evaluation
data).

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
ruff check src tests notebooks
mypy src tests
```
