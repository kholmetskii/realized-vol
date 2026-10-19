# EUR/USD research

This document covers dataset construction and supporting diagnostics, and
preserves the original Q1 2024 forecasting study. Current results are in the
project README.

## Build the daily dataset

    python -m rvol.data.sources.dukascopy EURUSD 2024-01-01 2024-03-31

Downloaded quotes are turned into the daily dataset and audited in
[`notebooks/01_data.ipynb`](../notebooks/01_data.ipynb).

The builder carries FX sessions across file boundaries, removes overlaps and
writes atomically. The daily table contains session dates, RV, log RV and
sample counts. The audit checks ordering, invalid values, observation counts,
annual coverage and gaps longer than seven days. RV feature construction
rejects disconnected periods by default.

The data-preparation pipeline is:

    data.market + estimators.variance
        -> estimators.realized -> data.volatility.dataset
        -> data.volatility.features + data.preparation -> ready forecast table

Numerical variance formulas contain no pandas or market-calendar conventions. Forecasting
consumes the prepared numerical timeline. Additional noise- and jump-robust
estimators are not yet implemented.

## Reproduce supporting plots

The diagnostics figures below come from
[`notebooks/01_data.ipynb`](../notebooks/01_data.ipynb) on the Q1 2024 tick file,
and the simulation figures from
[`notebooks/02_simulation.ipynb`](../notebooks/02_simulation.ipynb). Both save
their figures in `figures/`.

## Results

The archived results below are the original four-model Q1 2024 experiment.
Current results come from `notebooks/03_forecasting.ipynb`; see the project README. Evaluation
on an already inspected period is exploratory rather than a fresh confirmatory
test.

### 1. How far can you sample before noise takes over?

Sample: EURUSD tick quotes from Dukascopy, 1 January to 31 March 2024 —
5,359,908 ticks over 64 trading sessions, each ending at 17:00 New York
(22:00 UTC in winter and 21:00 UTC in summer). Median quoted spread 0.19 bps,
no crossed quotes.

Realized variance is computed per session at sampling intervals from 1 second
to 1 hour, then averaged across sessions and annualised.

![Volatility signature, mid prices](../figures/signature_plot.png)

Annualised volatility is about **5.6%** at five-minute sampling, and the curve
is flat within one standard error across the whole grid: every interval from
1 second to 1 hour gives an answer consistent with every other. Sampling
EURUSD mid prices is close to unbiased even at the tick.

That flatness is not a failure to measure. Running the same estimator on the
bid series, where bid-ask bounce is not averaged away, produces the upward
slope the theory predicts:

![Volatility signature, bid prices](../figures/signature_bid.png)

### 2. Pairing by session makes the small effect measurable

Comparing two averages is weak here, because a volatile week raises the
estimate at every frequency at once, and that shared variation dominates the
standard errors. Pairing — one ratio RV(1s)/RV(5min) per session, then a
one-sided t-test on the log ratios — removes it
(`rvol.estimators.diagnostics.noise_test`):

| price | RV(1s)/RV(5min) | t | p | implied noise sd |
|---|---|---|---|---|
| mid | 1.0780 | 2.55 | 0.0066 | 0.030 bps |
| bid | 1.1794 | 5.29 | 8.2e-07 | 0.049 bps |
| ask | 1.1811 | 5.43 | 4.8e-07 | 0.048 bps |

The noise standard deviation is backed out from E[RV_n] = IV + 2n*omega^2.

Three things follow.

**Mid-price noise is real but small.** Second-by-second sampling inflates
variance by 7.8%, i.e. volatility by 3.8%. The unpaired figure could not
resolve this; the effect was always there, buried under volatility swings.

**Bid and ask agree to within 2%** — 0.049 against 0.048 bps — as they should,
since neither side of the quote is special. A useful check on the pipeline.

**The mid is cleaner than averaging alone explains.** Independent noise on each
quote would give 0.049/sqrt(2) = 0.035 bps for the mid; the measured 0.030 bps
is lower, so the two are negatively correlated. The spread widens and narrows
around the efficient price, and those moves cancel in the mid while showing up
in each quote. Quote noise at 0.049 bps is also about half the 0.095 bps
half-spread, so consecutive quotes are positively autocorrelated rather than
alternating between the two sides.

**Practically:** the five-minute convention is far more conservative than this
instrument requires. Estimators robust to microstructure noise have little to
correct on EURUSD mid quotes, which is a statement about this market rather
than about the estimators; they are validated against simulated paths with
known answers instead.

### 3. HAR-RV improves out-of-sample forecasts

The forecasting sample contains **323 complete EUR/USD sessions** from 2
January 2023 to 29 March 2024. The experiment uses 2023 as the initial history
and freezes Q1 2024 as the evaluation period: 64 one-session-ahead forecasts
from 2 January to 29 March. After the 22-session feature warm-up, the first
forecast has 237 eligible training rows. Each later forecast uses an expanding
window and admits an outcome only after its target session has ended.

Four fixed specifications are compared: the historical mean, naïve persistence,
AR(1), and HAR-RV. No model was changed after examining the Q1 results.

| model | mean QLIKE | reduction vs naïve | mean log-RV MSE | reduction vs naïve |
|---|---:|---:|---:|---:|
| historical mean | 0.2267 | 13.3% | 0.5796 | -28.1% |
| naïve | 0.2616 | — | 0.4525 | — |
| AR(1) | 0.1801 | 31.2% | 0.4120 | 8.9% |
| **HAR-RV** | **0.1400** | **46.5%** | **0.2607** | **42.4%** |

Negative reduction means worse performance than naïve. HAR-RV has the lowest
mean loss under both metrics. Relative to each benchmark:

| HAR-RV compared with | QLIKE reduction | DM | one-sided p | log-MSE reduction | DM | one-sided p |
|---|---:|---:|---:|---:|---:|---:|
| naïve | 46.5% | 3.297 | 0.00049 | 42.4% | 3.150 | 0.00082 |
| AR(1) | 22.3% | 2.564 | 0.00517 | 36.7% | 4.071 | 2.34e-05 |
| historical mean | 38.3% | 2.555 | 0.00531 | 55.0% | 3.607 | 0.00016 |

The loss differential is benchmark loss minus HAR loss, so positive DM
statistics favour HAR. The main table uses three Newey–West lags, selected by
the sample-size rule in the evaluator. Every comparison remains significant at
5% when the lag choice is varied from zero through five.

![Out-of-sample forecast evaluation](../figures/forecast_evaluation.png)

The advantage is meaningful but not uniform. HAR beats AR(1) and the
historical mean on 48 of 64 sessions (75%), but beats naïve on only 33 sessions
(51.6%). Its large average improvement over naïve comes from avoiding several
costly misses rather than winning almost every day. HAR is best under both
losses in February and March; in January the historical mean narrowly leads
under QLIKE (0.0940 against 0.0968), while HAR still leads under log-RV MSE.

## Limitations

- **One instrument and one out-of-sample quarter.** The dataset spans 15
  months, but forecast evaluation contains only 64 Q1 2024 sessions. EUR/USD
  was quiet, at roughly 5.6% annualised during that quarter. Nothing here shows
  the conclusions hold for a turbulent regime, a less liquid pair, or another
  asset class.
- **The noise measurement is indirect.** The implied noise sd assumes the
  standard model — an efficient price plus i.i.d. noise — and inherits its
  bias if the noise is autocorrelated or dependent on the price. The numbers
  above already hint that it is.
- **1-second sampling is close to the tick.** At about 85,000 ticks a session,
  roughly one per second, sampling finer than 1 second adds almost no
  observations, so the left end of the signature curve flattens for want of
  data rather than for want of noise.
- **Sessions are dropped, not adjusted.** Any session covering under 12 hours
  is excluded, which removes the Sunday-evening opens and holidays instead of
  modelling them.
- **The statistical tests are asymptotic.** Newey–West adjustment addresses
  serial correlation in loss differences, but 64 forecasts remain a small
  sample. The results support this fixed Q1 experiment, not universal HAR
  superiority.
