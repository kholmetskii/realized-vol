# Realized Volatility

Estimating realized volatility from intraday data, forecasting it, and comparing
the forecasts with proper statistical tests.

## Question

How predictable is daily volatility, and can competing models be told apart
statistically — given that the target itself is unobservable and is estimated
with error?

## Layout

    src/rvol/data/         tick data download and parsing
    src/rvol/estimators/   RV, subsampled RV, two-scale RV, realized kernel,
                           bipower variation, jump test
    src/rvol/models/       HAR, HARQ, GARCH, Markov-switching
    src/rvol/evaluation/   loss functions, Diebold-Mariano, bootstrap, MCS
    src/rvol/simulation/   generators with known true volatility
    scripts/               reproducible entry points
    tests/                 checks against synthetic data with a known answer

## Install

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

## Results

### 1. How far can you sample before noise takes over?

Sample: EURUSD tick quotes from Dukascopy, 1 January to 31 March 2024 —
5,359,908 ticks over 64 trading sessions, each running 21:00 to 21:00 UTC
(17:00 New York). Median quoted spread 0.19 bps, no crossed quotes.

Realized variance is computed per session at sampling intervals from 1 second
to 1 hour, then averaged across sessions and annualised.

![Volatility signature, mid prices](figures/signature_plot.png)

Annualised volatility is about **5.6%** at five-minute sampling, and the curve
is flat within one standard error across the whole grid: every interval from
1 second to 1 hour gives an answer consistent with every other. Sampling
EURUSD mid prices is close to unbiased even at the tick.

That flatness is not a failure to measure. Running the same estimator on the
bid series, where bid-ask bounce is not averaged away, produces the upward
slope the theory predicts:

![Volatility signature, bid prices](figures/signature_bid.png)

### 2. Pairing by session makes the small effect measurable

Comparing two averages is weak here, because a volatile week raises the
estimate at every frequency at once, and that shared variation dominates the
standard errors. Pairing — one ratio RV(1s)/RV(5min) per session, then a
one-sided t-test on the log ratios — removes it (`rvol.estimators.signature.noise_test`):

| price | RV(1s)/RV(5min) | t | p | implied noise sd |
|---|---|---|---|---|
| mid | 1.0763 | 2.49 | 0.0077 | 0.029 bps |
| bid | 1.1771 | 5.22 | 1.1e-06 | 0.049 bps |
| ask | 1.1777 | 5.31 | 7.6e-07 | 0.048 bps |

The noise standard deviation is backed out from E[RV_n] = IV + 2n*omega^2.

Three things follow.

**Mid-price noise is real but small.** Second-by-second sampling inflates
variance by 7.6%, i.e. volatility by 3.7%. The unpaired figure could not
resolve this; the effect was always there, buried under volatility swings.

**Bid and ask agree to within 2%** — 0.049 against 0.048 bps — as they should,
since neither side of the quote is special. A useful check on the pipeline.

**The mid is cleaner than averaging alone explains.** Independent noise on each
quote would give 0.049/sqrt(2) = 0.035 bps for the mid; the measured 0.029 bps
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

## Limitations

- **One instrument, one quarter.** EURUSD in Q1 2024 was quiet, at roughly 5.6%
  annualised. Nothing here shows the conclusions hold for a less liquid pair,
  a turbulent period, or an asset class with wider spreads.
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
- **No forecasting yet.** Everything so far measures volatility in-sample;
  nothing has been predicted or compared out-of-sample.
