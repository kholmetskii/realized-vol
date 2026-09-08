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

TODO as they land: signature plot, out-of-sample loss table, MCS composition.

## Limitations

TODO: an honest list of what this work does not show.
