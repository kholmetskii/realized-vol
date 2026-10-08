# Scripts

Run these commands from the repository root with the project environment active.
Input and output paths are relative to the current working directory.

```text
scripts/
├── run_experiment.py                 Forecast, evaluate, and save selected reports
├── data/
│   ├── build_daily_dataset.py        Build daily variance from downloaded ticks
│   └── audit_daily_dataset.py        Check daily dataset coverage and quality
├── diagnostics/
│   ├── noise_report.py               Compare fine and coarse sampling on real data
│   ├── signature_plot.py             Plot one volatility signature
│   └── diagnostics_plots.py          Plot signatures and noise diagnostics together
└── simulation/
    ├── sampling_plot.py              Illustrate sampling-grid sensitivity
    ├── returns_and_rv_plot.py        Show the path from prices to realized variance
    ├── cumulative_iv_vs_rv_plot.py   Compare simulated integrated and realized variance
    ├── bid_ask_bounce_plot.py        Illustrate transaction-price noise
    ├── daily_rv_plot.py              Plot daily variance from a multi-day simulation
    ├── heston_plot.py                Plot one Heston price and variance path
    └── simulation_plots.py           Compare estimators on clean and noisy simulations
```

The simulation scripts use generated data. The diagnostics scripts use real tick
datasets; they are separate from the optional forecast robustness reports.

Build and audit a daily dataset:

```bash
python scripts/data/build_daily_dataset.py "data/EURUSD_*.parquet" \
  --frequency 5min \
  --out data/derived/EURUSD_daily_rv_5min.parquet

python scripts/data/audit_daily_dataset.py \
  data/derived/EURUSD_daily_rv_5min.parquet
```

Run a forecast experiment with optional outputs:

```bash
python scripts/run_experiment.py \
  data/derived/EURUSD_daily_rv_5min.parquet \
  --config experiments/har.toml \
  --output-dir outputs/har \
  --plot \
  --robustness
```

Inspect tick-data noise or generate an illustrative figure:

```bash
python scripts/diagnostics/noise_report.py \
  data/EURUSD_2024-01-01_2024-03-31.parquet

python scripts/simulation/heston_plot.py --out outputs/examples/heston.png
```

Each command supports `--help`. The main project README contains the full
experiment configuration and figure-reproduction instructions.
