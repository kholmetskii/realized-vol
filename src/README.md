# Source layout

The package is organized around forecasting, data preparation, and presentation.
Shared contracts remain in `rvol.domain`. Command coordination lives in
`rvol.cli`; numerical implementations do not depend on the command line.

```text
rvol/
├── domain/                       Shared protocols, settings and immutable results
├── forecasting/
│   ├── execution/
│   │   ├── preparation.py        Align features and future targets by origin
│   │   ├── walk_forward.py       Expanding or rolling chronological execution
│   │   ├── holdout.py            Fit once at a fixed historical cutoff
│   │   └── _data.py              Internal validation and array preparation
│   ├── composition/
│   │   ├── forecasting.py        ExperimentDefinition, ExperimentRun and defaults
│   │   └── configuration.py      Resolve registered components from settings
│   ├── models/                   HAR, AR(1), EWMA, naive and historical mean
│   ├── evaluation/               Losses, comparison tests and diagnostic checks
│   ├── features.py               Past-only HAR feature builder
│   └── targets.py                Session and cumulative variance target builder
├── data/
│   ├── dukascopy.py              Download and decode tick observations
│   ├── dataset.py                Build compact daily variance datasets
│   ├── coverage.py               Audit session coverage and data quality
│   ├── realized.py               Adapt timestamped prices to numerical estimators
│   └── market/                   FX sessions, sampling and daily-series validation
├── estimators/                   Variance formulas on arrays
├── diagnostics/                  Tick-data noise and volatility signatures
├── simulation/                   Synthetic price paths, noise and Monte Carlo
├── presentation/
│   ├── plotting/                 Figures built from supplied research results
│   └── reporting/                CSV, JSON and Markdown artifact writers
├── infrastructure/               Parquet and TOML input/output adapters
└── cli/                          Configure and coordinate the experiment command
```

## Working with an experiment

```python
from rvol.domain import ExperimentConfig, WalkForwardSettings
from rvol.forecasting.composition import standard_experiment_definition

definition = standard_experiment_definition()
config = ExperimentConfig(execution=WalkForwardSettings(min_train_size=252))
```

The standard definition selects all five built-in models. Use a saved
configuration or construct a definition to select different components.
Given a daily variance DataFrame:

```python
experiment = definition.run(daily, config)
evaluation = definition.evaluator().evaluate(experiment)
robustness = definition.robustness_analyzer().analyze(
    experiment, evaluation=evaluation,
)
```

Other common imports:

```python
from rvol.data.dataset import build_daily_dataset
from rvol.data.coverage import audit_daily_dataset
from rvol.forecasting.execution import HoldoutExperiment, WalkForwardExperiment
from rvol.forecasting.features import HARFeatureBuilder
from rvol.forecasting.models import HARForecaster
from rvol.forecasting.targets import RealizedVarianceTargetBuilder
from rvol.presentation.plotting.forecasting import plot_forecast_evaluation
from rvol.presentation.reporting import ForecastArtifactWriter
```

## Dependency boundaries

- `domain` imports only its own modules and general-purpose dependencies.
- Forecast execution, models and evaluation depend on `domain`, without importing
  each other. Composition selects and connects their implementations.
- Forecast feature and target builders use `data.market` to validate the observed
  daily timeline. Their outputs are aligned separately by execution preparation.
- Data preparation uses market rules and pure estimators. It does not import
  forecasting models or experiment configuration.
- Tick-data diagnostics use prepared market observations. Simulation and pure
  estimators remain independent of forecasting and presentation.
- Plotting and artifact writers consume results; they do not fit models.
- CLI modules connect composition, persistence and presentation.

The architecture tests check nested boundaries, including `forecasting.models`,
`forecasting.execution`, and `presentation.reporting`, so folder grouping does
not weaken the dependency rules.

Python import paths changed in this reorganization. Saved TOML component names,
forecasting behavior, and the commands under `scripts/` remain unchanged.
Artifact metadata records the new implementation module paths; numeric results
and output schemas retain their existing meaning.
