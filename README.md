# Realized Volatility

[![CI](https://github.com/kholmetskii/realized-vol/actions/workflows/ci.yml/badge.svg)](https://github.com/kholmetskii/realized-vol/actions/workflows/ci.yml)

Estimating realized volatility from intraday data, forecasting it, and comparing
the forecasts with proper statistical tests.

## Question

How predictable is daily volatility, and can competing models be told apart
statistically — given that the target itself is unobservable and is estimated
with error?

## Architecture

    src/rvol/domain/          settings, contracts and immutable results
    src/rvol/data/            ticks, market sessions and daily datasets
    src/rvol/estimators/      variance formulas on arrays
    src/rvol/forecasting/
        features.py          past-only inputs
        targets.py           future variance outcomes
        models/              fit and predict
        execution/           chronological training and forecasting
        evaluation/          losses, comparisons and robustness
        composition/         Experiment and standard components
    src/rvol/presentation/   plots and saved reports
    src/rvol/infrastructure/ Parquet input/output
    src/rvol/cli/            command coordination
    src/rvol/diagnostics/    tick-data noise and volatility signatures
    src/rvol/simulation/     price paths, noise and Monte Carlo checks

Models own fitting, execution owns observation availability and refitting, and
evaluation consumes the saved forecasts. Composition connects these steps
through one `Experiment` object. Tests enforce these boundaries, including
between nested packages. Shared contracts and results stay in `domain`;
numerical code stays independent of files, command arguments and reports.

Use functions for stateless calculations, dataclasses for settings and records,
and classes for configured behavior or fitted model state. Protocols describe
interchangeable components; abstract base classes are not required.

Additional noise- and jump-robust estimators are not yet implemented.

Installed environments expose `rvol-run-experiment`. The equivalent
repository entry point is `python scripts/run_experiment.py`. It runs the
experiment once and saves numerical results, with optional plots and robustness
reports generated from the same forecasts and losses.

Supporting commands are grouped by purpose under `scripts/data`,
`scripts/diagnostics`, and `scripts/simulation`. Run commands from the repository
root with the project environment active so input and output paths resolve
consistently. Each command supports `--help`.

The independent data-preparation pipeline is:

    data.market + estimators ──> data.realized ──> data.dataset

`estimators` contains no pandas, timestamps, FX conventions, or reporting.
`data.market` contains no variance formulas. `data.realized` combines the
timestamp rules with estimators; `data.dataset` assembles the daily observations
consumed by forecasting features and targets. Forecast models implement the
domain `ForecastModel` protocol; persistence stays outside the numerical code.

To add a model, implement `name`, `feature_names`, `specification`, and `fit`,
then include the object in `Experiment.models`. Execution and evaluation use
the common interface without model-specific branches.

All models share one array-based interface: `fit(features, target)` returns a
`Predictor` with `predict(features)`. The `ForecastModel` contract describes model
configuration and fitting; `Predictor` describes the fitted object's forecasting
interface. Features are two-dimensional arrays
with columns in `model.feature_names` order; training targets are
one-dimensional arrays of observed log realised variance. Both operations
leave their inputs unchanged. Predictions are one-dimensional arrays with one
finite log variance per row, representing finite positive variances after
exponentiation.

Convert DataFrames to arrays at the call site:

    from rvol.forecasting.models import HARModel

    model = HARModel()
    columns = list(model.feature_names)
    predictor = model.fit(
        training.loc[:, columns].to_numpy(dtype="float64"),
        training["target"].to_numpy(dtype="float64"),
    )
    predicted_log_rv = predictor.predict(current.loc[:, columns].to_numpy(dtype="float64"))

Fitted coefficients are available as `predictor.intercept` and
`predictor.coefficients`, with slopes in `model.feature_names` order.

`Experiment` holds the configuration, feature and target builders, models, and
evaluation choices. `standard_experiment(config)` selects the five built-in
models. Load data, run once, evaluate, and export explicitly:

    from pathlib import Path
    from rvol.domain import ExperimentConfig, WalkForwardSettings
    from rvol.forecasting.composition import standard_experiment
    from rvol.infrastructure import ParquetDatasetRepository
    from rvol.presentation.reporting import DatasetSnapshot, ForecastArtifactWriter

    source = Path("data/derived/EURUSD_daily_rv_5min.parquet")
    daily = ParquetDatasetRepository(source).load()
    config = ExperimentConfig(execution=WalkForwardSettings(min_train_size=252))
    experiment = standard_experiment(config)
    result = experiment.run(daily)
    evaluation = experiment.evaluator().evaluate(result)
    paths = ForecastArtifactWriter("outputs/experiment").write(
        result, evaluation, config=experiment.config,
        dataset=DatasetSnapshot.from_frame(source, daily),
        specification=experiment.specification,
        configuration=experiment.resolved_configuration,
    )

`ExperimentConfig.execution` accepts `WalkForwardSettings` or `HoldoutSettings`;
the experiment selects the matching execution method directly. Each call to
`run(daily)` creates fresh fitted state and returns an `ExperimentResult`.
Use `dataclasses.replace(experiment, config=new_config)` to reuse the selected
components with different dates or execution settings. Scientific settings are
ordinary Python objects, and saved outputs retain the same schemas and metadata.

Editable experiments live in [`experiments/eurusd.py`](experiments/eurusd.py)
and [`experiments/holdout.py`](experiments/holdout.py). Each `build_experiment()`
returns fresh configured components. The EUR/USD baseline uses expanding history,
daily refitting, at least 252 training observations, and reports target dates
from 1 January 2025 through 30 September 2026. Run it from the repository root:

    python -m experiments.eurusd \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --output-dir outputs/eurusd \
        --plot \
        --robustness

Numerical results are always saved: `forecasts.csv`, `loss_summary.csv`,
`comparisons.csv`, and `experiment.json`. `--plot` adds `forecasts.png` and
`forecasts.png.json`. `--robustness` adds the selected diagnostic CSV files,
`robustness.md`, and `robustness.json` under the output directory's `robustness/`
subdirectory. The default output directory is `outputs/experiment`; choose a
different directory for each experiment whose results you want to retain.

Select one model from the same Python experiment:

    python -m experiments.eurusd \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --models EWMA \
        --run-id ewma_only \
        --output-dir outputs/ewma \
        --plot \
        --robustness

Choose `naive`, `AR1`, `historical_mean`, or `HAR` instead to run another model.
A single-model run automatically uses that model as candidate, has no pairwise
comparisons, and retains descriptive robustness checks such as monthly losses
and largest errors. To run HAR:

    python -m experiments.eurusd \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --models HAR \
        --run-id har_only \
        --output-dir outputs/har \
        --plot

The command line overrides run labels, reporting dates and model selection.
It selects the existing configured model objects, preserving their parameters.
For multiple models, the experiment's candidate must remain selected or be
replaced with `--candidate NAME`. `--baselines NAME ...` selects comparison
baselines; `--baselines` with no names disables those comparisons. Unknown or
duplicate models, invalid references and reversed reporting dates are rejected
before loading the dataset.

Edit research settings directly in a Python experiment. For example, change
EWMA's decay while retaining the remaining selected models:

    from dataclasses import replace
    from experiments.eurusd import build_experiment
    from rvol.forecasting.models import EWMAModel

    experiment = build_experiment()
    experiment = replace(
        experiment,
        models=tuple(
            EWMAModel(decay=0.90) if model.name == "EWMA" else model
            for model in experiment.models
        ),
    )

The same object exposes feature and target builders, metrics, comparison pairs,
the statistical test, diagnostic checks and plot metric. For example,
`HARFeatureBuilder(weekly_window=5, monthly_window=22, max_gap_days=7)` configures
past-only inputs; `max_gap_days=None` disables its gap limit. Feature and target
builders each accept their own `date_col` and `rv_col`, so renamed columns or
alternative label series remain explicit. Metrics are objects such as
`QLikeMetric()` and `LogMSEMetric()`; comparison settings belong on
`DieboldMarianoTest(hac_lags=..., min_hac_lags=...)`.

Use the repository experiment from Python without a separate parsing layer:

    from experiments.eurusd import build_experiment

    experiment = build_experiment()
    result = experiment.run(daily)
    evaluation = experiment.evaluator().evaluate(result)
    robustness = experiment.robustness_analyzer().analyze(result, evaluation=evaluation)
    settings = experiment.resolved_configuration

To use the command's loading and export workflow from a new Python experiment,
call `main(experiment)` under `if __name__ == "__main__":`. The installed
`rvol-run-experiment` command and `scripts/run_experiment.py` use the standard
components with default execution settings; the repository modules above bind
the EUR/USD settings explicitly.

`Experiment.resolved_configuration` snapshots actual component implementations,
parameters, execution policies and workflow choices, including CLI overrides.
The command stores it in `experiment.json` and `robustness.json`; forecast
figures receive a `<image filename>.json` sidecar with settings, dataset
fingerprint and coverage. Direct artifact-writer calls can pass
`configuration=experiment.resolved_configuration` to retain the same snapshot.

`ExperimentConfig.instrument` and `--instrument` are explicit labels; no
instrument is inferred from filenames. The label appears in metadata,
evaluation, robustness, reports and plot labels. Without it, forecast titles
are generic.

The target is selected separately from execution settings:

    from rvol.domain import TargetHorizon

    experiment = standard_experiment(
        config,
        target_horizon=TargetHorizon(sessions=5, kind="cumulative"),
    )
    result = experiment.run(daily)

For a forecast made after session `t`, the two target kinds mean:

| Target | Variance being forecast | Log target used in fitting |
|---|---|---|
| `TargetHorizon(h, "session")` | `RV[t+h]` | `log(RV[t+h])` |
| `TargetHorizon(h, "cumulative")` | `RV[t+1] + ... + RV[t+h]` | `log(RV[t+1] + ... + RV[t+h])` |

The default is the existing next-session target, `TargetHorizon(1, "session")`.
Horizons count observed sessions, including across weekends and holidays.
Cumulative targets sum variance, rather than averaging it or adding log values.
Their features still contain only information available at the origin. Training
and observation updates wait until the **final** target session is known;
partially observed intervals cannot enter either operation. Trailing rows
without a complete target are excluded. Reporting bounds refer to that final
session, rather than the origin or interval start.

Historical mean, EWMA, AR1 and HAR fit the selected target directly. The standard
naive benchmark uses today's variance for a future-session target, and
`h × today's variance` for a cumulative target; its `variance_multiplier` is
recorded in the model specification. HAR's smearing correction uses residuals
of the selected log target. Models retain their common fit/predict interfaces.

Prepared DataFrames carry `horizon_sessions`, `target_kind`, and
`target_start_date`. For a session target the start and end are the same;
for a cumulative target the start is the next observed session. Legacy inputs
without horizon columns retain the one-session interpretation. Custom builders
must supply consistent target metadata, including the `target_horizon` DataFrame
attribute for empty frames. Results preserve the target even when no forecasts
can be produced.

Evaluation keeps the default naive-versus-HAR comparison. Use
`replace(experiment, comparison_pairs=...)` to select other ordered pairs.
Plots and robustness use the configured candidate and baseline list, with the
HAC lags and top-error count configured on the diagnostic objects.

Direct `ForecastEvaluator` calls calculate loss summaries without implicit
model comparisons; pass `comparison_pairs` explicitly when needed. Direct
`plot_forecast_evaluation` calls require their candidate explicitly. Comparison
diagnostics require a candidate and baselines; descriptive checks can run without
them. These numerical and plotting
components do not select the project's model names.

Comparison tests implement `ForecastComparisonTest`: a `specification` property
and `compare(baseline_loss, candidate_loss) -> ComparisonOutcome`. Loss vectors
are aligned chronologically. The common alternative is that the candidate has
lower expected loss; each test validates its own sample requirements.
`DieboldMarianoTest(hac_lags=None)` remains the default. It wraps the original
one-sided DM calculation and records the resolved Newey–West lag count in each
outcome. Generic results expose `test_name`, `statistic`, `p_value`, and immutable
`test_parameters`, alongside the model names and mean losses.
For horizon `h`, standard composition and the direct evaluator's default test
use at least `h - 1` HAC lags, alongside the existing sample-size rule, to account
for overlapping multi-session errors. An explicit `hac_lags` overrides that
minimum; the resolved count is always recorded. Comparisons require at least
three observations and at least two more observations than the lag count.
Injected tests retain their own settings and sample requirements.

    from dataclasses import replace
    from rvol.forecasting.evaluation import DieboldMarianoTest

    experiment = replace(
        experiment,
        comparison_test=DieboldMarianoTest(hac_lags=5),
    )
    evaluation = experiment.evaluator().evaluate(result)

Replace that object with an independent test to use the same evaluator and
artifact writer. Each test carries its own settings.

Robustness analysis receives independent `DiagnosticCheck` objects. The built-in
checks are `LagSensitivityCheck`, `MonthlyLossCheck`, `WinRateCheck`, and
`LargestErrorsCheck`. The analyzer accepts an existing evaluation, or calculates
a shared loss table for an independent Python call, then runs only the selected
checks. Descriptive checks work even when a sample is too short for a statistical
comparison.

    from rvol.forecasting.evaluation import MonthlyLossCheck, LargestErrorsCheck

    experiment = replace(
        experiment,
        diagnostic_checks=(MonthlyLossCheck(), LargestErrorsCheck(top_errors=3)),
    )
    robustness = experiment.robustness_analyzer().analyze(result)

Each check supplies a `DiagnosticResult` with immutable rows, CSV columns,
report labels and formatting templates. `RobustnessArtifactWriter` renders these
tables without branches for particular checks. Independent checks implement
the same protocol. `robustness.check(name)` selects a completed table;
`paths.csv(name)` selects its artifact.

Run with `--robustness` to save the configured checks. `robustness.json` schema 4
records their settings and nested comparison-test specifications. The sensitivity
CSV uses the generic `statistic` column.

Run and fold identities belong to shared configuration:

    from dataclasses import replace
    from datetime import date
    from rvol.domain import RunIdentity
    from rvol.presentation.reporting import ValidationArtifactWriter

    first_config = replace(
        experiment.config, identity=RunIdentity("eurusd", "fold_a"),
        forecast_start=date(2025, 1, 1), forecast_end=date(2025, 6, 30),
    )
    second_config = replace(
        experiment.config, identity=RunIdentity("eurusd", "fold_b"),
        forecast_start=date(2025, 4, 1), forecast_end=date(2025, 9, 30),
    )
    first = replace(experiment, config=first_config).run(daily)
    second = replace(experiment, config=second_config).run(daily)
    validation = experiment.evaluator().evaluate_runs((first, second))
    ValidationArtifactWriter("outputs/validation").write(validation)

The pair `(run_id, fold_id)` identifies one independently scored run. The
experiment command accepts `--run-id` and optional `--fold-id`; the deterministic
standalone default is `("default", None)`. Collections reject duplicate
identities, so independent default runs must be named before combining them.
Each run keeps its own forecasts, loss alignment and comparison tests, even
when fold dates overlap. This API scores explicitly constructed folds; it does
not select settings or generate train/test splits automatically.

The default `aggregation="equal_run"` averages per-run mean losses, giving each
identified run/fold equal weight. `aggregation="pooled"` weights those means by
their forecast counts. Overlapping dates remain separate forecast cases in
both modes; `n_forecasts` counts those cases, not independent observations.
Runs must contain identical model sets, metric sets and target horizons. Statistical tests are
reported within each run/fold; p-values are not pooled across folds.

Validation artifacts retain run/fold IDs in per-date losses, per-run summaries
and comparisons, alongside aggregate loss summaries. `validation.json` schema 3
records identities, per-run instrument labels, targets and aggregation rules.
Single-run forecast artifacts use schema 9; robustness metadata uses schema 4.
All CSVs carry `run_id`/`fold_id`
where applicable, plus `horizon_sessions`/`target_kind`. Forecasts and per-date
validation losses also retain the forecast origin and target interval start.
Evaluation and plotting reject mismatched run identities or target definitions.

Models, feature builders and target builders expose an immutable
`ComponentSpecification` with an implementation identifier, ordered feature
names and scalar settings. `FeatureBuilder[pd.DataFrame]` and
`TargetBuilder[pd.DataFrame]` are separate callable contracts. A feature builder
returns `origin_date` and past-only inputs, without target columns. A target
builder supplies its `horizon` and labels for the full observed session timeline,
independently of feature warm-up windows or missing feature rows.

`HARFeatureBuilder` and `build_har_features` return HAR inputs, including
the latest observed session. `RealizedVarianceTargetBuilder` owns the target horizon,
source columns, interval dates and log-variance labels. It retains incomplete
trailing outcomes with missing labels and end dates. `target_date` is the final
session of the interval and determines when the outcome becomes available.

`experiment.prepare(daily)` builds both tables and joins them by `origin_date`.
It validates unique origins, matching target metadata and valid intervals, and
omits incomplete future outcomes before execution. A feature builder can omit
warm-up or unavailable rows without changing which sessions each target covers.
For direct executor calls, assemble inputs explicitly:

    from rvol.domain import TargetHorizon
    from rvol.forecasting.execution import WalkForwardExperiment, prepare_forecast_data
    from rvol.forecasting.features import HARFeatureBuilder
    from rvol.forecasting.targets import RealizedVarianceTargetBuilder

    features = HARFeatureBuilder()(daily)
    targets = RealizedVarianceTargetBuilder(TargetHorizon(5, "cumulative"))(daily)
    prepared = prepare_forecast_data(features, targets)
    executor = WalkForwardExperiment(experiment.models, experiment.config)
    result = executor.run(prepared)

Custom feature and target builders can be injected independently through
`replace(experiment, feature_builder=..., target_builder=...)`; both use
the same alignment and executor availability rules. Use `standard_experiment`'s
`target_horizon` argument to also derive
the standard naive multiplier and minimum comparison-test lag.

Use `experiment.specification` to snapshot the selected feature builder,
target builder and models before running an experiment. Custom components describe their own
settings; reporting depends on these domain values rather than importing
concrete models or duplicating their defaults. Direct `ForecastArtifactWriter`
calls require this snapshot as the `specification` keyword argument.

Execution policies select the strategy used by a walk-forward run:

    from rvol.forecasting.composition import standard_experiment
    from rvol.domain import EveryNSessions, ExperimentConfig, RollingWindow, WalkForwardSettings

    settings = WalkForwardSettings(
        training_window=RollingWindow(size=504),
        retrain=EveryNSessions(5),
        min_train_size=252,
    )
    config = ExperimentConfig(execution=settings)
    experiment = standard_experiment(config)
    result = experiment.run(daily)

`TrainingWindowPolicy.select(n_available)` returns a slice of chronological,
already eligible training rows. `ExpandingWindow` selects all of them;
`RollingWindow` selects up to its configured size. Sizes count observations.
`RefitSchedule.should_refit(eligible_origin_index)` counts forecast origins
starting at zero for the first origin meeting the training requirement.
`EveryNSessions(5)` is due at indices 0, 5, 10, and so on, using indices from
the full dataset even when only a later date range is reported. Calendar gaps
do not advance the count. `WalkForwardSettings()` defaults to expanding
history and refitting every session.

Fitted models may additionally implement `UpdatablePredictor`. Its
`update(features, target)` method consumes new observations in chronological
order and returns updated state, preserving its parameters, the previous
predictor and the input arrays. Targets are observed log realized variances;
the caller ensures availability and supplies each observation once. Empty
batches preserve the state. EWMA implements this capability with the same
variance-scale recursion used by `fit`; other fitted models retain the common
`predict` interface. Fitting EWMA again initializes its state from the supplied
training window.

The runner starts once the selected window meets `min_train_size`, growing a
rolling window up to its configured size. Set the minimum equal to the window
size to require a full window from the first fit. A bounded window must be at
least the minimum training size. Without a strategy override, runs retain
expanding history and daily refitting.

Between refits, HAR and AR1 reuse their fitted coefficients with fresh features,
and HAR retains its fitted smearing correction. EWMA consumes only newly
available targets, once each; a refit rebuilds its state from the selected
window. Both fitting and updates use outcomes with `target_date <= origin_date`.
Reporting bounds preserve the global schedule: the runner initializes from
the last scheduled fit before the requested range and applies subsequent
updates before reporting its first forecast. All fitted state belongs to one
run. Origin dates must advance strictly in chronological order.

Forecast records carry `fit_date`, `train_start_date`, and `train_end_date`.
Training boundaries refer to target dates in the last fit, and `n_train` is
the count used by that fit, including when the model is reused or updated.
`ExperimentResult` also retains the run configuration and immutable `execution`
metadata. `ExecutionMetadata` records the actual executor implementation, an
`ExecutionSpecification` snapshot and scalar runtime `facts`. Settings describe
their own component and any named policies. Walk-forward supplies its schedule
anchor and observation-update rule as runtime facts; the artifact writer adds
no execution rules. Custom window and refit policies provide component
specifications.

Chronological holdout uses a separate executor and settings type:

    from datetime import date
    from rvol.forecasting.composition import standard_experiment
    from rvol.domain import ExperimentConfig, HoldoutSettings

    config = ExperimentConfig(
        execution=HoldoutSettings(
            train_end=date(2024, 12, 31),
            min_train_size=252,
            update_observations=False,
        ),
        forecast_start=date(2025, 1, 1),
    )
    experiment = standard_experiment(config)
    result = experiment.run(daily)
    evaluation = experiment.evaluator().evaluate(result)

Holdout fits once using every prepared row whose target date is on or before
`train_end`, and requires the minimum training count at that cutoff. Forecast
origins must be on or after the cutoff; targets must follow it. A nontrading
cutoff is allowed. Every forecast uses fresh input features with fixed fitted
parameters. By default, predictor state is also fixed. Set
`update_observations=True` to let stateful predictors consume newly available
outcomes, once each, at subsequent origins. EWMA then updates its variance
state; models without an update method keep their fitted predictors.

Reporting dates do not change fitting or the observations available at an
origin. `fit_date` remains the configured cutoff, and `n_train` and training
boundaries describe the single fit. Holdout uses the same evaluator, plotting
and artifact writer as walk-forward. Its metadata records the cutoff and update
rule, without a walk-forward schedule anchor.

The experiment modules, `scripts/run_experiment.py`, and the installed
`rvol-run-experiment` entry point accept these options:

| Option | Meaning | Default |
|---|---|---|
| `daily` | Daily realized-variance Parquet dataset | Required |
| `--output-dir PATH` | Directory for all saved outputs | `outputs/experiment` |
| `--plot` | Save forecast plot and metadata | Disabled |
| `--robustness` | Save the experiment's diagnostic reports | Disabled |
| `--models NAME ...` | Select configured model objects | Experiment's models |
| `--candidate NAME` | Plot and comparison candidate | Experiment's candidate; sole model for a single-model run |
| `--baselines NAME ...` | Comparison baselines; no names disables comparisons | Experiment's surviving baselines |
| `--instrument LABEL` | Instrument used in metadata and labels | Experiment's label |
| `--forecast-start YYYY-MM-DD` | First target date to report | Experiment's start date |
| `--forecast-end YYYY-MM-DD` | Last target date to report | Experiment's end date |
| `--run-id NAME` | Identity of this experiment run | Experiment's run ID |
| `--fold-id NAME` | Optional validation fold identity | Experiment's fold ID |

Training windows, refit intervals, horizons, model parameters, metrics,
comparison tests and diagnostic settings belong in the Python experiment.
The typed examples above show `TargetHorizon`, `WalkForwardSettings`,
`RollingWindow`, `EveryNSessions` and the evaluation objects directly.
Sensitivity diagnostics use their configured lag grid, which may include lags
below `h - 1`.

Run the fixed-cutoff holdout experiment with:

    python -m experiments.holdout \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --output-dir outputs/holdout_2024 \
        --plot \
        --robustness

It fits using outcomes through 31 December 2024 and reports the same 2025–2026
target period as the baseline. Set `update_observations=True` in its
`HoldoutSettings` to let EWMA update its state after the cutoff. Plots and
robustness reports consume the same holdout forecasts.

## Reproduce the plots

    python scripts/simulation/sampling_plot.py
    python scripts/simulation/returns_and_rv_plot.py
    python scripts/simulation/cumulative_iv_vs_rv_plot.py
    python scripts/simulation/bid_ask_bounce_plot.py
    python scripts/simulation/daily_rv_plot.py
    python scripts/simulation/heston_plot.py
    python scripts/simulation/simulation_plots.py
    python scripts/diagnostics/diagnostics_plots.py \
        data/EURUSD_2024-01-01_2024-03-31.parquet

For the forecast figure's Q1 2024 reporting period, configure the earlier dates
and 200-observation training minimum in Python, then use the same runner:

    from datetime import date
    from rvol.cli.run_experiment import main
    from rvol.domain import ExperimentConfig, WalkForwardSettings
    from rvol.forecasting.composition import standard_experiment

    experiment = standard_experiment(ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=200),
        forecast_start=date(2024, 1, 1),
        forecast_end=date(2024, 3, 31),
    ))
    main(experiment, argv=[
        "data/derived/EURUSD_daily_rv_5min.parquet",
        "--output-dir", "outputs/forecast_figure", "--plot",
    ])

Copy the generated figure and its metadata into the figure collection:

    cp outputs/forecast_figure/forecasts.png figures/forecast_evaluation.png
    cp outputs/forecast_figure/forecasts.png.json figures/forecast_evaluation.png.json

The scripts write `figures/sampling_grids.png`, `figures/returns_and_rv.png`,
`figures/cumulative_iv_vs_rv.png`, `figures/bid_ask_bounce.png`,
`figures/daily_rv.png`, `figures/heston_simulation.png`,
`figures/simulation_overview.png`, `figures/forecast_evaluation.png`, and
`figures/diagnostics_overview.png`.
Use `scripts/diagnostics/signature_plot.py` when only one mid, bid, or ask signature is
needed, or `python scripts/diagnostics/noise_report.py TICKS.parquet` for a
numerical report comparing fine and coarse sampling. Simulation scripts use
generated data; diagnostics scripts use real tick datasets.

## Build the daily dataset

Keep downloaded tick data in separate chunks, then combine them into a compact
daily realised-variance dataset without loading every chunk at once:

    python scripts/data/build_daily_dataset.py "data/EURUSD_*.parquet" \
        --frequency 5min \
        --out data/derived/EURUSD_daily_rv_5min.parquet

The builder carries FX sessions across file boundaries, removes overlapping
timestamps and session dates, and writes the output atomically. The result has
one row per complete session with the session date, realised variance, log
realised variance, and the number of sampled observations.

Audit coverage before forecasting:

    python scripts/data/audit_daily_dataset.py \
        data/derived/EURUSD_daily_rv_5min.parquet

The audit checks ordering, duplicate and invalid values, observation counts,
annual session coverage, and gaps longer than seven calendar days. HAR feature
construction rejects such gaps by default so disconnected periods cannot be
treated as consecutive trading sessions.

`rvol.forecasting.features.build_har_features` converts that daily table into
past-only HAR-RV inputs: the current session's log variance, the latest five
sessions' mean log variance, and the latest 22 sessions' mean log variance.
`rvol.forecasting.targets.RealizedVarianceTargetBuilder` independently creates future
log-variance labels and their interval dates. The standard experiment combines
these by forecast origin and defaults to predicting the next observed session.

`rvol.forecasting.execution.WalkForwardExperiment` runs historical-mean, naïve,
fixed-decay EWMA, AR(1), and HAR-RV models over identical expanding windows.
EWMA uses a fixed 0.94 decay on the variance scale,
evaluated in log space for numerical stability. Training outcomes are admitted
only after their target dates have become observable, preventing future
information from leaking into a forecast.

`HAR` fits the selected log variance target by OLS and applies Duan's residual
correction to target mean variance:

    residual_i = actual_log_rv_i - fitted_log_rv_i
    log_correction = log(mean(exp(residual_i)))
    predicted_log_rv = OLS_log_forecast + log_correction
    predicted_rv = exp(predicted_log_rv)

The correction uses only residuals from the eligible training window and is
recomputed at each fit, then retained until the next fit. A log-sum-exp
calculation avoids overflowing the exponentials. The correction assumes that
the training residual distribution represents current forecast uncertainty,
so it can improve QLIKE while worsening log-MSE. It is part of the single
`HARModel.fit` implementation.

The common output scale does not imply a common statistical target:

| Model | Meaning of the forecast after exponentiating its log output |
|---|---|
| Historical mean | Geometric mean of training realised variances |
| Naïve | Current session's realised variance; multiplied by the horizon for cumulative targets |
| EWMA | Exponentially weighted average of past realised variances |
| AR(1) | Geometric-mean variance forecast from the log regression |
| HAR | Mean-variance estimate using the training-residual correction |

Using the Q1 2024 experiment configured in the reproduction example above,
compare HAR with every other model and save robustness reports in one run:

    from dataclasses import replace

    experiment = replace(experiment, comparison_pairs=experiment.all_comparison_pairs)
    main(experiment, argv=[
        "data/derived/EURUSD_daily_rv_5min.parquet",
        "--output-dir", "outputs/final_evaluation", "--robustness",
    ])

The report shows mean QLIKE and log-RV squared loss for every model, followed
by one-sided Diebold-Mariano tests of whether each requested candidate has
lower expected loss than its baseline. The standard experiment's default pair
compares HAR with the naïve benchmark. The tests use a Newey-West long-run
variance estimate for serially correlated loss differences; a positive
difference favours the candidate.
Every run writes normalized forecasts, mean losses, pairwise comparisons,
and JSON experiment metadata under `--output-dir`. The metadata
records dataset coverage and a SHA-256 fingerprint so the exact input can be
verified later. JSON schema version 9 records a `specification` object
containing the independent feature and target builders and each model's
implementation, ordered feature
names and settings, including EWMA decay and HAR's training-residual smearing
correction. HAR windows count observed sessions; `max_gap_days` is a calendar-day
gap limit. These are configured settings, rather than fitted coefficients.
The `execution` object records the actual executor, its settings and policies,
and executor-supplied runtime facts. For walk-forward, those facts include the
full dataset's schedule anchor and the observation-update rule.
`forecasts.csv` includes `fit_date`, `train_start_date`, and
`train_end_date`. The training count
and boundaries describe the last actual fit, even when a predictor has since
consumed new observations. Forecasts also include `target_start_date`,
`horizon_sessions` and `target_kind`; loss summaries and comparisons include the
two target-definition columns. JSON's top-level `target` object describes the
horizon (session count and kind); `specification.target` identifies the builder
and label source columns. The dataset object includes a nullable `instrument`
label; CLI artifacts also include `resolved_configuration`.
`min_train_size` belongs to execution settings, rather than shared experiment
bounds. Repeating an unchanged run produces byte-identical artifacts.
`evaluation.comparison_test` identifies the configured statistical test.
`comparisons.csv` uses `test_name` and `statistic` columns; resolved test
settings are JSON in `test_parameters`.

The artifact writer requires a completed runner result with its execution
snapshot and fit boundaries. It rejects configuration or strategy metadata
that differs from the completed run before creating output files.

The robustness report above checks Newey–West lags from zero through five,
monthly losses, session-level HAR win rates, and each model's five largest
absolute log-RV errors. Detailed CSV files and a combined Markdown report are written under
`outputs/final_evaluation/robustness`, alongside the main numerical results
in `outputs/final_evaluation`. Add `--plot` to produce the figure in that run too.

## Install

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

Run the offline quality checks with:

    pytest -m "not slow"
    ruff check src tests scripts experiments
    mypy src tests experiments

## Results

The archived results below are the original four-model Q1 2024 experiment,
before EWMA and HAR's variance correction were added. Run the workflow above
for results from the current implementation. Evaluation on an already inspected
period is exploratory rather than a fresh confirmatory test.

### 1. How far can you sample before noise takes over?

Sample: EURUSD tick quotes from Dukascopy, 1 January to 31 March 2024 —
5,359,908 ticks over 64 trading sessions, each ending at 17:00 New York
(22:00 UTC in winter and 21:00 UTC in summer). Median quoted spread 0.19 bps,
no crossed quotes.

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
one-sided t-test on the log ratios — removes it
(`rvol.diagnostics.microstructure.noise_test`):

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

![Out-of-sample forecast evaluation](figures/forecast_evaluation.png)

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
