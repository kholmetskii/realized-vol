# Realized Volatility

[![CI](https://github.com/kholmetskii/realized-vol/actions/workflows/ci.yml/badge.svg)](https://github.com/kholmetskii/realized-vol/actions/workflows/ci.yml)

Estimating realized volatility from intraday data, forecasting it, and comparing
the forecasts with proper statistical tests.

## Question

How predictable is daily volatility, and can competing models be told apart
statistically — given that the target itself is unobservable and is estimated
with error?

## Architecture

    src/rvol/domain/       immutable configuration, results, and protocols
    src/rvol/application/  forecast data alignment and execution use cases
    src/rvol/cli/          packaged command-line entry points
    src/rvol/composition/  experiment components and comparison defaults
    src/rvol/models/       interchangeable volatility forecasting models
    src/rvol/evaluation/   loss metrics and statistical model comparison
    src/rvol/reporting/    deterministic experiment artifacts
    src/rvol/infrastructure/ Parquet persistence adapters
    src/rvol/data/         external data download and parsing
    src/rvol/market/       FX sessions and timestamp sampling rules
    src/rvol/estimators/   pure numerical variance estimators
    src/rvol/features/     daily features built from market data and estimators
    src/rvol/targets/      future outcomes built from observed daily series
    src/rvol/diagnostics/  volatility signatures and microstructure tests
    src/rvol/plotting/     static research figures
    src/rvol/simulation/   latent prices, observation noise, Monte Carlo checks
    scripts/               thin command-line and reproducibility entry points
    tests/                 checks against synthetic data with a known answer

Additional noise- and jump-robust estimators are not yet implemented.

The forecasting dependency direction points toward stable contracts:

    cli (scripts delegate here)
       ├── composition ─┬──> application ──> domain
       │               ├──> models ────────> domain
       │               ├──> features ──────> domain
       │               ├──> targets ───────> domain
       │               └──> evaluation ──> domain
       ├── evaluation ────────────────> domain
       ├── reporting ──> evaluation + domain
       └── infrastructure

Installed environments expose `rvol-run-experiment`. The equivalent
repository entry point is `python scripts/run_experiment.py`. It runs the
experiment once and saves numerical results, with optional plots and robustness
reports generated from the same forecasts and losses.

The independent data-preparation pipeline is:

    data + market + estimators ──> features ──> composition

`estimators` contains no pandas, timestamps, FX conventions, or reporting.
`market` contains no variance formulas. `features` is the adapter that combines
the two, while higher layers consume the resulting daily quantities. Forecast
models implement the domain `Forecaster` protocol, and Parquet persistence
implements `DatasetRepository`; neither is hard-coded into the experiment.

To add a model, implement `name`, `feature_names`, `specification`, and `fit`,
then inject the new object into `WalkForwardExperiment`. The experiment and
evaluator do not need model-specific branches. Tests enforce the dependency
direction so an inner layer cannot accidentally import an outer adapter.

All models share one array-based interface: `fit(features, target)` returns a
fitted object with `predict(features)`. Features are two-dimensional arrays
with columns in `model.feature_names` order; training targets are
one-dimensional arrays of observed log realised variance. Both operations
leave their inputs unchanged. Predictions are one-dimensional arrays with one
finite log variance per row, representing finite positive variances after
exponentiation.

The HAR-specific `fit_har` function and `HarModel` DataFrame wrapper have been
removed. Convert DataFrames to arrays at the call site and use the common API:

    from rvol.models import HARForecaster

    model = HARForecaster()
    columns = list(model.feature_names)
    fitted = model.fit(
        training.loc[:, columns].to_numpy(dtype="float64"),
        training["target"].to_numpy(dtype="float64"),
    )
    predicted_log_rv = fitted.predict(current.loc[:, columns].to_numpy(dtype="float64"))

Fitted coefficients are available as `fitted.intercept` and
`fitted.coefficients`, with slopes in `model.feature_names` order.

The experiment command resolves an `ExperimentRun` through
`rvol.composition.experiment_run_from_mapping`, using
`standard_experiment_definition` for defaults. Its definition selects the feature
builder, target builder, models, executor factory, metrics, evaluation comparison
pairs, plot candidate, and robustness baselines together. Construct a different
definition to use other features, targets, models, execution methods, metrics,
or comparison choices:

    from rvol.composition import standard_experiment_definition
    from rvol.domain import ExperimentConfig, WalkForwardSettings

    definition = standard_experiment_definition()
    config = ExperimentConfig(execution=WalkForwardSettings(min_train_size=252))
    experiment = definition.run(daily, config)
    evaluation = definition.evaluator().evaluate(experiment)

`ExperimentExecutor[Dataset]` is the execution contract: `run(features)` returns
an `ExperimentResult`. `ExperimentDefinition` receives an `executor_factory`
that accepts `(models, config)` and creates a fresh executor for each call to
`definition.run`. The definition aligns features with targets and delegates
execution; the executor owns fitting, observation availability and forecasting.
Executors preserve the prepared input and keep fitted state local to each run.

The standard factory is `WalkForwardExperiment`. Select another implementation
through the same composition entry point:

    definition = standard_experiment_definition(executor_factory=CustomExecutor)

`CustomExecutor(models, config)` must implement `run(features)`. Independently
constructed `ExperimentDefinition` objects also require an `executor_factory`.
Executors return the common forecast records so evaluation and plotting can be
reused; artifact export additionally requires the run configuration, execution
snapshot and fit boundaries described below. Executors can use independent
settings types and retain run/fold identities and target definitions in the
common results.

`ExperimentConfig[Settings]` contains shared reporting dates and a required
`execution` settings object. `ExperimentDefinition[Settings]` binds its executor
factory to that settings type. Each execution method validates its own settings;
the shared configuration does not impose training windows or refitting rules.
`ExperimentRun(definition, config)` binds a compatible definition and settings
for repeated calls to `run.run(daily)`. The command uses this same object when
selecting an executor.
`holdout_experiment_definition()` assembles the standard models and features
with holdout execution; composition owns concrete executor selection.

Saved experiments use TOML configuration version 1. The complete baseline is
in [`experiments/eurusd.toml`](experiments/eurusd.toml). Run it once to generate
numerical results, a plot, and robustness reports:

    python scripts/run_experiment.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --config experiments/eurusd.toml \
        --output-dir outputs/eurusd \
        --plot \
        --robustness

Numerical results are always saved: `forecasts.csv`, `loss_summary.csv`,
`comparisons.csv`, and `experiment.json`. `--plot` adds `forecasts.png` and
`forecasts.png.json`. `--robustness` adds the selected diagnostic CSV files,
`robustness.md`, and `robustness.json` under the output directory's `robustness/`
subdirectory. Omit either flag when its output is not needed. The default output
directory is `outputs/experiment`; use a different directory for each experiment
whose results you want to retain.

To run only HAR, use [`experiments/har.toml`](experiments/har.toml):

    python scripts/run_experiment.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --config experiments/har.toml \
        --output-dir outputs/har \
        --plot

This forecasts next-session variance with expanding history and daily refitting.
Its comparison list is empty, and its optional robustness report contains
monthly losses and largest errors. With no saved configuration, `--models HAR`
also selects a single-model run; choose `--candidate NAME` when selecting a
model other than HAR.

The input dataset and output destinations are command arguments. The saved
configuration selects the experiment's components and settings:

| TOML section | Settings |
|---|---|
| `dataset` | Optional `instrument` label |
| `run` | `run_id`, optional `fold_id`, `forecast_start`, `forecast_end` |
| `target` | `name`, label source columns, `sessions`, `kind` |
| `execution` | `name`, `min_train_size`, window/refit or holdout options |
| `features` | `name`, column names, HAR windows, gap limit |
| `[[models]]` | One table per selected model, with its constructor parameters |
| `evaluation` | Metric names, `candidate`, `baselines`, `comparison_pairs` |
| `evaluation.test` | Test name and parameters |
| `robustness` | `checks`, `hac_lags`, `top_errors` |
| `plot` | Display `metric`, selected from evaluation metrics |

Registered model names are `historical_mean`, `naive`, `EWMA`, `AR1`, and `HAR`.
EWMA accepts `decay`; naive accepts an optional `variance_multiplier`, whose
default follows the target horizon. HAR always retains its variance correction.
The current feature builder is `name = "HAR"`, with `date_col`, `rv_col`,
`weekly_window`, `monthly_window`, and `max_gap_days`. Set `check_gaps = false`
to disable the gap limit. Targets use `name = "realized_variance"`, with their
own `date_col` and `rv_col` (defaults: `date` and `log_rv`), plus `sessions` and
`kind`. Feature and target source columns are independent: set both sections
when using renamed daily columns, or select a different variance series for
the labels. Metrics are `QLIKE` and `log-RV MSE`; the registered
comparison test is `name = "DM"`, with optional `hac_lags` and `min_hac_lags`.
The named registries live in composition; Python callers can still inject
independent implementations directly through `ExperimentDefinition`.

Precedence is **built-in defaults → saved configuration → explicit command
options**. For example, append `--horizon 5 --target-kind cumulative` to change
the target without editing the file. Explicit default values also override a
file: `--horizon 1` resets its horizon, and `--no-update-observations` disables
saved holdout updates. Repeated `--compare` or `--check` options replace the
corresponding saved list. `--models` retains saved parameters for selected names
and trims inherited baselines and comparison pairs to those models. If all
inherited pairs are removed, comparisons use the first surviving baseline
against the candidate. `--candidate` rebuilds inherited baselines from the other
selected models; changing `--candidate` or `--baselines` rebuilds inherited
comparison pairs. A saved empty comparison list remains disabled. Explicit
`--compare` and `--baselines` selections take precedence and are validated;
invalid explicit references are never silently removed.

`--metrics` selects the evaluated losses. If the saved plot metric is excluded,
the first selected metric becomes the plotting default; otherwise the saved
metric is retained. `--metric` overrides that default and must name a selected
metric. It requires `--plot`. Similarly, `--check`, `--max-hac-lag`, and
`--top-errors` require `--robustness`; unused report options are rejected before
data loading. `--compare` and `--hac-lags` configure the numerical evaluation,
which always runs. All requested outputs share one execution and one loss table.
Lag-sensitivity checks repeat the comparison tests at each lag using those
existing losses.

Switching `--executor` discards the previous executor's private settings while
retaining shared settings such as `min_train_size`. Switching a saved rolling
window to `--training-window expanding` removes its old window size. Explicit
incompatible flags are still rejected. With no saved candidate, the default
remains HAR, so a model selection excluding HAR must supply a candidate.
Omitted baselines use the other selected models in the standard baseline order;
omitted evaluation comparisons use the first baseline against the candidate.
Configuration validation names resolved fields such as `execution.window_size`
and `execution.training_window`, whether settings came from TOML or flags.
Explicit comparison references must name selected models. When no baselines
are selected, omitted robustness checks default to monthly losses and largest
errors. Explicit comparison diagnostics still require a baseline. A saved
single-model configuration can set empty `baselines` and
`comparison_pairs` lists, as in `experiments/har.toml`.

Unknown sections, components or parameters, incompatible options, invalid
counts/dates, model references and unavailable plot metrics are rejected before
the daily dataset is loaded. Configurations select known components and cannot
import arbitrary Python objects.

For the same composition from Python:

    from rvol.composition import experiment_run_from_mapping
    from rvol.infrastructure.configuration import load_experiment_configuration

    run = experiment_run_from_mapping(
        load_experiment_configuration("experiments/eurusd.toml"),
    )
    experiment = run.run(daily)
    evaluation = run.definition.evaluator().evaluate(experiment)
    robustness = run.definition.robustness_analyzer(
        checks=run.diagnostic_checks,
    ).analyze(experiment, evaluation=evaluation)
    settings = run.resolved_configuration

`ExperimentRun` also retains diagnostic selection and the plot metric. Its
`resolved_configuration` snapshots actual component implementations, parameters,
execution policies and workflow choices, including applied defaults and command
overrides. The command stores this snapshot in `experiment.json` and
`robustness.json`; forecast figures receive a `<image filename>.json` sidecar
with resolved settings, dataset fingerprint and coverage. Direct artifact-writer
calls can pass `configuration=run.resolved_configuration` to retain that snapshot.

`dataset.instrument` and `--instrument` are explicit labels; no instrument is
inferred from filenames. The label is retained in run metadata, evaluation and
robustness results, and report/plot labels. When absent, forecast titles are
generic. Python callers set `ExperimentConfig(instrument="EUR/USD", ...)`.

The target is selected separately from execution settings:

    from rvol.domain import TargetHorizon

    definition = standard_experiment_definition(
        target_horizon=TargetHorizon(sessions=5, kind="cumulative"),
    )
    experiment = definition.run(daily, config)

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

Evaluation keeps the default naive-versus-HAR comparison and accepts ordered
`--compare` overrides. Plots and robustness use the configured candidate and
baseline list, with the
requested HAC lags and top-error count. The existing
`run_standard_forecast_experiment` helper delegates to the same definition.

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
    from rvol.evaluation import DieboldMarianoTest

    definition = replace(
        standard_experiment_definition(),
        comparison_test=DieboldMarianoTest(hac_lags=5),
    )
    evaluation = definition.evaluator().evaluate(experiment)

Replace that object with an independent test to use the same evaluator and
artifact writer. The `hac_lags` convenience override applies only to DM; custom
tests carry their own settings.

Robustness analysis receives independent `DiagnosticCheck` objects. The built-in
checks are `LagSensitivityCheck`, `MonthlyLossCheck`, `WinRateCheck`, and
`LargestErrorsCheck`. The analyzer accepts an existing evaluation, or calculates
a shared loss table for an independent Python call, then runs only the selected
checks. Descriptive checks work even when a sample is too short for a statistical
comparison.

    from rvol.evaluation import MonthlyLossCheck, LargestErrorsCheck

    analyzer = definition.robustness_analyzer(
        checks=(MonthlyLossCheck(), LargestErrorsCheck(top_errors=3)),
    )
    robustness = analyzer.analyze(experiment)

Each check supplies a `DiagnosticResult` with immutable rows, CSV columns,
report labels and formatting templates. `RobustnessArtifactWriter` renders these
tables without branches for particular checks. Independent checks implement
the same protocol. `robustness.check(name)` selects a completed table;
`paths.csv(name)` selects its artifact. The former fixed result fields and path
attributes have been removed.

With `--robustness`, the experiment command accepts repeated `--check` options
(`hac_sensitivity`, `win_rates`, `monthly_losses`, `largest_errors`). Omitting
them retains the saved selection, or defaults to all four checks when baselines
are available. For example, add
`--robustness --check monthly_losses --check largest_errors` to
generate only those tables and report sections. `robustness.json` schema 4
records the selected checks, their settings, and nested comparison-test
specifications. The sensitivity CSV uses the generic `statistic` column.

Run and fold identities belong to shared configuration:

    from dataclasses import replace
    from datetime import date
    from rvol.domain import RunIdentity
    from rvol.reporting import ValidationArtifactWriter

    first = definition.run(daily, replace(
        config, identity=RunIdentity("eurusd", "fold_a"),
        forecast_start=date(2025, 1, 1), forecast_end=date(2025, 6, 30),
    ))
    second = definition.run(daily, replace(
        config, identity=RunIdentity("eurusd", "fold_b"),
        forecast_start=date(2025, 4, 1), forecast_end=date(2025, 9, 30),
    ))
    validation = definition.evaluator().evaluate_runs((first, second))
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

`HARFeatureBuilder` and `build_har_features` now return only HAR inputs, including
the latest observed session. They no longer accept a `horizon` argument or
produce future labels. `RealizedVarianceTargetBuilder` owns the target horizon,
source columns, interval dates and log-variance labels. It retains incomplete
trailing outcomes with missing labels and end dates. `target_date` is the final
session of the interval and determines when the outcome becomes available.

`definition.prepare(daily)` builds both tables and joins them by `origin_date`.
It validates unique origins, matching target metadata and valid intervals, and
omits incomplete future outcomes before execution. A feature builder can omit
warm-up or unavailable rows without changing which sessions each target covers.
For direct executor calls, assemble inputs explicitly:

    from rvol.application import prepare_forecast_data
    from rvol.features.forecasting import HARFeatureBuilder
    from rvol.targets.variance import RealizedVarianceTargetBuilder

    features = HARFeatureBuilder()(daily)
    targets = RealizedVarianceTargetBuilder(TargetHorizon(5, "cumulative"))(daily)
    prepared = prepare_forecast_data(features, targets)
    experiment = executor.run(prepared)

Custom feature and target builders can be injected independently through
`ExperimentDefinition(feature_builder=..., target_builder=..., ...)`; both use
the same alignment and executor availability rules. Use the standard definition
factory's `target_horizon` argument or saved `[target]` settings to also derive
the standard naive multiplier and minimum comparison-test lag.

Use `definition.specification` to snapshot the selected feature builder,
target builder and models before running an experiment. Custom components describe their own
settings; reporting depends on these domain values rather than importing
concrete models or duplicating their defaults. Direct `ForecastArtifactWriter`
calls now require this snapshot as the `specification` keyword argument.

Execution policies select the strategy used by a walk-forward run:

    from rvol.composition import standard_experiment_definition
    from rvol.domain import EveryNSessions, ExperimentConfig, RollingWindow, WalkForwardSettings

    settings = WalkForwardSettings(
        training_window=RollingWindow(size=504),
        retrain=EveryNSessions(5),
        min_train_size=252,
    )
    config = ExperimentConfig(execution=settings)
    definition = standard_experiment_definition()
    experiment = definition.run(daily, config)

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
no execution rules. Custom settings implement `DescribedExecutionSettings` for
export, and custom window and refit policies provide component specifications.

Chronological holdout uses a separate executor and settings type:

    from datetime import date
    from rvol.application import HoldoutExperiment
    from rvol.composition import standard_experiment_definition
    from rvol.domain import ExperimentConfig, HoldoutSettings

    definition = standard_experiment_definition(executor_factory=HoldoutExperiment)
    config = ExperimentConfig(
        execution=HoldoutSettings(
            train_end=date(2024, 12, 31),
            min_train_size=252,
            update_observations=False,
        ),
        forecast_start=date(2025, 1, 1),
    )
    experiment = definition.run(daily, config)
    evaluation = definition.evaluator().evaluate(experiment)

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

The experiment command and its installed `rvol-run-experiment` equivalent
accept these options:

| Option | Meaning | Default |
|---|---|---|
| `--config PATH` | Saved experiment TOML file | Built-in preset |
| `--output-dir PATH` | Directory for all saved outputs | `outputs/experiment` |
| `--plot` | Save forecast plot and metadata | Disabled |
| `--robustness` | Save selected diagnostic reports | Disabled |
| `--instrument LABEL` | Instrument used in metadata and labels | Generic label |
| `--models NAME ...` | Selected model names | Standard five models |
| `--metrics NAME ...` | Selected evaluation metrics | QLIKE and log-MSE |
| `--metric NAME` | Metric displayed in forecast plots; requires `--plot` | Saved or first selected metric |
| `--candidate NAME` | Plot and robustness candidate | HAR |
| `--baselines NAME ...` | Plot and robustness baselines | Other selected models |
| `--compare BASELINE CANDIDATE` | Evaluation comparison; repeatable | First baseline vs candidate |
| `--hac-lags N` | Explicit DM lag count | Automatic, accounting for horizon |
| `--check NAME` | Diagnostic to include; repeatable; requires `--robustness` | All four, or descriptive checks without baselines |
| `--max-hac-lag N` | Sensitivity grid from 0 through N; requires `--robustness` | 0 through 5 |
| `--top-errors N` | Largest errors per model; requires `--robustness` | 5 |
| `--executor walk-forward` or `holdout` | Execution method | `walk-forward` |
| `--horizon N` | Target horizon in observed sessions | `1` |
| `--target-kind session` or `cumulative` | Future-session variance or summed future variance | `session` |
| `--min-train-size N` | Minimum training observations | Selected executor's default (`252` for both) |
| `--training-window expanding` or `rolling` | History selected at each fit | `expanding` |
| `--window-size N` | Maximum training observations for a rolling window | Required for `rolling` |
| `--retrain-every N` | Refit every N eligible forecast sessions | `1` |
| `--train-end YYYY-MM-DD` | Holdout training cutoff | Required for `holdout` |
| `--update-observations` / `--no-update-observations` | Enable/disable holdout predictor updates | Disabled |

For example, forecast summed variance over the next five observed sessions:

    python scripts/run_experiment.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --min-train-size 252 \
        --horizon 5 \
        --target-kind cumulative \
        --forecast-start 2025-01-01 \
        --forecast-end 2026-09-30 \
        --output-dir outputs/five_session_cumulative

Add `--plot` and/or `--robustness` to include those outputs in the same run.
Sensitivity diagnostics still run their explicitly selected lag grid, which
may include lags below `h - 1`.

For example, use up to 504 training observations and refit every fifth eligible
session with:

    python scripts/run_experiment.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --min-train-size 252 \
        --training-window rolling \
        --window-size 504 \
        --retrain-every 5 \
        --forecast-start 2025-01-01 \
        --forecast-end 2026-09-30 \
        --output-dir outputs/rolling_504_refit_5

Add `--plot` and/or `--robustness` to generate reports from these same forecasts.
`--window-size` requires `--training-window rolling` and must be at least
`--min-train-size`. Counts and intervals must be positive integers. Invalid
options and reversed date bounds produce usage errors before loading the data.
Omitting the new options preserves expanding history and daily refitting.

Run chronological holdout with a fixed training cutoff:

    python scripts/run_experiment.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --executor holdout \
        --train-end 2024-12-31 \
        --min-train-size 252 \
        --forecast-start 2025-01-01 \
        --forecast-end 2026-09-30 \
        --output-dir outputs/holdout_2024

Add `--update-observations` to update EWMA state after the cutoff. Optional plots
and robustness reports use this same holdout result. Holdout rejects
`--training-window`, `--window-size` and `--retrain-every`, including explicitly
supplied default values. Walk-forward rejects `--train-end` and
`--update-observations`. Invalid combinations fail before data loading.

## Reproduce the plots

    python scripts/sampling_plot.py
    python scripts/returns_and_rv_plot.py
    python scripts/cumulative_iv_vs_rv_plot.py
    python scripts/bid_ask_bounce_plot.py
    python scripts/daily_rv_plot.py
    python scripts/heston_plot.py
    python scripts/simulation_plots.py
    python scripts/run_experiment.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --min-train-size 200 \
        --forecast-start 2024-01-01 \
        --forecast-end 2024-03-31 \
        --output-dir outputs/forecast_figure \
        --plot
    cp outputs/forecast_figure/forecasts.png figures/forecast_evaluation.png
    cp outputs/forecast_figure/forecasts.png.json figures/forecast_evaluation.png.json
    python scripts/diagnostics_plots.py \
        data/EURUSD_2024-01-01_2024-03-31.parquet

The scripts write `figures/sampling_grids.png`, `figures/returns_and_rv.png`,
`figures/cumulative_iv_vs_rv.png`, `figures/bid_ask_bounce.png`,
`figures/daily_rv.png`, `figures/heston_simulation.png`,
`figures/simulation_overview.png`, `figures/forecast_evaluation.png`, and
`figures/diagnostics_overview.png`.
Use `scripts/signature_plot.py` when only one mid, bid, or ask signature is
needed.

## Build the daily dataset

Keep downloaded tick data in separate chunks, then combine them into a compact
daily realised-variance dataset without loading every chunk at once:

    python scripts/build_daily_dataset.py "data/EURUSD_*.parquet" \
        --frequency 5min \
        --out data/derived/EURUSD_daily_rv_5min.parquet

The builder carries FX sessions across file boundaries, removes overlapping
timestamps and session dates, and writes the output atomically. The result has
one row per complete session with the session date, realised variance, log
realised variance, and the number of sampled observations.

Audit coverage before forecasting:

    python scripts/audit_daily_dataset.py \
        data/derived/EURUSD_daily_rv_5min.parquet

The audit checks ordering, duplicate and invalid values, observation counts,
annual session coverage, and gaps longer than seven calendar days. HAR feature
construction rejects such gaps by default so disconnected periods cannot be
treated as consecutive trading sessions.

`rvol.features.forecasting.build_har_features` converts that daily table into
past-only HAR-RV inputs: the current session's log variance, the latest five
sessions' mean log variance, and the latest 22 sessions' mean log variance.
`rvol.targets.variance.RealizedVarianceTargetBuilder` independently creates future
log-variance labels and their interval dates. The standard experiment combines
these by forecast origin and defaults to predicting the next observed session.

`rvol.application.WalkForwardExperiment` runs historical-mean, naïve,
fixed-decay EWMA, AR(1), and HAR-RV forecasters over identical expanding windows.
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
`HARForecaster.fit` implementation.

The common output scale does not imply a common statistical target:

| Model | Meaning of the forecast after exponentiating its log output |
|---|---|
| Historical mean | Geometric mean of training realised variances |
| Naïve | Current session's realised variance; multiplied by the horizon for cumulative targets |
| EWMA | Exponentially weighted average of past realised variances |
| AR(1) | Geometric-mean variance forecast from the log regression |
| HAR | Mean-variance estimate using the training-residual correction |

Run the complete out-of-sample comparison with:

    python scripts/run_experiment.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --min-train-size 200 \
        --forecast-start 2024-01-01 \
        --forecast-end 2024-03-31 \
        --compare naive HAR \
        --compare EWMA HAR \
        --compare AR1 HAR \
        --compare historical_mean HAR \
        --output-dir outputs/final_evaluation

The report shows mean QLIKE and log-RV squared loss for every model, followed
by one-sided Diebold-Mariano tests of whether each requested candidate has
lower expected loss than its baseline. Without `--compare`, the default test
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
`forecasts.csv` appends `fit_date`, `train_start_date`, and
`train_end_date`; its existing columns retain their order. The training count
and boundaries describe the last actual fit, even when a predictor has since
consumed new observations. Forecasts also append `target_start_date`,
`horizon_sessions` and `target_kind`; loss summaries and comparisons append the
two target-definition columns. JSON's top-level `target` object describes the
horizon (session count and kind); `specification.target` identifies the builder
and label source columns. Readers that check `schema_version` must accept
version 9. The dataset object includes a nullable `instrument` label;
CLI artifacts also include `resolved_configuration`. Execution metadata
now contains `executor`, `settings`, named `policies`, and runtime `facts`.
`min_train_size` belongs to execution settings, rather than shared experiment
bounds. Repeating an unchanged run produces byte-identical artifacts.
`evaluation.comparison_test` identifies the configured statistical test.
`comparisons.csv` uses generic `test_name` and `statistic` columns; resolved test
settings are JSON in `test_parameters`, replacing the DM-specific statistic and
HAC columns. Forecasts and loss-summary values are unchanged by this migration.

The artifact writer requires a completed runner result with its execution
snapshot and fit boundaries. It rejects configuration or strategy metadata
that differs from the completed run before creating output files.

Include the fixed robustness checks in the same experiment:

    python scripts/run_experiment.py \
        data/derived/EURUSD_daily_rv_5min.parquet \
        --min-train-size 200 \
        --forecast-start 2024-01-01 \
        --forecast-end 2024-03-31 \
        --output-dir outputs/final_evaluation \
        --robustness

This report checks Newey–West lags from zero through five, monthly losses,
session-level HAR win rates, and each model's five largest absolute log-RV
errors. Detailed CSV files and a combined Markdown report are written under
`outputs/final_evaluation/robustness`, alongside the main numerical results
in `outputs/final_evaluation`. Add `--plot` to produce the figure in that run too.

## Install

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

Run the offline quality checks with:

    pytest -m "not slow"
    ruff check src tests scripts
    mypy src tests

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
