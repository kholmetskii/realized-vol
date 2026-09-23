# Realized Volatility Project Plan

## 1. Project objective

Answer one focused research question:

> How predictable is next-day EUR/USD realized variance, and can competing
> forecasts be distinguished statistically when the evaluation target is
> itself an estimated, noisy quantity?

The finished project should demonstrate one complete chain of reasoning:

1. Intraday prices contain market-microstructure noise.
2. Different estimators respond differently to that noise and to jumps.
3. Daily realized variance is persistent enough to forecast.
4. Forecast rankings depend on a disciplined out-of-sample protocol and an
   appropriate loss function.

This is an empirical research project, not a trading strategy. It will not
claim profitable alpha or recommend trades.

## 2. Scope

### Included in version 1

- One market: EUR/USD quotes from Dukascopy.
- One-day-ahead variance forecasts.
- Estimators: naive RV, sparse/subsampled RV, two-scale RV, one realized
  kernel, realized quarticity, and bipower variation.
- Forecasts: historical mean, yesterday's RV, EWMA, HAR-RV, GARCH(1,1), and
  HARQ only if the preceding models are complete.
- Evaluation: MSE, QLIKE, loss tables, Diebold-Mariano comparisons, and
  Mincer-Zarnowitz diagnostics.
- A simulation study, a real-data study, automated tests, and a short paper.

### Explicitly deferred

- Machine learning and neural networks.
- Markov-switching and rough-volatility models.
- Options, implied volatility, and trading strategies.
- Multiple currencies or asset classes.
- Five- and 22-day forecast horizons.
- A large family of jump tests and HAR variants.
- Model Confidence Sets until there are enough defensible models to justify
  a multiple-comparison procedure.

Do not add a deferred item until version 1 is complete.

## 3. Current baseline — complete

The repository already has:

- A tested Dukascopy downloader and decoder.
- Heston-style simulation with observable simulated integrated variance.
- Separate IID log-price noise and bid-ask-bounce observation models.
- A reproducible Monte Carlo harness reporting estimator bias and RMSE.
- New York 17:00 FX session boundaries with daylight-saving handling.
- Realized variance and subsampled realized variance.
- Mid, bid, and ask volatility-signature plots.
- A paired fine-versus-coarse noise test.
- A written first empirical result and limitations.
- 34 passing tests.

Preserve these results as the first chapter of the final study. Do not rewrite
them merely to make the project appear new.

## 4. Working principles

1. **Learn immediately before implementing.** Each phase contains only the
   theory required for its code.
2. **Simulate before using real data.** An estimator must recover a known
   simulated quantity before it is trusted on EUR/USD.
3. **One canonical definition.** Variance scale, session boundary, target,
   annualisation convention, and forecast horizon must be defined once.
4. **Chronological evaluation only.** No random train/test splits.
5. **Freeze before testing.** The model list and refit protocol must be fixed
   before the final test period is evaluated.
6. **Baselines come first.** A sophisticated model has no value if it cannot
   improve on yesterday's RV or EWMA.
7. **Negative findings count.** “Models cannot be distinguished” is a valid
   result when supported by a sound experiment.

## 5. Milestones

### Milestone 1 — Time-series foundations

**Learn**

- Returns, log returns, variance, and volatility.
- Lags, rolling averages, autocorrelation, and stationarity.
- AR(1), linear regression, residuals, and one-step forecasts.
- Expanding versus rolling windows.
- Data leakage and chronological train/validation/test splits.

**Build**

- A small educational script or notebook that simulates white noise and
  AR(1) series.
- A walk-forward AR(1) forecast compared with a historical-mean forecast.
- Plots of each series and its autocorrelation function.

**Exit criteria**

- Explain why returns can be weakly autocorrelated while squared returns are
  strongly autocorrelated.
- Produce forecasts without using any future observations.
- Write a test that deliberately detects a shifted or leaking feature.

**Estimated effort:** 10–15 focused hours.

### Milestone 2 — Simulation harness and estimator contracts

**Learn**

- Integrated variance, quadratic variation, and jump variation.
- The observation model `observed price = efficient price + noise`.
- Bias, variance, consistency, and Monte Carlo error.

**Build**

- Separate efficient-price simulation from observation-noise simulation.
- Add configurable jumps.
- Add transaction-price bounce as a separate noise model from independent
  log-price noise.
- Run many replications rather than validating on one path.
- Define a common estimator interface and consistent units.

**Exit criteria**

- Naive RV approaches simulated integrated variance without noise or jumps.
- Tick-scale RV becomes increasingly biased as observation noise rises.
- Jump-contaminated quadratic variation equals continuous variation plus
  simulated squared jumps within numerical tolerance.
- Monte Carlo results include bias and RMSE with uncertainty, not a selected
  example path.

**Estimated effort:** 15–20 hours.

### Milestone 3 — Noise-robust and jump-robust estimators

Implement in this order:

1. Sparse RV on a declared sampling grid.
2. Existing subsampled RV, with stronger Monte Carlo tests.
3. Two-scale realized variance.
4. Realized quarticity.
5. One realized kernel with a documented bandwidth rule.
6. Bipower variation and nonnegative jump-variation estimate.

**Required comparisons**

- Clean diffusion.
- Low, medium, and high observation noise.
- Diffusion with jumps.
- Several sampling frequencies.
- Bias, RMSE, and runtime.

**Exit criteria**

- Every formula has a citation and a unit test.
- The noise-robust estimators improve on naive high-frequency RV when noise is
  sufficiently strong.
- Bipower variation is not presented as a formal jump test.
- Tests do not assume beforehand that five minutes is optimal.

**Estimated effort:** 20–30 hours.

### Milestone 4 — Reproducible daily EUR/USD dataset

**Data period**

Aim for at least five years of full FX sessions. Download and aggregate in
small batches so a failed request does not restart the entire job. Store the
compact daily feature table as the main reproducible input; raw tick files may
remain outside version control.

**Daily table**

One row per New York-close session, containing at minimum:

- Session date and observation count.
- Close-to-close log return.
- RV at selected frequencies.
- Subsampled RV.
- Two-scale RV.
- Realized kernel.
- Realized quarticity.
- Bipower and estimated jump variation.
- Median spread and data-quality flags.

**Cleaning report**

Record counts of:

- Raw and retained quotes.
- Duplicate and non-monotonic timestamps.
- Crossed or nonpositive quotes.
- Partial sessions.
- Missing intervals and abnormally wide spreads.
- Sessions removed, with reasons.

**Target decision**

Before forecasting, choose one primary daily variance proxy using theory and
the simulation evidence—not forecast performance. A practical initial choice
is five-minute subsampled RV, with realized kernel results as a robustness
check. Document the decision and freeze it.

**Exit criteria**

- Rebuilding the daily table is a single documented command.
- Repeated runs produce identical results from the same raw data.
- Every retained session has passed explicit quality checks.
- No raw-data or generated-output directory is accidentally committed.

**Estimated effort:** 15–25 hours, excluding unattended download time.

### Milestone 5 — Stylized facts and forecast design

**Analyze before forecasting**

- Daily returns, squared returns, RV, and log RV.
- Histograms and summary statistics.
- Autocorrelation functions.
- Volatility clustering and persistence.
- Continuous and jump components.

**Predeclare the experiment**

- Training, validation, and untouched final-test dates.
- Expanding or rolling estimation window.
- Refit frequency.
- Exact feature definitions.
- Primary and secondary losses.
- Rules for positivity and missing forecasts.
- Model list.

Use the validation period for debugging and limited specification decisions.
Do not inspect the final loss table until the experiment is frozen.

**Exit criteria**

- The design is saved in a configuration file or protocol document.
- Every feature used for day `t+1` is provably available by the end of day
  `t`.
- The final test period remains untouched.

**Estimated effort:** 8–12 hours.

### Milestone 6 — Forecasting models

Implement in increasing complexity:

1. Historical mean.
2. Yesterday's RV.
3. EWMA.
4. HAR-RV on log variance.
5. GARCH(1,1) on daily returns.
6. HARQ, only after realized quarticity and HAR-RV are validated.

Each model must expose the same walk-forward forecasting interface and return
a strictly positive variance forecast with a documented reason for how
positivity is achieved.

For log-HAR, test and document the level retransformation method. Fit GARCH
only with information that would have existed at the forecast origin.

**Exit criteria**

- All models forecast exactly the same dates.
- Forecast arrays contain no silent gaps, infinities, or nonpositive values.
- A synthetic AR/HAR process test verifies correct lag alignment.
- Coefficients and forecasts are reproducible.

**Estimated effort:** 15–25 hours.

### Milestone 7 — Forecast evaluation

**Implement**

- MSE on the variance scale.
- QLIKE on the variance scale as the primary loss.
- Mean loss and improvement relative to yesterday's RV.
- Diebold-Mariano comparisons with a justified long-run variance estimator.
- Mincer-Zarnowitz regressions with HAC standard errors.
- Optional bootstrap confidence intervals as a robustness check.

**Report**

- Point estimates and uncertainty.
- Statistical significance and effect size.
- Performance over time, not just a single average.
- Sensitivity to the alternative realized-variance proxy.
- Crisis/high-volatility and ordinary periods, if the sample supports the
  distinction without post-hoc cherry-picking.

**Exit criteria**

- A model is never called “better” solely because its average loss is lower.
- QLIKE inputs and orientation are verified with hand-calculated tests.
- Nested-model caveats and overlapping-loss issues are stated where relevant.
- Running one script regenerates the final tables.

**Estimated effort:** 15–20 hours.

### Milestone 8 — Paper and release

Write a short paper of roughly 10–15 pages:

1. Research question and contribution.
2. Data and cleaning.
3. Volatility measurement and simulation evidence.
4. Forecast models and experimental protocol.
5. Out-of-sample results.
6. Robustness checks.
7. Limitations and conclusion.

Minimum figures and tables:

- Mid/bid/ask signature plots.
- Simulation bias/RMSE comparison.
- RV and log-RV time series plus autocorrelation.
- Forecast performance through time.
- Estimator comparison table.
- Out-of-sample loss and statistical-comparison table.

Before release:

- Run tests, linting, and type checking.
- Reproduce every figure and table from documented commands.
- Pin or record the environment.
- Add data provenance and licensing notes.
- Ensure README claims match generated evidence.
- Tag version 1.0 only when the paper and reproducibility instructions work
  from a clean environment.

**Estimated effort:** 12–18 hours.

## 6. Proposed package structure

```text
src/rvol/
    data/
        dukascopy.py
        cleaning.py
        daily.py
    simulation/
        heston.py
        noise.py
        jumps.py
    estimators/
        realized.py
        two_scale.py
        kernel.py
        jumps.py
        signature.py
    models/
        baselines.py
        har.py
        garch.py
        harq.py
    evaluation/
        losses.py
        backtest.py
        dm.py
        diagnostics.py
scripts/
    01_download.py
    02_build_daily.py
    03_simulation_study.py
    04_stylized_facts.py
    05_forecast.py
    06_evaluate.py
    07_build_report.py
configs/
    experiment_v1.toml
tests/
```

Do not reorganize working code merely to match this tree. Move modules only
when the corresponding phase creates a genuine need.

## 7. Main risks and controls

| Risk | Control |
|---|---|
| Learning scope becomes overwhelming | Learn one phase at a time; defer unrelated finance topics |
| Tick downloads consume the project | Download in resumable batches and retain a compact daily table |
| Data leakage | Central walk-forward engine and explicit lag-alignment tests |
| Final test becomes another validation set | Freeze protocol and generate final results once |
| Too many models | Keep the six-model cap for version 1 |
| Estimator code appears correct but is not | Monte Carlo recovery tests against latent simulated quantities |
| Results depend on one proxy | Predeclare a primary proxy and one robustness proxy |
| Statistical tests are overinterpreted | Report effect sizes, uncertainty, assumptions, and power limitations |
| Project never reaches a written result | Update the paper at the end of every milestone |

## 8. Definition of done

Version 1 is complete when another person can:

1. Install the package from documented instructions.
2. Rebuild or obtain the cleaned daily dataset with documented provenance.
3. Run the simulation study and estimator validation.
4. Reproduce all forecast results without future information leakage.
5. Recreate every figure and table in the paper.
6. Understand which conclusions are supported, which are uncertain, and
   which questions were intentionally left for later work.

Expected remaining effort is approximately 110–165 focused hours. At ten
hours per week, that is roughly three to four months. The schedule should be
driven by milestone exit criteria rather than calendar deadlines.

## 9. Immediate next actions

1. Add configurable jumps and verify that quadratic variation recovers
   continuous variation plus squared jumps.
2. Expand the Monte Carlo scenarios to clean, IID-noise, bid-ask-bounce, and
   jump-contaminated paths.
3. Implement two-scale realized variance with synthetic recovery tests.
4. Complete the Milestone 1 AR(1) learning exercise before forecasting begins.
5. Keep the README's current empirical section intact while the next result
   is developed.
