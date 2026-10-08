import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rvol.diagnostics.microstructure import NoiseTest  # noqa: E402
from rvol.domain import ExperimentResult, ForecastRecord  # noqa: E402
from rvol.evaluation import EvaluationResult, LossRecord, ModelLossSummary  # noqa: E402
from rvol.plotting.daily import plot_daily_realized_variance  # noqa: E402
from rvol.plotting.diagnostics import (  # noqa: E402
    plot_diagnostics_overview,
    plot_volatility_signature,
)
from rvol.plotting.forecasting import plot_forecast_evaluation  # noqa: E402
from rvol.plotting.microstructure import plot_bid_ask_bounce  # noqa: E402
from rvol.plotting.returns import (  # noqa: E402
    plot_cumulative_iv_vs_rv,
    plot_returns_and_rv,
)
from rvol.plotting.sampling import plot_sampling_grids  # noqa: E402
from rvol.plotting.simulation import (  # noqa: E402
    plot_heston_path,
    plot_simulation_overview,
)
from rvol.simulation.monte_carlo import MonteCarloSummary  # noqa: E402


def sample_signature(scale: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "seconds": [1.0, 60.0, 300.0],
        "ann_vol_pct": np.array([6.0, 5.8, 5.6]) * scale,
        "ann_vol_se": [0.2, 0.2, 0.25],
    })


def sample_noise(ratio: float) -> NoiseTest:
    return NoiseTest(
        fine="1s",
        coarse="5min",
        n_days=20,
        mean_ratio=ratio,
        se_log_ratio=0.02,
        t_stat=3.0,
        p_value=0.003,
        implied_noise_bps=0.04,
    )


def sample_summary(name: str, bias: float, rmse: float) -> MonteCarloSummary:
    return MonteCarloSummary(
        estimator=name,
        n_replications=20,
        mean_estimate=0.01 + bias,
        mean_target=0.01,
        bias=bias,
        bias_se=0.001,
        rmse=rmse,
    )


def test_diagnostic_plots_build_expected_panels():
    signatures = {
        "mid": sample_signature(),
        "bid": sample_signature(1.02),
    }
    single = plot_volatility_signature(signatures)
    overview = plot_diagnostics_overview(
        signatures,
        {"mid": sample_noise(1.08), "bid": sample_noise(1.18)},
    )

    assert len(single.axes) == 1
    assert len(overview.axes) == 2
    plt.close(single)
    plt.close(overview)


def test_sampling_plot_builds_path_and_grid_panels():
    prices = 100 * np.exp(np.cumsum(np.r_[0.0, np.full(39, 0.001)]))
    figure = plot_sampling_grids(prices, step=4)

    assert len(figure.axes) == 2
    plt.close(figure)


def test_returns_plot_builds_four_transformation_panels():
    prices = 100 * np.exp(np.cumsum(np.r_[0.0, np.full(39, 0.001)]))
    figure = plot_returns_and_rv(prices)

    assert len(figure.axes) == 4
    plt.close(figure)


def test_cumulative_iv_vs_rv_plot_builds_comparison_and_error_panels():
    prices = 100 * np.exp(np.cumsum(np.r_[0.0, np.full(39, 0.001)]))
    variance = np.full(40, 0.04)
    figure = plot_cumulative_iv_vs_rv(prices, variance, dt=1 / 252 / 39)

    assert len(figure.axes) == 2
    plt.close(figure)


def test_bid_ask_bounce_plot_builds_quotes_trades_and_returns_panels():
    efficient = np.linspace(100.0, 100.1, 40)
    bid = efficient * np.exp(-0.0002)
    ask = efficient * np.exp(0.0002)
    mid = 0.5 * (bid + ask)
    trades = np.where(np.arange(40) % 2 == 0, bid, ask)
    figure = plot_bid_ask_bounce(efficient, bid, ask, mid, trades)

    assert len(figure.axes) == 3
    plt.close(figure)


def test_daily_rv_plot_builds_price_and_variance_panels():
    closes = np.linspace(100.0, 102.0, 40)
    daily_rv = np.linspace(0.0001, 0.0003, 40)
    figure = plot_daily_realized_variance(closes, daily_rv, rolling_window=10)

    assert len(figure.axes) == 2
    plt.close(figure)


def test_simulation_plot_builds_four_panels():
    prices = np.linspace(100.0, 101.0, 50)
    scenarios = {
        "Clean": (
            sample_summary("Naive RV", 0.0, 0.001),
            sample_summary("Subsampled RV", 0.0001, 0.0015),
        ),
        "Noisy": (
            sample_summary("Naive RV", 0.005, 0.006),
            sample_summary("Subsampled RV", 0.0002, 0.0016),
        ),
    }
    figure = plot_simulation_overview(
        prices,
        prices * 1.0001,
        np.linspace(0.03, 0.05, 50),
        scenarios,
    )

    assert len(figure.axes) == 4
    plt.close(figure)


def test_heston_plot_builds_price_variance_and_integrated_variance_panels():
    prices = np.linspace(100.0, 101.0, 50)
    variance = np.linspace(0.03, 0.05, 50)
    figure = plot_heston_path(prices, variance, dt=1 / 252 / 49, theta=0.04)

    assert len(figure.axes) == 3
    plt.close(figure)


def test_forecast_evaluation_plot_builds_paths_losses_and_advantage_panels():
    dates = pd.bdate_range("2024-01-02", periods=6)
    actual = np.array([-10.0, -9.7, -10.2, -9.8, -10.1, -9.6])
    errors = {
        "naive": np.array([0.4, -0.3, 0.5, -0.2, 0.35, -0.4]),
        "AR1": np.array([0.3, -0.2, 0.35, -0.15, 0.25, -0.3]),
        "HAR": np.array([0.1, -0.1, 0.15, -0.05, 0.1, -0.12]),
    }
    records = []
    losses = []
    summaries = []
    for model, model_errors in errors.items():
        model_losses = np.square(model_errors)
        summaries.append(ModelLossSummary(
            model=model,
            metric="QLIKE",
            n_obs=len(dates),
            mean_loss=float(np.mean(model_losses)),
        ))
        for index, target_date in enumerate(dates):
            prediction = actual[index] + model_errors[index]
            records.append(ForecastRecord(
                model=model,
                origin_date=(target_date - pd.Timedelta(days=1)).date(),
                target_date=target_date.date(),
                n_train=200 + index,
                actual_log_rv=float(actual[index]),
                predicted_log_rv=float(prediction),
            ))
            losses.append(LossRecord(
                model=model,
                metric="QLIKE",
                target_date=target_date.date(),
                loss=float(model_losses[index]),
            ))

    figure = plot_forecast_evaluation(
        ExperimentResult(records=tuple(records)),
        EvaluationResult(
            losses=tuple(losses),
            summaries=tuple(summaries),
            comparisons=(),
        ),
    )

    assert len(figure.axes) == 3
    assert figure.axes[0].get_title() == "Forecast paths"
    assert len(figure.axes[1].patches) == 3
    assert len(figure.axes[2].lines) == 3
    assert "HAR-RV" in figure.axes[0].get_legend_handles_labels()[1]
    plt.close(figure)
