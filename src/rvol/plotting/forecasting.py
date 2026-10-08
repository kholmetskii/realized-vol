"""Plots for out-of-sample volatility forecast evaluation."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from rvol.domain import ExperimentResult
from rvol.evaluation import EvaluationResult

MODEL_STYLES: dict[str, dict[str, object]] = {
    "historical_mean": {"color": "0.55", "linestyle": ":"},
    "naive": {"color": "tab:orange", "linestyle": "--"},
    "EWMA": {"color": "tab:purple", "linestyle": (0, (3, 1, 1, 1))},
    "AR1": {"color": "tab:blue", "linestyle": "-."},
    "HAR": {"color": "tab:green", "linestyle": "-"},
}
MODEL_LABELS = {
    "historical_mean": "Historical mean",
    "naive": "Naïve",
    "EWMA": "EWMA",
    "AR1": "AR(1)",
    "HAR": "HAR-RV",
}


def _model_order(models: set[str]) -> list[str]:
    preferred = ("historical_mean", "naive", "EWMA", "AR1", "HAR")
    return [model for model in preferred if model in models] + sorted(models.difference(preferred))


def plot_forecast_evaluation(
    experiment: ExperimentResult,
    evaluation: EvaluationResult,
    *,
    candidate_model: str = "HAR",
    metric: str = "QLIKE",
    title: str = "Out-of-sample realized-volatility forecasts",
) -> Figure:
    """Plot forecast paths, mean losses, and cumulative candidate advantage."""
    if not experiment.records:
        raise ValueError("experiment must contain forecast records")

    records_by_model = defaultdict(list)
    for record in experiment.records:
        records_by_model[record.model].append(record)
    for records in records_by_model.values():
        records.sort(key=lambda record: record.target_date)

    models = set(records_by_model)
    if candidate_model not in models:
        raise ValueError(f"unknown candidate model: {candidate_model}")
    ordered_models = _model_order(models)
    reference = records_by_model[candidate_model]
    target_dates = [record.target_date for record in reference]
    plot_dates = np.asarray(target_dates, dtype="datetime64[D]")
    actual = np.array([record.actual_log_rv for record in reference])
    for model, records in records_by_model.items():
        if [record.target_date for record in records] != target_dates:
            raise ValueError(f"forecast dates are not aligned for {model}")
        if not np.allclose([record.actual_log_rv for record in records], actual):
            raise ValueError(f"actual values disagree for {model}")

    summaries = {
        summary.model: summary
        for summary in evaluation.summaries
        if summary.metric == metric
    }
    if models.difference(summaries):
        raise ValueError(f"evaluation does not contain {metric} summaries for every model")

    losses_by_model: defaultdict[str, dict[date, float]] = defaultdict(dict)
    for loss in evaluation.losses:
        if loss.metric == metric:
            losses_by_model[loss.model][loss.target_date] = loss.loss
    candidate_losses = losses_by_model[candidate_model]
    if set(candidate_losses) != set(target_dates):
        raise ValueError(f"evaluation does not contain aligned {metric} losses")

    figure = plt.figure(figsize=(12.0, 8.2))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.34, wspace=0.28)
    forecast_axis = figure.add_subplot(grid[0, :])
    loss_axis = figure.add_subplot(grid[1, 0])
    cumulative_axis = figure.add_subplot(grid[1, 1])

    forecast_axis.plot(
        plot_dates,
        actual,
        color="black",
        linewidth=2.2,
        label="Actual log RV",
        zorder=5,
    )
    for model in ordered_models:
        records = records_by_model[model]
        predictions = [record.predicted_log_rv for record in records]
        style = MODEL_STYLES.get(model, {"color": None, "linestyle": "-"})
        forecast_axis.plot(
            plot_dates,
            predictions,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0 if model == candidate_model else 1.25,
            alpha=1.0 if model == candidate_model else 0.78,
            label=MODEL_LABELS.get(model, model),
        )
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    forecast_axis.xaxis.set_major_locator(locator)
    forecast_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator, show_offset=False)
    )
    forecast_axis.set_ylabel("Log realized variance")
    forecast_axis.set_title("Forecast paths")
    forecast_axis.grid(alpha=0.22)
    forecast_axis.legend(frameon=False, ncols=len(ordered_models) + 1, loc="upper center")

    mean_losses = [summaries[model].mean_loss for model in ordered_models]
    bar_colors = [
        MODEL_STYLES.get(model, {"color": "0.6"})["color"]
        for model in ordered_models
    ]
    model_labels = [MODEL_LABELS.get(model, model) for model in ordered_models]
    bars = loss_axis.bar(model_labels, mean_losses, color=bar_colors, alpha=0.85)
    loss_axis.set_ylabel(f"Mean {metric}")
    loss_axis.set_title("Average forecast loss — lower is better")
    loss_axis.grid(axis="y", alpha=0.22)
    loss_axis.tick_params(axis="x", rotation=18)
    loss_axis.set_ylim(0, max(mean_losses) * 1.2)
    for bar, value in zip(bars, mean_losses, strict=True):
        loss_axis.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    cumulative_axis.axhline(0, color="0.35", linewidth=0.9)
    for baseline in ordered_models:
        if baseline == candidate_model:
            continue
        baseline_losses = losses_by_model[baseline]
        if set(baseline_losses) != set(target_dates):
            raise ValueError(f"evaluation does not contain aligned {metric} losses")
        differences = np.array([
            baseline_losses[target_date] - candidate_losses[target_date]
            for target_date in target_dates
        ])
        style = MODEL_STYLES.get(baseline, {"color": None})
        cumulative_axis.plot(
            plot_dates,
            np.cumsum(differences),
            color=style["color"],
            linewidth=1.8,
            label=(
                f"{MODEL_LABELS.get(baseline, baseline)} − "
                f"{MODEL_LABELS.get(candidate_model, candidate_model)}"
            ),
        )
    cumulative_locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    cumulative_axis.xaxis.set_major_locator(cumulative_locator)
    cumulative_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(cumulative_locator, show_offset=False)
    )
    cumulative_axis.set_ylabel(f"Cumulative {metric} advantage")
    candidate_label = MODEL_LABELS.get(candidate_model, candidate_model)
    cumulative_axis.set_title(f"Positive values favour {candidate_label}")
    cumulative_axis.grid(alpha=0.22)
    cumulative_axis.legend(frameon=False, fontsize=9)

    figure.suptitle(title, fontsize=14)
    return figure
