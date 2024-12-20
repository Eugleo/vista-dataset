from typing import Optional

import numpy as np
import plotly.express as px
import polars as pl
import sklearn.metrics as skm
from lets_plot import *
from lets_plot.plot.core import PlotSpec
from polars import col as c
from scipy.stats import beta
from sklearn.preprocessing import LabelBinarizer


def levels_line_plot(data: pl.DataFrame) -> PlotSpec:
    """Data should have columns: model, level, score"""

    data = (
        data.with_columns(
            ci_high=c("score") + 1.96 * c("error"),
            ci_low=c("score") - 1.96 * c("error"),
        )
        .sort("model", "level")
        .with_columns(level=c("level").str.replace("Level ", ""))
    )

    chart = (
        ggplot(
            data,
            aes(x="level", y="score", color="model", linetype="model"),
        )
        + geom_line(size=1.5)
        + geom_ribbon(
            aes(ymin="ci_low", ymax="ci_high", fill="model"),
            alpha=0.2,
            linetype=0,
            show_legend=False,
        )
        + labs(
            x="Level",
            y="Mean Macro F1",
            title="Performance by level",
            color="Model",
            linetype="Model",
        )
        + ylim(0, 1.1)
        + theme(
            axis_text=element_text(size=15),
            axis_title=element_text(size=18),
            text=element_text(size=18),
            plot_title=element_text(size=20),
            legend_title=element_text(size=18, margin=(0, 50, 0, 0)),
            legend_position="bottom",
        )
        + ggsize(width=500, height=300)
    )

    assert isinstance(chart, PlotSpec)

    return chart


def grouped_levels_line_plot(data: pl.DataFrame) -> PlotSpec:
    data = (
        data.with_columns(
            ci_high=c("score") + 1.96 * c("error"),
            ci_low=c("score") - 1.96 * c("error"),
        )
        .sort("model", "level")
        .with_columns(level=c("level").str.replace("Level ", ""))
    )

    chart = (
        ggplot(
            data,
            aes(x="level", y="score", color="model", linetype="model"),
        )
        + geom_line(size=1.5)
        + geom_ribbon(
            aes(ymin="ci_low", ymax="ci_high", fill="model"),
            alpha=0.2,
            linetype=0,
            show_legend=False,
        )
        + labs(
            x="Level",
            y="Mean Macro F1",
            title="Performance by level",
            color="Model",
            linetype="Model",
        )
        + facet_wrap("group", ncol=2)
        + ylim(0, 1.1)
        + theme(
            axis_text=element_text(size=15),
            axis_title=element_text(size=18),
            text=element_text(size=18),
            plot_title=element_text(size=20),
            legend_title=element_text(size=18, margin=(0, 50, 0, 0)),
        )
        + ggsize(width=1000, height=250)
    )

    assert isinstance(chart, PlotSpec)

    return chart


def clip_levels_line_plot(data: pl.DataFrame) -> PlotSpec:
    data = (
        data.with_columns(
            ci_high=c("score") + 1.96 * c("error"),
            ci_low=c("score") - 1.96 * c("error"),
        )
        .sort("model", "level")
        .with_columns(level=c("level").str.replace("Level ", ""))
    )

    chart = (
        ggplot(
            data,
            aes(x="level", y="score", color="model", shape="model"),
        )
        + geom_line(size=1.5)
        + geom_point(size=4)
        + geom_errorbar(
            aes(ymin="ci_low", ymax="ci_high"),
            width=0,
            size=1,
        )
        + labs(
            x="Level",
            y="Mean Macro F1",
            title="Performance by level",
            color="Model",
            shape="Model",
        )
        + ylim(0, 1.1)
        + theme(
            axis_text=element_text(size=15),
            axis_title=element_text(size=18),
            text=element_text(size=18),
            plot_title=element_text(size=20),
            legend_title=element_text(size=18, margin=(0, 50, 0, 0)),
        )
        + ggsize(width=1000, height=250)
        + facet_wrap("group", ncol=2)
        + scale_color_manual(
            {
                "CLIP (2)": px.colors.sequential.Teal[2],
                "CLIP (4)": px.colors.sequential.Teal[3],
                "CLIP (8)": px.colors.sequential.Teal[4],
                "CLIP (16)": px.colors.sequential.Teal[5],
                "CLIP (32)": px.colors.sequential.Teal[6],
                "ViCLIP (8)": "#994EA4",
            }
        )
    )

    assert isinstance(chart, PlotSpec)

    return chart


def task_groups_bar_plot(
    data: pl.DataFrame, ncol: int = 3, color: str = "environment"
) -> PlotSpec:
    data = data.with_columns(
        ci_high=c("score") + 1.96 * c("error"),
        ci_low=c("score") - 1.96 * c("error"),
    ).sort("model", "group")

    chart = (
        ggplot(
            data,
            aes(
                x="model",
                y="score",
                fill=color,
                color=color,
                group=color,
                # linetype="environment",
            ),
        )
        + geom_bar(
            stat="identity",
            position=position_dodge(width=0.8),
            width=0.8,
            alpha=0.5,
            show_legend=False,
        )
        + geom_errorbar(
            aes(ymin="ci_low", ymax="ci_high"),
            width=0.8,
            size=1.5,
            position=position_dodge(width=0.8),
        )
        + facet_wrap("group", ncol=ncol)
        + labs(
            group="Task Group",
            y="Mean Macro F1",
            title="Performance in level 1 by group and env",
            linetype=color.capitalize(),
            color=color.capitalize(),
        )
        + theme(
            axis_title_x=element_blank(),
            axis_text=element_text(size=15),
            axis_title=element_text(size=18),
            text=element_text(size=18),
            plot_title=element_text(size=20),
            legend_title=element_text(size=18, margin=(0, 50, 0, 0)),
            legend_position="bottom",
        )
        + ggsize(width=500, height=300)
    )

    assert isinstance(chart, PlotSpec)

    return chart


def average_precision(task_labels: dict, group):
    labels = task_labels[group.struct.field("task")[0]]
    lb = LabelBinarizer()
    lb.fit(labels)

    true_labels = group.struct.field("true_label").to_numpy()
    scores = group.struct.field("score").to_numpy()

    n_samples = len(true_labels) // len(lb.classes_)
    try:
        y_true = lb.transform(
            true_labels.reshape(n_samples, len(lb.classes_))[:n_samples, 0]
        )
        if len(lb.classes_) == 2:
            # LabelBinarizer returns a 1-column array in this case
            y_true = np.concatenate([1 - y_true, y_true], axis=1)  # type: ignore
        y_score = scores.reshape(n_samples, len(lb.classes_))

        # print(group.struct.field("task")[0], task_labels[group.struct.field("task")[0]])
        # print(f"{y_true=}")
        # print(f"{y_score=}")
        # print(f"{group=}")
        # print()

        return skm.average_precision_score(y_true=y_true, y_score=y_score, average=None)
    except Exception:
        pass


def map_plot(per_label_map: pl.DataFrame, title: str):
    per_task_map = (
        per_label_map.group_by("task", "model")
        .agg(mAP=c("AP").mean(), error=c("AP").std() / c("AP").len().sqrt())
        .sort("task", "model", descending=[False, True])
    )
    fig = px.bar(
        per_task_map.to_pandas(),
        x="mAP",
        y="task",
        color="model",
        title=title,
        barmode="group",
        error_x="error",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(
        width=800,
        height=100
        * (1 + per_task_map["model"].n_unique() + per_task_map["task"].n_unique()),
    )

    return fig


def task_performance(
    metric_per_task: pl.DataFrame,
    predictions: pl.DataFrame,
    tasks: list,
    task_labels: dict,
    title: str,
    baseline_per_task: Optional[dict] = None,
):
    models = metric_per_task["model"].unique().sort().to_list()
    avg_data = (
        metric_per_task.group_by("model")
        .agg(c("metric").mean())
        .with_columns(task=pl.lit("average"))
        .select(["task", "model", "metric"])
    )
    metric_per_task = pl.concat([metric_per_task, avg_data])
    tasks.append("average")
    tasks = sorted(tasks)
    metric_per_task = metric_per_task.sort("model")

    nrows = 1 + len(models)
    ncols = len(tasks)

    model_titles = [model for model in models for _ in range(ncols)]
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[t.removeprefix(title) for t in tasks] + model_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
    )

    for task_idx, task in enumerate(tasks, start=1):
        subplot = px.bar(
            metric_per_task.filter(c("task") == task).to_pandas(),
            x="model",
            color="model",
            y="metric",
            range_y=[0, 1],
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        subplot.update_layout(showlegend=False)
        fig.add_traces(subplot["data"], rows=1, cols=task_idx)
        if task != "average":
            if baseline_per_task:
                fig.add_hline(y=baseline_per_task[task], line_dash="dot", col=task_idx)  # type: ignore
            for model_idx, model in enumerate(models, start=2):
                task_df = predictions.filter(c("task") == task, c("model") == model)
                labels = task_labels[task]
                matrix = skm.confusion_matrix(
                    task_df["true_label"].to_numpy(),
                    task_df["label"].to_numpy(),
                    labels=labels,
                )

                heatmap = px.imshow(
                    matrix,
                    x=labels,
                    y=labels,
                    labels=dict(x="Candidate Label", y="Video True Label"),
                    text_auto=True,
                )
                heatmap.update_layout(showlegend=False)

                if model_idx < len(models) + 1:
                    fig.update_xaxes(showticklabels=False, row=model_idx, col=task_idx)
                fig.add_traces(heatmap["data"], rows=model_idx, cols=task_idx)

    fig.update_layout(
        coloraxis_showscale=False,
        width=500 * ncols,
        height=330 * nrows,
        showlegend=False,
        title=title,
        coloraxis_colorscale=px.colors.sequential.Viridis_r,
    )
    fig.update_yaxes(range=[0, 1], row=1)
    return fig


def overall_performance(
    metric_per_task: pl.DataFrame, y_label: str, title: str, baseline_per_task: dict
):
    ERROR_MODE = "std"

    if ERROR_MODE == "std":
        avg_data = (
            metric_per_task.group_by("model")
            .agg(
                pl.col("metric").mean(), error=pl.col("metric").std() / pl.len().sqrt()
            )
            .with_columns(task=pl.lit("average"))
            .sort("model")
        )
    # elif ERROR_MODE == "bayesian":
    #     avg_data = (
    #         metric_per_task.group_by("model")
    #         .agg(
    #             pl.col("metric").mean(),
    #             error_low=pl.col("metric").apply(
    #                 lambda x: bayesian_confidence_low(x, confidence=0.682),
    #                 return_dtype=pl.Float64,
    #             ),
    #             error_high_minus_mean=pl.col("metric").apply(
    #                 lambda x: bayesian_confidence_high_minus_mean(x, confidence=0.682),
    #                 return_dtype=pl.Float64,
    #             ),
    #         )
    #         .with_columns(task=pl.lit("average"))
    #         .sort("model")
    #     )
    else:
        raise ValueError(f"Unknown error mode: {ERROR_MODE}")

    fig = px.bar(
        avg_data.to_pandas(),
        x="model",
        y="metric",
        color="model",
        range_y=[0, 1],
        range_color=[0, 1],
        color_discrete_map={
            "clip": "#22D3EE",
            "gpt-4o": "#059669",
            "viclip": "#A78BFA",
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
        title=title,
        labels={"metric": y_label},
        error_y="error" if ERROR_MODE == "std" else "error_high_minus_mean",
        error_y_minus=None if ERROR_MODE == "std" else "error_low",
    )
    fig.add_hline(
        y=pl.Series(baseline_per_task.values()).mean(),
        line_dash="dot",
        line_color="black",
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        width=220,
        height=330,
    )
    return fig


def overall_performance_clip(
    metric_per_task: pl.DataFrame, y_label: str, title: str, baseline_per_task: dict
):
    ERROR_MODE = "std"

    metric_per_task = metric_per_task.with_columns(
        group=pl.when(c("task").str.starts_with("foundation/objects"))
        .then(pl.lit("Objects"))
        .when(c("task").str.starts_with("foundation/containers"))
        .then(pl.lit("Containers"))
        .when(c("task").str.starts_with("foundation/pick_v_put"))
        .then(pl.lit("Picking"))
        .when(c("task").str.starts_with("foundation/slice/"))
        .then(pl.lit("Slicing"))
        .when(c("task").str.starts_with("foundation/toggle"))
        .then(pl.lit("Toggling"))
        .when(c("task").str.starts_with("foundation/clean"))
        .then(pl.lit("Cleaning"))
        .when(c("task").str.starts_with("foundation/heat"))
        .then(pl.lit("Heating"))
        .when(c("task").str.starts_with("foundation/cool"))
        .then(pl.lit("Cooling"))
        .when(c("task").str.starts_with("foundation/on_v_off"))
        .then(pl.lit("On/Off?"))
        .when(c("task").str.starts_with("foundation/sliced_v_whole"))
        .then(pl.lit("Sliced?"))
        .otherwise(pl.lit("Other"))
    )

    # metric_per_task = metric_per_task.with_columns(
    #     group=pl.when(c("task").str.starts_with("foundation/objects"))
    #     .then(pl.lit("Object"))
    #     .when(c("task").str.starts_with("foundation/containers"))
    #     .then(pl.lit("Object"))
    #     .when(c("task").str.starts_with("foundation/pick_v_put"))
    #     .then(pl.lit("Action"))
    #     .when(c("task").str.starts_with("foundation/slice/"))
    #     .then(pl.lit("Action"))
    #     .when(c("task").str.starts_with("foundation/toggle"))
    #     .then(pl.lit("Action"))
    #     .when(c("task").str.starts_with("foundation/clean"))
    #     .then(pl.lit("Action"))
    #     .when(c("task").str.starts_with("foundation/heat"))
    #     .then(pl.lit("Action"))
    #     .when(c("task").str.starts_with("foundation/cool"))
    #     .then(pl.lit("Action"))
    #     .when(c("task").str.starts_with("foundation/on_v_off"))
    #     .then(pl.lit("Object state"))
    #     .when(c("task").str.starts_with("foundation/sliced_v_whole"))
    #     .then(pl.lit("Object state"))
    #     .otherwise(pl.lit("Object state"))
    # )

    if ERROR_MODE == "std":
        avg_data = metric_per_task.group_by("model", "group").agg(
            c("metric").mean(),
            error=pl.col("metric").std() / pl.len().sqrt(),
        )
    elif ERROR_MODE == "bayesian":
        avg_data = (
            metric_per_task.group_by("model")
            .agg(
                pl.col("metric").mean(),
                error_low=pl.col("metric").apply(
                    lambda x: bayesian_confidence_low(x, confidence=0.682),
                    return_dtype=pl.Float64,
                ),
                error_high_minus_mean=pl.col("metric").apply(
                    lambda x: bayesian_confidence_high_minus_mean(x, confidence=0.682),
                    return_dtype=pl.Float64,
                ),
            )
            .with_columns(task=pl.lit("average"))
            .sort("model")
        )
    else:
        raise ValueError(f"Unknown error mode: {ERROR_MODE}")

    avg_data = avg_data.with_columns(
        n_frames=pl.when(c("model") == "viclip")
        .then(pl.lit("8"))
        .when(c("model") == "gpt-4o")
        .then(pl.lit("16"))
        .when(c("model").str.starts_with("clip"))
        .then(c("model").str.split("-").list.get(1))
        .cast(pl.Int32),
    ).sort("n_frames", "model", "group")

    fig = px.bar(
        avg_data,
        x="model",
        y="metric",
        color="model",
        range_y=[0, 1],
        range_color=[0, 1],
        color_discrete_map={
            f"clip-{n}": px.colors.sequential.Teal[i + 2]
            for i, n in enumerate([2, 4, 8, 16, 32])
        }
        | {"clip": "#06B6D4", "gpt-4o": "#16A34A", "viclip": "#A78BFA"},
        error_y="error" if ERROR_MODE == "std" else "error_high_minus_mean",
        error_y_minus=None if ERROR_MODE == "std" else "error_low",
        facet_col="group",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(title_text=None)

    # fig.add_hline(
    #     y=pl.Series(baseline_per_task.values()).mean(),
    #     line_dash="dot",
    #     line_color="black",
    # )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        # width=250 + 60 * metric_per_task["group"].n_unique(),
        width=200 + 60 * metric_per_task["group"].n_unique(),
        height=300,
        yaxis=dict(title=y_label),
    )
    return fig


def incorrect_video_labels(predictions: pl.DataFrame):
    incorrect_count = (
        predictions.filter(c("label") != c("true_label"))
        .group_by("video", "task")
        .agg(c("model").n_unique().alias("count"))
    )

    df = incorrect_count.sort("count", descending=True)

    return df


def bayesian_confidence_low(col, confidence):
    # to find the confidence interval, we need to find upper and lower bounds with
    # confidence equal to 1 - (1 - confidence) / 2 (i.e. half the probability of error each,
    # so that combining them will yield the correct confidence).
    # so, we need to solve for b in
    # (n + 1) Beta(b, 1+k, 1-k+n) (n choose k) = (1 - confidence) / 2         (lower bound)
    # (n + 1) Beta(b, 1+k, 1-k+n) (n choose k) = 1 - (1 - confidence) / 2     (upper bound)
    # which is what we do below.
    return bayesian_quantile(col, (1 - confidence) / 2)


def bayesian_confidence_high_minus_mean(col, confidence):
    return bayesian_quantile(col, 1 - (1 - confidence) / 2) - col.mean()


def bayesian_quantile(col, quantile):
    """Assuming a uniform prior on [0,1] for the accuracy, do a Bayesian update based on the measurements in col and return the desired quantile for the resulting distribution of accuracy"""
    # for this to work correctly, col must be boolean, i.e. all values must be 0 or 1
    # assert all(col.is_in([0, 1])), f"col must be boolean. col: {col}"
    n = len(col)
    k = sum(col)  # number correct
    # the posterior distribution P(acc | data) is P(data | acc) P(acc) / P(data)
    # P(data | acc) is just the binomial distribution acc^k (1-acc)^{n-k} (n choose k)
    # P(acc) is constant since the prior was uniform.
    # P(data) depends only on n since \int_0^1 acc^k (1-acc)^{n-k} (n choose k) dacc = 1/(n+1).
    # (this makes sense since there are n+1 possible values of k: 0, 1, ..., n)
    # so the posterior PDF is just P(data | acc) with a normalizing factor of (n + 1).
    # we integrate symbolically to find the CDF:
    # \int_0^b (n + 1) (acc^k (1-acc)^{n-k} (n choose k)) dacc = (n + 1) Beta(b, 1+k, 1-k+n) (n choose k)
    return beta.ppf(quantile, 1 + k, 1 - k + n)
