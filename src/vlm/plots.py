from typing import Optional

import numpy as np
import plotly.express as px
import polars as pl
import sklearn.metrics as skm
from plotly.subplots import make_subplots
from polars import col as c
from sklearn.preprocessing import LabelBinarizer


def levels_line_plot(data: pl.DataFrame):
    """Data should have columns: model, group, level, score"""

    return px.line(
        data.sort("group", "model", "level").to_pandas(),
        x="level",
        y="score",
        color="model",
        facet_row="group",
        title="mAP by level",
        error_y="error",
        height=1000,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )


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
    predictions_per_task: pl.DataFrame,
    scores: pl.DataFrame,
    metric: str,
    title: str,
    baselines: Optional[dict],
    labels: dict,
    tasks: list,
):
    models = metric_per_task["model"].unique().sort().to_list()
    avg_data = (
        metric_per_task.group_by("model")
        .agg(pl.col(metric).mean())
        .with_columns(task=pl.lit("average"))
        .select(["task", "model", metric])
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
            y=metric,
            range_y=[0, 1],
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        subplot.update_layout(showlegend=False)
        fig.add_traces(subplot["data"], rows=1, cols=task_idx)
        if task != "average":
            if baselines:
                fig.add_hline(y=baselines[task], line_dash="dot", col=task_idx)  # type: ignore
            for model_idx, model in enumerate(models, start=2):
                # task_df = predictions_per_task.filter(
                #     c("task") == task, c("model") == model
                # )
                task_labels = labels[task]
                # matrix = skm.confusion_matrix(
                #     task_df["true_label"].to_numpy(),
                #     task_df["label"].to_numpy(),
                #     labels=task_labels,
                # )
                # print(tasres.filter(c("task") == task, c("model") == model))
                rankings = (
                    scores.filter(c("task") == task, c("model") == model)
                    .with_columns(
                        pl.col("score")
                        .rank("max", descending=True)
                        .over("label")
                        .alias("rank")
                    )
                    .pivot(values="rank", index="true_label", columns="label")
                    .sort("true_label")
                    .select(pl.all().exclude("true_label"))
                )

                heatmap = px.imshow(
                    rankings.to_numpy(),
                    x=task_labels,
                    y=task_labels,
                    labels=dict(x="Candidate Label", y="Video True Label"),
                    text_auto=True,
                )
                heatmap.update_layout(showlegend=False)
                # if task_idx > 1:
                #     fig.update_yaxes(showticklabels=False, row=model_idx, col=task_idx)
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
    metrics: pl.DataFrame, metric: str, metric_label: str, title: str
):
    avg_data = (
        metrics.group_by("model")
        .agg(pl.col(metric).mean(), error=pl.col(metric).std() / pl.len().sqrt())
        .with_columns(task=pl.lit("average"))
        .sort("model")
    )
    fig = px.bar(
        avg_data.to_pandas(),
        x="model",
        y=metric,
        color=metric,
        range_y=[0, 1],
        range_color=[0, 1],
        color_continuous_scale="YlGn",
        title=title,
        labels={metric: metric_label},
        error_y="error",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig


def incorrect_video_labels(predictions: pl.DataFrame):
    incorrect_count = (
        predictions.filter(c("label") != c("true_label"))
        .groupby("video", "task")
        .agg(c("model").n_unique().alias("count"))
    )

    df = incorrect_count.sort("count", descending=True)

    return df
