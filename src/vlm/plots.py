from typing import Optional
from warnings import warn
import itertools

import numpy as np
import plotly.express as px
import polars as pl
import sklearn.metrics as skm
from plotly.subplots import make_subplots
from polars import col as c
from scipy.stats import beta
from sklearn.preprocessing import LabelBinarizer


def levels_line_plot(data: pl.DataFrame):
    """Data should have columns: model, group, level, score"""

    fig = px.line(
        data.sort("group", "model", "level").to_pandas(),
        x="level",
        y="score",
        color="model",
        facet_row="group",
        title="Mean macro F1 by level",
        error_y="error",
        # color_discrete_map={
        #     "clip": "#22D3EE",
        #     "gpt-4o": "#059669",
        #     "viclip": "#A78BFA",
        #     "clip-32": "#22D3EE",
        #     "clip-4": "#2DD4BF",
        # },
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(line={"width": 5})
    fig.update_layout(
        # showlegend=False,
        width=1000,
        height=1200,
        yaxis=dict(tickmode="linear", tick0=0, dtick=0.25),
        # plot_bgcolor="#F5F5F5",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


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
    # TODO: make bayesian error

    print(f"MAP PLOT: {per_label_map}")

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
    print(f"metric per task: {metric_per_task}")
    
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


    matrix_columns_per_task = []
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

            matrix_column = []
            for model_idx, model in enumerate(models, start=2):
                task_df = predictions.filter(c("task") == task, c("model") == model)
                labels = task_labels[task]
                matrix = skm.confusion_matrix(
                    task_df["true_label"].to_numpy(),
                    task_df["label"].to_numpy(),
                    labels=labels,
                )

                matrix_column.append(matrix)

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
            
            matrix_columns_per_task.append((task, matrix_column))
    
    # check to see if we can sum any of the confusion matrices
    other_task_labels = sorted([(task_labels[t], t) for t in tasks if t != "average"])
    task_groups = [(tl, [t for t in tg]) for tl, tg in itertools.groupby(other_task_labels, key=lambda x: x[0])]

    # get non-singleton task groups
    ns_task_groups = [(tl, tg) for tl, tg in task_groups if len(tg) > 1]

    if len(ns_task_groups) > 1:
        warn(f"Currently only supports a single summed group; skipping {ns_task_groups[1:]}")

    if ns_task_groups:
        task_labels, task_group = ns_task_groups[0]
  
        tasks = [t[1] for t in task_group]

        # for each model, make a heatmap which is the sum of the confusion matrices for all the tasks in the group
        for model_idx, model in enumerate(models, start=2):
            matrix = np.zeros((len(task_labels), len(task_labels)))
            for t in tasks:
                matrix += matrix_columns_per_task[tasks.index(t)][1][model_idx - 2]

            heatmap = px.imshow(
                matrix,
                x=task_labels,
                y=task_labels,
                labels=dict(x="Candidate Label", y="Video True Label"),
                text_auto=True,
            )
            heatmap.update_layout(showlegend=False)
            # add it to the first column
            fig.add_traces(heatmap["data"], rows=model_idx, cols=1)

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
            "clip-32": "#22D3EE",
            "clip-4": "#2DD4BF",
            "default (16f, 1-shot)": "#059669",
            "16f, 0-shot": "#84CC16",
            "5f, 1-shot": "#22C55E",
        },
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


def incorrect_video_labels(predictions: pl.DataFrame):
    incorrect_count = (
        predictions.filter(c("label") != c("true_label"))
        .groupby("video", "task")
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


def get_real_life_special_groups():
    scrambled_objs = [
        "avocado",
        "chopsticks",
        "potato",
        "tomato",
    ]

    def _maker(df, get_tasks):
        scramble_tasks = get_tasks("extrapyramidal/object_tracking")

        # now divide up the tasks based on number of objects; the last part of the path (split by '/') are in the format scramble_<n>_objects
        lvls = {}
        for i in range(2, 9):
            lvls[f"Object tracking: Scramble {i}"] = [s for s in scramble_tasks if f"scramble_{i}" in s.split("/")[-1]]

        objs = {}
        for obj in scrambled_objs:
            objs[f"Object tracking: Object {obj}"] = [s for s in scramble_tasks if obj in s.split("/")[-1]]
          
        general = {
            "Object tracking overall": scramble_tasks,
            "Object tracking: Object identity": get_tasks("extrapyramidal/object_tracking/object_identity"),
            "Object tracking: Relative position": get_tasks("extrapyramidal/object_tracking/relative_position"),
        }

        other = {
            "Action: Opening v. Closing": get_tasks("extrapyramidal/opening_v_closing"),
            "Object Recognition (Real)": get_tasks("extrapyramidal/recognize_small_object"),
        }

        return {**lvls, **objs, **general, **other}
    
    return _maker