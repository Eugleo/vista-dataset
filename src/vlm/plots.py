import plotly.express as px
import polars as pl
import sklearn.metrics as skm
from plotly.subplots import make_subplots
from polars import col as c


def performance_per_task(
    data: pl.DataFrame, df: pl.DataFrame, metric: str, title: str, baselines: dict
):
    data = data.sort("task", "model")
    tasks = data["task"].unique().sort().to_list()
    models = data["model"].unique().sort().to_list()
    if len(tasks) > 1:
        avg_data = (
            data.group_by("model")
            .agg(pl.col(metric).mean())
            .with_columns(task=pl.lit("average"))
            .select(["task", "model", metric])
        )
        data = pl.concat([data, avg_data])
        tasks.append("average")

    model_titles = [model for model in models for _ in range(len(tasks))]
    fig = make_subplots(
        rows=1 + len(models),
        cols=len(tasks),
        subplot_titles=tasks + model_titles,
        # specs=[[{}, {}]],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    for task_idx, task in enumerate(tasks, start=1):
        subplot = px.bar(
            data.filter(c("task") == task).to_pandas(),
            x="model",
            color="model",
            y=metric,
            range_y=[0, 1],
        )
        subplot.update_layout(showlegend=False)
        fig.add_traces(subplot["data"], rows=1, cols=task_idx)
        if task != "average":
            fig.add_hline(y=baselines[task], line_dash="dot", col=task_idx)  # type: ignore
            for model_idx, model in enumerate(models, start=2):
                task_df = df.filter(c("task") == task, c("model") == model)
                task_labels = task_df["true_label"].sort().to_list()
                matrix = skm.confusion_matrix(
                    task_df["true_label"].to_numpy(),
                    task_df["label"].to_numpy(),
                    labels=task_labels,
                )
                heatmap = px.imshow(
                    matrix,
                    x=task_labels,
                    y=task_labels,
                    labels=dict(x="Predicted label", y="True label"),
                    text_auto=True,
                )
                heatmap.update_layout(showlegend=False)
                if task_idx > 1:
                    fig.update_yaxes(showticklabels=False, row=model_idx, col=task_idx)
                if model_idx < len(models) + 1:
                    fig.update_xaxes(showticklabels=False, row=model_idx, col=task_idx)
                fig.add_traces(heatmap["data"], rows=model_idx, cols=task_idx)
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(
        width=300 * len(tasks) if len(tasks) > 1 else 500,
        height=300 * (1 + len(models)),
        showlegend=False,
    )
    return fig


def confusion_matrix(data: pl.DataFrame):
    # Compute confusion matrix
    cm = skm.confusion_matrix(data["true_label"].to_numpy(), data["label"].to_numpy())

    # Create heatmap
    heatmap = px.imshow(
        cm,
        labels=dict(x="Predicted label", y="True label"),
        color_continuous_scale="Viridis",
    )
    return heatmap
