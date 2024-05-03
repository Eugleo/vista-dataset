import json

import plotly.express as px
import polars as pl
import sklearn.metrics as skm
import torch as t


def serialize_dict(d) -> str:
    return json.dumps(d, sort_keys=True)


def subsample(x: t.Tensor, n_frames: int) -> t.Tensor:
    total_frames, *_ = x.shape
    if total_frames <= n_frames:
        raise ValueError("Video is too short to subsample.")
    step = total_frames // n_frames
    x = x[::step, ...][:n_frames, ...]
    return x


# A wrapper around a sklearn metric function
def mcc(group: pl.Series):
    # group is a pl.Series object with two named fields, true_label and predicted_label
    # we can access those fields using group.struct.field
    return skm.matthews_corrcoef(
        y_true=group.struct.field("true_label").to_numpy(),
        y_pred=group.struct.field("label").to_numpy(),
    )


def get_predictions(df: pl.DataFrame):
    return df.group_by(["video", "task", "model"]).agg(
        # Extract the label with the highest probability
        pl.col("label").sort_by("prob").last(),
        pl.col("true_label").first(),
    )


# A helper function used to extract label colums from the dataframe,
# package them as structs, and then map matric_fun over them
def compute_metric(metric_fun):
    return pl.struct("true_label", "label").map_batches(metric_fun).first()


def performance_per_task(data: pl.DataFrame):
    return px.bar(
        data.to_pandas(),
        x="model",
        color="model",
        facet_col="task",
        y="mcc",
        title="Performance of each model per task",
    )


def confusion_matrix(data: pl.DataFrame):
    cm = skm.ConfusionMatrixDisplay.from_predictions(
        data["true_label"].to_numpy(),
        data["label"].to_numpy(),
        xticks_rotation="vertical",
    )
    return cm
