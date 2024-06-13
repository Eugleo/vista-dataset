from functools import partial
from pathlib import Path

import jsonlines
import numpy as np
import plotly.express as px
import polars as pl
import sklearn.metrics as skm
import streamlit as st
import typer
from polars import col as c
from vlm import plots, utils
from vlm.models import GPTModel
from vlm.objects import Task


def load_logs(log_dir):
    logs = []
    for file in (log_dir).glob("gpt*/*.jsonl"):
        # if "59a" in str(file):
        with jsonlines.open(file) as reader:
            for line in reader:
                logs.append(line)
    return logs


def display_message(msg):
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
            text = (
                msg["content"]
                .replace("# FIRST TASK", "**FIRST TASK**")
                .replace("# SECOND TASK", "**SECOND TASK**")
                .replace("## EXAMPLE", "**EXAMPLE**\n")
            )
            st.markdown(text)
        else:
            with st.container():
                for content in msg["content"]:
                    if isinstance(content, str):
                        text = (
                            content.replace("# FIRST TASK", "**FIRST TASK**")
                            .replace("# SECOND TASK", "**FIRST TASK**")
                            .replace("## EXAMPLE", "**EXAMPLE**\n")
                        )
                        st.markdown(text)
                    elif content["type"] == "text":
                        text = (
                            content["text"]
                            .replace("# FIRST TASK", "**FIRST TASK**")
                            .replace("# SECOND TASK", "**FIRST TASK**")
                            .replace("## EXAMPLE", "**EXAMPLE**\n")
                        )
                        st.markdown(text)
                images = [
                    content["image_url"]["url"]
                    for content in msg["content"]
                    if content["type"] == "image_url"
                ]
                if images:
                    st.image(
                        images, caption=[f"Frame {n + 1}" for n in range(len(images))]
                    )


def select_true_label(label):
    st.session_state.selected_label = label


def ranking_matrix(predictions: pl.DataFrame):
    labels = predictions["true_label"].unique().sort().to_list()

    matrix = skm.confusion_matrix(
        predictions["true_label"].to_numpy(),
        predictions["label"].to_numpy(),
        labels=labels,
    )

    heatmap = px.imshow(
        matrix,
        x=labels,
        y=labels,
        labels=dict(x="Candidate Label", y="Video True Label"),
        text_auto=True,
        height=300,
        color_continuous_scale=px.colors.sequential.Purples,
    )
    heatmap.update_layout(showlegend=False)

    heatmap.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=60, r=50, t=0, b=0, pad=0),
    )

    return heatmap


def task_picker(logs, scores):
    st.sidebar.markdown("## 1) Select Task")

    predictions = utils.get_predictions(scores)

    task_performance_df = (
        predictions.group_by("task", "model")
        .agg(metric=utils.compute_metric(utils.f1))
        .sort("task", "model")
        .drop("model")
    )

    event = st.sidebar.dataframe(
        task_performance_df,
        column_config={
            "task": st.column_config.TextColumn(width="medium"),
            "metric": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.2f"
            ),
        },
        on_select="rerun",
        selection_mode="single-row",
    )

    task_ids = list(log["task"] for log in logs)
    task_id = (
        task_performance_df["task"][event.selection["rows"][0]]  # type: ignore
        if event.selection["rows"]  # type: ignore
        else task_ids[0]
    )

    st.sidebar.markdown("#### Selected task")
    st.sidebar.markdown(task_id)

    st.sidebar.markdown("#### Task performance overview")

    st.sidebar.plotly_chart(
        ranking_matrix(predictions.filter(c("task") == task_id)),
        use_container_width=True,
    )

    return [log for log in logs if log["task"] == task_id]


def label_picker(logs):
    st.sidebar.markdown("## 2) Filter videos by their true label")

    log = logs[0]
    label_descriptions = log["label_descriptions"]

    for i, (label, description) in enumerate(label_descriptions.items()):
        c1, _, c2 = st.sidebar.columns([1, 4, 1])
        label_txt = (
            f"**{label}**" if label == st.session_state.selected_label else label
        )
        c1.markdown(label_txt)
        c2.button("Select", key=i, on_click=select_true_label, args=(label,))
        st.sidebar.markdown(description)

    return [log for log in logs if log["true_label"] == st.session_state.selected_label]


def video_picker(logs):
    if len(logs) == 1:
        log = logs[0]
    else:
        st.sidebar.markdown("## 3) Select video")
        valid_videos = list(set(log["video"] for log in logs))
        video = st.sidebar.selectbox("Select Video", valid_videos)
        log = [log for log in logs if log["video"] == video][0]
    st.session_state.selected_log = log
    st.session_state.selected_label = log["true_label"]
    return log


def initialize_state():
    if "selected_label" not in st.session_state:
        st.session_state.selected_label = None


def get_scores(logs):
    scores = pl.DataFrame(
        [
            {
                "task": log["task"] if "task" in log else "unknown",
                "video": log["video"],
                "true_label": log["true_label"],
                "label": label,
                "score": score,
                "model": "gpt-4o",
            }
            for log in logs
            for label, score in log["parsed_scores"].items()
        ]
    )
    scores = utils.add_rescaling(scores).filter(c("rescaling") == "v+l")
    return scores


def main(log_dir: str):
    st.set_page_config(layout="wide")

    initialize_state()

    logs, task_labels = [], {}
    for log in load_logs(Path(log_dir)):
        # if "task" not in log:
        #     continue
        labels = list(sorted(log["label_descriptions"].keys()))
        task_labels.setdefault(log["task"], labels)

        scores = GPTModel.parse_and_cache_scores(
            log["history"][-1]["content"], "", task_labels[log["task"]], {}
        )
        log["parsed_scores"] = scores
        logs.append(log)

    scores = get_scores(logs)

    logs = task_picker(logs, scores)
    if logs:
        logs = label_picker(logs)
        if logs:
            log = video_picker(logs)

            left_col, right_col = st.columns([1, 3])
            with left_col:
                st.video(log["video"])

                print(
                    scores.filter(c("task") == log["task"], c("video") == log["video"])
                )

                fig = px.bar(
                    scores.filter(c("task") == log["task"], c("video") == log["video"]),
                    x="label",
                    y="score",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with right_col:
                with st.container(height=1200):
                    conversation = log["history"]
                    for msg in conversation:
                        display_message(msg)


if __name__ == "__main__":
    typer.run(main)
