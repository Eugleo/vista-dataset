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
from vlm.objects import Task


def load_logs(log_dir):
    logs = []
    for file in (log_dir).glob("gpt*/*.jsonl"):
        if "gpt-4o-5" in str(file):
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


def prev_video():
    st.session_state.video_index -= 1


# Function to increment video index
def next_video():
    st.session_state.video_index += 1


# Function to update video index based on selectbox
def select_video(index):
    st.session_state.video_index = index


def select_task(index):
    print(index)
    st.session_state.task_index = index


def ranking_matrix(scores):
    task_labels = scores["label"].unique().sort().to_list()
    rankings = (
        scores.with_columns(
            # add a small random number to each score to break ties
            score=c("score") + pl.Series(np.random.rand(len(scores)) * 1e-6)
        )
        .with_columns(
            pl.col("score").rank("max", descending=True).over("label").alias("rank")
        )
        .pivot(values="rank", index="true_label", columns="label")
        .sort("true_label")
        .select(task_labels)
    )

    fig = px.imshow(
        rankings.to_numpy(),
        x=task_labels,
        y=task_labels,
        labels=dict(
            x="Candidate Label (a single ranking problem)",
            y="Video (aliased by its true label)",
        ),
        text_auto=True,
        height=300,
        color_continuous_scale=px.colors.sequential.Viridis_r,
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=60, r=50, t=0, b=0, pad=0),
    )
    return fig


def main(log_dir: str):
    st.set_page_config(layout="wide")

    logs = load_logs(Path(log_dir))

    if "video_index" not in st.session_state:
        st.session_state.video_index = 0

    if "task_index" not in st.session_state:
        st.session_state.task_index = 0

    st.sidebar.markdown("## Select Task")

    task_labels = {}
    task_dir = Path("tasks") / "alfred"
    for t in set(log["task"] for log in logs):
        try:
            task_labels[t] = Task.from_file(t, Path(task_dir) / f"{t}.yaml").labels
        except FileNotFoundError:
            continue

    scores_df = pl.DataFrame(
        [
            {
                "task": log["task"] if "task" in log else "unknown",
                "video": log["video"],
                "true_label": log["true_label"],
                "label": label,
                "score": score,
                "model": "gpt-4o-16",
            }
            for log in logs
            for label, score in log["parsed_scores"].items()
        ]
    )
    scores_df = utils.rescale(scores_df, in_each="video")

    task_performance_df = (
        scores_df.sort("task", "video", "label")
        .group_by("task", maintain_order=True)
        .agg(
            AP=pl.struct("task", "label", "score", "true_label").map(
                partial(plots.average_precision, task_labels)
            )
        )
        .explode("AP")
        .group_by("task")
        .agg(mAP=pl.col("AP").mean())
        .sort("task")
    )

    event = st.sidebar.dataframe(
        task_performance_df.to_pandas(),
        column_config={
            "task": st.column_config.TextColumn(width="medium"),
            "mAP": st.column_config.ProgressColumn(
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

    st.sidebar.markdown(f"Selected: {task_id}")

    videos = sorted([log["video"] for log in logs if log["task"] == task_id])
    video = videos[st.session_state.video_index]
    print(videos)
    print(st.session_state.video_index, video)
    # video = st.sidebar.selectbox(
    #     "Select Video", videos, index=st.session_state.video_index
    # )
    log = next(log for log in logs if log["task"] == task_id and log["video"] == video)

    task_logs = [log for log in logs if log["task"] == task_id]
    print(list(task_logs[0].keys()))

    st.sidebar.markdown("---")
    st.sidebar.plotly_chart(
        ranking_matrix(scores_df.filter(c("task") == task_id)), use_container_width=True
    )

    left_col, right_col = st.columns([1, 3])
    with left_col:
        st.video(log["video"])

        true_label = log["true_label"]
        predicted_label = log["predicted_label"]
        label_descriptions = log["label_descriptions"]

        fig = px.bar(
            scores_df.filter(c("task") == task_id, c("video") == video).to_pandas(),
            x="label",
            y="score",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        for i, (label, description) in enumerate(label_descriptions.items()):
            color = "green" if label == true_label else "gray"
            label_txt = f"**{label}** 👈" if label == predicted_label else label
            c1, _, c2 = st.sidebar.columns([1, 4, 1])
            c1.markdown(f"{label_txt}")
            idx = [label in v for v in videos].index(True)
            c2.button("Select", key=i, on_click=select_video, args=(idx,))
            st.sidebar.markdown(
                f"<span style='color:{color}'>{description}</span>",
                unsafe_allow_html=True,
            )

    with right_col:
        with st.container(height=1200):
            conversation = log["history"]
            for msg in conversation:
                display_message(msg)

    # # Previous and Next buttons, going to previous/next video in the same task

    # st.sidebar.markdown("---")
    # if st.session_state.video_index > 0:
    #     st.sidebar.button("Previous video", on_click=prev_video)

    # if st.session_state.video_index < len(videos) - 1:
    #     st.sidebar.button("Next video", on_click=next_video)


if __name__ == "__main__":
    typer.run(main)
