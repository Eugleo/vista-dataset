from pathlib import Path

import jsonlines
import plotly.express as px
import polars as pl
import sklearn.metrics as skm
import streamlit as st
import typer


def load_logs(log_dir):
    logs = []
    for file in (log_dir).glob("gpt*/*.jsonl"):
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


def confusion_matrix(true_labels, predicted_labels, labels):
    matrix = skm.confusion_matrix(
        true_labels,
        predicted_labels,
        labels=labels,
    )
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Purples",
        height=200,
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

    task_ids = sorted(set(log["task"] for log in logs))
    task_id = st.sidebar.selectbox("Select Task ID", task_ids)
    videos = sorted(set(log["video"] for log in logs if log["task"] == task_id))
    video = st.sidebar.selectbox(
        "Select Video", videos, index=st.session_state.video_index
    )
    log = next(log for log in logs if log["task"] == task_id and log["video"] == video)

    task_logs = [log for log in logs if log["task"] == task_id]
    true_labels = [log["true_label"] for log in task_logs]
    predicted_labels = [log["predicted_label"] for log in task_logs]
    labels = list(log["label_descriptions"].keys())

    st.sidebar.markdown("---")
    st.sidebar.plotly_chart(
        confusion_matrix(true_labels, predicted_labels, labels),
        use_container_width=True,
    )

    left_col, right_col = st.columns([1, 3])
    with left_col:
        st.video(log["video"])

        parsed_scores = log["parsed_scores"]
        true_label = log["true_label"]
        predicted_label = log["predicted_label"]
        label_descriptions = log["label_descriptions"]

        # parsed_scores = GPTModel.parse_and_cache_scores(
        #     log["history"][-1]["content"], "", labels, {}
        # )

        scores_df = (
            pl.DataFrame(
                [
                    {"label": label, "score": score}
                    for label, score in parsed_scores.items()
                ]
            )
            .with_columns(
                pl.col("label")
                .apply(
                    lambda x: "green"
                    if x == true_label
                    else ("blue" if x == predicted_label else "gray")
                )
                .alias("color")
            )
            .sort("label")
        )

        fig = px.bar(
            scores_df.to_pandas(),
            x="label",
            y="score",
            color="color",
            color_discrete_map={"green": "green", "blue": "blue", "gray": "gray"},
        )
        fig.update_yaxes(range=[0, 5])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        for label, description in label_descriptions.items():
            color = "green" if label == true_label else "gray"
            label = f"**{label}** 👈" if label == predicted_label else label
            st.sidebar.markdown(
                f"{label}<br><span style='color:{color}'>{description}</span>",
                unsafe_allow_html=True,
            )

    with right_col:
        with st.container(height=1200):
            conversation = log["history"]
            for msg in conversation:
                display_message(msg)

    # Previous and Next buttons, going to previous/next video in the same task

    st.sidebar.markdown("---")
    if st.session_state.video_index > 0:
        st.sidebar.button("Previous video", on_click=prev_video)

    if st.session_state.video_index < len(videos) - 1:
        st.sidebar.button("Next video", on_click=next_video)


if __name__ == "__main__":
    typer.run(main)
