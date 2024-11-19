import json
import logging
import subprocess
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Annotated, Optional

import dotenv
import jsonlines
import openai
import plotly.express as px
import polars as pl
import typer
from lets_plot import ggsize, labs, scale_color_manual, scale_fill_manual, ylim
from polars import col as c
from polars import first
from rich.pretty import pprint
from rich.progress import Progress, SpinnerColumn, TextColumn

import log_viewer
from vlm import plots, utils
from vlm.config import ExperimentConfig
from vlm.models import GPTModel
from vlm.objects import Task

app = typer.Typer()


@app.command()
def view_logs(
    experiment: Annotated[Optional[str], typer.Option()] = None,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
):
    exp_dir = utils.get_experiment_dir(experiment_dir, experiment)
    subprocess.run(
        ["streamlit", "run", "src/log_viewer.py", "--", str(exp_dir / "logs")]
    )


def get_tasks(df, prefix):
    return [t for t in df.get_column("task").unique() if t.startswith(prefix)]


@app.command()
def cancel_batch(
    batch_ids: Annotated[list[str], typer.Argument()],
):
    dotenv.load_dotenv()
    client = openai.Client()
    for batch_id in batch_ids:
        client.batches.cancel(batch_id)
        print(f"Batch {batch_id} is cancelled.")


@app.command()
def download_batch(
    batch_ids: Annotated[list[str], typer.Argument()],
    log_dir: Annotated[str, typer.Option()],
    experiment: Annotated[Optional[str], typer.Option()] = None,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
    task_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/tasks",
    video_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/videos",
    verbose: Annotated[bool, typer.Option()] = False,
):
    dotenv.load_dotenv()
    client = openai.Client()
    dir = utils.get_experiment_dir(experiment_dir, experiment)
    task_true_labels = {}

    logs = log_viewer.load_logs(dir / "logs" / log_dir)

    for batch_id in batch_ids:
        batch_file = client.batches.retrieve(batch_id)

        if batch_file.status != "completed":
            print(f"{batch_id}: {batch_file.status}")
            if verbose:
                pprint(batch_file)
            continue

        assert batch_file.output_file_id is not None
        batch_response_dir = dir / "logs" / "gpt_reponses"
        batch_response_dir.mkdir(exist_ok=True, parents=True)
        batch_reponse_file = batch_response_dir / f"{batch_id}.jsonl"
        if not batch_reponse_file.exists():
            print(f"Downloading batch response for {batch_id}...")
            content = client.files.content(batch_file.output_file_id)
            content.write_to_file(batch_reponse_file)
        else:
            print(f"{batch_id}: downloaded")

        results = []
        with jsonlines.open(batch_reponse_file) as reader:
            for response in reader:
                model = response["response"]["body"]["model"]
                if response["response"]["status_code"] != 200:
                    print(f"WARNING: Request failed: {response}")
                    continue
                path, task_id = response["custom_id"].split(",")
                answer = response["response"]["body"]["choices"][0]["message"][
                    "content"
                ]

                if task_id not in task_true_labels:
                    task = Task.from_file(task_id, Path(task_dir) / f"{task_id}.yaml")
                    with open(Path(task_dir) / f"{task_id}_data.json") as f:
                        true_labels = {
                            str(Path(video_dir) / obj["path"]): obj["label"]
                            for obj in json.load(f)
                        }
                        task_true_labels[task_id] = true_labels
                true_labels = task_true_labels[task_id]

                label_scores = GPTModel.parse_and_cache_scores(
                    answer, path, task.labels, cache={}, verbose=verbose
                )

                for label_idx, (label, score) in enumerate(label_scores.items()):
                    results.append(
                        {
                            "task": task_id,
                            "model": model,
                            "video": path,
                            "label": label,
                            "label_idx": label_idx,
                            "score": score,
                            "true_label": true_labels[path],
                            "true_label_idx": task.labels.index(true_labels[path]),
                        }
                    )

                for log in logs:
                    if (
                        log["task"] == task_id
                        and log["video"] == path
                        and "predicted_label" not in log
                    ):
                        log["parsed_scores"] = label_scores
                        log["predicted_label"] = max(label_scores, key=label_scores.get)
                        log["label_descriptions"] = task.labels
                        log["true_label"] = true_labels[path]

        (dir / "logs" / model).mkdir(exist_ok=True, parents=True)
        with jsonlines.open(dir / "logs" / model / "responses.jsonl", "w") as f:
            f.write_all(log)

        results = pl.DataFrame(results)
        result_dir = dir / "results"
        result_dir.mkdir(exist_ok=True, parents=True)
        results.write_json(result_dir / f"{model}_{batch_id}.json")


@app.command()
def evaluate(config: Annotated[str, typer.Argument()]):
    experiment_config = ExperimentConfig.from_file(config)
    experiment = ExperimentConfig.to_experiment(experiment_config)
    experiment_dir = Path(experiment.output_dir) / experiment.id
    experiment_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename=experiment_dir / "log.txt",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    experiment.run()


def load_task_labels(task_dir, tasks):
    task_labels = {}
    for t in tasks:
        try:
            task_labels[t] = Task.from_file(t, Path(task_dir) / f"{t}.yaml").labels
        except FileNotFoundError:
            continue
    return task_labels


def _grouped_levels_plot(df, get_tasks):
    performance_by_level = pl.concat(
        [
            df.filter(c("task").is_in(get_tasks(f"level_{n}/{group}")))
            .with_columns(
                model=pl.when(c("model") == "viclip_cosine")
                .then(pl.lit("ViCLIP"))
                .when(c("model") == "clip-32_cosine")
                .then(pl.lit("CLIP"))
                .when(c("model") == "gpt-4o")
                .then(pl.lit("GPT-4o"))
                .when(c("model") == "majority_baseline")
                .then(pl.lit("Baseline"))
                .otherwise(c("model")),
                group=pl.lit("General") if group == "remix" else pl.lit("Permutation"),
            )
            .filter(c("model").is_in(["ViCLIP", "CLIP", "GPT-4o", "Baseline"]))
            .group_by("model", "group")
            .agg(
                score=c("metric").mean(),
                error=c("metric").std() / (c("metric").len().sqrt() + 1e-6),
            )
            .with_columns(level=pl.lit(f"Level {n}"))
            for n in range(2, 9)
            for group in ["remix", "permutation"]
        ]
    )
    fig = plots.grouped_levels_line_plot(performance_by_level)

    return fig


def _levels_plot(df, get_tasks, min_level=1):
    performance_by_level = pl.concat(
        [
            df.filter(
                c("task").is_in(
                    get_tasks(f"level_{n}")
                    if n > 1
                    else (get_tasks("foundation") + get_tasks("extrapyramidal"))
                )
            )
            .with_columns(
                model=pl.when(c("model") == "viclip_cosine")
                .then(pl.lit("ViCLIP"))
                .when(c("model") == "clip-32_cosine")
                .then(pl.lit("CLIP"))
                .when(c("model") == "gpt-4o")
                .then(pl.lit("GPT-4o"))
                .when(c("model") == "majority_baseline")
                .then(pl.lit("Baseline"))
                .otherwise(c("model")),
            )
            .filter(c("model").is_in(["ViCLIP", "CLIP", "GPT-4o", "Baseline"]))
            .group_by("model")
            .agg(
                score=c("metric").mean(),
                error=c("metric").std() / (c("metric").len().sqrt() + 1e-6),
            )
            .with_columns(level=pl.lit(f"Level {n}"))
            for n in range(min_level, 9)
        ]
    )
    fig = plots.levels_line_plot(performance_by_level)

    return fig


def _problem_set_bar_plot(df, groups, ncol=4, color="model"):
    performance_by_group = pl.concat(
        [
            df.filter(c("task").is_in(tasks))
            .with_columns(
                model=pl.when(c("model") == "viclip_cosine")
                .then(pl.lit("ViCLIP"))
                .when(c("model") == "clip-32_cosine")
                .then(pl.lit("CLIP"))
                .when(c("model") == "gpt-4o")
                .then(pl.lit("GPT-4o"))
                .when(c("model") == "majority_baseline")
                .then(pl.lit("Baseline"))
                .otherwise(c("model")),
            )
            .filter(c("model").is_in(["ViCLIP", "CLIP", "GPT-4o"]))
            .group_by("model", "environment")
            .agg(
                score=c("metric").mean(),
                error=c("metric").std() / (c("metric").len().sqrt() + 1e-6),
            )
            .with_columns(group=pl.lit(name))
            for name, tasks in groups.items()
        ]
    )
    fig = plots.task_groups_bar_plot(performance_by_group, ncol=ncol, color=color)

    return fig


@app.command()
def plot_alfred(
    experiment: Annotated[Optional[str], typer.Argument()] = "Exp_reprint",
    experiment_dir: Annotated[str, typer.Option()] = "experiments",
    # task_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/tasks",
):
    dir = utils.get_experiment_dir(experiment_dir, experiment)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Creating plots...")
        df = pl.concat(
            [
                pl.read_json(file).cast({"true_label_idx": pl.Int64})
                for file in (dir / "results").glob("*.json")
            ]
        ).unique()
        df = df.filter(
            ~c("task").str.contains("permuted"),
            ~c("task").str.contains("substituted"),
        )

        scores = (
            utils.add_rescaling(df)
            .filter(c("rescaling") == "v+l")
            .with_columns(
                group=pl.concat_str(
                    c("task").str.split("/").list.get(0),
                    c("task").str.split("/").list.get(1),
                    separator="/",
                )
            )
        )

        num_nulls = len(scores.filter(c("score").is_null()))
        if num_nulls > 0:
            print(f"WARNING: {num_nulls} null scores found")
            scores = scores.filter(c("score").is_not_null())

        predictions = utils.get_predictions(scores)
        predictions = utils.add_majority_baseline(predictions)

        metric_per_task_with_baseline = (
            predictions.with_columns(
                environment=pl.when(c("video").str.contains("alfred"))
                .then(pl.lit("Virtual home"))
                .when(c("video").str.contains("minecraft"))
                .then(pl.lit("Minecraft"))
                .when(c("video").str.contains("real_life"))
                .then(pl.lit("Real world"))
                .otherwise(pl.lit("Unknown"))
            )
            .group_by("task", "model", "environment")
            .agg(metric=utils.compute_metric(utils.f1))
            .sort("task", "model", "environment")
        )
        metric_per_task_with_baseline.write_csv(dir / "per_label_metrics.csv")

        def get_tasks(prefix):
            return [t for t in df.get_column("task").unique() if t.startswith(prefix)]

        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)

        fig = _problem_set_bar_plot(
            metric_per_task_with_baseline,
            {
                "Objects": get_tasks("foundation/objects")
                + get_tasks("foundation/containers"),
                "Object properties": get_tasks("foundation/on_v_off")
                + get_tasks("foundation/sliced_v_whole"),
                "Actions": get_tasks("foundation/clean")
                + get_tasks("foundation/heat")
                + get_tasks("foundation/cool")
                + get_tasks("foundation/pick_v_put")
                + get_tasks("foundation/slice")
                + get_tasks("foundation/toggle"),
            },
            ncol=3,
            color="environment",
        ) + labs(title="Performance in level 1 by group and env")
        fig.to_pdf(str(plot_dir / "level_1_overall.pdf"))
        fig.to_png(str(plot_dir / "level_1_overall.png"))
        fig = (
            _problem_set_bar_plot(
                metric_per_task_with_baseline.filter(
                    c("environment") == "Virtual home"
                ),
                {
                    "Objects": get_tasks("foundation/objects"),
                    "Containers": get_tasks("foundation/containers"),
                    "On/off": get_tasks("foundation/on_v_off"),
                    "Sliced/whole": get_tasks("foundation/sliced_v_whole"),
                    "Cleaning": get_tasks("foundation/clean"),
                    "Heating": get_tasks("foundation/heat"),
                    "Cooling": get_tasks("foundation/cool"),
                    "Pick/put": get_tasks("foundation/pick_v_put"),
                    "Slicing": get_tasks("foundation/slice"),
                    "Toggling": get_tasks("foundation/toggle"),
                },
                ncol=10,
            )
            + labs(title="Performance in level 1 by task group (virtual home)")
            + ggsize(width=1000, height=300)
        )
        fig.to_pdf(str(plot_dir / "level_1_virtual_home_expanded.pdf"))

        fig = (
            _problem_set_bar_plot(
                metric_per_task_with_baseline.filter(c("environment") == "Real world"),
                {
                    "Mimic": get_tasks("foundation"),
                    "Object interaction": get_tasks(
                        "extrapyramidal/object_interaction"
                    ),
                    "Object tracking": get_tasks(
                        "extrapyramidal/object_tracking/scramble"
                    ),
                    "Doors (kinetics)": get_tasks("extrapyramidal/opening_v_closing"),
                },
                ncol=6,
            )
            + labs(
                title="Performance in level 1 by task group (real world)",
            )
            + ggsize(width=700, height=300)
        )
        fig.to_pdf(str(plot_dir / "level_1_real_world.pdf"))

        fig = _grouped_levels_plot(metric_per_task_with_baseline, get_tasks) + labs(
            title="Performance by level and problem group (all environments)",
        )
        fig.to_pdf(str(plot_dir / "levels_overall.pdf"))
        fig.to_png(str(plot_dir / "levels_overall.png"))

        fig = _levels_plot(
            metric_per_task_with_baseline, get_tasks, min_level=1
        ) + labs(
            title="Performance by level (all environments)",
        )
        fig.to_pdf(str(plot_dir / "levels_overall_ungrouped.pdf"))
        fig.to_png(str(plot_dir / "levels_overall_ungrouped.png"))

        fig = (
            _levels_plot(
                metric_per_task_with_baseline.filter(c("environment") == "Minecraft"),
                get_tasks,
            )
            + labs(title="Performance by level (Minecraft)")
            + ylim(0, 1.2)
        )
        fig.to_pdf(str(plot_dir / "levels_minecraft_ungrouped.pdf"))

        for env in ["Real world", "Virtual home"]:
            fig = (
                _grouped_levels_plot(
                    metric_per_task_with_baseline.filter(c("environment") == env),
                    get_tasks,
                )
                + labs(
                    title=f"Performance by level and problem group ({env.lower()})",
                )
                + ylim(-0.1, 1.1)
            )
            fig.to_pdf(str(plot_dir / f"levels_{env.lower().replace(' ', '_')}.pdf"))

        ## CLIPs below

        performance_by_level = pl.concat(
            [
                metric_per_task_with_baseline.filter(
                    c("task").is_in(get_tasks(f"level_{n}/{group}"))
                )
                .with_columns(
                    model=pl.when(c("model") == "clip-8_cosine")
                    .then(pl.lit("CLIP (8)"))
                    .when(c("model") == "clip-32_cosine")
                    .then(pl.lit("CLIP (32)"))
                    .when(c("model") == "clip-16_cosine")
                    .then(pl.lit("CLIP (16)"))
                    .when(c("model") == "clip-4_cosine")
                    .then(pl.lit("CLIP (4)"))
                    .when(c("model") == "clip-2_cosine")
                    .then(pl.lit("CLIP (2)"))
                    .when(c("model") == "viclip_cosine")
                    .then(pl.lit("ViCLIP (8)"))
                    .otherwise(c("model")),
                    group=pl.lit("General")
                    if group == "remix"
                    else pl.lit("Permutation"),
                )
                .filter(
                    c("model").is_in(
                        [
                            "CLIP (2)",
                            "CLIP (4)",
                            "CLIP (8)",
                            "CLIP (16)",
                            "CLIP (32)",
                            "ViCLIP (8)",
                        ]
                    )
                )
                .group_by("model", "group")
                .agg(
                    score=c("metric").mean(),
                    error=c("metric").std() / (c("metric").len().sqrt() + 1e-6),
                )
                .with_columns(level=pl.lit(f"Level {n}"))
                for n in range(2, 9)
                for group in ["remix", "permutation"]
            ]
        )
        fig = plots.clip_levels_line_plot(performance_by_level) + labs(
            title="Performance by level, problem group, and frame rate (virtual home)",
        )
        fig.to_pdf(str(plot_dir / "levels_virtual_home_expanded_clips.pdf"))

        groups = {
            "Objects": get_tasks("foundation/objects"),
            "Containers": get_tasks("foundation/containers"),
            "On/off": get_tasks("foundation/on_v_off"),
            "Sliced/whole": get_tasks("foundation/sliced_v_whole"),
            "Cleaning": get_tasks("foundation/clean"),
            "Heating": get_tasks("foundation/heat"),
            "Cooling": get_tasks("foundation/cool"),
            "Pick/put": get_tasks("foundation/pick_v_put"),
            "Slicing": get_tasks("foundation/slice"),
            "Toggling": get_tasks("foundation/toggle"),
        }

        performance_by_group = pl.concat(
            [
                metric_per_task_with_baseline.filter(
                    c("task").is_in(tasks), c("environment") == "Virtual home"
                )
                .with_columns(
                    model=pl.when(c("model") == "clip-8_cosine")
                    .then(pl.lit("CLIP (8)"))
                    .when(c("model") == "clip-32_cosine")
                    .then(pl.lit("CLIP (32)"))
                    .when(c("model") == "clip-16_cosine")
                    .then(pl.lit("CLIP (16)"))
                    .when(c("model") == "clip-4_cosine")
                    .then(pl.lit("CLIP (4)"))
                    .when(c("model") == "clip-2_cosine")
                    .then(pl.lit("CLIP (2)"))
                    .when(c("model") == "viclip_cosine")
                    .then(pl.lit("ViCLIP (8)"))
                    .otherwise(c("model")),
                )
                .filter(
                    c("model").is_in(
                        [
                            "CLIP (2)",
                            "CLIP (4)",
                            "CLIP (8)",
                            "CLIP (16)",
                            "CLIP (32)",
                            "ViCLIP (8)",
                        ]
                    )
                )
                .cast(
                    {
                        "model": pl.Enum(
                            [
                                "CLIP (2)",
                                "CLIP (4)",
                                "CLIP (8)",
                                "CLIP (16)",
                                "CLIP (32)",
                                "ViCLIP (8)",
                            ]
                        )
                    }
                )
                .group_by("model", "environment")
                .agg(
                    score=c("metric").mean(),
                    error=c("metric").std() / (c("metric").len().sqrt() + 1e-6),
                )
                .with_columns(group=pl.lit(name))
                for name, tasks in groups.items()
            ]
        )
        fig = (
            plots.task_groups_bar_plot(performance_by_group, ncol=5, color="model")
            + labs(
                title="Performance in level 1 by problem group, and frame rate (virtual home)"
            )
            + ggsize(width=1000, height=600)
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
            + scale_fill_manual(
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
        fig.to_pdf(str(plot_dir / "level_1_virtual_home_expanded_clips.pdf"))


@app.command()
def plot_alfred_clip(
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
    task_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/tasks",
):
    dir = utils.get_experiment_dir(experiment_dir, experiment)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Creating plots...")
        df = pl.concat(
            [pl.read_json(file) for file in (dir / "results").glob("*.json")]
        )
        df.write_csv(dir / "unified_results.csv")

        return

        df = df.filter(
            ~c("task").str.contains("permuted"), ~c("task").str.contains("substituted")
        )

        scores = utils.add_rescaling(df).with_columns(
            group=pl.concat_str(
                c("task").str.split("/").list.get(0),
                c("task").str.split("/").list.get(1),
                separator="/",
            )
        )
        tasks_of_interest = (
            scores.sort("task")
            .group_by("group")
            .agg(task_of_interest=c("task").take(pl.len() // 2 + 1))
        )
        selected_rescaling = (
            scores.join(tasks_of_interest, on="group")
            .filter(c("score").is_not_nan())
            .filter(c("task").is_in("task_of_interest"))
            .with_columns(score=c("score").mean().over("model", "group"))
            .group_by("model", "group")
            .agg(best_rescaling=c("rescaling").sort_by("score").last())
        )
        scores = (
            scores.join(selected_rescaling, on=["model", "group"])
            .join(tasks_of_interest, on="group")
            .filter(
                c("rescaling") == c("best_rescaling"),
                ~c("task").is_in("task_of_interest"),
            )
            .with_columns(model=c("model").str.strip_suffix("_cosine"))
            .with_columns(
                model=pl.when(c("model") == "clip-32")
                .then(pl.lit("clip"))
                .otherwise("model")
            )
        )

        num_nulls = len(scores.filter(c("score").is_null()))
        if num_nulls > 0:
            print(f"WARNING: {num_nulls} null scores found")
            scores = scores.filter(c("score").is_not_null())

        predictions = utils.get_predictions(scores)
        predictions = utils.add_majority_baseline(predictions)

        metric_per_task_with_baseline = (
            predictions.group_by("task", "model")
            .agg(metric=utils.compute_metric(utils.f1))
            .sort("task", "model")
        )
        metric_per_task_with_baseline.write_csv(dir / "per_label_metrics.csv")

        baseline_per_task = {
            task: metric
            for task, metric in metric_per_task_with_baseline.filter(
                c("model") == "majority_baseline"
            )
            .select("task", "metric")
            .iter_rows()
        }
        metric_per_task = metric_per_task_with_baseline.filter(
            c("model") != "majority_baseline",
            c("model") != "s3d",
        )

        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"

        print(f"Creating plots in {plot_dir}...")
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "misclassification_num.csv")

        def get_tasks(prefix):
            return [t for t in df.get_column("task").unique() if t.startswith(prefix)]

        import plotly.io as pio

        pio.kaleido.scope.mathjax = None

        groups = {
            "The whole Foundation Level": get_tasks("foundation"),
            "Object recognition": get_tasks("foundation/objects")
            + get_tasks("foundation/containers"),
            "Action understanding": get_tasks("foundation/pick_v_put")
            + get_tasks("foundation/slice")
            + get_tasks("foundation/toggle")
            + get_tasks("foundation/clean")
            + get_tasks("foundation/heat")
            + get_tasks("foundation/cool"),
            "Object state recognition": get_tasks("foundation/on_v_off")
            + get_tasks("foundation/sliced_v_whole"),
            # "Overview": get_tasks("foundation"),
        }

        group_task = progress.add_task("Creating group plots...", total=len(groups))
        for name, tasks in groups.items():
            progress.advance(group_task, 1)
            filename = name.replace(": ", "_").replace(" ", "-")

            if (
                len(tasks) == 0
                or len(metric_per_task.filter(pl.col("task").is_in(tasks))) == 0
            ):
                continue

            plot = plots.overall_performance_clip(
                metric_per_task=metric_per_task.filter(pl.col("task").is_in(tasks)),
                y_label="Average Macro F1",
                title=f"{name} (standardized)",
                baseline_per_task={
                    t: v for t, v in baseline_per_task.items() if t in tasks
                },
            )
            plot.write_image(plot_dir / f"{filename}_f1.pdf", scale=2)

        for problem in ["remix", "permutation"]:
            performance_by_level = pl.concat(
                [
                    metric_per_task_with_baseline.filter(
                        c("task").is_in(get_tasks(f"level_{n}/{problem}"))
                    )
                    .filter(~c("model").str.contains("baseline"))
                    .group_by("model")
                    .agg(
                        score=c("metric").mean(),
                        error=c("metric").std() / (c("metric").len().sqrt() + 1e-6),
                    )
                    .with_columns(level=pl.lit(f"Level {n}"), group=pl.lit(name))
                    for n in range(2, 9)
                ]
            )
            fig = plots.levels_line_plot(performance_by_level)
            fig.write_image(plot_dir / f"_{problem}.pdf", scale=2)


if __name__ == "__main__":
    typer.run(plot_alfred)
