import json
import logging
import subprocess
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Annotated, Optional

import dotenv
import jsonlines
import log_viewer
import openai
import polars as pl
import typer
from polars import col as c
from rich.pretty import pprint
from rich.progress import Progress, SpinnerColumn, TextColumn
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


@app.command()
def plot_alfred(
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
        ).filter(
            ~c("task").str.contains("permuted"), ~c("task").str.contains("substituted")
        )

        # scores = utils.rescale(df, in_each="label")
        scores = utils.rescale(df, in_each="video")
        scores = utils.add_random_baseline(scores)

        num_nulls = len(scores.filter(c("score").is_null()))
        if num_nulls > 0:
            print(f"WARNING: {num_nulls} null scores found")
            scores = scores.filter(c("score").is_not_null())

        task_labels = {}
        per_label_baselines = []
        for t in df.get_column("task").unique():
            try:
                task_labels[t] = Task.from_file(t, Path(task_dir) / f"{t}.yaml").labels
            except FileNotFoundError:
                continue
            with open(Path(task_dir) / f"{t}_data.json") as f:
                label_counts = Counter(v["label"] for v in json.load(f))
            for l in task_labels[t]:
                per_label_baselines += [
                    {
                        "task": t,
                        "model": "Ω random",
                        "AP": label_counts[l] / sum(label_counts.values()),
                    },
                    {
                        "task": t,
                        "model": "ξ random",
                        "AP": utils.get_baseline_ap(
                            sum(label_counts.values()), label_counts[l]
                        ),
                    },
                ]
        per_label_baselines = pl.DataFrame(per_label_baselines)

        per_label_metrics = (
            # This sort is very important
            scores.sort("task", "model", "video", "label")
            .group_by("task", "model", maintain_order=True)
            .agg(
                AP=pl.struct("task", "label", "score", "true_label").map(
                    partial(plots.average_precision, task_labels)
                )
            )
            .explode("AP")
            .sort("task", "model")
        )
        per_label_metrics = pl.concat([per_label_metrics, per_label_baselines])
        per_label_metrics.write_csv(dir / "per_label_metrics.csv")

        predictions = utils.get_predictions(scores)
        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "misclassification_num.csv")

        def get_tasks(prefix):
            return [t for t in df.get_column("task").unique() if t.startswith(prefix)]

        level_groups = (
            {
                f"Level {n}: permutation": get_tasks(f"level_{n}/permutation")
                for n in range(2, 9)
            }
            | {f"Level {n}: remix": get_tasks(f"level_{n}/remix") for n in range(2, 9)}
            | {f"Level {n}: overall": get_tasks(f"level_{n}") for n in range(2, 9)}
        )

        groups = {
            "The whole Foundation Level": get_tasks("foundation"),
            "Object Recognition": get_tasks("foundation/objects"),
            "Container Recognition": get_tasks("foundation/containers"),
            "State: On v. Off": get_tasks("foundation/on_v_off"),
            "State: Sliced v. Whole": get_tasks("foundation/sliced_v_whole"),
            "Action: Cleaning": get_tasks("foundation/clean"),
            "Action: Heating": get_tasks("foundation/heat"),
            "Action: Cooling": get_tasks("foundation/cool"),
            "Action: Putting down v. Picking up": get_tasks("foundation/pick_v_put"),
            "Action: Slicing": get_tasks("foundation/slice"),
            "Action: Toggling On v. Off": get_tasks("foundation/toggle"),
        } | level_groups

        group_task = progress.add_task("Creating group plots...", total=len(groups))
        for name, tasks in groups.items():
            progress.advance(group_task, 1)
            filename = name.replace(": ", "_").replace(" ", "-")
            # plot = plots.map_plot(
            #     per_label_metrics.filter(pl.col("task").is_in(tasks)),
            #     f"{name} (standardized)",
            # )
            # plot.write_image(plot_dir / f"{filename}_mAP.png", scale=2)

            plot_dir = dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)
            plot = plots.overall_performance(
                per_label_metrics.filter(pl.col("task").is_in(tasks)),
                metric="AP",
                metric_label="Mean AP over all labels",
                title=f"{name} (standardized)",
            )
            plot.write_image(plot_dir / f"{filename}_mAP.png", scale=2)

            details_dir = plot_dir / "details"
            details_dir.mkdir(exist_ok=True, parents=True)
            plot = plots.task_performance(
                metric_per_task=per_label_metrics.filter(pl.col("task").is_in(tasks))
                .group_by("task", "model")
                .agg(mAP=c("AP").mean())
                .filter(~pl.col("model").str.contains("random")),
                predictions_per_task=predictions.filter(
                    pl.col("task").is_in(tasks)
                ).filter(~pl.col("model").str.contains("random")),
                scores=scores.filter(pl.col("task").is_in(tasks)).filter(
                    ~pl.col("model").str.contains("random")
                ),
                metric="mAP",
                title=f"{name} (standardized)",
                baselines=None,
                labels=task_labels,
                tasks=tasks,
            )
            plot.write_image(details_dir / f"{filename}_matrices.png", scale=2)
        print(per_label_metrics)
        performance_by_level = pl.concat(
            [
                per_label_metrics.filter(c("task").is_in(tasks))
                .group_by("model")
                .agg(
                    score=c("AP").mean(),
                    error=c("AP").std() / (c("AP").len().sqrt() + 1e-6),
                )
                .with_columns(level=pl.lit(n), group=pl.lit(name))
                for n in range(2, 9)
                for name, tasks in [
                    ("Permutation", get_tasks(f"level_{n}/permutation")),
                    ("Remix", get_tasks(f"level_{n}/remix")),
                    ("Overall", get_tasks(f"level_{n}")),
                ]
            ]
        )
        fig = plots.levels_line_plot(performance_by_level)
        fig.write_image(plot_dir / "_levels_line_plot.png", scale=2)


@app.command()
def plot(
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

        # scores = utils.rescale(df, in_each="label")
        scores = utils.rescale(df, in_each="video")
        scores = utils.add_random_baseline(scores)

        num_nulls = len(scores.filter(c("score").is_null()))
        if num_nulls > 0:
            print(f"WARNING: {num_nulls} null scores found")
            scores = scores.filter(c("score").is_not_null())

        task_labels = {}
        per_label_baselines = []
        for t in df.get_column("task").unique():
            try:
                task_labels[t] = Task.from_file(t, Path(task_dir) / f"{t}.yaml").labels
            except FileNotFoundError:
                continue
            with open(Path(task_dir) / f"{t}_data.json") as f:
                label_counts = Counter(v["label"] for v in json.load(f))
            for l in task_labels[t]:
                per_label_baselines += [
                    {
                        "task": t,
                        "model": "Ω random",
                        "AP": label_counts[l] / sum(label_counts.values()),
                    },
                    {
                        "task": t,
                        "model": "ξ random",
                        "AP": utils.get_baseline_ap(
                            sum(label_counts.values()), label_counts[l]
                        ),
                    },
                ]
        per_label_baselines = pl.DataFrame(per_label_baselines)

        per_label_metrics = (
            # This sort is very important
            scores.sort("task", "model", "video", "label")
            .group_by("task", "model")
            .agg(
                AP=pl.struct("task", "label", "score", "true_label").map(
                    partial(plots.average_precision, task_labels)
                )
            )
            .explode("AP")
            .sort("task", "model")
        )
        per_label_metrics = pl.concat([per_label_metrics, per_label_baselines])
        per_label_metrics.write_csv(dir / "per_label_metrics.csv")

        predictions = utils.get_predictions(scores)
        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "misclassification_num.csv")

        groups = {
            f"Task: {t}": [t]
            for t in df.get_column("task").unique()
            if (Path(task_dir) / f"{t}.yaml").exists()
        }

        group_task = progress.add_task("Creating group plots...", total=len(groups))
        for name, tasks in groups.items():
            progress.advance(group_task, 1)
            # TODO: plot_alfred uses the commented-out line below; I'm not sure why "/" is replaced with "_" here
            # filename = name.replace(": ", "_").replace(" ", "-")
            filename = name.replace(": ", "_").replace(" ", "-").replace("/", "_")
            plot = plots.map_plot(
                per_label_metrics.filter(pl.col("task").is_in(tasks)),
                f"{name} (standardized)",
            )
            plot_dir = dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)
            
            plot.write_image(plot_dir / f"{filename}_mAP.png", scale=2)

            plot = plots.overall_performance(
                per_label_metrics.filter(pl.col("task").is_in(tasks)),
                metric="AP",
                metric_label="Mean AP over all labels",
                title=f"{name} (standardized)",
            )
            plot.write_image(plot_dir / f"{filename}_mAP.png", scale=2)

            details_dir = plot_dir / "details"
            details_dir.mkdir(exist_ok=True, parents=True)
            plot = plots.task_performance(
                metric_per_task=per_label_metrics.filter(pl.col("task").is_in(tasks))
                .group_by("task", "model")
                .agg(mAP=c("AP").mean())
                .filter(~pl.col("model").str.contains("random")),
                predictions_per_task=predictions.filter(
                    pl.col("task").is_in(tasks)
                ).filter(~pl.col("model").str.contains("random")),
                scores=scores.filter(pl.col("task").is_in(tasks)).filter(
                    ~pl.col("model").str.contains("random")
                ),
                metric="mAP",
                title=f"{name} (standardized)",
                baselines=None,
                labels=task_labels,
                tasks=tasks,
            )
            plot.write_image(details_dir / f"{filename}_matrices.png", scale=2)
