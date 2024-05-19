import logging
from functools import partial
from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import typer
from polars import col as c
from rich.progress import Progress, SpinnerColumn, TextColumn
from vlm import plots, utils
from vlm.config import ExperimentConfig
from vlm.objects import Task

app = typer.Typer()


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
def plot(
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
    task_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/tasks",
):
    if experiment is None:
        experiments = [d for d in Path(experiment_dir).iterdir() if d.is_dir()]
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
        print(f"Loading the most recent experiment: {dir}...")
    else:
        dir = Path(experiment_dir) / experiment

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Creating plots...")
        df = pl.read_json(dir / "results.json")

        scores = utils.standardize(df)
        predictions = utils.get_predictions(scores)
        metrics = (
            predictions.group_by(["task", "model"])
            .agg(accuracy=utils.compute_metric(utils.accuracy))
            .sort("task", "model")
        )

        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "problems.csv")

        group_names = set(t.split("/")[1] for t in df.get_column("task").unique())
        groups = {
            name: [t for t in df.get_column("task").unique() if t.startswith(name)]
            for name in group_names
        }

        for name, tasks in groups.items():
            tasks = [t for t in tasks if len(df.filter(c("task") == t)) > 0]
            if not tasks:
                continue
            labels = {
                t: list(
                    Task.from_file(t, Path(task_dir) / f"{t}.yaml").label_prompts.keys()
                )
                for t in tasks
            }
            baselines = {t: 1 / len(labels[t]) for t in tasks for t in tasks}
            plot = plots.task_performance(
                metrics.filter(pl.col("task").is_in(tasks)),
                predictions.filter(pl.col("task").is_in(tasks)),
                metric="accuracy",
                title=f"{name}",
                baselines=baselines,
                labels=labels,
                tasks=tasks,
            )
            plot_dir = dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)
            plot.write_image(plot_dir / f"{name}_performance.pdf")

            plot = plots.overall_performance(
                metrics.filter(pl.col("task").is_in(tasks)),
                metric="accuracy",
                title=f"{name} (standardized)",
            )
            plot.write_image(plot_dir / f"{name}_overall.pdf")

        print("Plots seed")


@app.command()
def plot_alfred(
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
    task_dir: Annotated[str, typer.Option()] = "/data/datasets/vlm_benchmark/tasks",
):
    if experiment is None:
        experiments = [d for d in Path(experiment_dir).iterdir() if d.is_dir()]
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
        print(f"Loading the most recent experiment: {dir}...")
    else:
        dir = Path(experiment_dir) / experiment

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Creating plots...")
        df = pl.read_json(dir / "results.json")
        scores = utils.standardize(df)
        scores = utils.add_random_baseline(scores)

        task_labels = {
            t: list(
                Task.from_file(t, Path(task_dir) / f"{t}.yaml").label_prompts.keys()
            )
            for t in df.get_column("task").unique()
        }

        per_label_metrics = (
            # This sort is very important
            scores.group_by("task", "model")  # .sort("task", "model", "video", "label")
            .agg(
                mAP=pl.struct("task", "label", "score", "true_label").map_batches(
                    partial(plots.mean_average_precision, task_labels)
                )
            )
            .explode("mAP")
            .sort("task", "model")
        )

        predictions = utils.get_predictions(df)
        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "problems.csv")

        clean_tasks = [
            t
            for t in df.get_column("task").unique()
            if t.startswith("foundation/clean")
        ]
        heat_tasks = [
            t for t in df.get_column("task").unique() if t.startswith("foundation/heat")
        ]
        cool_tasks = [
            t for t in df.get_column("task").unique() if t.startswith("foundation/cool")
        ]

        groups = {
            "The whole Foundation Level": [
                "foundation/object_recognition/pick_from_counter_top",
                "foundation/object_recognition/pick_from_dining_table",
                "foundation/object_recognition/pick_from_somewhere",
                "foundation/container_recognition/place_butter_knife",
                "foundation/container_recognition/place_soap_bar",
                "foundation/container_recognition/place_mug",
                "foundation/container_recognition/place_key_chain",
            ]
            + clean_tasks
            + heat_tasks
            + cool_tasks,
            "Cleaning v. just putting into a sink": clean_tasks,
            "Heating v. just putting into a microwave": heat_tasks,
            "Cooling v. just standing in front of a fridge": cool_tasks,
            "Object Recognition": [
                "foundation/object_recognition/pick_from_counter_top",
                "foundation/object_recognition/pick_from_dining_table",
                "foundation/object_recognition/pick_from_somewhere",
            ],
            "Container Recognition": [
                "foundation/container_recognition/place_butter_knife",
                "foundation/container_recognition/place_soap_bar",
                "foundation/container_recognition/place_mug",
                "foundation/container_recognition/place_key_chain",
            ],
            # "Location Recognition, kinda broken": [
            #     "level_1/object_recognition/goto/location"
            # ],
        }

        for name, tasks in groups.items():
            plot = plots.map_plot(
                per_label_metrics.filter(pl.col("task").is_in(tasks)),
                f"{name} (standardized)",
            )
            plot_dir = dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)
            plot.write_image(plot_dir / f"{name} mAP.pdf")

            plot = plots.overall_performance(
                per_label_metrics.filter(pl.col("task").is_in(tasks)),
                metric="mAP",
                metric_label="Mean AP over all labels",
                title=f"{name} (standardized)",
            )
            plot.write_image(plot_dir / f"{name} mAP, overall.pdf")


@app.command()
def plot_habitat(
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    interactive: Annotated[bool, typer.Option()] = False,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
):
    if experiment is None:
        experiments = [d for d in Path(experiment_dir).iterdir() if d.is_dir()]
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
        print(f"Loading the most recent experiment: {dir}...")
    else:
        dir = Path(experiment_dir) / experiment

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Creating plots...")
        df = pl.read_json(dir / "results.json")

        predictions = utils.get_predictions(df, standardize=True)
        metrics = predictions.group_by(["task", "model"]).agg(
            accuracy=utils.compute_metric(utils.accuracy)
        )

        problems = plots.incorrect_video_labels(predictions)
        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        problems.write_csv(plot_dir / "problems.csv")

        groups = {
            "Static v. Changing": [
                "closed_or_closing/cabinet",
                "closed_or_closing/drawer",
                "closed_or_closing/fridge",
                "opening_or_closing/cabinet",
                "opening_or_closing/drawer",
                "opening_or_closing/fridge",
            ],
            "Open v. Opening": [
                "open_or_opening/cabinet",
                "open_or_opening/drawer",
                "open_or_opening/fridge",
            ],
            "Open v. Closed": [
                "open_or_closed/cabinet",
                "open_or_closed/drawer",
                "open_or_closed/fridge",
            ],
            "Opening v. Closing Specific Container": ["opening_or_closing/container"],
            "Container Type": [
                "container_type/apple",
                "container_type/can",
                "container_type/hammer",
            ],
            "Near v. Far": ["near_or_far/chair", "near_or_far/plant", "near_or_far/tv"],
            "Small Object": ["object/small"],
            "Large Object": ["object/large"],
            "Object in Container": ["object/small_in_container"],
            "Object in Room": ["object/large_in_room"],
            "Room": ["room/model", "room/scan"],
            "Find Room": ["room/find"],
            "Sequence of Rooms": ["room/sequence"],
            "Move Can": ["move_can"],
            "Walking Towards v. Away": [
                "walking_towards_or_away/chair",
                "walking_towards_or_away/plant",
                "walking_towards_or_away/tv",
            ],
        }

        for name, tasks in groups.items():
            tasks = [t for t in tasks if len(df.filter(c("task") == t)) > 0]
            if not tasks:
                continue
            baselines = {
                task: 1
                / df.filter(c("task") == task).get_column("true_label").n_unique()
                for task in tasks
            }
            plot = plots.task_performance(
                metrics.filter(pl.col("task").is_in(tasks)),
                predictions.filter(pl.col("task").is_in(tasks)),
                metric="accuracy",
                title=f"{name} (standardized)",
                baselines=baselines,
                tasks=tasks,
            )
            plot_dir = dir / "plots"
            plot_dir.mkdir(exist_ok=True, parents=True)
            plot.write_image(plot_dir / f"{name}_performance.pdf")

            plot = plots.overall_performance(
                metrics.filter(pl.col("task").is_in(tasks)),
                metric="accuracy",
                title=f"{name} (standardized)",
            )
            plot.write_image(plot_dir / f"{name}_overall.pdf")

        print("Plots seed")
