import logging
from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import typer
from polars import col as c
from rich.progress import Progress, SpinnerColumn, TextColumn
from vlm import plots, utils
from vlm.config import ExperimentConfig

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
            task.replace("/", "-"): [task] for task in df.get_column("task").unique()
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


@app.command()
def plot_minecraft(
    experiment: Annotated[Optional[str], typer.Argument()] = None,
    interactive: Annotated[bool, typer.Option()] = False,
    experiment_dir: Annotated[
        str, typer.Option()
    ] = "/data/datasets/vlm_benchmark/experiments",
    standardize: bool = False,
):
    if experiment is None:
        experiments = Path(experiment_dir).iterdir()
        dir = sorted(experiments, key=lambda d: d.stat().st_mtime)[-1]
    else:
        dir = Path(experiment_dir) / experiment

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Creating plots...")
        df = pl.read_json(dir / "results.json")

        predictions = utils.get_predictions(df, standardize=standardize)
        metrics = predictions.group_by(["task", "model"]).agg(
            mcc=utils.compute_metric(utils.mcc)
        )
        metrics = metrics.sort("model")
        metrics = metrics.sort("task")

        plot_dir = dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)

        performance_plot = utils.performance_per_task(metrics)
        performance_plot.write_image(plot_dir / "task_performance.pdf")
        print("Performance plot saved")
        if interactive:
            performance_plot.show()

        for task in df["task"].unique().to_list():
            for model in df["model"].unique().to_list():
                task_df = predictions.filter(pl.col("task") == task).filter(
                    pl.col("model") == model
                )
                confusion_matrix = utils.confusion_matrix(task_df)
                confusion_matrix.figure_.set_figwidth(12)
                confusion_matrix.figure_.set_figheight(12)
                confusion_matrix.figure_.savefig(
                    plot_dir / f"{task}_{model}_confusion_matrix.pdf"
                )
        print("Confusion matrices saved")


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
