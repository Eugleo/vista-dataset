import time
from typing import Annotated, Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from vlm.config import ExperimentConfig

app = typer.Typer()


@app.command()
def evaluate(config: Annotated[str, typer.Argument()]):
    experiment_config = ExperimentConfig.from_file(config)
    experiment = ExperimentConfig.to_experiment(experiment_config)
    experiment.run()


@app.command()
def plot(experiment: Annotated[Optional[str], typer.Argument()]):
    raise NotImplementedError("Plotting is not yet implemented")
