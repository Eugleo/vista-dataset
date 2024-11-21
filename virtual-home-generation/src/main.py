from pathlib import Path
from typing import Annotated

import typer
from alfred import utils

app = typer.Typer()


@app.command()
def generate(
    metadata: Annotated[str, typer.Argument()] = "metadata",
    output: Annotated[
        str, typer.Argument()
    ] = "/Users/eugen/Downloads/Projects/mats/vlm-benchmark",
    dry_run: Annotated[bool, typer.Option()] = False,
    levels: Annotated[int, typer.Option()] = 6,
):
    metadata_path = Path(metadata)
    trajectories = utils.load_trajectories(metadata_path)
