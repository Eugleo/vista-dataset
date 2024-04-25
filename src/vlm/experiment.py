import time

from human_id import generate_id
from pydantic.dataclasses import dataclass

from vlm.datamodel import Task
from vlm.models import Model


@dataclass
class Experiment:
    id: str
    tasks: list[Task]
    datasets: list[str]
    models: list[Model]

    data_dir: str = "data"
    cache_dir: str = ".cache"
    output_dir: str = "experiments"

    @staticmethod
    def from_config(config):
        id = generate_id(word_count=2) + "_" + time.strftime("%Y%m%d%H%M%S")
        return Experiment(
            id,
            [Task.from_dict(task) for task in config["tasks"]],
            config["datasets"],
            config["models"],
        )

    def run(self): ...
