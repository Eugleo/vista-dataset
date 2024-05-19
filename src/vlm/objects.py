from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

import polars as pl
import yaml


@dataclass
class Task:
    id: str
    label_prompts: dict[str, str]
    concepts: dict[str, Any] = field(default_factory=dict)

    prompt_gpt: Optional[str] = None
    prompt_baseline: Optional[str] = None

    @property
    def labels(self) -> list[str]:
        return list(self.label_prompts.keys())

    @staticmethod
    def from_file(id: str, path: Union[str, Path]) -> "Task":
        with open(path) as f:
            data = yaml.safe_load(f)
        return Task(id=id, **data)


@dataclass
class Video:
    path: str
    labels: dict[str, str]


class Model(Protocol):
    id: str

    def predict(self, videos: list[Video], tasks: list[Task]) -> Optional[pl.DataFrame]:
        """Predict the probability of each label in each task for each video.
        Returns a DataFrame with the following columns:
        - task: The task ID
        - head: The head ID
        - video: The path to the video
        - label: The label for which the probability is predicted
        - prob: The predicted probability
        - true_prob: The true probability (1.0 if the label is correct, 0.0 otherwise)
        """
        ...


@dataclass
class Experiment:
    id: str
    tasks: list[Task]
    videos: list[Video]
    models: list[Callable[[], Model]]
    config_file: str

    output_dir: str

    def run(self) -> pl.DataFrame:
        print(
            f"Running {len(self.models)} models on {len(self.tasks)} tasks ({len(self.videos)} unique videos)"
        )
        results: list[pl.DataFrame] = []
        for get_model in self.models:
            model = get_model()
            print(f"Running model {model.id}")
            result = model.predict(self.videos, self.tasks)
            if result is None:
                continue
            results.append(result.drop("metadata"))
        result = pl.concat(results)

        output_dir = Path(self.output_dir) / self.id
        output_dir.mkdir(exist_ok=True, parents=True)
        (output_dir / "config.yaml").write_text(Path(self.config_file).read_text())
        result.write_json(output_dir / "results.json")
        return result
