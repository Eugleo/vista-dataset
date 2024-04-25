import torch
import torch.nn as nn
from torch import Tensor

from vlm.datamodel import Task
from vlm.encoders import TextEncoder

Head = nn.Module


def _compute_projection(direction: Tensor, alpha: float) -> Tensor:
    projection = direction.T @ direction / torch.norm(direction) ** 2
    identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
    projection = alpha * projection + (1 - alpha) * identity
    return projection


class ProjectionHead(Head):
    def __init__(self, baseline, target, direction, projection, alpha):
        super().__init__()
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)
        self.register_buffer("projection", projection)
        self.alpha = alpha

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        return y

    @staticmethod
    def for_task(task: Task, encoder: TextEncoder, alpha: float) -> "ProjectionHead":
        target_prompts = list(task.label_prompts.values())
        baseline_prompts = [task.prompt_baseline]
        target = encoder.encode_text(target_prompts).mean(dim=0, keepdim=True)
        baseline = encoder.encode_text(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        projection = _compute_projection(direction, alpha)

        return ProjectionHead(baseline, target, direction, projection, alpha)


def _logit_reward(x: Tensor, labels: Tensor, target: Tensor) -> Tensor:
    return (x @ labels.T).softmax(dim=-1)[:, target]


class CosineHead(Head):
    def __init__(self, baselines, target):
        super().__init__()
        self.register_buffer("options", torch.cat([target, baselines]))

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.options = self.options.to(x.device)
        return _logit_reward(x, self.options, target=torch.tensor(0).to(x.device))

    @staticmethod
    def from_model(task: Task, encoder: TextEncoder) -> "CosineHead":
        target_prompts = list(task.label_prompts.values())
        baseline_prompts = [task.prompt_baseline]
        target = encoder.encode_text(target_prompts).mean(dim=0, keepdim=True)
        baselines = encoder.encode_text(baseline_prompts)

        return CosineHead(baselines, target)
