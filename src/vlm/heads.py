import torch as t
import torch.nn as nn

from vlm.encoders import TextEncoder
from vlm.objects import Task


class Head(nn.Module):
    id: str


# def _compute_projection(direction: Tensor, alpha: float) -> Tensor:
#     projection = direction.T @ direction / t.norm(direction) ** 2
#     identity = t.diag(t.ones(projection.shape[0])).to(projection.device)
#     projection = alpha * projection + (1 - alpha) * identity
#     return projection


# class ProjectionHead(Head):
#     def __init__(self, target, direction, projection, alpha):
#         super().__init__()
#         self.register_buffer("target", target)
#         self.register_buffer("direction", direction)
#         self.register_buffer("projection", projection)
#         self.alpha = alpha
#         self.id = "projection"

#     @t.inference_mode()
#     def forward(self, x: t.Tensor) -> t.Tensor:
#         x = x / t.norm(x, dim=-1, keepdim=True)
#         y = 1 - (t.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
#         return y

#     @staticmethod
#     def for_task(task: Task, encoder: TextEncoder, alpha: float) -> "ProjectionHead":
#         target_prompts = list(task.label_prompts.values())
#         assert task.prompt_baseline is not None
#         labels = encoder.encode_text(target_prompts)
#         baseline = encoder.encode_text([task.prompt_baseline])
#         directions = labels - baseline
#         projections = t.Tensor([_compute_projection(d, alpha) for d in directions])

#         return ProjectionHead(labels, directions, projections, alpha)


class CosineHead(Head):
    def __init__(self, labels):
        super().__init__()
        self.id = "cosine"
        self.register_buffer("options", labels)

    @t.inference_mode()
    def forward(self, x: t.Tensor) -> t.Tensor:
        self.options = self.options.to(x.device)
        return (x @ self.options.T).softmax(dim=-1)

    @staticmethod
    def for_task(task: Task, encoder: TextEncoder) -> "CosineHead":
        target_prompts = list(task.label_prompts.values())
        target = encoder.encode_text(target_prompts)
        return CosineHead(target)
