from typing import Annotated

import typer
import yaml
from vlm import utils
from vlm.experiment import Experiment

# def _load_encoder(model_name: str, args, device: torch.device) -> Encoder:
#     assert model_name in ["viclip", "s3d", "clip"]

#     logging.info(f"Loading encoder {model_name}")

#     if model_name == "viclip":
#         encoder = ViCLIP(args.cache_dir)
#     elif model_name == "s3d":
#         encoder = S3D(args.cache_dir)
#     elif model_name == "clip":
#         if args.n_frames is None:
#             raise ValueError("Number of frames must be provided when using CLIP.")
#         model = "ViT-bigG-14/laion2b_s39b_b160k"
#         model_name_prefix, pretrained = model.split("/")
#         encoder = CLIP(
#             model_name_prefix,
#             pretrained,
#             args.cache_dir,
#             expected_n_frames=int(args.n_frames),
#         )
#     return encoder.to(device)


app = typer.Typer()


@app.command()
def evaluate(config: Annotated[str, typer.Argument()]):
    experiment = Experiment.from_config(yaml.safe_load(open(config)))
    results = experiment.run()
    overall_performance = utils.performance_per_task(results)
    overall_performance.save(experiment.output_dir)
    for task in experiment.tasks:
        confusion_matrix = utils.confusion_matrix(results, task.id)
        confusion_matrix.save(experiment.output_dir)
