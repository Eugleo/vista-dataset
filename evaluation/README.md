# ViSTa Evaluation Code

This repo hosts the evaluation code for the ViSTa dataset. Generally, we'd recommend you to use the dataset as a standalone with your own evaluation code, for full flexibility.

## Usage with Your Own Data

_Note: Throughout the following instructions, "task" and "problem set" are used interchangeably._

Clone the repo. First, you will need to download the models you want to test. Assuming your cache directory is `.cache`:

- **S3D**: Run

  ```shell
  mkdir -p .cache/encoders/s3d
  wget -P .cache/encoders/s3d https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
  wget -P .cache/encoders/s3d https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
  ```

- **ViCLIP**: Download `ViCLIP-L_InternVid-FLT-10M.pth` from [HuggingFace](https://huggingface.co/OpenGVLab/ViCLIP/tree/main) and put it into `.cache/encoders/viclip`.

- **CLIP**: Nothing needs to be done, CLIP will get downloaded automatically.

- **OpenAI models**: Put your OpenAI API key into `.env` file in the project root:

  ```shell
  # inside .env
  OPENAI_API_KEY=
  ```

  You'll get to configure what model will be used in a config file (see below).

- Other models: It should be relatively easy to add additional encoder-based models (i.e. models that can embed videos and text, and whose predictions will be computed as a cosine similarity between the two embeddings). You can start with the `ViCLIP` class in `src/vlm/encoders.py` as an example.

You can now run the evaluation.

1. Create a virtual environment and install dependencies. We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for this.

2. Move the video files into a local directory. You can choose the folder structure within that directory as you like.
3. Create task definitons and labeled task data in a local directory (see section Tasks below). All files should be placed into this directory, without any additional folder structure. **The naming scheme is also important.** Check out the section Tasks below.
4. Define an experiment config file in `configs/`. You can use the one below as an inspiration

   ```yaml
   # Optional: Directory where encoders (viclip, ...) and cache are stored
   cache_dir: /data/datasets/vlm_benchmark/.cache
   # Optional: Directory where experiment results will be saved
   output_dir: /data/datasets/vlm_benchmark/experiments
   # Directory task definitions (yaml) and task data (json) are stored
   task_dir: /data/datasets/vlm_benchmark/tasks/habitat
   # Directory where the videos are stored
   # In labeled task data, the video path is taken relative to this directory
   video_dir: /data/datasets/vlm_benchmark/videos/habitat
   # The tasks to test
   # The script will look for [task_name].yaml in task_dir
   tasks:
       - open_cabinet
   # The models to test
   models:
       # We support two kinds of models: encoder and gpt
       - kind: encoder
           # Which encoder to use
           # We support: viclip, s3d, clip
           encoder: viclip
           # Which head(s) to use (currently only cosine is supported)
           heads:
           - kind: cosine
           # Optional, is 8 by default
           batch_size: 16
       - kind: encoder
           encoder: s3d
           heads:
           - kind: cosine
       - kind: encoder
           encoder: clip
           heads:
           - kind: cosine
           # CLIP models require the specific model that is to be used
           hf_model: ViT-bigG-14/laion2b_s39b_b160k
           # CLIP models also require the number of frames to average over
           n_frames: 32
       - kind: gpt
          # GPT models require the number of frames that are used as input
          n_frames: 5
          # GPT models also require the specific model to use (in the format you'd specify it in the OAI API)
          model: gpt-4o
   ```

   If you're using one of the GPT models, don't forget to put your OPENAI_API_KEY in the `.env` file.

5. Run the experiment by running the `vlm` executable with the `evaluate` command. E.g. if you used `uv` above:

   ```shell
   uv run vlm evaluate [path to config file]
   ```

   This will create a new folder in `output_dir`, and inside of it, a `results.csv` with results for all the tasks, models, and videos.

6. Generate the plots by running the `vlm` executable with the `plot` command. E.g. if you used `uv` above:

   ```shell
   uv run vlm plot --experiment-dir [...] --task-dir [...] [optional: experiment name]
   ```

   If no experiment name is given, the latest experiment in `experiment-dir` is loaded by default. This creates a folder `plots` inside of the experiment's directory (where `results.csv` is).

   We recommend you to create your own plotting function that will group the tasks in your dataset in the way you think is best (by default, each task is plotted separately). You can take a look at `plot_alfred` in `main.py` for ideas.

## Tasks

A task is one multiclass classification problem. The definition of a task `[task id]` has to be stored in `[task dir]/[task id].yaml`, and has to have the following structure:

```yaml
# in [task dir]/open_cabinet.yaml

# Required
label_prompts:
  # label id (used in the task data, see below): label description (given to the evaluation head / to gpt as a part of the prompt)
  actively_opening: a video where a kitchen cabinet is being opened
  observing_opened: a video where we observe an open kitchen cabinet
  observing_closed: a video where we observe a closed kitchen cabinet
  actively_closing: a video where a kitchen cabinet is being closed
# Optional, only needed for the GPT models
prompt_gpt: [...]
# Optional, only needed for the projection head that we currently don't support
prompt_baseline: a container
# Optional, not used by the evaluate code, but can be useful for custom plotting
metadata: [any dictionary]
```

Each task will contain some videos with descriptions. The data are stored in a separate directory, but the possible descriptions are stored with the task definition in `task_dir`. The data file for task `[task id]` have to be stored in `[task id]_data.json`. For example:

```js
// [task dir]/open_cabinet_data.json
[
  {
    path: "open_cabinet/cabinet_close_1.mp4",
    label: "actively_closing",
  },
  {
    path: "open_cabinet/cabinet_open_1.mp4",
    label: "actively_opening",
  },
  {
    path: "open_cabinet/cabinet_observe_opened_1.mp4",
    label: "observing_opened",
  },
];
```

The video paths are taken relative to the `video_dir`, which is specified in the experiment config. The labels (or more precisely, the label ids here) reference one of the label ids in `label_prompts` from the task definiton.
