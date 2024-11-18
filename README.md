# ViSTa Supplementary Material

ViSTa is a dataset for evaluating <b>Vi</b>sion-based understanding of <b>S</b>equential <b>Ta</b>sks.
[todo: arxiv link]
ViSTa comprises over 4,000 videos with step-by-step descriptions in virtual home, Minecraft, and real-world environments.
It has a hierarchical structure: basic single-step tasks compose into more and more complex sequential tasks.
ViSTa is intended to allow precise quantification of models' sequential reasoning capabilities across sequences of different lengths,
with a focus on capabilities required for using vision-language models (VLMs) as reward models in reinforcement learning.

![Level 1 of ViSTa has videos containing a single action, like "We pick up a banana" or "We put a banana in a closet"; Level 2 has videos of two actions in sequence, like "We pick up a banana, then we put the banana in a closet"; this continues through Level 8, which has videos of sequences of eight actions, like "We pick up a banana, then [...], then we put the banana in a closet, then [...]".](./.assets/dataset_overview.png?raw=true)

## Usage

[todo: video data link]

[todo: update the old description:


The videos have to be downloaded separately (>7GB in size), but the tasks and metadata are included directly in the supplementary material. 

The directory `evaluation` contains code that can be used to replicate the results from the paper. For people who would like to use the dataset directly with _their_ evaluation code, we also supply all the video metadata in `dataset.json`, although our code uses a slightly different structure of the same data.

For more information, see the documentation:

- `documentation.pdf` contains description of the dataset, the data collection protocols, and details about evaluation
- `datasheet.pdf` contains documentation in the datasheet for datasets format
- `croissant.json` contains the croissant spec for the dataset]


## Levels
### Single-action videos (level 1)
These test if a model can identify fundamental actions like "mine a wooden block", "open a door", or "put a banana into the closet". 
The actions are sometimes quite complex: for example, the video "heat up an apple" shows the agent putting the apple in a microwave, turning it on, waiting, then picking the apple back up.

### Multiple-action videos (levels 2 through 8)
These use sequences of the fundamental actions like "pick up an apple, then put the apple in the drawer" to test if a model understands action order and if it notices when we swap out actions for different ones.
Sequences in level $n$ contain $n$ fundamental actions.


## Problem sets

ViSTa groups the video-description pairs into <i>problem sets</i>: classification problems testing specific capabilities.
During the evaluation of a problem set, models get a video and must score how well it matches each description from the problem set.

[todo: description of where to find the problem sets here]

### Objects
These are Level 1 (single-action) problem sets which test object recognition and contain videos such as "We pick up an <b>apple</b>" and "We pick up a <b>hammer</b>".

### Object properties
These are Level 1 (single-action) problem sets which test detection of specific object properties—open/closed, turned on/turned off, etc.
They have videos such as "We observe an <b>open</b> drawer" and "We observe a <b>closed</b> drawer".


### Actions
These are Level 1 (single-action) problem sets which test understanding of particular actions (heating, cooling, cleaning, etc.).
The videos include "We <b>heat up</b> a banana", or "We put a banana in a microwave <b>without turning it on</b>".


### General problems
Level $n$ (multiple-action) problem sets which test general sequential task understanding, e.g. "We open a drawer, then we pick up a banana from the drawer."
Models must determine which of several possible sequences of actions matches the video.

### Permutation problems

Level $n$ (multiple-action) problem sets testing whether the model can understand action order. 
In a given problem set, the videos are permutations of the same actions, differing only in their order. 

![An example problem set for action-order understanding in ViSTa: two videos which have the same actions in different orders. In this case, the video descriptions are "We put a fork in a sink, then we turn the water on, then we pick up the fork from the sink" and "We pick up a fork from a sink, then we put the fork into a sink, then we turn the water on".](./.assets/problem_set.png?raw=true)

## Environments

Some videos in ViSTa are from existing datasets; most are manually filmed or edited. 

### Virtual home
ViSTa contains more than 3,000 videos in the virtual home environment, in levels 1–8.
The videos are clips from [ALFRED](https://github.com/askforalfred/alfred) and combinations thereof.

[todo: a slightly more detailed breakdown of the numbers of videos here]

### Real world
ViSTa contains more than 1,100 videos in the real world environment, of which 200 are sourced from [Kinetics-700](https://github.com/cvdfoundation/kinetics-dataset); the rest were created specifically for ViSTa.

* 810 videos (in levels 1–5 and 8) are directly analogous to the virtual home videos: they show the agent (us) doing tasks in the real world.
* 95 videos test object tracking: they show similar objects being shuffled.
* 18 videos test understanding of object interactions: they show us pinning fabric, or falsely seeming to do so.
* 200 videos test action recognition in complex contexts, sourced from [Kinetics-700](https://github.com/cvdfoundation/kinetics-dataset): they show either a door opening or a door closing.

### Minecraft
In Minecraft, ViSTa has 53 videos in levels 1–3.
Most were created manually; the rest were sourced from the [BASALT benchmark](https://github.com/minerllabs/basalt-benchmark).


## Results on current VLMs

In 2024, we used ViSTa to evaluate three current VLMs: [CLIP](https://github.com/openai/CLIP), [ViCLIP](https://github.com/OpenGVLab/InternVideo/tree/main), and [GPT-4o](https://openai.com/index/hello-gpt-4o/).
Unsurprisingly, GPT-4o was significantly better than the open-source models.
All models were good at recognizing objects, but had a harder time recognizing object properties and actions.
None of the models were able to understand sequences of tasks well.

[todo: the figures from the main body of the paper]

## Authors

[Evžen Wybitul](https://github.com/Eugleo), [Evan Ryan Gunter](https://github.com/evgunter), [Mikhail Seleznyov](https://github.com/Dont-Care-Didnt-Ask), [David Lindner](https://github.com/david-lindner)