import json

path_prefix = "/data/datasets/vlm_benchmark/tasks/real_life"
path_suffix = "foundation/slice"

objects = ['apple', 'potato']
actions = ['slice', 'dontslice']
locations = ['c', 't']

for obj in objects:
    data = []
    for action in actions:
        for location in locations:
            for i in range(1, 3):
                data.append({
                    "path": f"{path_suffix}/{obj}_{action}_{i}_{location}.mp4",
                    "label": f"{action}",
                })

    with open(f'{path_prefix}/{path_suffix}/{obj}_data.json', 'w') as f:
        json.dump(data, f, indent=4)
