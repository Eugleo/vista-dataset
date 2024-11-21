import json
import itertools

path_prefix = "/data/datasets/vlm_benchmark/tasks/real_life"
path_suffix = "foundation/cool"

objects = ['apple', 'mug']
actions = [('freeze', 'freeze'), ('put_no_close', 'freezer_no_close'), ('open_no_put', 'no_freeze'), ('approach', 'freezer_approach')]
hands = ['lh', 'rh']
locations = ['c', 't']

for obj in objects:
    data = []
    for hand, source, sink, (action_label, action_name) in itertools.product(hands, locations, locations, actions):
        data.append({
            "path": f"foundation/cool/{obj}/{action_label}/{obj}_{action_name}_{hand}_{source}-{sink}.mp4",
            "label": action_label,
        })

    with open(f'{path_prefix}/{path_suffix}/{obj}_data.json', 'w') as f:
        json.dump(data, f, indent=4)


