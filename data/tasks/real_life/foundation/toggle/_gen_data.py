import json

path_prefix = "/data/datasets/vlm_benchmark/tasks/real_life"
path_suffix = "foundation/toggle"

objects = ['desk_lamp', 'floor_lamp', 'microwave', 'sink']
actions = ['turn_on', 'turn_off']

for obj in objects:
    data = []
    for action in actions:
        if obj == 'floor_lamp':
            for i in range(1, 4):
                data.append({
                    "path": f"{path_suffix}/{obj}_{action}_{i}.mp4",
                    "label": action,
                })
        else:
            data.append({
                "path": f"{path_suffix}/{obj}_{action}.mp4",
                "label": action,
            })

    with open(f'{path_prefix}/{path_suffix}/{obj}_data.json', 'w') as f:
        json.dump(data, f, indent=4)
