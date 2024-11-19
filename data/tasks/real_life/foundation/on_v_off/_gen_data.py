import json

path_prefix = "/data/datasets/vlm_benchmark/tasks/real_life"
path_suffix = "foundation/on_v_off"

objects = ['desk_lamp', 'floor_lamp', 'microwave', 'sink']
states = ['on', 'off']

# bash script to rename every file containing "desklamp" to "desk_lamp":
# for f in *desklamp*; do mv -- "$f" "${f//desklamp/desk_lamp}"; done

for obj in objects:
    data = []
    for state in states:
        if obj == 'floor_lamp':
            for i in range(1, 4):
                data.append({
                    "path": f"{path_suffix}/{obj}/{state}/{obj}_{state}_{i}.mp4",
                    "label": f"state_{state}",
                })
        else:
            data.append({
                "path": f"{path_suffix}/{obj}/{state}/{obj}_{state}.mp4",
                "label": f"state_{state}",
            })

    with open(f'{path_prefix}/{path_suffix}/{obj}_data.json', 'w') as f:
        json.dump(data, f, indent=4)
