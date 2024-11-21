import json

path_prefix = "/data/datasets/vlm_benchmark/tasks/real_life"
path_suffix = "foundation/pick_v_put"


# first do pick

objects = ['apple', 'butter_knife', 'can', 'hammer', 'mug', 'fork']
locations = ['c', 'f', 'm', 's']
hands = ['lh', 'rh']

location_names = {
    'c': 'counter',
    'f': 'freezer',
    'm': 'microwave',
    's': 'sink',
}

for obj in objects:
    if obj == 'fork':
        locs = ['c']
    else:
        locs = locations
    for loc in locs:
        data = []
        for hand in hands:
            if obj == 'fork':
                for i in range(1, 3):
                    data.append({
                        "path": f"{path_suffix}/{obj}_pick_{hand}_c{i}.mp4",
                        "label": "pick",
                    })
                    data.append({
                        "path": f"{path_suffix}/{obj}_put_{hand}_c{i}.mp4",
                        "label": "put",
                    })
            else:
                for ll in [l for l in locations if l != loc]:
                    data.append({
                        "path": f"{path_suffix}/pick_{obj}_{hand}_{loc}-{ll}.mp4",
                        "label": "pick",
                    })

                    # put is basically the same as for containers (since all of these are putting in a container), except for the labels
                    loc_name = location_names[loc]
                    data.append({
                        "path": f"foundation/put_object/{loc_name}/{obj}/{obj}_{hand}_{ll}-{loc}.mp4",
                        "label": 'put',
                    })

        if loc == 'c':
            loc_name = 'countertop'

        with open(f'{path_prefix}/{path_suffix}/{obj}_{loc_name}_data.json', 'w') as f:
            json.dump(data, f, indent=4)
            print(f'- foundation/pick_v_put/{obj}_{loc_name}')
