import itertools
import json

path_prefix = "/data/datasets/vlm_benchmark/tasks/real_life"
path_suffix = "foundation/sliced_v_whole"

objects = ['apple', 'potato']
states = [('slice', 'slice'), ('whole', '')]
hands = ['lh', 'rh']
locations = ['c', 't']

for obj in objects:
    data = []
    for (state_label, state_name), location, hand in itertools.product(states, locations, hands):
        if obj == 'potato':
            for i in range(1, 3):
                if state_label == 'slice':
                    data.append({
                        # not sliced_v_whole because some videos are shared with pick_object
                        "path": f"foundation/pick_object/{obj}{state_name}_{hand}_{location}_{i}.mp4",
                        "label": f"{state_label}",
                    })
                else:
                    # sub potato_lh_c_1 and potato_lh_c_2 with potato_lh_c1 and potato_lh_c2
                    # (the formatting is inconsistent because pick_object uses different formatting)
                    if hand == 'lh' and location == 'c':
                        data.append({
                            # not sliced_v_whole because some videos are shared with pick_object
                            "path": f"foundation/pick_object/{obj}{state_name}_{hand}_{location}{i}.mp4",
                            "label": f"{state_label}",
                        })
                    else:
                        data.append({
                            # not sliced_v_whole because some videos are shared with pick_object
                            "path": f"foundation/pick_object/{obj}{state_name}_{hand}_{location}_{i}.mp4",
                            "label": f"{state_label}",
                        })
        else:
            data.append({
                # not sliced_v_whole because some videos are shared with pick_object
                "path": f"foundation/pick_object/{obj}{state_name}_{hand}_{location}.mp4",
                "label": f"{state_label}",
            })



    with open(f'{path_prefix}/{path_suffix}/{obj}_data.json', 'w') as f:
        json.dump(data, f, indent=4)
