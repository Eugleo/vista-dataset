import json

objects = ['apple', 'butter_knife', 'can', 'hammer', 'mug']
hands = ['lh', 'rh']
goals = ['c', 'f', 'm', 's']
goal_names = {
    'c': 'counter',
    'f': 'freezer',
    'm': 'microwave',
    's': 'sink',
}

for obj in objects:
    data = []
    for hand in hands:
        for goal in goals:
            # every other location
            sources = [g for g in goals if g != goal]
            goal_name = goal_names[goal]
            for source in sources:
                data.append({
                    "path": f"foundation/put_object/{goal_name}/{obj}/{obj}_{hand}_{source}-{goal}.mp4",
                    "label": goal_name,
                })

    with open(f'/data/datasets/vlm_benchmark/tasks/real_life/foundation/containers/{obj}_data.json', 'w') as f:
        json.dump(data, f, indent=4)
