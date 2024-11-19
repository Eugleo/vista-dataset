import json

hands = ['lh', 'rh']

# start out with countertop

countertop_objects = [
    'apple',
    'appleslice',
    'bowl',
    'butterknife',
    'cup',
    'handtowel',
    'kettle',
    'knife',
    'ladle',
    'pan',
    'pen',
    'plate',
    'pot',
    'potato',
    'potatoslice',
    'soapbottle',
    'spatula',
    'spoon',
]

data = []

for obj in countertop_objects:
    for hand in hands:
        if obj in ['apple', 'appleslice']:
            data.append({
                "path": f"foundation/pick_object/{obj}_{hand}_c.mp4",
                "label": obj,
            })
        else:
            for i in range(1, 3):
                if obj != 'potatoslice':
                    data.append({
                        "path": f"foundation/pick_object/{obj}_{hand}_c{i}.mp4",
                        "label": obj,
                    })

                if obj in ['potato', 'potatoslice']:  # potato and potatoslice have both this and above
                    if not (obj == 'potato' and hand == 'lh'):
                        data.append({
                            "path": f"foundation/pick_object/{obj}_{hand}_c_{i}.mp4",
                            "label": obj,
                        })
                    
with open('/data/datasets/vlm_benchmark/tasks/real_life/foundation/objects/countertop_data.json', 'w') as f:
    json.dump(data, f, indent=4)


# now do table

table_objects = [
    'apple',
    'appleslice',
    'book',
    'pencil',
    'potato',
    'potatoslice',
]

data = []

for obj in table_objects:
    for hand in hands:
        if obj in ['apple', 'appleslice']:
            data.append({
                "path": f"foundation/pick_object/{obj}_{hand}_t.mp4",
                "label": obj,
            })
        else:
            for i in range(1, 3):
                if obj in ['potato', 'potatoslice']:
                    data.append({
                        "path": f"foundation/pick_object/{obj}_{hand}_t_{i}.mp4",
                        "label": obj,
                    })
                else:
                    data.append({
                        "path": f"foundation/pick_object/{obj}_{hand}_t{i}.mp4",
                        "label": obj,
                    })



with open('/data/datasets/vlm_benchmark/tasks/real_life/foundation/objects/table_data.json', 'w') as f:
    json.dump(data, f, indent=4)
