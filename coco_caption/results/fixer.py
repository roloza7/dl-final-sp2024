import json
import random
with open('captions_val2017_fakecap_results.json', 'r') as f:
    data = json.load(f)

with open(r'C:\Users\justi\Documents\GaTech\Spring2024\dl-final-sp2024\coco_caption\annotations\captions_train2017.json', 'r') as f:
    good_data = json.load(f)

print(data[0])

valid_id = []
for i in good_data['annotations']:
    valid_id.append(i['image_id'])

for i in data:
    if (i['image_id']) == 6818:
        breakpoint()
    idx = random.randint(0, len(valid_id))
    i['image_id'] = valid_id[idx]
    valid_id.pop(idx)


with open('captions_val2017_fakefixed.json', 'w') as f:
    json.dump(data, f)