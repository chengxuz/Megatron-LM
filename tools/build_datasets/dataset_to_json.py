from llm_devo.datasets import miniBERTa
import json
import ipdb
import os

input_dataset = miniBERTa.get_sw_150M_subset(
        just_dataset=True)
target_path = os.path.join(
        '/om2/group/evlab/llm_dataset/Megatron_datasets/',
        'sw_150M_raw.json')

with open(target_path, 'w') as fout:
    for idx, item in enumerate(input_dataset):
        item['src'] = 'Smashwords'
        item['type'] = 'Eng'
        item['id'] = idx
        now_str = json.dumps(item)
        fout.write(now_str + '\n')
