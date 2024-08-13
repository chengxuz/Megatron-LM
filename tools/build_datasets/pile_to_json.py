from datasets import load_dataset
import json
import ipdb
import os

input_dataset = load_dataset(
        'EleutherAI/the_pile_deduplicated',
        streaming=True)
dataset_iter = iter(input_dataset['train'])
num_splits = 165
per_sp_len = 81405
all_idx = 0

for sp_idx in range(num_splits):
    print(sp_idx)
    target_path = os.path.join(
            '/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/',
            f'split_{sp_idx:05}-of-01650.json')
    with open(target_path, 'w') as fout:
        for idx in range(per_sp_len):
            item = next(dataset_iter)
            item['src'] = 'Pile'
            item['type'] = 'Eng'
            item['id'] = all_idx
            now_str = json.dumps(item)
            fout.write(now_str + '\n')
            all_idx += 1
