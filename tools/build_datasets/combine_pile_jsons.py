import json
import ipdb
import os
from tqdm import tqdm

#num_splits = 165
num_splits = 41
#num_splits = 1
output_path = os.path.join(
        '/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/',
        f'pile_up_to_{num_splits}-of-01650.json')

with open(output_path, 'w') as fout:
    for sp_idx in tqdm(list(range(num_splits))):
        input_path = os.path.join(
                '/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/',
                f'split_{sp_idx:05}-of-01650.json')
        with open(input_path, 'r') as fin:
            all_lines = fin.readlines()
            fout.writelines(all_lines)
