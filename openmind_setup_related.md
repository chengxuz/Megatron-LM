# How to use Singularity to load the docker image from nvidia

First load the module: `module load openmind8/apptainer`, then:
```
singularity pull docker://nvcr.io/nvidia/pytorch:24.06-py3
```

This will start the shell inside the image:
```
singularity shell --writable-tmpfs -B /om,/om2 --nv /om2/user/chengxuz/docker_images/pytorch_24.06-py3.sif
```

## Data preprocess

NLTK is needed for data preprocessing. This can help install it:
```
pip install --user nltk
```

Preprocess command is:
```
python tools/preprocess_data.py --input "/om2/group/evlab/llm_dataset/Megatron_datasets/sw_150M_raw.json" --output-prefix "sw_150M_gpt2" --vocab-file gpt2-vocab.json  --tokenizer-type GPT2BPETokenizer --merge-file gpt2-merges.txt --append-eod
```

### Pile data preprocess

First run the `tools/build_datasets/pile_to_json.py` to download the splits into json files.

Then run the `tools/build_datasets/combine_pile_jsons.py` to combine the small jsons into one large jsons. The `num_splits` argument in the file needs to be correspondingly adjusted.

Then run this:
```
python tools/preprocess_data.py --input "/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/pile_up_to_165-of-01650.json" --output-prefix "/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/pile_up_to_165-of-01650" --vocab-file "/om2/user/chengxuz/megatron_related/gpt_ckpts/gpt2-vocab.json"  --tokenizer-type GPT2BPETokenizer --merge-file "/om2/user/chengxuz/megatron_related/gpt_ckpts/gpt2-merges.txt" --append-eod  --workers 10
```

## Test train

```
bash examples/gpt3/train_gpt2_345m.sh "/om2/user/chengxuz/megatron_related/gpt_test_train/gpt2_345m/ckpts" "/om2/user/chengxuz/megatron_related/gpt_test_train/gpt2_345m/tensorboards" "/om2/user/chengxuz/megatron_related/gpt_ckpts/gpt2-vocab.json" "/om2/user/chengxuz/megatron_related/gpt_ckpts/gpt2-merges.txt" "/om2/group/evlab/llm_dataset/Megatron_datasets/sw_150M_gpt2_text_document"
```

```
export ROOT_DIR="/om2/user/chengxuz/megatron_related" ; bash examples/gpt3/train_gpt2_1d3b.sh "${ROOT_DIR}/gpt_test_train/gpt2_1d3b/ckpts" "${ROOT_DIR}/gpt_test_train/gpt2_1d3b/tensorboards" "${ROOT_DIR}/gpt_ckpts/gpt2-vocab.json" "${ROOT_DIR}/gpt_ckpts/gpt2-merges.txt" "/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/pile_up_to_165-of-01650_text_document"
```

## General train

Something like below:
```
sbatch --job-name=gpt2_2d7b --export=SETTING="baselines.py:change_to_2d7b_from_1d3b",NUM_GPUS=8 sb_scripts/train_gnrl.sh
```

# Evaluation related 

## How to build Sandbox from sif file

```
singularity build --sandbox /om2/user/chengxuz/docker_images/megatron_sbox /om2/user/chengxuz/docker_images/pytorch_24.06-py3.sif
```
Sandbox is needed for installing additional repos. Otherwise, you will get a No Space Left error when installing.

## How to run the sandbox file to install additional repo

```
export TMPDIR='/om2/user/chengxuz/tmp' ; singularity shell -B /om,/om2 --nv /om2/user/chengxuz/docker_images/megatron_sbox
```
This TMPDIR may not be needed. But just to be safe.

To get the cuda right, I need to do `ln -s lib.real lib` in the `/om2/user/chengxuz/docker_images/megatron_sbox/usr/local/cuda-12.5/compat/` folder.


## Test evaluation command

```
export ROOT_DIR="/om2/user/chengxuz/megatron_related" ; bash examples/gpt3/eval_gpt2_1d3b.sh "${ROOT_DIR}/gpt_test_train/gpt2_1d3b/ckpts" "${ROOT_DIR}/gpt_test_train/gpt2_1d3b/tensorboards" "${ROOT_DIR}/gpt_ckpts/gpt2-vocab.json" "${ROOT_DIR}/gpt_ckpts/gpt2-merges.txt" "/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/pile_up_to_165-of-01650_text_document"
```

## General Test command

```
bash examples/gpt3/gnrl_eval_from_1d3b.sh --setting "baselines.py:change_to_w1d7b_from_1d3b"
```
