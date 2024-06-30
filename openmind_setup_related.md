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

## Test train

```
bash examples/gpt3/train_gpt2_345m.sh "/om2/user/chengxuz/megatron_related/gpt_test_train/gpt2_345m/ckpts" "/om2/user/chengxuz/megatron_related/gpt_test_train/gpt2_345m/tensorboards" "/om2/user/chengxuz/megatron_related/gpt_ckpts/gpt2-vocab.json" "/om2/user/chengxuz/megatron_related/gpt_ckpts/gpt2-merges.txt" "/om2/group/evlab/llm_dataset/Megatron_datasets/sw_150M_gpt2_text_document"
```

```
export ROOT_DIR="/om2/user/chengxuz/megatron_related" ; bash examples/gpt3/train_gpt2_1d3b.sh "${ROOT_DIR}/gpt_test_train/gpt2_1d3b/ckpts" "${ROOT_DIR}/gpt_test_train/gpt2_1d3b/tensorboards" "${ROOT_DIR}/gpt_ckpts/gpt2-vocab.json" "${ROOT_DIR}/gpt_ckpts/gpt2-merges.txt" "/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/pile_up_to_165-of-01650_text_document"
```
