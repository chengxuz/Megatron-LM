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

## Test train

```
bash examples/gpt3/train_gpt2_345m.sh "/om2/user/chengxuz/megatron_related/gpt_test_train/gpt2_345m/ckpts" "/om2/user/chengxuz/megatron_related/gpt_test_train/gpt2_345m/tensorboards" "/om2/user/chengxuz/megatron_related/gpt_ckpts/gpt2-vocab.json" "/om2/user/chengxuz/megatron_related/gpt_ckpts/gpt2-merges.txt" "/om2/group/evlab/llm_dataset/Megatron_datasets/sw_150M_gpt2_text_document"
```
