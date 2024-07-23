#!/bin/bash 
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=400G
#SBATCH -t 12:0:0
#SBATCH -c 50
#SBATCH -p multi-gpu
#SBATCH -o /om2/user/chengxuz/sbatch_logs/slurm_%j.out

. ~/.bash_profile

#INNER_NUM_GPUS="${NUM_GPUS:-1}"

cd /om2/user/chengxuz/repos/Megatron-LM/
singularity exec\
    --writable-tmpfs -B /om,/om2 --nv /om2/user/chengxuz/docker_images/pytorch_24.06-py3.copy1.sif\
    bash examples/gpt3/gnrl_pretrain_from_1d3b.sh --setting ${SETTING}
# "baselines.py:change_to_1d7b_from_1d3b"
