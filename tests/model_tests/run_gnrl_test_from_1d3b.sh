#!/bin/bash

# Runs the general setting model

. examples/gpt3/gnrl_gpt2_1d3b_basics.sh
#. sb_scripts/get_rand_port
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

#torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt_from_cfgs.py \
MASTER_ADDR=localhost MASTER_PORT=${PORT} python tests/model_tests/test_att_model.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} "$@"
