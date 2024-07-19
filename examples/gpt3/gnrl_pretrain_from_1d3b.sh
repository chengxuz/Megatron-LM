#!/bin/bash

# Runs the general setting model

. examples/gpt3/gnrl_gpt2_1d3b_basics.sh

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt_from_cfgs.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} "$@" \
