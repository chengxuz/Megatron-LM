# Runs the "1.3B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE="${NUM_GPUS:-4}"
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

ROOT_DIR="/om2/user/chengxuz/megatron_related"
CHECKPOINT_PATH="${ROOT_DIR}/gpt_test_train/gpt2_1d3b/ckpts"
TENSORBOARD_LOGS_PATH="${ROOT_DIR}/gpt_test_train/gpt2_1d3b/tensorboards"
VOCAB_FILE="${ROOT_DIR}/gpt_ckpts/gpt2-vocab.json"
MERGE_FILE="${ROOT_DIR}/gpt_ckpts/gpt2-merges.txt"
DATA_PATH="/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/pile_up_to_165-of-01650_text_document"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --max-position-embeddings 2048
    --num-layers 24
    --hidden-size 2048
    --num-attention-heads 32
    --seq-length 1024
)

TRAINING_ARGS=(
    --micro-batch-size 16
    --global-batch-size  128
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)
