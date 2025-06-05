#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-16}

START_ALPHA_BETA=${4-1.0}  # alpha_beta 起始值
END_ALPHA_BETA=${5-1.0}    # alpha_beta 终止值
START_ALPHA=${6-0.5}  # alpha 起始值
END_ALPHA=${7-0.5}    # alpha 终止值

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT_NAME="openllama2-3b"
CKPT="${BASE_PATH}/${CKPT_NAME}/"
PEFT_CKPT_NAME="714"
PEFT_CKPT="${BASE_PATH}/results/openllama2/train/minillm_init/openllama2-3B/init/${PEFT_CKPT_NAME}/"
TEACHER_CKPT_NAME="openllama2-7b"
TEACHER_CKPT="${BASE_PATH}/${TEACHER_CKPT_NAME}/"
TEACHER_PEFT_CKPT_NAME="sft_7B/e20-bs4-lr0.0005-G1-N4-NN1-lora-16-64-0.1/14280"
TEACHER_PEFT_CKPT="${BASE_PATH}/results/openllama2/train/sft/${TEACHER_PEFT_CKPT_NAME}/"
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/openllama2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/openllama2/512/1M/"
# hp
BATCH_SIZE=2
LR=0.00025
GRAD_ACC=1
EVAL_BATCH_SIZE=4
# length
MAX_LENGTH=512
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 10"
OPTS+=" --kd-ratio 1.0"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
# OPTS+=" --save ${SAVE_PATH}"
# lora
OPTS+=" --peft lora"
OPTS+=" --do-train"
OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
OPTS+=" --peft-path ${PEFT_CKPT}"
OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type alphanet"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# GKD
OPTS+=" --student-gen"
OPTS+=" --init-threshold 0.2"
OPTS+=" --loss-eps 0.2"

#export CUDA_VISIBLE_DEVICES=3,4,5,6
export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
# CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

for alpha_beta in $(seq ${START_ALPHA_BETA} 0.1 ${END_ALPHA_BETA}); do
   for alpha in $(seq ${START_ALPHA} 0.1 ${END_ALPHA}); do
       beta=$(echo "$alpha_beta - $alpha" | bc)

              # 跳过 alpha == 0.2 和 beta == 0.7
    #    if [ $(echo "$alpha == 0.2" | bc) -eq 1 ] && [ $(echo "$beta == 0.7" | bc) -eq 1 ]; then
    #        echo "Skipping alpha=${alpha} and beta=${beta}"
    #        continue
    #    fi
       
       # runtime
    #    SAVE_PATH="${BASE_PATH}/results/openllama2/train/alphanet/3B_7B_0.2_0.8"
       SAVE_PATH="${BASE_PATH}/results/openllama2/train/alphanet/3B_7B_${alpha}_${beta}"
       mkdir -p ${SAVE_PATH}

       CURRENT_OPTS="${OPTS}"
       CURRENT_OPTS+=" --save ${SAVE_PATH}"
       CURRENT_OPTS+=" --ab_alpha ${alpha}"
       CURRENT_OPTS+=" --ab_beta ${beta}"

       CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${CURRENT_OPTS} $@"
       echo ${CMD}
       echo "PYTHONPATH=${PYTHONPATH}"
       ${CMD}
   done
done

# echo ${CMD}
# echo "PYTHONPATH=${PYTHONPATH}"
# mkdir -p ${SAVE_PATH}
# ${CMD}
