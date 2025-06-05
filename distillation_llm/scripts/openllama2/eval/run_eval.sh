#!/bin/bash

MASTER_PORT=2622
DEVICE=${1}
ckpt=${2}

# dolly eval
for seed in 20 30 40 50
do
    bash ./scripts/openllama2/eval/eval_main_dolly_lora.sh ./ ${MASTER_PORT} 2 openllama2-7b ${ckpt} --seed $seed  --eval-batch-size 8
    bash ./scripts/openllama2/eval/eval_main_self_inst_lora.sh ./ ${MASTER_PORT} 2 openllama2-7b ${ckpt} --seed $seed  --eval-batch-size 8
    bash ./scripts/openllama2/eval/eval_main_vicuna_lora.sh ./ ${MASTER_PORT} 2 openllama2-7b ${ckpt} --seed $seed  --eval-batch-size 8
    bash ./scripts/openllama2/eval/eval_main_sinst_lora.sh ./ ${MASTER_PORT} 2 openllama2-7b ${ckpt} --seed $seed  --eval-batch-size 8
    bash ./scripts/openllama2/eval/eval_main_uinst_lora.sh ./ ${MASTER_PORT} 2 openllama2-7b ${ckpt} --seed $seed  --eval-batch-size 8
done