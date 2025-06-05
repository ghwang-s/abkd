#!/bin/bash

# custom config
DATA="/data/huacong/DATA"
TRAINER=PromptKD
DISTILL="ab"
DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
start_alpha_beta=$2  # Starting value of alpha_beta
end_alpha_beta=$3    # Ending value of alpha_beta
start_alpha=$4       # Starting value of alpha
end_alpha=$5         # Ending value of alpha
gpu_id=$6
SEED=$7
KD_WEIGHT=$8
CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}

# sh scripts/promptkd/base2new_train_ab.sh dtd 1.0 1.3 0.5 1.2  1 1 100.0
# sh scripts/promptkd/base2new_train_ab.sh fgvc_aircraft 1.0 1.3 0.5 1.2 2 1  100.0

for alpha_beta in $(seq $start_alpha_beta 0.1 $end_alpha_beta); do
    for alpha in $(seq $start_alpha 0.1 $end_alpha); do
            beta=$(echo "$alpha_beta - $alpha" | bc)
            echo "Running with alpha=$alpha, beta=$beta"
            DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${DISTILL}/seed_${SEED}/KD_WEIGHT:${KD_WEIGHT}/alpha:${alpha}_beta:${beta}
            CUDA_VISIBLE_DEVICES=${gpu_id} python3 train.py \
                                    --root ${DATA} \
                                    --seed ${SEED} \
                                    --trainer ${TRAINER} \
                                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                                    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                                    --output-dir ${DIR} \
                                    --distill ${DISTILL} \
                                    --ab_alpha $alpha \
                                    --ab_beta $beta \
                                    DATASET.NUM_SHOTS ${SHOTS} \
                                    TRAINER.MODAL base2novel \
                                    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
                                    TRAINER.PROMPTKD.KD_WEIGHT ${KD_WEIGHT}
            done
        done
