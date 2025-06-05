#!/bin/bash

# custom config
DATA="/data/huacong/DATA"
TRAINER=PromptKD
DISTILL="dkd"
DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2

KD_WEIGHT=100.0
CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0

# sh scripts/promptkd/base2new_train_dkd.sh fgvc_aircraft 1

# sh scripts/promptkd/base2new_train_dkd.sh sun397 1

# sh scripts/promptkd/base2new_train_dkd.sh ucf101 1

# sh scripts/promptkd/base2new_train_dkd.sh eurosat 1

# sh scripts/promptkd/base2new_train_dkd.sh oxford_flowers 1

# sh scripts/promptkd/base2new_train_dkd.sh stanford_cars 1

# sh scripts/promptkd/base2new_train_dkd.sh food101 1

# fgvc_aircraft, oxford_flowers, dtd: KD_WEIGHT:200
# imagenet, caltech101, eurosat, food101, oxford_pets, stanford_cars, sun397, ucf101, KD_WEIGHT:1000
for warmup in $(seq 1 1 1); do
        for beta in 0.5 1; do
#            beta=$(echo "$alpha_beta - $alpha" | bc)
#            echo "Running with alpha=$alpha, beta=$beta"
            DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${DISTILL}/seed_${SEED}/${KD_WEIGHT}/warmup:${warmup}_beta:${beta}
            CUDA_VISIBLE_DEVICES=4 python3 train.py \
                                    --root ${DATA} \
                                    --seed ${SEED} \
                                    --trainer ${TRAINER} \
                                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                                    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                                    --output-dir ${DIR} \
                                    --distill ${DISTILL} \
                                    --dkd_warmup $warmup \
                                    --dkd_beta $beta \
                                    DATASET.NUM_SHOTS ${SHOTS} \
                                    TRAINER.MODAL base2novel \
                                    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
                                    TRAINER.PROMPTKD.KD_WEIGHT ${KD_WEIGHT}
            done
        done
