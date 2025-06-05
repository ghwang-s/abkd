#!/bin/bash

# custom config
DATA="/data/huacong/DATA"
TRAINER=PromptKD
DISTILL="ttm"
DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2

KD_WEIGHT=1000.0
CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0
LR=0.005
# sh scripts/promptkd/base2new_train_dkd.sh fgvc_aircraft 1

# fgvc_aircraft, oxford_flowers, dtd: KD_WEIGHT:200
# imagenet, caltech101, eurosat, food101, oxford_pets, stanford_cars, sun397, ucf101, KD_WEIGHT:1000

            DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${DISTILL}/seed_${SEED}/$LR_${LR}_KD_WEIGHT_${KD_WEIGHT}/warmup:${warmup}_beta:${beta}
            CUDA_VISIBLE_DEVICES=3 python3 train.py \
                                    --root ${DATA} \
                                    --seed ${SEED} \
                                    --trainer ${TRAINER} \
                                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                                    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                                    --output-dir ${DIR} \
                                    --distill ${DISTILL} \
                                    OPTIM.LR ${LR} \
                                    DATASET.NUM_SHOTS ${SHOTS} \
                                    TRAINER.MODAL base2novel \
                                    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
                                    TRAINER.PROMPTKD.KD_WEIGHT ${KD_WEIGHT}
