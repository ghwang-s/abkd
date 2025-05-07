#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
    --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth \
    --model_s resnet44 \
    --distill kd  \
    -r 1 -b 2  -a 0 \
    --trial 2 \
    --kd_T 4
