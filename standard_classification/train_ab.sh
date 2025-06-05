#!/bin/bash

# Get input parameters
start_alpha_beta=$1  # Starting value of alpha_beta
end_alpha_beta=$2    # Ending value of alpha_beta
start_alpha=$3       # Starting value of alpha
end_alpha=$4         # Ending value of alpha
teacher_model=$5     # Teacher model name
student_model=$6     # Student model name
gpu_id=$7            # GPU ID
b=$8

# ========================================================
#  bash train_ab.sh 1.1 1.1 0.8 0.8 resnet56 resnet20 1 64


for alpha_beta in $(seq $start_alpha_beta 0.1 $end_alpha_beta); do
    for alpha in $(seq $start_alpha 0.1 $end_alpha); do
        beta=$(echo "$alpha_beta - $alpha" | bc)

        CUDA_VISIBLE_DEVICES=$gpu_id python3 train_student.py \
            --path_t ./save/models/"$teacher_model"_vanilla/ckpt_epoch_240.pth \
            --model_s "$student_model" \
            --distill ab  \
            -r 1 -b "$b"  -a 0 \
            --trial 1 \
            --ab_alpha "$alpha" \
            --ab_beta "$beta"
    done
done