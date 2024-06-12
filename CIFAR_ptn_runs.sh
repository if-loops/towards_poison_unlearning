#!/bin/bash

# Set the values for the other parameters
export CUDA_VISIBLE_DEVICES=2
CUDA_VISIBLE_DEVICES=2
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

dataset_method="poisoning"
patch_size=3

unlearn_method3="ASSD"
unlearn_method2="ALFSSD"
unlearn_method="XALFSSD"

p_cuts=(0.00 0.05 0.10 0.15 0.20 0.25 0.50 0.75)
rm -r wandb/*

VERSION_I="VERSION_X"

project_name_prefix="poison_10_"

dataset="CIFAR10"
num_classes=10
model="resnet9"
pretrain_iters=4000


#-----------------
# Loop over the deletion_size values
for p_cut in "${p_cuts[@]}"; do

    project_name=$project_name_prefix+$VERSION_I+$p_cut
    echo $project_name
    # Clear to avoid problems with similar names for logs
    rm -r logs/*
    rm -r results.tsv

    forget_set_sizes=(500)
    deletion_sizes=(1 50 100 150 200 250 300 350 400 450 500)

    for forget_set_size in "${forget_set_sizes[@]}"; do
        # Nested loop over the fractions
        for deletion_size in "${deletion_sizes[@]}"; do
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method2
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method3
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=EU --unlearn_iters=$pretrain_iters --k=-1
            python visualize.py --project_name=$project_name --p_cut=$p_cut
        done
    done

    forget_set_sizes=(1000)
    deletion_sizes=(1 100 200 300 400 500 600 700 800 900 1000)

    for forget_set_size in "${forget_set_sizes[@]}"; do
        # Nested loop over the fractions
        for deletion_size in "${deletion_sizes[@]}"; do
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=EU --unlearn_iters=$pretrain_iters --k=-1
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method2
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method3
            python visualize.py --project_name=$project_name --p_cut=$p_cut
        done
    done

    forget_set_sizes=(100)
    deletion_sizes=(1 10 20 30 40 50 60 70 80 90 100)

    for forget_set_size in "${forget_set_sizes[@]}"; do
        # Nested loop over the fractions
        for deletion_size in "${deletion_sizes[@]}"; do
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=EU --unlearn_iters=$pretrain_iters --k=-1
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method2
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method3
            python visualize.py --project_name=$project_name --p_cut=$p_cut
        done
    done

done
#-----------------


dataset="CIFAR100"
num_classes=100
model="resnetwide28x10"
pretrain_iters=6000

project_name_prefix="poison_100_"
#project_name+=$VERSION_I
#-----------------
# Loop over the deletion_size values
for p_cut in "${p_cuts[@]}"; do
    project_name=$project_name_prefix+$VERSION_I+$p_cut
    echo $project_name
    # Clear to avoid problems with similar names for logs
    rm -r logs/*
    rm -r results.tsv

    forget_set_sizes=(500)
    deletion_sizes=(1 50 100 150 200 250 300 350 400 450 500)

    for forget_set_size in "${forget_set_sizes[@]}"; do
        # Nested loop over the fractions
        for deletion_size in "${deletion_sizes[@]}"; do
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=EU --unlearn_iters=$pretrain_iters --k=-1
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method2
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method3
            python visualize.py --project_name=$project_name --p_cut=$p_cut
        done
    done

    forget_set_sizes=(1000)
    deletion_sizes=(1 100 200 300 400 500 600 700 800 900 1000)

    for forget_set_size in "${forget_set_sizes[@]}"; do
        # Nested loop over the fractions
        for deletion_size in "${deletion_sizes[@]}"; do
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=EU --unlearn_iters=$pretrain_iters --k=-1
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method2
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method3
            python visualize.py --project_name=$project_name --p_cut=$p_cut
        done
    done

    forget_set_sizes=(100)
    deletion_sizes=(1 10 20 30 40 50 60 70 80 90 100)

    for forget_set_size in "${forget_set_sizes[@]}"; do
        # Nested loop over the fractions
        for deletion_size in "${deletion_sizes[@]}"; do
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=EU --unlearn_iters=$pretrain_iters --k=-1
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method2
            python src/main.py --p_cut=$p_cut --dataset=$dataset --num_classes=$num_classes --model=$model --pretrain_iters=$pretrain_iters --dataset_method=$dataset_method --forget_set_size=$forget_set_size --deletion_size=$deletion_size --patch_size=$patch_size --unlearn_method=$unlearn_method3
            python visualize.py --project_name=$project_name --p_cut=$p_cut
        done
    done

done
#-----------------

