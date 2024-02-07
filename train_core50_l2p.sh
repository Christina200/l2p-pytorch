#!/bin/bash

#SBATCH --job-name=l2p_core50
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH -w agi1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=14-0
#SBATCH -o %N_%x_%j.out
#SBTACH -e %N_%x_%j.err

python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env main.py \
        core50_l2p \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path ./datasets \
        --output_dir ./output 