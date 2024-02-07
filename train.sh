nohup python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env main.py \
    cifar100_l2p \
    --model vit_base_patch16_224 \
    --batch-size 4 \
    --data-path ./datasets \
    --output_dir ./output \
    > ./results/train/cifar100_online_bs.log 2>&1 &
