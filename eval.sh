nohup python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py cifar10_l2p --eval > ./results/eval/cifar10_eval_online.log