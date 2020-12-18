#train rounds in order, killing distrib processes in between

#python train_LM_round0.py --n_cpus 12

#kill $(ps aux | grep "train_LM_round0.py" | grep -v grep | awk '{print $2}')

#python train_LM_round1.py --n_cpus 12

#kill $(ps aux | grep "train_LM_round1.py" | grep -v grep | awk '{print $2}')

CUDA_VISIBLE_DEVICES=1 python train_LM_round2.py --n_cpus 1
