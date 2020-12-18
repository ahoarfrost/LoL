#train rounds in order, killing distrib processes in between

#python train_LM_round0.py --n_cpus 6

#kill $(ps aux | grep "train_LM_round0.py" | grep -v grep | awk '{print $2}')

python train_LM_round1.py --n_cpus 6

kill $(ps aux | grep "train_LM_round1.py" | grep -v grep | awk '{print $2}')

python train_LM_round2_fastlr.py --n_cpus 6

