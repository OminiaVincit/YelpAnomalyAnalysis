python train_gpu.py --gpu 0 --features topics --num_features 64 --epoch 2000 --snapshot 500 --norm 1 --data_index 3 --site tripadvisor --opt MomentumSGD --lr 0.001 --lr_decay_freq 1900 --seed 42311
python train_gpu.py --gpu 0 --features topics --num_features 64 --epoch 2000 --snapshot 500 --norm 1 --data_index 3 --site tripadvisor --opt Adam --alpha 0.001 --seed 23231
python train_gpu.py --gpu 0 --features topics --num_features 64 --epoch 2000 --snapshot 500 --norm 1 --data_index 3 --site tripadvisor --opt AdaDelta --seed 23231

