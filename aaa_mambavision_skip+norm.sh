#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
source ~/.bashrc
conda activate pytch
cd /gs/fs/tga-j/zhou.y.ak/aaa_mambavision_skip+norm

python train.py --gpu 0 --wandb --continue

python test.py --test_epoch 41 --gpu 0
python test.py --test_epoch 40 --gpu 0
python test.py --test_epoch 39 --gpu 0
python test.py --test_epoch 38 --gpu 0
python test.py --test_epoch 37 --gpu 0
python test.py --test_epoch 36 --gpu 0
python test.py --test_epoch 35 --gpu 0
python test.py --test_epoch 34 --gpu 0
python test.py --test_epoch 33 --gpu 0
python test.py --test_epoch 32 --gpu 0
python test.py --test_epoch 31 --gpu 0
python test.py --test_epoch 30 --gpu 0
python test.py --test_epoch 29 --gpu 0
python test.py --test_epoch 28 --gpu 0
python test.py --test_epoch 27 --gpu 0
python test.py --test_epoch 26 --gpu 0
python test.py --test_epoch 25 --gpu 0
python test.py --test_epoch 24 --gpu 0
python test.py --test_epoch 23 --gpu 0
python test.py --test_epoch 22 --gpu 0
python test.py --test_epoch 21 --gpu 0
python test.py --test_epoch 20 --gpu 0

