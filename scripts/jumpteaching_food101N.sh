#!/bin/bash
gpu='2'
seed=1
config_path='./configs/jumpteaching_food101n.py'
batch_size=32

python main.py -c=$config_path \
               --gpu=$gpu \
               --seed=$seed \
               --batch_size=$batch_size \
               --save_result=1 \
               --save_log=1


