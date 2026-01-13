#!/bin/bash

gpu='1'
seed=1
batch_size=32
config_path='./configs/jumpteaching_webvision.py'

python main.py -c=$config_path \
               --gpu=$gpu \
               --seed=$seed \
               --batch_size=$batch_size \
               --save_result=1 \
               --save_log=1




