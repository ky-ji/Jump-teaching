gpu='0'
seed=12
config_path='./configs/jumpteaching_clothing1M.py'
save_log=false
save_result=false

python main.py -c=$config_path\
               --gpu=$gpu --seed=$seed\
               --save_log=$save_log --save_result=$save_result

