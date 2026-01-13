#cd /root
gpu='0'
seed='12'
config_path='./configs/jumpteaching_cifar.py'

declare -A noise_configs
noise_configs[sym]="0.2 0.5 0.8"
noise_configs[asym]="0.4"
#noise_configs[ins]="0.2 0.5 0.8"

for noise_type in "${!noise_configs[@]}"
do
    noise_rates=${noise_configs[$noise_type]}
    for noise_rate in $noise_rates
    do
        python main.py -c=$config_path --noise_type=$noise_type --seed=$seed --gpu=$gpu --percent=$noise_rate
    done
done



