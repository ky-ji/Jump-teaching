gpu='2'
seed='12'
config_path='./configs/jumpteaching_CIFAR.py'

declare -A noise_configs
noise_configs[pairflip]="0.45"
#noise_configs[asym]="0.4"

for noise_type in "${!noise_configs[@]}"
do
    noise_rates=${noise_configs[$noise_type]}
    for noise_rate in $noise_rates
    do
        python main.py -c=$config_path --noise_type=$noise_type --seed=$seed --gpu=$gpu --percent=$noise_rate --dataset='cifar-100' --num_classes=100
    done
done


