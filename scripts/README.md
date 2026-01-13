# Experiment Scripts

This directory contains shell scripts for running experiments with Jump-teaching.

## Available Scripts

### CIFAR Experiments

| Script | Description |
|--------|-------------|
| `jumpteaching_cifar10.sh` | Run Jump-teaching on CIFAR-10 with symmetric/asymmetric noise |
| `jumpteaching_cifar100.sh` | Run Jump-teaching on CIFAR-100 with symmetric/asymmetric noise |
| `jumpteaching_cifar10_pairflip.sh` | Run Jump-teaching on CIFAR-10 with pairflip noise |
| `jumpteaching_cifar100_pairflip.sh` | Run Jump-teaching on CIFAR-100 with pairflip noise |

### Real-world Noisy Datasets

| Script | Description |
|--------|-------------|
| `jumpteaching_clothing1M.sh` | Run Jump-teaching on Clothing1M |
| `jumpteaching_food101N.sh` | Run Jump-teaching on Food-101N |
| `jumpteaching_webvision.sh` | Run Jump-teaching on WebVision |

## Usage

### Quick Start

```bash
# CIFAR-10 experiments
bash scripts/jumpteaching_cifar10.sh

# CIFAR-100 experiments  
bash scripts/jumpteaching_cifar100.sh

# Real-world datasets
bash scripts/jumpteaching_clothing1M.sh
bash scripts/jumpteaching_food101N.sh
bash scripts/jumpteaching_webvision.sh
```

## Customization

You can modify the scripts to change:

- `gpu`: GPU ID to use
- `seed`: Random seed for reproducibility
- `noise_type`: Type of noise (`sym`, `asym`, `ins`)
- `percent`: Noise rate (e.g., 0.2, 0.5, 0.8)
- `batch_size`: Training batch size

See the main README.md for detailed parameter descriptions.
