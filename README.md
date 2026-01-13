# Jump-teaching: Combating Sample Selection Bias via Temporal Disagreement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/conference/aaai/aaai-26/)

Official implementation of **Jump-teaching**, accepted at **AAAI 2026**.


## Repository Structure

```
Jump-teaching/
├── algorithms/          # Algorithm implementations
│   └── jumpteaching.py # Main Jump-teaching implementation
├── configs/            # Configuration files for different datasets
│   ├── jumpteaching_cifar.py
│   ├── jumpteaching_clothing1M.py
│   ├── jumpteaching_food101n.py
│   └── jumpteaching_webvision.py
├── datasets/           # Dataset loaders and preprocessing
├── models/             # Neural network architectures 
├── losses/             # Loss function implementations
├── utils/              # Utility functions
├── labels/             # Pre-computed Hadamard hash codes
├── scripts/            # Shell scripts for experiments
├── main.py             # Main training script
└── generate_labelcodes.py  # Generate hash codes
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+ (for GPU support)

### Setup

```bash
git clone https://github.com/ky-ji/Jump-teaching.git
cd Jump-teaching
pip install -e .
```

## Quick Start

### 1. Download Datasets

Download CIFAR-10 and CIFAR-100:
```bash
python download_cifar.py
```

### 2. Run Jump-teaching on CIFAR-10

```bash
# Using the script
bash scripts/jumpteaching_cifar10.sh

# Or run directly
python main.py -c=./configs/jumpteaching_cifar.py \
    --gpu 0 \
    --seed 1 \
    --noise_type sym \
    --percent 0.5
```

### 3. Run Jump-teaching on CIFAR-100

```bash
# Using the provided script
bash scripts/jumpteaching_cifar100.sh

# Or run directly
python main.py -c=./configs/jumpteaching_cifar.py \
    --gpu 0 \
    --seed 1 \
    --noise_type sym \
    --percent 0.5 \
    --dataset cifar-100 \
    --num_classes 100
```

## Label Hash Codes

Jump-teaching uses Hadamard-based hash codes to represent class labels efficiently. Pre-computed codes are stored in `./labels/`. To generate hash codes for new datasets:

```bash
python generate_labelcodes.py \
    --hashbits <BITS> \
    --dataset <DATASET_NAME> \
    --num_classes <NUM_CLASSES> \
    [--output_dir ./labels] \
    [--verify]
```

Common setups:

```bash
# CIFAR-10 (10 classes, 32-bit codes)
python generate_labelcodes.py --hashbits 32 --dataset cifar-10 --num_classes 10

# CIFAR-100 (100 classes, 64-bit codes)
python generate_labelcodes.py --hashbits 64 --dataset cifar-100 --num_classes 100

# Clothing1M (14 classes, 32-bit codes)
python generate_labelcodes.py --hashbits 32 --dataset clothing1M --num_classes 14
```

The script saves tensors as `<hashbits>_<dataset>_<num_classes>_class.pkl`. Ensure your config points to the directory containing those files:

```python
config['labelhashcodes_path'] = './labels/'
# Automatically loads: f"{hashbits}_{dataset}_{num_classes}_class.pkl"
```

Use `--verify` to confirm the Hadamard properties (balanced, orthogonal, ±1 values).

## Configuration

All hyperparameters are defined in config files (`configs/*.py`). Key parameters:

### Dataset Configuration
- `dataset`: Dataset name (`cifar-10`, `cifar-100`, `clothing1M`, `food101N`, `webvision`)
- `num_classes`: Number of classes

### Model Configuration
- `model1_type`: Backbone architecture (`PreResNet18SH`, `resnet50SH`, `InceptionResNetV2SH`)
- `hashbits`: Length of hash codes (32 for CIFAR-10, 64 for CIFAR-100/WebVision, 128 for Food-101N)
- `tau`: Threshold for sample selection (default: 0.001)
- `T`: Temperature parameter for softmax (default: 2)
- `Step`: Update frequency parameter (default: 2)

### Command Line Overrides
You can override config values via command line:
- `--gpu`: GPU ID
- `--seed`: Random seed
- `--noise_type`: Noise type (`sym`, `asym`, `ins`)
- `--percent`: Noise rate (0.2, 0.5, 0.8)
- `--batch_size`: Override batch size

## Experiments

We provide shell scripts for reproducing paper results:

### CIFAR Experiments
```bash
# CIFAR-10 
bash scripts/jumpteaching_cifar10.sh

# CIFAR-100 
bash scripts/jumpteaching_cifar100.sh

# CIFAR 
bash scripts/jumpteaching_cifar10_pairflip.sh
bash scripts/jumpteaching_cifar100_pairflip.sh
```

### Real-world Noisy Datasets
```bash
# Clothing1M 
bash scripts/jumpteaching_clothing1M.sh

# Food-101N
bash scripts/jumpteaching_food101N.sh

# WebVision
bash scripts/jumpteaching_webvision.sh
```


## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{ji2024jump,
  title={Jump-teaching: Ultra Efficient and Robust Learning with Noisy Label},
  author={Ji, Kangye and Cheng, Fei and Wang, Zeqing and Huang, Bohu},
  journal={arXiv preprint arXiv:2405.17137},
  year={2024}
}
```

You can also use the provided `CITATION.bib`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank the authors of the repositories [Co-teaching](https://github.com/bhanML/Co-teaching), [DivideMix](https://github.com/LiJunnan1992/DivideMix) and [DISC](https://github.com/JackYFL/DISC) for their excellent work!



## Contact

For questions or issues, please:
- Open an issue in this repository
- Contact: kangyejics@gmail.com

