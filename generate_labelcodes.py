import argparse
import os
import numpy as np
import torch
from scipy.linalg import hadamard


def generate_hash_codes(hashbits, num_classes):
    if hashbits & (hashbits - 1) != 0:
        raise ValueError(f"hashbits must be a power of 2, got {hashbits}")

    ha_d = hadamard(hashbits)
    ha_2d = np.concatenate((ha_d, -ha_d), axis=0)

    if num_classes <= hashbits:
        hash_targets = torch.from_numpy(ha_d[0:num_classes]).float()
    elif num_classes <= 2 * hashbits:
        hash_targets = torch.from_numpy(ha_2d[0:num_classes]).float()
    else:
        raise ValueError(
            f"num_classes ({num_classes}) exceeds maximum supported "
            f"by hashbits ({2 * hashbits}). Increase hashbits."
        )

    return hash_targets


def verify_hash_codes(hash_codes):
    print(f"\nHash codes shape: {hash_codes.shape}")
    column_sums = hash_codes.sum(0)
    print(f"Column sums: {column_sums}")
    bit_encoded = ((hash_codes + 1) / 2).to(torch.uint8)
    print(f"\nBinary representation (first 10 classes):")
    print(bit_encoded[:10])

    return True


def save_hash_codes(hash_codes, hashbits, dataset, num_classes, output_dir='./labels'):
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{hashbits}_{dataset}_{num_classes}_class.pkl"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "wb") as f:
        torch.save(hash_codes, f)

    print(f"\nHash codes saved to: {file_path}")
    return file_path


def load_and_verify(file_path):
    with open(file_path, 'rb') as f:
        loaded_codes = torch.load(f)

    print(f"\nLoaded hash codes from: {file_path}")
    print(f"Shape: {loaded_codes.shape}")
    print(f"First 5 codes:\n{loaded_codes[:5]}")

    return loaded_codes


def main():
    parser = argparse.ArgumentParser(
        description='Generate label hash codes for jumpteaching algorithm'
    )
    parser.add_argument(
        '--hashbits',
        type=int,
        required=True,
        help='Length of hash codes (must be power of 2: 8, 16, 32, 64, 128, etc.)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., cifar-10, cifar-100, clothing1M, etc.)'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        required=True,
        help='Number of classes in the dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./labels',
        help='Output directory for saved hash codes (default: ./labels)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the generated hash codes'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Generating Label Hash Codes")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Hash bits: {args.hashbits}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    hash_codes = generate_hash_codes(args.hashbits, args.num_classes)
    verify_hash_codes(hash_codes)
    file_path = save_hash_codes(
        hash_codes,
        args.hashbits,
        args.dataset,
        args.num_classes,
        args.output_dir
    )

    if args.verify:
        print("\n" + "=" * 60)
        print("Verification: Loading saved hash codes")
        print("=" * 60)
        load_and_verify(file_path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
