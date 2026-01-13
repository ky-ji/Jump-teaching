from setuptools import setup, find_packages

setup(
    name="jump-teaching",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.4",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "pillow>=8.0.0",
        "tqdm>=4.64.0",
        "tensorboard>=2.8.0",
        "addict>=2.4.0",
    ],
)

