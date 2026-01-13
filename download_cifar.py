import torchvision
import torchvision.transforms as transforms

data_root = './data'

# Download CIFAR-10
torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.ToTensor())
torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transforms.ToTensor())

# Download CIFAR-100
torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transforms.ToTensor())
torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transforms.ToTensor())

print("CIFAR-10 and CIFAR-100 datasets downloaded to", data_root)