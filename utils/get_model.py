import models
import torch

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))


def get_model(
    model_name: str = 'resnet18',
    input_channel: int = 3,
    num_classes: int = 10,
    hashbits: int=-1,
    device=None,
):
    """
    Load specified model and move it to specified device

    Args:
        model_name: Model name
        input_channel: Number of input channels
        num_classes: Number of classes
        hashbits: Hash bits for jumpteaching model
        device: Device, e.g., "cuda:0" or "cuda:1"

    Returns:
        Model instance on specified device
    """
    assert model_name in model_names
    if model_name == "PreResNet18SH":
        model = models.__dict__[model_name]( num_class=num_classes,
                                            low_dim=20,
                                        hashbits = hashbits)
    elif model_name == "PreResNet34SH":
        model = models.__dict__[model_name]( num_class=num_classes,
                                            low_dim=20,
                                        hashbits = hashbits)
    elif model_name == "PreResNet18":
        model = models.__dict__[model_name]( num_class=num_classes,
                                            low_dim=20)
    elif model_name == "InceptionResNetV2SH":
        model = models.__dict__[model_name](input_channel=input_channel,
                                            num_classes=num_classes,
                                            hash_bit=hashbits)
    else:
        model = models.__dict__[model_name](input_channel=input_channel,
                                            num_classes=num_classes,
                                            hashbits = hashbits)
    return model.to(device if device is not None else "cuda:0")