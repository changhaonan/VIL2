from .backbones.resetnet import *
from .noise_nets.cond_unet_1d import ConditionalUnet1D
from .mlp.mlp import ConditionalUnetMLP


def build_vision_encoder(name: str, **kwargs):
    if name.startswith("resnet"):
        resnet = get_resnet(name, weights="IMAGENET1K_V1", **kwargs)
        # replace all BatchNorm modules with GroupNorm
        resnet = replace_bn_with_gn(resnet)
        return resnet
    else:
        raise NotImplementedError(f"Unknown vision encoder: {name}")


def build_noise_pred_net(name: str, **kwargs):
    if name.startswith("UNET1D"):
        return ConditionalUnet1D(**kwargs)
    elif name.startswith("MLP"):
        return ConditionalUnetMLP(**kwargs)
    else:
        raise NotImplementedError(f"Unknown noise net: {name}")
