from .backbones.resetnet import *
from .noise_nets.cond_unet_1d import ConditionalUnet1D
from .network.cond_unet_mlp import ConditionalUnetMLP
from .network.mlp import MLP
from .network.transformer import Transformer
from .network.parallel_mlp import ParallelMLP

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
    elif name.startswith("UNETMLP"):
        return ConditionalUnetMLP(**kwargs)
    elif name.startswith("MLP"):
        return MLP(**kwargs)
    elif name.startswith("TRANSFORMER"):
        return Transformer(**kwargs)
    elif name.startswith("PARALLELMLP"):
        return ParallelMLP(**kwargs)
    else:
        raise NotImplementedError(f"Unknown noise net: {name}")
