from network.model_resnet import BasicBlock1D, ResNet1D
from network.model_tcn import TlioTcn
import torch

def get_model(arch, net_config, input_dim=3, output_dim=3):
    if arch in ["resnet"]:
        network = ResNet1D(BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"])
    else:
        raise ValueError("Invalid architecture: ", arch)
    return network
