from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Optional


__all__ = ['freeze_bn', 'freeze_model']


def freeze_bn(mod: Module) -> Module:
    """Prevents parameter and stats from updating in Batchnorm layers that are frozen
    Args:
        mod (torch.nn.Module): model to train
    Returns:
        torch.nn.Module: model
    """

    # Loop on modules
    for m in mod.modules():
        if isinstance(m, _BatchNorm) and m.affine and all(not p.requires_grad for p in m.parameters()):
            # Switch back to commented code when https://github.com/pytorch/pytorch/issues/37823 is resolved
            m.track_running_stats = False
            m.eval()

    return mod


def freeze_model(model: Module, last_frozen_layer: Optional[str] = None, frozen_bn_stat_update: bool = False) -> Module:
    """Freeze a specific range of model layers
    Args:
        model (torch.nn.Module): model to train
        last_frozen_layer (str, optional): last layer to freeze. Assumes layers have been registered in forward order
        frozen_bn_stat_update (bool, optional): force stats update in BN layers that are frozen
    Returns:
        torch.nn.Module: model
    """

    # Loop on parameters
    if isinstance(last_frozen_layer, str):
  
        for n, p in model.named_parameters():
            if n.startswith(last_frozen_layer):
                p.requires_grad_(True)

    return model