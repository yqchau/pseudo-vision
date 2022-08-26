from collections import OrderedDict

import torch


def lightning_to_torch(PATH):
    """
    Pytorch Lightning State Dict to Torch State Dict converter
    create new OrderedDict that does not contain `model.`
    """

    state_dict = torch.load(PATH)["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[6:]  # remove `model.`
        new_state_dict[name] = v

    return new_state_dict
