from __future__ import annotations

import torch
import torch.nn.functional as F

import typing
if typing.TYPE_CHECKING:
    from jaxtyping import Float
    
import enum

class FiltrationType(enum.Enum):
    SUPERLEVEL = "superlevel"
    SUBLEVEL = "sublevel"
    BOTHLEVELS = "bothlevels"

class ActivationType(enum.Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"

class DiceType(enum.Enum):
    DICE = "dice"
    CLDICE = "cldice"

def convert_to_one_vs_rest( # this is more of a one-vs-max strategy
        prediction: Float[torch.Tensor, "batch channel *spatial_dimensions"],
) -> Float[torch.Tensor, "batch channel *spatial_dimensions"]:
    """
    Converts a multi-class prediction tensor into a one-vs-rest format by building 
    the softmax over each class (one) and the max of all other classes (rest).

    Args:
        prediction (torch.Tensor): The input prediction tensor of shape (batch, channel, *spatial_dimensions).

    Returns:
        torch.Tensor: The converted prediction tensor of shape (batch, channel, *spatial_dimensions).
    """
    converted_prediction = torch.zeros_like(prediction)

    for channel in range(prediction.shape[1]):
        # Get logits for the channel class
        channel_logits = prediction[:,channel].unsqueeze(1)

        # For each pixel, get the class with the highest probability but exclude the channel class
        rest_logits = torch.max(prediction[:, torch.arange(prediction.shape[1]) != channel], dim=1).values.unsqueeze(1)

        # Apply softmax to get probabilities and select the probability of the channel class
        converted_prediction[:, channel] = torch.softmax(torch.cat([rest_logits, channel_logits], dim=1), dim=1)[:,1]

    return converted_prediction


def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    else:
        raise ValueError('input tensor must have 4D with shape: (batch, channel, height, width)')


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    else:
        raise ValueError('input tensor must have 4D with shape: (batch, channel, height, width)')


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel