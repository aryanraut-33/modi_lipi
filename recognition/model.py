"""Shared training/inference model and stroke-preserving preprocessing."""
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet18

ARCH = 'resnet18-gray64-v1'


def build_model(classes):
    # Small-image stem: avoid the normal 7x7/stride-2 + max-pool loss of detail.
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def prepare_image(image):
    """Black strokes on white; preserve the full crop including vowel modifiers."""
    image = ImageOps.grayscale(image)
    array = np.asarray(image)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    if np.median(border) < 127:
        image = ImageOps.invert(image)
    image = ImageOps.contain(image, (58, 58), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (64, 64), 255)
    canvas.paste(image, ((64-image.width)//2, (64-image.height)//2))
    return canvas


def tensor_image(image):
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).unsqueeze(0).sub_(0.5).div_(0.5)
