from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms

import copy
import os


""" Transform tensor back to frame """
def recover_frame(frame):
    frame = frame.cpu().squeeze(0)
    denormalizer = tensor_denormalizer()
    frame = denormalizer(frame)
    frame.data.clamp_(0, 1)
    toPIL = transforms.Compose([transforms.ToPILImage(), transforms.Resize((540, 304))])
    frame = toPIL(frame)
    return frame


""" Image loader, loads image from file using PIL and converts it to torch tensor """
def image_loader(image_name, size=512):
    image = Image.open(image_name).convert('RGB')
    loader = transforms.Compose([transforms.Resize(size),
                                 transforms.ToTensor(),
                                 tensor_normalizer()])
    image = loader(image).unsqueeze(0)   # If only one image, add a fake dimension in front to augment a batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return image.to(device, torch.float)


""" Image loader for the style image, returns 3 versions with resolutions in sizes_list """
def style_loader(image_name, device, sizes_list):
    image = Image.open(image_name).convert('RGB')
    out = []
    for size in sizes_list:
        loader = transforms.Compose([transforms.Resize(size),
                                     transforms.CenterCrop(size),
                                     transforms.ToTensor(),
                                     tensor_normalizer()])
        style_img = loader(image).unsqueeze(0)
        out.append(style_img.to(device, torch.float))
    return out


""" Imshow, displays image using matplotlib """
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    denormalizer = tensor_denormalizer()
    image = denormalizer(image)
    image.data.clamp_(0, 1)
    toPIL = transforms.ToPILImage()
    image = toPIL(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


""" Saves image in the /output folder with a specified name as .jpg """
def save_image(tensor, title="output"):
    image = tensor.cpu().clone()  # clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    denormalizer = tensor_denormalizer()
    image = denormalizer(image)
    image.data.clamp_(0, 1)
    toPIL = transforms.ToPILImage()
    image = toPIL(image)
    scriptDir = os.path.dirname(__file__)
    image.save("{}.jpg".format(title))


""" Returns the gram matrix of a feature map """
def gram_matrix(input):
    b, ch, h, w = input.size()

    features = input.view(b, ch, h * w)  # change input to vectorized feature map K x N
    features_t = features.transpose(1, 2)

    # the gram matrix needs to be normalized because otherwise the early layers with a bigger N
    # will result in higher values of the gram matrix.
    gram = features.bmm(features_t) / (ch * h * w)  # compute the gram matrix bmm = batch matrix-matrix product -> K x K

    return gram


""" Transforms to normalize the image while transforming it to a torch tensor """
def tensor_normalizer():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


""" Denormalizes image to save or display it """
def tensor_denormalizer():
    return transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                               transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
