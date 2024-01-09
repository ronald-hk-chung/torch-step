import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from duckduckgo_search import DDGS
import uuid
import os
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
from typing import Callable, Type
from pathlib import Path
import requests

def get_pretrained_model(name: str, pretrained_weights: str | None = None):
  """Get pretrained model and pretrained transformation (forward and reverse)

  Args:
  model[str]: name of pretrained model
  weights[str]: name of pretrained model weights

  Returns:
  A tuple of (model, forward_transformation, reverse_transformation)

  Example usage:
  model, ftfms, rtfms = get_prerained_model(name='resnet18',
                                            weights='ResNet18_Weights.IMAGENET1K_V1')
  """

  # Change get_state_dict from Torch Hub
  def get_state_dict_from_hub(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return torch.hub.load_state_dict_from_url(self.url, *args, **kwargs)
  torchvision.models._api.WeightsEnum.get_state_dict = get_state_dict_from_hub

  # Get default transformation and re-construct forward transformation and reverse transformation using V2
  if pretrained_weights is not None:
    weights = torchvision.models.get_weight(pretrained_weights)
    pretrained_transforms = weights.transforms()
    forward_transforms = T.Compose([T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                                    T.Resize(size=pretrained_transforms.resize_size,
                                             interpolation=pretrained_transforms.interpolation,
                                             antialias=True),
                                    T.CenterCrop(size=pretrained_transforms.crop_size),
                                    T.Normalize(mean=pretrained_transforms.mean, std=pretrained_transforms.std)
                                    ])
    reverse_transforms = T.Compose([T.Normalize(mean=[0.0] * 3,
                                                std=list(map(lambda x: 1 / x, pretrained_transforms.std))),
                                    T.Normalize(mean=list(map(lambda x: -x, pretrained_transforms.mean)),
                                                std=[1.0] * 3),
                                    T.ToPILImage()
                                    ])
  else:
    weights = None
    forward_transforms = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    reverse_transforms = T.ToPILImage()

  # Get model using torchvision.models.get_model
  model = torchvision.models.get_model(name=name, weights=weights)

  return model, forward_transforms, reverse_transforms


def collect_images(keywords: str,
                   path: str,
                   max_results: int = 30,
                   timeout: tuple = (3, 5)):
  """Function to collect images using DDGS
  
  Args:
    keywords [str]: keywords for query
    path [str]: images path to be saved to
    max_results [int]: max number of results. If None, returns results only from the first response. Defaults to None.
    timeout [tuple[float, float]]: timeout for request (connect_timeout, read_timeout). Default to (3, 5)

  Returns:
    image_results [dict[str: str]]: {'image': image_url,
                                     'url': site_url,
                                     'path': image_path,
                                     'height': image_height,
                                     'width': image_width,
                                     'source': source_of_search}
  """
  Path(path).mkdir(parents=True, exist_ok=True)
  image_results = []
  with DDGS() as ddgs:
    results = list(ddgs.images(keywords=keywords, max_results=max_results))

  for result in tqdm(results):
    try:
      r = requests.get(result['image'], timeout=timeout)
    except:
      continue

    if r.status_code == 200:
      image_name = uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=result['image'])
      with open(f'{path}/{image_name}.jpg','wb') as img:
        img.write(r.content)
      try:
        img = Image.open(f'{path}/{image_name}.jpg').convert('RGB')
        image_result = {'image': result['image'],
                        'url': result['url'],
                        'path': f'{path}/{image_name}.jpg',
                        'height': result['height'],
                        'width': result['width'],
                        'source': result['source']}
        image_results.append(image_result)
      except:
        os.remove(f'{path}/{image_name}.jpg')

  print(f'[INFO] Downloaded {len(image_results)} images into {path}')
  return image_results

def validate_images(path: str):
  """Function to validate images within a path and remove broken images
  
  Args: path [str]: path to validate
  """
  paths = Path(path).rglob('*.jpg')
  for path in paths:
    try:
      img = Image.open(path).convert('RGB')
      if np.array(img).shape[2] != 3:
        os.remove(path)
        print(f'[INFO] Removed invalid path: {path}')
    except:
      os.remove(path)
      print(f'[INFO] Removed invalid path: {path}')


def show_batch(dataloader: torch.utils.data.DataLoader,
               transforms: transforms.Compose = T.ToPILImage(),
               labelling: Callable = None):
  """Function to show a batch of images with custom labelling
  
  Args:
    dataloader [DataLoader]: DataLoader of images to show
    transforms [transforms]: transform of image tensor to PIL
    labelling [Callable]: function to return label given labels generated from dataloader
  """
  fig = plt.figure(figsize=(20, 10))
  imgs, *labels = next(iter(dataloader))
  nrows, ncolumns = 4, 8
  for i, label in enumerate(list(zip(*labels))):
    plt.subplot(nrows, ncolumns, i + 1)
    plt.imshow(transforms(imgs[i]))
    if labelling:
      plt.title(labelling(label))
    else:
      plt.title(list(map(lambda x: round(x.item(), 2), label)))
    plt.axis(False)


def get_cam(img_path: str,
            tfms: transforms.Compose,
            rtfms: transforms.Compose,
            classifier: Callable,
            layers_to_activate: list,
            class_to_activate: int):
  """Method to get Class Activation Maps(CAM) for in image
  
  Args:
    img_path [str]: path of img to be analysed
    tfms [transforms]: transform of image, PIL to Tensor
    rtfms [transforms]: reverse transform of image, Tensor to PIL
    classifier [TSEngine]: TSEngine class classifier
    layers_to_activate [list]: layers to get activation from model in classifier
    class_to_activate [int]: class for Class Activation Maps (CAM) analysis
  """
  gradients = None
  activations = None

  def backward_hook(module, grad_input, grad_output):
    nonlocal gradients
    gradients = grad_output

  def forward_hook(module, args, output):
    nonlocal activations
    activations = output
  
  # unfreeze layers and attach hooks to layers_to_activate
  classifier.unfreeze()
  classifier.attach_forward_hooks(layers_to_activate, forward_hook)
  classifier.attach_backward_hooks(layers_to_activate, backward_hook)

  # Open image and perform forward and backward pass on class to activates
  img = tfms(Image.open(img_path).convert('RGB'))
  y_logits = classifier.model(img.unsqueeze(dim=0).to('cuda'))
  print(f'Prediction: {y_logits}')
  y_logits[class_to_activate].backward()

  # freeze classifier and remove hooks
  classifier.remove_hooks()
  classifier.freeze()

  # Calculate gradient heatmap
  pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
  for i in range(activations.size()[1]):
    activations[:, i, : :] *= pooled_gradients[i]
  heatmap = torch.mean(activations, dim=1).squeeze()
  heatmap = torch.nn.functional.relu(heatmap)
  heatmap /= torch.max(heatmap)

  # Plot out original image and image with heatmap
  fig, (ax1, ax2) = plt.subplots(1,2)
  ax1.imshow(rtfms(img))
  ax1.axis(False)
  overlay = to_pil_image(heatmap.detach(), mode='F').resize((img.shape[1],img.shape[2]), 
                                                            resample=Image.BICUBIC)
  cmap = colormaps['jet']
  overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
  ax2.imshow(rtfms(img))
  ax2.imshow(overlay, alpha=0.4, interpolation='nearest')
  ax2.axis(False)
  plt.show()