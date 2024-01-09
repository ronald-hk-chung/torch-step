"""
Contains class torchstep for train and valid step for PyTorch Model
"""

import datetime
import random
import os
import pathlib
import shutil
from pathlib import Path
from copy import copy, deepcopy
from tqdm.auto import tqdm
from typing import Callable, Type
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torchinfo
from PIL import Image


class TSEngine:
  """
  TorchStep class contains a number of useful functions for Pytorch Model Training
  """

  def __init__(self,
               model: torch.nn.Module,
               optim: tuple[torch.optim.Optimizer, dict[str, float]],
               loss_fn: torch.nn.Module,
               metric_fn: Callable,
               train_dataloader: torch.utils.data.DataLoader,
               valid_dataloader: torch.utils.data.DataLoader):
    self.model = deepcopy(model)
    self.optimizer = optim[0](params=self.model.parameters(), **optim[1])
    self.loss_fn = loss_fn
    self.metric_fn = metric_fn
    self.train_dataloader = train_dataloader
    self.valid_dataloader = valid_dataloader
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device)
    self.writer = None
    self.scheduler = None
    self.is_batch_lr_scheduler = False
    self.clipping = None
    self.results = {"train_loss": [],
                    "train_metric": [],
                    "valid_loss": [],
                    "valid_metric": []}
    self.learning_rates = []
    self.total_epochs = 0
    self.modules = list(self.model.named_modules())
    self.layers = {name: layer for name, layer in self.modules[1:]}
    self.forward_hook_handles = []
    self.backward_hook_handles = []
    self.callbacks = [self.SaveResults,
                      self.PrintResults,
                      self.TBWriter,
                      self.LearningRateScheduler,
                      self.GradientClipping]

  @staticmethod
  def set_seed(seed=42):
    """Function to set random seed for torch, numpy and random
    
    Args: seed [int]: random_seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

  def set_tensorboard(self,
                      name: str,
                      folder: str = "runs"):
    """Method to set TSEngine tensorboard
    
    Args:
      name [str]: name of project
      folder [str]: name of folder to run tensorboard logs, Defaults to 'runs'
    """
    suffix = datetime.datetime.now().strftime("%Y%m%d")
    self.writer = SummaryWriter(f"{folder}/{name}_{suffix}")

  def set_loaders(self,
                  train_dataloader: torch.utils.data.DataLoader,
                  valid_dataloader: torch.utils.data.DataLoader):
    """Method to set dataloaders
    
    Args: train_dataloader, valid_dataloader
    """
    self.train_dataloader = train_dataloader
    self.valid_dataloader = valid_dataloader

  def train_step(self):
    self.model.train()
    self.callback_handler.on_epoch_begin(self)
    train_loss, train_metric = 0, 0
    for batch, (X, *y) in enumerate(self.train_dataloader):
      X = X.to(self.device)
      y = y[0].to(self.device) if len(y)==1 else [item.to(self.device) for item in y]
      self.callback_handler.on_batch_begin(self)
      y_logits = self.model(X)
      loss = self.loss_fn(y_logits, y)
      self.callback_handler.on_loss_end(self)
      train_loss += np.array(loss.item())
      train_metric += np.array(self.metric_fn(y_logits, y))
      loss.backward()
      self.callback_handler.on_step_begin(self)
      self.optimizer.step()
      self.callback_handler.on_step_end(self)
      self.optimizer.zero_grad()
      self.callback_handler.on_batch_end(self)
    train_loss /= len(self.train_dataloader)
    train_metric /= len(self.train_dataloader)
    return train_loss, train_metric

  def valid_step(self):
    self.model.eval()
    valid_loss, valid_metric = 0, 0
    with torch.inference_mode():
      for batch, (X, *y) in enumerate(self.valid_dataloader):
        X = X.to(self.device)
        y = y[0].to(self.device) if len(y)==1 else [item.to(self.device) for item in y]
        y_logits = self.model(X)
        loss = self.loss_fn(y_logits, y)
        valid_loss += np.array(loss.item())
        valid_metric += np.array(self.metric_fn(y_logits, y))
    valid_loss /= len(self.valid_dataloader)
    valid_metric /= len(self.valid_dataloader)
    return valid_loss, valid_metric

  def train(self, epochs: int):
    """Method for TSEngine to run train and valid loops
    
    Args: epochs [int]: num of epochs to run
    """
    self.callback_handler.on_train_begin(self)
    for epoch in tqdm(range(epochs)):
      self.total_epochs += 1
      train_loss, train_metric = self.train_step()
      valid_loss, valid_metric = self.valid_step()
      self.callback_handler.on_epoch_end(self,
                                         train_loss=train_loss,
                                         train_metric=train_metric if isinstance(train_metric, np.ndarray) else [train_metric],
                                         valid_loss=valid_loss,
                                         valid_metric=valid_metric if isinstance(valid_metric, np.ndarray) else [valid_metric])

  @staticmethod
  def make_lr_fn(start_lr: float,
                 end_lr: float,
                 num_iter: int,
                 step_mode: str = "exp"):
    """Method to generate learning rate function (internal only)"""
    if step_mode == "linear":
      factor = (end_lr / start_lr - 1) / num_iter
      def lr_fn(iteration):
        return 1 + iteration * factor
    else:
      factor = (np.log(end_lr) - np.log(start_lr)) / num_iter
      def lr_fn(iteration):
        return np.exp(factor) ** iteration
    return lr_fn

  def set_lr(self,
             lr: float):
    """Method to set learning rate
    
    Args: lr [float]: learning rate
    """
    for g in self.optimizer.param_groups:
      g["lr"] = lr

  def lr_range_test(self,
                    end_lr: float,
                    start_lr: float | None = None,
                    num_iter: int = 100,
                    step_mode: str = "exp",
                    alpha: float = 0.05,
                    show_graph: bool = True):
    """Method to perform LR Range Test
    Reference: Leslie N. Smith 'Cyclical Learning Rates for Training Neual Networks'

    Args:
      end_lr [float]: upper boundary for the LR Range test
      start_lr [float]: lower boundary for the LR Range test, Defaults to current optimizer LR
      num_iter [int]: number of interations to move from start_lr to end_lr
      step_mode [str]: show LR range test linear or log scale, Defaults to 'exp' 
      alpha [float]: alpha term for smoothed loss (smooth_loss = alpha * loss + (1-alpha) * prev_loss)
      show_graph [bool]: to show LR Range Test result in plot

    Return:
      max_grad_lr [float]: LR with maximum loss gradient (steepest)
      min_loss_lr [float]: LR with minium loss (minimum)

    """
    previous_states = {"model": deepcopy(self.model.state_dict()),
                       "optimizer": deepcopy(self.optimizer.state_dict())}
    if start_lr is not None: self.set_lr(start_lr)
    start_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
    lr_fn = self.make_lr_fn(start_lr, end_lr, num_iter)
    scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)
    tracking = {"loss": [], "lr": []}
    iteration = 0
    while iteration < num_iter:
      for X, *y in self.train_dataloader:
        X = X.to(self.device)
        y = y[0].to(self.device) if len(y)==1 else [item.to(self.device) for item in y]
        y_logits = self.model(X)
        loss = self.loss_fn(y_logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        tracking["lr"].append(scheduler.get_last_lr()[0])
        if iteration == 0:
          tracking["loss"].append(loss.item())
        else:
          prev_loss = tracking["loss"][-1]
          smoothed_loss = alpha * loss.item() + (1 - alpha) * prev_loss
          tracking["loss"].append(smoothed_loss)
        iteration += 1
        if iteration == num_iter:
          break
        self.optimizer.step()
        scheduler.step()
    max_grad_idx = np.gradient(np.array(tracking["loss"])).argmin()
    min_loss_idx = np.array(tracking["loss"]).argmin()
    max_grad_lr = tracking["lr"][max_grad_idx]
    min_loss_lr = tracking["lr"][min_loss_idx]
    self.optimizer.load_state_dict(previous_states["optimizer"])
    self.model.load_state_dict(previous_states["model"])
    if show_graph:
      print(f"Max Gradient: {max_grad_lr:.2E} | Lowest Loss: {min_loss_lr:.2E}")
      fig, ax = plt.subplots(1, 1, figsize=(6, 4))
      ax.plot(tracking["lr"], tracking["loss"])
      ax.scatter(tracking["lr"][max_grad_idx], tracking["loss"][max_grad_idx], c="g", label="Max Gradient")
      ax.scatter(tracking["lr"][min_loss_idx], tracking["loss"][min_loss_idx], c="r", label="Min Loss")
      if step_mode == "exp": ax.set_xscale("log")
      ax.set_xlabel("Learning Rate")
      ax.set_ylabel("Loss")
      ax.legend()
      fig.tight_layout()
    else:
      return max_grad_lr, min_loss_lr

  def save_checkpoint(self, filename: str):
    """Method to save model checkpoint
    
    Args: filename [str]: filename in pt/pth of model, e.g. 'model_path/model.pt'
    """
    checkpoint = {"epoch": self.total_epochs,
                  "model_state_dict": self.model.state_dict(),
                  "optimizer_state_dict": self.optimizer.state_dict(),
                  "results": self.results}
    torch.save(checkpoint, filename)

  def load_checkpoint(self, filename: str):
    """Method to load model checkpoint
    
    Args: file path of checkpoint to load in pt/pth format, e.g. 'model_path/model.pt'
    """
    checkpoint = torch.load(filename)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.total_epochs = checkpoint["epoch"]
    self.results = checkpoint["results"]
    self.model.train()

  def set_lr_scheduler(self, scheduler, is_batch_lr_scheduler=False):
    """Method to set LR scheduler
    
    Args:
      scheduler [torch.optim.scheduler]
      is_batch_lr_scheduler [bool]: True for batch scheduler, False for epoch scheduler
    """
    self.scheduler = scheduler
    self.is_batch_lr_scheduler = is_batch_lr_scheduler

  def model_info(self,
                 col_names: list[str] = ["input_size", "output_size", "num_params", "trainable"],
                 col_width: int = 20,
                 row_settings: list[str] = ["var_names"]):
    """Method to utilise torchinfo to shwo model summary
    Reference: https://github.com/TylerYep/torchinfo

    Args:
      col_names (Iterable[str]): Specify which columns to show in the output
        Currently supported: ("input_size",
                              "output_size",
                              "num_params",
                              "params_percent",
                              "kernel_size",
                              "mult_adds",
                              "trainable")
        Default: ["input_size", "output_size", "num_params", "trainable"]

      col_width (int): Width of each column. Default: 20
    """
    print(torchinfo.summary(model=self.model,
                            input_size=next(iter(self.train_dataloader))[0].shape,
                            verbose=0,
                            col_names=col_names,
                            col_width=col_width,
                            row_settings=row_settings))

  def freeze(self, layers: list[str] = None):
    """Method to change requires_grad to False for layers
    
    Args: layers [list[str]]: list of layers to freeze, freeze all if None
    """
    if layers is None:
      layers = [name for name, module in self.model.named_modules() if "." not in name]

    for layer in layers:
      for name, module in self.model.named_modules():
        if layer in name:
          for param in module.parameters():
            param.requires_grad = False

  def unfreeze(self, layers: list[str] = None):
    """Method to change requires_grad to True for layers
    
    Args: layers [list[str]]: list of layers to unfreeze, unfreeze all if None
    """
    if layers is None:
      layers = [name for name, module in self.model.named_modules() if "." not in name]

    for layer in layers:
      for name, module in self.model.named_modules():
        if layer in name:
          for param in module.parameters():
            param.requires_grad = True

  def set_optimizer(self, optim: tuple[torch.optim.Optimizer, dict[str, float]]):
    """Method to set optimizer
    
    Args:
      optim [tuple[torch.optim.Opimizer, dictionary of parameters]]
      Example usage: optim=(torch.optim.Adam, {'lr': 1e-3})
    """
    self.optimizer = optim[0](params=self.model.parameters(), **optim[1])

  def predict(self, X: torch.Tensor):
    """Method for TSEngine to predict in inference_mode"""
    self.model.eval()
    with torch.inference_mode():
      y_logits = self.model(X)
    self.model.train()
    return y_logits

  def add_graph(self):
    """Method to add graph for TensorBoard"""
    if self.train_dataloader and self.writer:
      X, y = next(iter(self.train_dataloader))
      self.writer.add_graph(self.model, X.to(self.device))

  def plot_loss_curve(self):
    """Method to plot loss curve"""
    plt.plot(self.results["train_loss"], label="Train Loss")
    plt.plot(self.results["valid_loss"], label="Valid Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

  def fit_one_cycle(self, epochs, max_lr=None, min_lr=None):
    """Method to perform fit one cycle polcy 
    Reference: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    Reference: https://arxiv.org/abs/1708.07120

    Sets the learning rate of each parameter group according to the 1cycle learning rate policy. 
    The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate 
    and then from that maximum learning rate to some minimum learning rate

    Args:
      epochs [int]: The number of epochs to train for
      max_lr [float]: Upper learning rate boundaries in the cycle for each parameter group
      min_lr [float]: Lower learning rate boundaries in the cycle for each parameter group

      if max_lr and min_lr is not specified,
      lr_range_test will be performed 
      with max_lr set to min_loss_lr and min_lr set to max_grad_lr
    """
    if max_lr is None or min_lr is None:
      max_grad_lr, min_loss_lr = self.lr_range_test(end_lr=1,
                                                    num_iter=100,
                                                    step_mode="exp",
                                                    show_graph=False)
      if max_lr is None: max_lr = min_loss_lr
      if min_lr is None: min_lr = max_grad_lr
    
    print(f"Max LR: {max_lr:.1E} | Min LR: {min_lr:.1E}")
    pervious_optimizer = deepcopy(self.optimizer)
    self.set_lr(min_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                    max_lr=max_lr,
                                                    total_steps=int(len(self.train_dataloader) * epochs * 1.05))
    self.set_lr_scheduler(scheduler=scheduler, is_batch_lr_scheduler=True)
    self.train(epochs=epochs)
    self.set_lr_scheduler(scheduler=None)
    self.optimizer = pervious_optimizer

  def set_clip_grad_value(self, clip_value):
    """Method to perform Value Clipping
    Clips gradietns element-wise so that they stay inside the [-clip_value, +clip_value]
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
    Executed in GradientClipping Callback
    Args:
      clip_value [float]: max and min gradient value
    """
    self.clipping = lambda: nn.utils.clip_grad_value_(self.model.parameters(),
                                                      clip_value=clip_value)

  def set_clip_grad_norm(self, max_norm, norm_type=2):
    """Method to perform Norm Clipping
    Norm clipping computes the norm for all gradeints together if they were concatedated into a single vector
    if the norm exceeds teh clipping value, teh gradients are scaled down to match the desired norm
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
    Executed in GradientClipping Callback

    Args:
      max_norm [float]: max norm of the gradients
      norm_type [float]: type of the used p-norm. Can be 'inf' for infinity norm
    """
    self.clipping = lambda: nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     max_norm=max_norm,
                                                     norm_type=norm_type)

  def set_clip_backprop(self, clip_value):
    """Method to set clip gradient on the fly using backward hook (register_hook)
    clamp all grad using torch.clamp between [-clip_value, +clip_value]

    Args:
      clip_value [float]: max and min gradient value
    
    """
    if self.clipping is None:
      self.clipping = []
    for p in self.model.parameters():
      if p.requires_grad:
        def func(grad):
          return torch.clamp(grad, -clip_value, clip_value)
        handle = p.register_hook(func)
        self.clipping.append(handle)

  def remove_clip(self):
    """Method to remove gradient clipping in backward hook"""
    if isinstance(self.clipping, list):
      for handle in self.clipping:
        handle.remove()
    self.clipping = None

  def attach_forward_hooks(self, layers_to_hook, hook_fn):
    """Method to attach custom forward hooks
    
    Args:
      layers_to_hook [list]: list of layers to hook
      hook_fn [Callable]: custom hook_fn in during forward pass
    """
    for name, layer in self.modules:
      if name in layers_to_hook:
        handle = layer.register_forward_hook(hook_fn)
        self.forward_hook_handles.append(handle)

  def attach_backward_hooks(self, layers_to_hook, hook_fn):
    """Method to attach custom backward hooks
    
    Args:
      layers_to_hook [list]: list of layers to hook
      hook_fn [Callable]: custom hook_fn in during backward pass
    """
    for name, layer in self.modules:
      if name in layers_to_hook:
        handle = layer.register_full_backward_hook(hook_fn)
        self.backward_hook_handles.append(handle)

  def remove_hooks(self):
    """Method to remove both custom forward and backward hook"""
    for handle in self.forward_hook_handles:
      handle.remove()
    self.forward_hook_handles = []
    for handle in self.backward_hook_handles:
      handle.remove()
    self.backward_hook_handles = []


  class Callback:
    def __init__(self, **kwargs): pass
    def on_train_begin(self, **kwargs): pass
    def on_train_end(self, **kwargs): pass
    def on_epoch_begin(self, **kwargs): pass
    def on_epoch_end(self, **kwargs): pass
    def on_batch_begin(self, **kwargs): pass
    def on_batch_end(self, **kwargs): pass
    def on_loss_begin(self, **kwargs): pass
    def on_loss_end(self, **kwargs): pass
    def on_step_begin(self, **kwargs): pass
    def on_step_end(self, **kwargs): pass

  class callback_handler:
    def on_train_begin(self, **kwargs):
      for callback in self.callbacks: callback.on_train_begin(self, **kwargs)
    def on_train_end(self, **kwargs):
      for callback in self.callbacks: callback.on_train_end(self, **kwargs)
    def on_epoch_begin(self, **kwargs):
      for callback in self.callbacks: callback.on_epoch_begin(self, **kwargs)
    def on_epoch_end(self, **kwargs):
      for callback in self.callbacks: callback.on_epoch_end(self, **kwargs)
    def on_batch_begin(self, **kwargs):
      for callback in self.callbacks: callback.on_batch_begin(self, **kwargs)
    def on_batch_end(self, **kwargs):
      for callback in self.callbacks: callback.on_batch_end(self, **kwargs)
    def on_loss_begin(self, **kwargs):
      for callback in self.callbacks: callback.on_loss_begin(self, **kwargs)
    def on_loss_end(self, **kwargs):
      for callback in self.callbacks: callback.on_loss_end(self, **kwargs)
    def on_step_begin(self, **kwargs):
      for callback in self.callbacks: callback.on_step_begin(self, **kwargs)
    def on_step_end(self, **kwargs):
      for callback in self.callbacks: callback.on_step_end(self, **kwargs)

  class PrintResults(Callback):
    def on_epoch_end(self, **kwargs):
      print(
        f"Epoch: {self.total_epochs} "
        + f"| LR: {np.array(self.learning_rates).mean():.1E} "
        + f"| train_loss: {np.around(kwargs['train_loss'], 3)} "
        + f"| valid_loss: {np.around(kwargs['valid_loss'], 3)} "
        + f"| train_metric: {np.around(kwargs['train_metric'], 3)} "
        + f"| valid_metric: {np.around(kwargs['valid_metric'], 3)} "
      )
      self.learning_rates = []

  class TBWriter(Callback):
    def on_epoch_end(self, **kwargs):
      if self.writer:
        loss_scalars = {"train_loss": kwargs["train_loss"],
                        "valid_loss": kwargs["valid_loss"]}
        self.writer.add_scalars(main_tag="loss",
                                tag_scalar_dict=loss_scalars,
                                global_step=self.total_epochs)

        for i, train_acc in enumerate(kwargs["train_metric"]):
          acc_scalars = {"train_metric": kwargs["train_metric"][i],
                         "valid_metric": kwargs["valid_metric"][i]}
          self.writer.add_scalars(main_tag=f"metric_{i}",
                                  tag_scalar_dict=acc_scalars,
                                  global_step=self.total_epochs)
        self.writer.close()

  class SaveResults(Callback):
    def on_epoch_end(self, **kwargs):
      self.results["train_loss"].append(kwargs["train_loss"])
      self.results["train_metric"].append(kwargs["train_metric"])
      self.results["valid_loss"].append(kwargs["valid_loss"])
      self.results["valid_metric"].append(kwargs["valid_metric"])

  class LearningRateScheduler(Callback):
    def on_batch_end(self, **kwargs):
      if self.scheduler and self.is_batch_lr_scheduler:
        self.scheduler.step()
      self.learning_rates.append(self.optimizer.state_dict()["param_groups"][0]["lr"])

    def on_epoch_end(self, **kwargs):
      self.learning_rates = []
      if self.scheduler and not self.is_batch_lr_scheduler:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
          self.scheduler.step(kwargs["valid_loss"])
        else:
          self.scheduler.step()

  class GradientClipping(Callback):
    def on_step_begin(self, **kwargs):
      if callable(self.clipping): self.clipping()
