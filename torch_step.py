"""
Contains class torchstep for training and testing a PyTorch Model
"""

import datetime
import random
import os, pathlib, shutil
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


class TSImageDataset(Dataset):
    """Torch Utils Dataset class to create datasets for image data

    Args:
      paths[list]: list of image paths
      labels[list]: list of image labels
      classes[list]: list of classes from labels, Default to None which will be induced from labels
      transforms[transforms.Compose]: transformation of image, Default to ToTensor()
    """

    def __init__(
        self,
        paths: list,
        labels: list,
        classes: list = None,
        transforms: transforms.Compose = T.Compose(
            [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        ),
    ):
        self.paths = paths
        self.labels = labels
        self.classes = classes if classes else sorted(list(set(self.labels)))
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transforms(Image.open(self.paths[idx])), self.classes.index(
            self.labels[idx]
        )


def get_pretrained_model(name: str, pretrained_weights: str | None = None):
    """Get pretrained model and pretrained transformation (forward and reverse)

  Args:
    model[str]: name of pretrained model
    weights[str]: name of pretrained model weights

  Returns:
    A tuple of (model, forward_transformation, reverse_transformation)
  
  Example usage:
    model, forward_transformation, reverse_transformation = \
      get_prerained_model(name='resnet18',
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
        forward_transforms = T.Compose(
            [
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                T.Resize(
                    size=pretrained_transforms.resize_size,
                    interpolation=pretrained_transforms.interpolation,
                    antialias=True,
                ),
                T.CenterCrop(size=pretrained_transforms.crop_size),
                T.Normalize(
                    mean=pretrained_transforms.mean, std=pretrained_transforms.std
                ),
            ]
        )
        reverse_transforms = T.Compose(
            [
                T.Normalize(
                    mean=[0.0] * 3,
                    std=list(map(lambda x: 1 / x, pretrained_transforms.std)),
                ),
                T.Normalize(
                    mean=list(map(lambda x: -x, pretrained_transforms.mean)),
                    std=[1.0] * 3,
                ),
                T.ToPILImage(),
            ]
        )
    else:
        weights = None
        forward_transforms = T.Compose(
            [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        )
        reverse_transforms = T.ToPILImage()

    # Get model using torchvision.models.get_model
    model = torchvision.models.get_model(name=name, weights=weights)

    return model, forward_transforms, reverse_transforms


def show_batch(
    dataloader: torch.utils.data.DataLoader,
    classes: list,
    transforms: transforms.Compose = T.ToPILImage(),
):
    fig = plt.figure(figsize=(20, 10))
    imgs, labels = next(iter(dataloader))
    nrows, ncolumns = 4, 8
    for i, label in enumerate(labels):
        plt.subplot(nrows, ncolumns, i + 1)
        plt.imshow(transforms(imgs[i]))
        plt.title(classes[label])
        plt.axis(False)


class TSEngine:
    """
    TorchStep class contains a number of useful functions for Pytorch Vision Model Training
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: tuple[torch.optim.Optimizer, dict[str, float]],
        loss_fn: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        metric_fn: Callable | None = None,
    ):
        self.model = deepcopy(model)
        self.optimizer = optim[0](params=self.model.parameters(), **optim[1])
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn if metric_fn else self.accuracy
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.writer = None
        self.scheduler = None
        self.is_batch_lr_scheduler = False
        self.clipping = None
        self.results = {
            "train_loss": [],
            "train_metric": [],
            "valid_loss": [],
            "valid_metric": [],
        }
        self.learning_rates = []
        self.total_epochs = 0
        self.callbacks = [
            self.SaveResults,
            self.PrintResults,
            self.TBWriter,
            self.LearningRateScheduler,
            self.GradientClipping,
        ]

    @staticmethod
    def accuracy(y_logits, y):
        return (y_logits.argmax(dim=1) == y).sum().item() / len(y_logits)

    @staticmethod
    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def set_tensorboard(self, name: str, folder: str = "runs"):
        suffix = datetime.datetime.now().strftime("%Y%m%d")
        self.writer = SummaryWriter(f"{folder}/{name}_{suffix}")

    def set_loaders(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
    ):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

    def train_step(self):
        self.model.train()
        self.callback_handler.on_epoch_begin(self)
        train_loss, train_metric = 0, 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
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
            for batch, (X, y) in enumerate(self.valid_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                loss = self.loss_fn(y_logits, y)
                valid_loss += np.array(loss.item())
                valid_metric += np.array(self.metric_fn(y_logits, y))
        valid_loss /= len(self.valid_dataloader)
        valid_metric /= len(self.valid_dataloader)
        return valid_loss, valid_metric

    def train(self, epochs: int):
        self.callback_handler.on_train_begin(self)
        for epoch in tqdm(range(epochs)):
            self.total_epochs += 1
            train_loss, train_metric = self.train_step()
            valid_loss, valid_metric = self.valid_step()
            self.callback_handler.on_epoch_end(
                self,
                train_loss=train_loss,
                train_metric=train_metric
                if isinstance(train_metric, np.ndarray)
                else [train_metric],
                valid_loss=valid_loss,
                valid_metric=valid_metric
                if isinstance(valid_metric, np.ndarray)
                else [valid_metric],
            )

    @staticmethod
    def make_lr_fn(
        start_lr: float, end_lr: float, num_iter: int, step_mode: str = "exp"
    ):
        if step_mode == "linear":
            factor = (end_lr / start_lr - 1) / num_iter

            def lr_fn(iteration):
                return 1 + iteration * factor

        else:
            factor = (np.log(end_lr) - np.log(start_lr)) / num_iter

            def lr_fn(iteration):
                return np.exp(factor) ** iteration

        return lr_fn

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def lr_range_test(
        self,
        end_lr: float,
        start_lr: float | None = None,
        num_iter: int = 100,
        step_mode: str = "exp",
        alpha: float = 0.05,
        show_graph: bool = True,
    ):
        previous_states = {
            "model": deepcopy(self.model.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
        }
        if start_lr is not None:
            self.set_lr(start_lr)
        start_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        lr_fn = self.make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)
        tracking = {"loss": [], "lr": []}
        iteration = 0
        while iteration < num_iter:
            for X, y in self.train_dataloader:
                X, y = X.to(self.device), y.to(self.device)
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
            ax.scatter(
                tracking["lr"][max_grad_idx],
                tracking["loss"][max_grad_idx],
                c="g",
                label="Max Gradient",
            )
            ax.scatter(
                tracking["lr"][min_loss_idx],
                tracking["loss"][min_loss_idx],
                c="r",
                label="Min Loss",
            )
            if step_mode == "exp":
                ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Loss")
            ax.legend()
            fig.tight_layout()
        else:
            return max_grad_lr, min_loss_lr

    def save_checkpoint(self, filename: str):
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "results": self.results,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_epochs = checkpoint["epoch"]
        self.results = checkpoint["results"]
        self.model.train()

    def set_lr_scheduler(self, scheduler, is_batch_lr_scheduler=False):
        self.scheduler = scheduler
        self.is_batch_lr_scheduler = is_batch_lr_scheduler

    def model_info(
        self,
        col_names: list[str] = ["input_size", "output_size", "num_params", "trainable"],
        col_width: int = 20,
        row_settings: list[str] = ["var_names"],
    ):
        print(
            torchinfo.summary(
                model=self.model,
                input_size=next(iter(self.train_dataloader))[0].shape,
                verbose=0,
                col_names=col_names,
                col_width=col_width,
                row_settings=row_settings,
            )
        )

    def freeze(self, layers: list[str] = None):
        if layers is None:
            layers = [
                name for name, module in self.model.named_modules() if "." not in name
            ]

        for layer in layers:
            for name, module in self.model.named_modules():
                if layer in name:
                    for param in module.parameters():
                        param.requires_grad = False

    def unfreeze(self, layers: list[str] = None):
        if layers is None:
            layers = [
                name for name, module in self.model.named_modules() if "." not in name
            ]

        for layer in layers:
            for name, module in self.model.named_modules():
                if layer in name:
                    for param in module.parameters():
                        param.requires_grad = True

    def set_optimizer(self, optim: tuple[torch.optim.Optimizer, dict[str, float]]):
        self.optimizer = optim[0](params=self.model.parameters(), **optim[1])

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            y_logits = self.model(X)
        self.model.train()
        return y_logits

    def add_graph(self):
        if self.train_dataloader and self.writer:
            X, y = next(iter(self.train_dataloader))
            self.writer.add_graph(self.model, X.to(self.device))

    def plot_loss_curve(self):
        plt.plot(self.results["train_loss"], label="Train Loss")
        plt.plot(self.results["valid_loss"], label="Valid Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

    def fit_one_cycle(self, epochs, max_lr=None, min_lr=None):
        max_grad_lr, min_loss_lr = self.lr_range_test(
            end_lr=1, num_iter=100, step_mode="exp", show_graph=False
        )
        if max_lr is None:
            max_lr = min_loss_lr
        if min_lr is None:
            min_lr = max_grad_lr
        print(f"Max LR: {max_lr:.1E} | Min LR: {min_lr:.1E}")
        pervious_optimizer = deepcopy(self.optimizer)
        self.set_lr(min_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=int(len(self.train_dataloader) * epochs * 1.05),
        )
        self.set_lr_scheduler(scheduler=scheduler, is_batch_lr_scheduler=True)
        self.train(epochs=epochs)
        self.set_lr_scheduler(scheduler=None)
        self.optimizer = pervious_optimizer

    def set_clip_grad_value(self, clip_value):
        self.clipping = lambda: nn.utils.clip_grad_value_(
            self.model.parameters(), clip_value=clip_value
        )

    def set_clip_grad_norm(self, max_norm, norm_type=2):
        self.clipping = lambda: nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=max_norm, norm_type=norm_type
        )

    def set_clip_backprop(self, clip_value):
        if self.clipping is None:
            self.clipping = []
        for p in self.model.parameters():
            if p.requires_grad:
                func = lambda grad: torch.clamp(grad, -clip_value, clip_value)
                handle = p.register_hook(func)
                self.clipping.append(handle)

    def remove_clip(self):
        if isinstance(self.clipping, list):
            for handle in self.clipping:
                handle.remove()
        self.clipping = None

    class Callback:
        def __init__(self, **kwargs):
            pass

        def on_train_begin(self, **kwargs):
            pass

        def on_train_end(self, **kwargs):
            pass

        def on_epoch_begin(self, **kwargs):
            pass

        def on_epoch_end(self, **kwargs):
            pass

        def on_batch_begin(self, **kwargs):
            pass

        def on_batch_end(self, **kwargs):
            pass

        def on_loss_begin(self, **kwargs):
            pass

        def on_loss_end(self, **kwargs):
            pass

        def on_step_begin(self, **kwargs):
            pass

        def on_step_end(self, **kwargs):
            pass

    class callback_handler:
        def on_train_begin(self, **kwargs):
            for callback in self.callbacks:
                callback.on_train_begin(self, **kwargs)

        def on_train_end(self, **kwargs):
            for callback in self.callbacks:
                callback.on_train_end(self, **kwargs)

        def on_epoch_begin(self, **kwargs):
            for callback in self.callbacks:
                callback.on_epoch_begin(self, **kwargs)

        def on_epoch_end(self, **kwargs):
            for callback in self.callbacks:
                callback.on_epoch_end(self, **kwargs)

        def on_batch_begin(self, **kwargs):
            for callback in self.callbacks:
                callback.on_batch_begin(self, **kwargs)

        def on_batch_end(self, **kwargs):
            for callback in self.callbacks:
                callback.on_batch_end(self, **kwargs)

        def on_loss_begin(self, **kwargs):
            for callback in self.callbacks:
                callback.on_loss_begin(self, **kwargs)

        def on_loss_end(self, **kwargs):
            for callback in self.callbacks:
                callback.on_loss_end(self, **kwargs)

        def on_step_begin(self, **kwargs):
            for callback in self.callbacks:
                callback.on_step_begin(self, **kwargs)

        def on_step_end(self, **kwargs):
            for callback in self.callbacks:
                callback.on_step_end(self, **kwargs)

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
                loss_scalars = {
                    "train_loss": kwargs["train_loss"],
                    "valid_loss": kwargs["valid_loss"],
                }
                self.writer.add_scalars(
                    main_tag="loss",
                    tag_scalar_dict=loss_scalars,
                    global_step=self.total_epochs,
                )

                for i, train_acc in enumerate(kwargs["train_metric"]):
                    acc_scalars = {
                        "train_metric": kwargs["train_metric"][i],
                        "valid_metric": kwargs["valid_metric"][i],
                    }
                    self.writer.add_scalars(
                        main_tag=f"metric_{i}",
                        tag_scalar_dict=acc_scalars,
                        global_step=self.total_epochs,
                    )
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
            self.learning_rates.append(
                self.optimizer.state_dict()["param_groups"][0]["lr"]
            )

        def on_epoch_end(self, **kwargs):
            self.learning_rates = []
            if self.scheduler and not self.is_batch_lr_scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(kwargs["valid_loss"])
                else:
                    self.scheduler.step()

    class GradientClipping(Callback):
        def on_step_begin(self, **kwargs):
            if callable(self.clipping):
                self.clipping()
