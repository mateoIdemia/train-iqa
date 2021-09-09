import os
import time
import math
import datetime
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR

import random

from torch import optim
import torchvision

from fastprogress import master_bar, progress_bar
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
import PIL

import holocron
from collections import defaultdict
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


from contiguous_params import ContiguousParams
import wandb
from utils import freeze_model



class Trainer:

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 gpu=None, output_file='checkpoint.pth', acc_threshold=0.05, configwb=None):

        self.model = model
        self.configwb = configwb
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.acc_threshold = acc_threshold

        # Output file
        self.output_file = output_file

        # Initialize
        self.example_ct=0
        self.step = 0
        self.start_epoch = 0
        self.epoch = 0
        self.train_loss = 0
        self.train_loss_recorder = []
        self.val_loss_recorder = []
        self.min_loss = math.inf
        self.gpu = gpu
        self._params = None
        self.lr_recorder, self.loss_recorder = [], []
        self.set_device(gpu)
        self._reset_opt(self.optimizer.defaults['lr'])

    def set_device(self, gpu):
        """
        Move tensor objects to the target GPU

        Args:
            gpu (int): index of the target GPU device
        """
        if isinstance(gpu, int):
            if not torch.cuda.is_available():
                raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
            if gpu >= torch.cuda.device_count():
                raise ValueError("Invalid device index")
            torch.cuda.set_device(gpu)
            self.model = self.model.cuda()
            if isinstance(self.criterion, torch.nn.Module):
                self.criterion = self.criterion.cuda()

    def save(self, output_file):
        """
        Save a trainer checkpoint

        Args:
            output_file (str): destination file path
        """
        
        torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, output_file))


    def load(self, state):
        """
        Resume from a trainer state

        Args:
            state (dict): checkpoint dictionary
        """
        self.start_epoch = state['epoch']
        self.epoch = self.start_epoch
        self.step = state['step']
        self.min_loss = state['min_loss']
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['model'])

    def _fit_epoch(self, freeze_until, mb):
        """
        Fit a single epoch

        Args:
            freeze_until (str): last layer to freeze
            mb (fastprogress.master_bar): primary progress bar
        """
        # self.model = freeze_bn(self.model.train())
        self.train_loss = 0
        self.model.train()
        pb = progress_bar(self.train_loader, parent=mb)
        for x, target in pb:
            x, target = self.to_cuda(x, target)
            self.example_ct +=  x.shape[0]
            # Forward
            batch_loss = self._get_loss(x, target)
            self.train_loss += batch_loss.item()

            # Backprop
            self._backprop_step(batch_loss)
            # Update LR
            self.scheduler.step()
            pb.comment = f"Training loss: {batch_loss.item():.4}"

            self.step += 1

            # Report metrics every 20th batch
            if self.step % 5 == 0:
                # where the magic happens
                wandb.log({"epoch": self.epoch, "train_loss": batch_loss.item()}, step=self.example_ct)
   

        self.epoch += 1
        # print(self.train_loss,len(self.train_loader),self.train_loss/len(self.train_loader))
        self.train_loss /= len(self.train_loader)
        self.train_loss_recorder.append(self.train_loss)

    def to_cuda(self, x, target):
        """Move input and target to GPU !"""
        if isinstance(self.gpu, int):
            if self.gpu >= torch.cuda.device_count():
                raise ValueError("Invalid device index")
            return self._to_cuda(x, target)
        else:
            return x, target

    @staticmethod
    def _to_cuda(x, target):
        """Move input and target to GPU !"""
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        return x, target

    def _backprop_step(self, loss):
        # Clean gradients
        self.optimizer.zero_grad()
        # Backpropate the loss
        loss.backward()
        # Update the params
        self.optimizer.step()

    def _get_loss(self, x, target):
        # Forward
        out = self.model(x)
        # Loss computation
        return self.criterion(out, target)

    def _set_params(self):
        self._params = ContiguousParams([p for p in self.model.parameters() if p.requires_grad])

    def _reset_opt(self, lr):
        """Reset the target params of the optimizer !"""
        self.optimizer.defaults['lr'] = lr
        self.optimizer.state = defaultdict(dict)
        self.optimizer.param_groups = []
        self._set_params()
        self.optimizer.add_param_group(dict(params=self._params.contiguous()))

    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

    @staticmethod
    def _eval_metrics_str(eval_metrics):
        raise NotImplementedError

    def _reset_scheduler(self, lr, num_epochs, sched_type='onecycle'):
        if sched_type == 'onecycle':
            self.scheduler = OneCycleLR(self.optimizer, lr, num_epochs * len(self.train_loader))
        elif sched_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, num_epochs * len(self.train_loader), eta_min=lr / 25e4)
        else:
            raise ValueError(f"The following scheduler type is not supported: {sched_type}")

    def fit_n_epochs(self, num_epochs, lr, freeze_until=None, sched_type='onecycle'):
        """
        Train the model for a given number of epochs

        Args:
            num_epochs (int): number of epochs to train
            lr (float): learning rate to be used by the scheduler
            freeze_until (str, optional): last layer to freeze
            sched_type (str, optional): type of scheduler to use
        """

        if self.configwb:
          wandb.watch(self.criterion, log="all", log_freq=200)

        self.epoch = 0
        self.train_loss_recorder = []
        self.val_loss_recorder = []

        self.model = freeze_model(self.model, freeze_until)
        # Update param groups & LR
        self._reset_opt(lr)
        # Scheduler
        self._reset_scheduler(lr, num_epochs, sched_type)

        mb = master_bar(range(num_epochs))
        for _ in mb:

            self._fit_epoch(freeze_until, mb)
            # Check whether ops invalidated the buffer
            self._params.assert_buffer_is_valid()
            eval_metrics = self.evaluate()

            # master bar
            mb.main_bar.comment = f"Epoch {self.start_epoch + self.epoch}/{self.start_epoch + num_epochs}"
            mb.write(f"Epoch {self.start_epoch + self.epoch}/{self.start_epoch + num_epochs} - "
                     f"{self._eval_metrics_str(eval_metrics)}")

            if eval_metrics['loss_val'] < self.min_loss:
                print(f"Validation loss decreased {self.min_loss:.4} --> "
                      f"{eval_metrics['loss_val']:.4}: saving state...")
                self.min_loss = eval_metrics['loss_val']
                wandb.log({"best_val_loss": self.min_loss})
                wandb.log({"best_val_accB": eval_metrics['acc_valB']})
                #wandb.log({"best_val_accN": eval_metrics['acc_valN']})
       
                self.save(self.output_file)

    def lr_find(self, freeze_until=None, start_lr=1e-7, end_lr=1, num_it=100):
        """
        Gridsearch the optimal learning rate for the training

        Args:
           freeze_until (str, optional): last layer to freeze
           start_lr (float, optional): initial learning rate
           end_lr (float, optional): final learning rate
           num_it (int, optional): number of iterations to perform
        """
        if len(self.train_loader) < num_it:
            print("Can't reach", num_it, "iterations, num_it is now", len(self.train_loader))
            num_it = len(self.train_loader)

        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(start_lr)
        gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
        scheduler = MultiplicativeLR(self.optimizer, lambda step: gamma)

        self.lr_recorder = [start_lr * gamma ** idx for idx in range(num_it)]
        self.loss_recorder = []

        for batch_idx, (x, target) in enumerate(self.train_loader):
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss = self._get_loss(x, target)
            self._backprop_step(batch_loss)
            # Update LR
            scheduler.step()

            # Record
            self.loss_recorder.append(batch_loss.item())
            # Stop after the number of iterations
            if batch_idx + 1 == num_it:
                break

    def showBatch(self, nb_images=None, nrow=4, fig_size_X=15, fig_size_Y=15, normalize=True):

        x, target = next(iter(self.train_loader))

        if(nb_images):
            x = x[:nb_images, :, :, :]
            target = target[:nb_images]

        images = make_grid(x, nrow=nrow)  # the default nrow is 8

        # Inverse normalize the images
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        if normalize:
            im_inv = inv_normalize(images)
        else:
            im_inv = images

        # Print the images
        plt.figure(figsize=(fig_size_X, fig_size_Y))
        plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))

    def plot_recorder(self, beta=0.95, block=True):
        """
        Display the results of the LR grid search

        Args:
            beta (float, optional): smoothing factor
            block (bool, optional): whether the plot should block execution
        """
        if len(self.lr_recorder) != len(self.loss_recorder) or len(self.lr_recorder) == 0:
            raise AssertionError("Please run the `lr_find` method first")

        # Exp moving average of loss
        smoothed_losses = []
        avg_loss = 0
        for idx, loss in enumerate(self.loss_recorder):
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

        plt.plot(self.lr_recorder[10:-5], smoothed_losses[10:-5])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Training loss')
        plt.grid(True, linestyle='--', axis='x')
        plt.show(block=block)

    def plot_losses(self):

        plt.plot(self.val_loss_recorder, label='val loss')
        plt.plot(self.train_loss_recorder, label='train loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower right')
        plt.show()

    def check_setup(self, freeze_until=None, lr=3e-4, num_it=100):
        """
        Check whether you can overfit one batch

        Args:
            freeze_until (str, optional): last layer to freeze
            lr (float, optional): learning rate to be used for training
            num_it (int, optional): number of iterations to perform
        """
        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(lr)

        prev_loss = math.inf

        x, target = next(iter(self.train_loader))
        x, target = self.to_cuda(x, target)

        for _ in range(num_it):
            # Forward
            batch_loss = self._get_loss(x, target)
            # Backprop
            self._backprop_step(batch_loss)

            # Check that loss decreases
            if batch_loss.item() > prev_loss:
                return False
            prev_loss = batch_loss.item()

        return True


class ClassificationTrainer(Trainer):

    """
    Image classification trainer class

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training loader
        val_loader (torch.utils.data.DataLoader): validation loader
        criterion (torch.nn.Module): loss criterion
        optimizer (torch.optim.Optimizer): parameter optimizer
        gpu (int, optional): index of the GPU to use
        output_file (str, optional): path where checkpoints will be saved
    """

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate the model on the validation set

        Returns:
            dict: evaluation metrics
        """
        self.model.eval()
        sigmoid = nn.Sigmoid()

        loss_val, top_valB, top_valN, num_samples = 0, 0, 0, 0
        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            # Forward
            out = self.model(x)
            # Loss computation
            loss_val += self.criterion(out, target).item()

            out = torch.sigmoid(out)
            top_valB += torch.sum((torch.abs(target - out) <= 0.08)).item()
            # top_valB += torch.sum((torch.abs(target[:,0] - out[:,0]) <= 0.08)).item()
            # top_valN += torch.sum((torch.abs(target[:,1] - out[:,1]) <= 0.08)).item()

            num_samples += x.shape[0]

            self.val_loss_recorder.append(loss_val / num_samples)

        loss_val /= len(self.val_loader)
        acc_valB = top_valB/ num_samples
        #acc_valN = top_valN/ num_samples
        wandb.log({"val_accB": acc_valB})
        #wandb.log({"val_accN": acc_valN})
        wandb.log({"val_loss": loss_val})
      

        return dict(train_loss=self.train_loss, loss_val=loss_val, acc_valB=acc_valB)

    @staticmethod
    def _eval_metrics_str(eval_metrics):
        return (f"Training loss: {eval_metrics['train_loss']:.4} "
                f"Validation loss: {eval_metrics['loss_val']:.4} "
                f"Acc Val Blur: {eval_metrics['acc_valB']:.2%}")
                #f"Acc Val Noise: {eval_metrics['acc_valN']:.2%}")
