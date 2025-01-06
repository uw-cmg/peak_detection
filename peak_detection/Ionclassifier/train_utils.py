import math
import os
import random
from copy import deepcopy
import torch
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.nn import Conv2d, ConvTranspose2d
import subprocess

def init_seeds(seed=0, deterministic=True):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)  # warn if deterministic is not possible
        torch.use_deterministic_algorithms(True, warn_only=True)  # warn if deterministic is not possible
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)

    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)) else model

def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement."""

    def __init__(self, patience=50):
        """
        Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        # self.best_fitness = 0.0  # i.e. mAP
        self.best_fitness = 1e5  # i.e. loss

        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch. Here I would use the chi_loss

        Returns:
            (bool): True if training should stop, False otherwise
        """

        if fitness <= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            print(
                f"Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop


def weights_init(module):
    imodules = (Conv2d, ConvTranspose2d)
    if isinstance(module, imodules):
        torch.nn.init.xavier_uniform_(module.weight.data)
        torch.nn.init.zeros_(module.bias)


def plot_losses(train_loss, test_loss, savepath) -> None:

    """
    Plots train and test losses
    """
    print('Plotting training history')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(train_loss, label='Train')
    ax.plot(test_loss, label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    # plt.show()
    plt.savefig(savepath + '/history.png')

def get_gpu_info(cuda_device: int) -> int:
    """
    Get the current GPU memory usage
    Adapted with changes from
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--id=' + str(cuda_device),
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_usage = [int(y) for y in result.split(',')]
    return gpu_usage[0:2]

from types import SimpleNamespace
class Parameters(SimpleNamespace):
    """refered to Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'.
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)



# https://github.com/ThFriedrich/airpi/blob/main/ap_training/lr_scheduler.py
class lr_schedule:

    def __init__(self, prms):
        self.learning_rate = prms.learning_rate_0
        self.epochs = prms.epochs
        self.epochs_cycle_1 = prms.epochs_cycle_1
        self.epochs_cycle = prms.epochs_cycle
        self.epochs_ramp = prms.epochs_ramp
        self.lr_fact = prms.lr_fact
        self.lr_bottom = self.learning_rate * self.lr_fact
        # self.b_gridSearch = 'learning_rate_rng' in prms
        self.cooldown = prms.cooldown
        self.warmup = prms.warmup
        # if self.b_gridSearch:
        #     self.learning_rate_rng = prms['learning_rate_rng']
        self.schedule = []
        self.build_lr_schedule()

    # def grid_search(self, epoch):
    #     self.learning_rate = self.learning_rate_rng[epoch]
    #     return self.learning_rate

    def s_transition(self, epochs, epochs_cycle_1, epochs_cycle, epochs_ramp, lr_fact, cooldown, warmup, epoch):
        '''Changes Learning Rate with a continuous transition
            (Cubic spline interpolation between 2 Values)'''

        if epoch >= epochs_cycle_1:
            cycle = epochs_cycle
            ep = epoch - epochs_cycle_1
        else:
            cycle = epochs_cycle_1
            ep = epoch

        cycle_pos = ep % cycle
        ep_cd = cycle - epochs_ramp

        if cycle_pos == 0:
            if epoch == 0:
                self.lr_bottom = self.learning_rate * lr_fact
            else:
                self.lr_bottom = self.learning_rate * lr_fact * lr_fact
                self.learning_rate = self.learning_rate * lr_fact

        if cycle_pos >= ep_cd and cooldown is True:
            lr_0 = self.learning_rate
            lr_1 = self.lr_bottom
            cs = self.s_curve_interp(lr_0, lr_1, epochs_ramp)
            ip = cycle_pos - ep_cd
            return cs(ip)
        elif cycle_pos < epochs_ramp and warmup is True and epoch < epochs_cycle_1:
            lr_1 = self.learning_rate
            cs = self.s_curve_interp(1e-8, lr_1, epochs_ramp)
            ip = cycle_pos
            return cs(ip)
        else:
            return self.learning_rate

    def build_lr_schedule(self):
        lr = np.ones(self.epochs)
        for lr_stp in range(self.epochs):
            # if self.b_gridSearch:
            #     lr[lr_stp] = self.grid_search(lr_stp)

            lr[lr_stp] = self.s_transition(self.epochs, self.epochs_cycle_1, self.epochs_cycle, self.epochs_ramp,
                                           self.lr_fact, self.cooldown, self.warmup, lr_stp)

        self.schedule = lr
        # self.plot()

    def plot(self):
        plt.figure(figsize=(6.5, 4))
        plt.plot(np.linspace(1, self.epochs, self.epochs), self.schedule)
        # plt.savefig('lr.png')

    def s_curve_interp(self, lr_0, lr_1, interval):
        '''Cubic spline interpolation between 2 Values'''
        x = (0, interval)
        y = (lr_0, lr_1)
        cs = CubicSpline(x, y, bc_type=((1, 0.0), (1, 0.0)))
        return cs
