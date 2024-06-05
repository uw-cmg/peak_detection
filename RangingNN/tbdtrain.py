import math
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from peak_detection.RangingNN.YOLO1D import DetectionModel
from peak_detection.RangingNN.dataset import BaseDataset, build_dataloader
from peak_detection.RangingNN.trainer import BaseTrainer
from peak_detection.RangingNN.validator import DetectionValidator

from peak_detection.RangingNN.utils import RANK, attempt_load_one_weight, LOGGER, plt_settings





@plt_settings()
def plot_results(file="path/to/results.csv", dir="", segment=False, pose=False, classify=False, on_plot=None):
    """
    Plot training results from a results CSV file. The function supports various types of data including segmentation,
    pose estimation, and classification. Plots are saved as 'results.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to 'path/to/results.csv'.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided. Defaults to ''.
        segment (bool, optional): Flag to indicate if the data is for segmentation. Defaults to False.
        pose (bool, optional): Flag to indicate if the data is for pose estimation. Defaults to False.
        classify (bool, optional): Flag to indicate if the data is for classification. Defaults to False.
        on_plot (callable, optional): Callback function to be executed after plotting. Takes filename as an argument.
            Defaults to None.

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results('path/to/results.csv', segment=True)
        ```
    """
    import pandas as pd  # scope for faster 'import ultralytics'
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
        index = [1, 4, 2, 3]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 8, 9, 12, 13]
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 8, 9, 10, 6, 7]
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate(index):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in {8, 9, 10}:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.warning(f"WARNING: Plotting error for {f}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, root_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            root_path (str): Path to the folder containing data.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        # stride = max(int(self.model.stride.max() if self.model else 0), 32)
        cfg = self.args
        dataset = BaseDataset
        return dataset(
            root_path=root_path,
            spectrumsz=cfg.imgsz,
            batch_size=batch,
            augment=False,  # augmentation
            prefix='mode',
            data=self.data,
            fraction=1.0,
        )

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    # def plot_training_samples(self, batch, ni):
    #     """Plots training samples with their annotations."""
    #     plot_images(
    #         images=batch["img"],
    #         batch_idx=batch["batch_idx"],
    #         cls=batch["cls"].squeeze(-1),
    #         bboxes=batch["bboxes"],
    #         paths=batch["im_file"],
    #         fname=self.save_dir / f"train_batch{ni}.jpg",
    #         on_plot=self.on_plot,
    #     )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png




