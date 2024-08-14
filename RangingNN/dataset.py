from torch.utils.data import dataloader, distributed
import torch
import os
from RangingNN.utils import RANK, PIN_MEMORY, DEFAULT_CFG, LOGGER, TQDM
import numpy as np
import random
from copy import deepcopy
from torch.utils.data import Dataset
from pathlib import Path
import glob
from skimage.transform import rescale
import hashlib
from multiprocessing.pool import ThreadPool
import psutil
from typing import Optional
import h5py

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads


##################################################
# The idea is to load the labels and dataset initiated and load the spectrums on the fly.
# The labels from h5 file and incorprated here is two elements, no cls value
##################################################


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def verify_load_label(file):
    """
    Verify label in one file
    Just check the labels and load them by the way
    return list of  all the labels from a file
    """
    # Number (missing, found, empty, corrupt), message
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ""
    try:
        # Verify labels
        if os.path.isfile(file):
            nf = 1  # label found
            with h5py.File(file, "r") as f:
                key = range(np.asarray(f['label']).shape[0])

                lb = [np.array(f['label'], dtype=np.float32)[k] for k in key]
            nl = len(lb)
            if nl:
                # random spot check
                check = random.randint(0, nl - 1)  # start end included
                assert lb[check].shape[1] == 2, f"labels require 2 columns, {lb.shape[1]} columns detected"
                assert lb[check].max() <= 1, f"non-normalized or out of bounds coordinates {lb[check][lb[check] > 1]}"
                assert lb[check].min() >= 0, f"negative label values {lb[check][lb[check] < 0]}"

            else:
                ne = 1  # label empty
                lb = [np.zeros((0, 2), dtype=np.float32)]
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 2), dtype=np.float32)
        # np.array below avoiding uppacking list value error
        return file, np.array(key), np.array(lb), nm, nf, ne, nc, msg

    except Exception as e:
        nc = 1
        msg = f"WARNING ⚠️ {file}: ignoring corrupt spectrum/label: {e}"
        return [None, None, None, nm, nf, ne, nc, msg]


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing spectrum data.

    Args:
        root_path (str): Path to the folder containing spectrums.
        spectrumsz (int, optional): spectrum size. Defaults to 640.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        batch_size (int, optional): Size of batches. Defaults to None.
        # single_cls (bool, optional): If True, single class training is used. Defaults to False.
        # classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        files (list): List of spectrum file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of spectrums in the dataset.
        transforms (callable): spectrum transformation function.
    """

    def __init__(
            self,
            root_path,
            spectrumsz=30720,
            augment=False,
            prefix="",
            batch_size=16,
            fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.root_path = root_path
        self.spectrumsz = spectrumsz
        self.augment = augment
        self.prefix = prefix
        self.fraction = fraction
        self.files = self.get_files(self.root_path)
        ##########################################################################
        # Operated once initiated
        self.labels = self.get_labels()
        # self.update_labels(include_class=classes)  # single_cls and include_class
        ##########################################################################
        self.ni = len(self.labels)  # number of data points
        self.batch_size = batch_size

        # Transforms
        # self.transforms = self.build_transforms(hyp=hyp)

    def get_files(self, root_path):
        """
        A safe transform from a root path to list of file paths.
        Read files.
        Can be a single folder or a list of folders
        """
        try:
            f = []  # spectrum files
            for p in root_path if isinstance(root_path, list) else [root_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            files = sorted(x.replace("/", os.sep) for x in f)  # if x.split(".")[-1].lower() in IMG_FORMATS
            assert files, f"{self.prefix}No spectrums found in {root_path}. "
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {root_path}\n") from e
        if self.fraction < 1:
            files = files[: round(len(files) * self.fraction)]  # retain a fraction of the dataset
        return files

    # def update_labels(self, include_class: Optional[list]):
    #     """Update labels to include only these classes (optional)."""
    #     include_class_array = np.array(include_class).reshape(1, -1)
    #     for i in range(len(self.labels)):
    #         if include_class is not None:
    #             cls = self.labels[i]["cls"]
    #             bboxes = self.labels[i]["bboxes"]
    #             j = (cls == include_class_array).any(1)
    #             self.labels[i]["cls"] = cls[j]
    #             self.labels[i]["bboxes"] = bboxes[j]
    #
    #         if self.single_cls:
    #             self.labels[i]["cls"][:, 0] = 0

    def load_spectrum(self, f, ki):
        """Loads 1 spectrum, returns(spectrum, original size, resizedsz)."""

        with h5py.File(f, "r") as f:
            sp = np.array(f['input'], dtype=np.float32)[ki]

        if sp.ndim == 1:
            sp = sp[..., None]
        if sp.shape[0] > sp.shape[1]:
            # set default shape (1, N) in case the spectrums are saved as (N,1)
            # so the first dimension is one channel
            sp = sp.T
            sz0 = sp.shape[1]
        else:
            sz0 = sp.shape[1]

        if not (sz0 == self.spectrumsz):  # resize if different m/c binning is required

            sp = rescale(sp, (1, self.spectrumsz / sz0)) # keep the channel dimention not rescaled

        return torch.tensor(sp), sz0, sp.shape[1]

    def get_labels(self):
        """Returns dictionary of labels for training."""

        labels = []
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {self.root_path}..."  # prefix for the progress bar
        total = len(self.files)

        with ThreadPool(NUM_THREADS) as pool:
            results = list(
                pool.imap(func=verify_load_label, iterable=self.files))  # LIST to make the IMapIterator iterable !!!

            pbar = TQDM(results, desc=desc, total=total)

            for (file, key, lb, nm_f, nf_f, ne_f, nc_f, msg) in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if file:
                    lb_ = [lb_1 for lb_1 in lb]
                    key_ = [key_1 for key_1 in key]  # array to list
                    for single, k in zip(lb_, key_):
                        labels.append(
                            {
                                "file": file,
                                "dataset_key": k,
                                "batch_idx": torch.zeros((1, single.shape[0])),  # number of instance zeros
                                "cls": torch.zeros(single[:, 0:1].shape),  # 0 is single class
                                "bboxes": torch.tensor(single),  # n, 2
                                "normalized": True,
                                "bbox_format": "center_width",
                            }
                        )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} spectrums, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {self.root_path}.")

        #########
        lengths = ((len(lb["cls"]), len(lb["bboxes"])) for lb in labels)
        len_cls, len_boxes = (sum(x) for x in zip(*lengths))
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found ")
        return labels

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        # return self.transforms(self.get_spectrum_and_label(index))
        return self.get_spectrum_and_label(index)

    def get_spectrum_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label["spectrum"], label["ori_shape"], label["resized_shape"] = self.load_spectrum(label['file'],
                                                                                           label['dataset_key'])
        label["ratio_pad"] = (
            label["resized_shape"] / label["ori_shape"],
        )  # for evaluation

        return label

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def build_transforms(self, hyp=None):
        """
        Normalization of denoise here.
        """

        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))  # key list, batch tuple
        ######################################################
        # Concat the specturm and labels from the whole batch together
        for i, k in enumerate(keys):
            value = values[i]
            if k == "spectrum":
                value = torch.stack(value, 0)
            if k in {"bboxes", "cls", }:
                value = torch.cat(value, 0)
            # if not those and it is batch_idx
            new_batch[k] = value
        #######################################################
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 1)
        return new_batch
