import gc
import math
import multiprocessing
import os
import time
import warnings
from copy import deepcopy
from datetime import datetime
from logging import raiseExceptions
from typing import Dict
import json
import numpy as np
import torch
from torch import optim, nn
import torch.utils.data as data

from peak_detection.Ionclassifier.RNN import IonRNN, WeightedFocalLoss
from peak_detection.Ionclassifier.dataset import Dataset
from peak_detection.Ionclassifier.train_utils import Parameters, init_seeds, weights_init, EarlyStopping, ModelEMA, get_gpu_info, plot_losses

def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

def collate_fn(batch):
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: len(x), reverse=True)

    # Get lengths of each sequence
    lengths = [len(seq) for seq in batch]

    # Pad sequences
    padded_seqs = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

    return padded_seqs, lengths

class BaseTrainer:
    """
    Example usage:
    import yaml
    from pathlib import Path
    from peak_detection.Ionclassifier.trainer import BaseTrainer

    hyperdict = yaml.safe_load(Path("/srv/home/jwei74/APT_ML/ranging/IonRNN.yaml").read_text())
    trainer = BaseTrainer(hyperdict)
    trainer.train()
    """
    def __init__(self, hyperdict: Dict):

        self.d_train, self.d_test = None, None
        self.lr_history, self.momentum_history = [],[]
        self.trainloss_total, self.testloss_total = [],[]
        self.accumulate = None
        self.lr = None
        self.epoch = None
        self.model = None
        self.optimizer = None
        self.lf = None
        self.scheduler = None
        self.pms = Parameters(**hyperdict)
        self.device = torch.device(self.pms.device)
        self.patience = self.pms.patience
        self.save_path = self.pms.save_path
        self.data_path = self.pms.data_path
        self.subset = self.pms.subset
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def train(self):

        # Initialize model
        init_seeds(seed=self.pms.seed, deterministic=self.pms.deterministic)
        self.model = IonRNN(input_size=self.pms.input_size, hidden_size=self.pms.hidden_size,
                            num_layers=self.pms.num_layers, num_classes=self.pms.num_classes,
                            dropout=self.pms.dropout).to(self.device)

        self.model.to(self.device)
        self.model.apply(weights_init)

        self.stopper = EarlyStopping(patience=self.patience)  #########################################
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.pms.amp)  # automatic mixed precision training for speeding up and save memory
        self.ema = ModelEMA(self.model)
        self.accumulate = max(round(self.pms.nbs / self.pms.batchsize),1) # accumulate loss before optimizing, nbs nominal batch size
        weight_decay = self.pms.weight_decay * self.pms.batchsize * self.accumulate / self.pms.nbs  # scale weight_decay

        self.optimizer = self.build_optimizer(model=self.model, lr=self.pms.lr0, momentum=self.pms.momentum,decay=weight_decay)
        self.setup_scheduler()
        self.scheduler.last_epoch = - 1  # do not move


        # Initialize dataset
        dataset = Dataset(data_dir=self.data_path, filestart=0, normalize_c=self.pms.normalize_c,
                          subset = self.pms.subset)

        # print("The input data shape is ", dataset.data_shape())

        indices = torch.randperm(len(dataset)).tolist()

        dataset_train = torch.utils.data.Subset(dataset,
                                                indices[:-int(0.4 * len(dataset))])
        dataset_test = torch.utils.data.Subset(dataset, indices[-int(0.4 * len(dataset)):])

        pool = multiprocessing.Pool()
        # define training and validation data loaders
        self.d_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.pms.batchsize, shuffle=True, pin_memory=True,
            num_workers=int(pool._processes / 2), collate_fn=collate_fn)

        self.d_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=self.pms.batchsize, shuffle=True, pin_memory=True,
            num_workers=int(pool._processes / 2), collate_fn=collate_fn)

        print('##############################START TRAINING ######################################')

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.optimizer.zero_grad()
        self.train_cell(self.d_train, self.d_test)
        torch.save({"date": datetime.now().isoformat(),'ema': deepcopy(self.ema.ema), 'state_dict': self.model.state_dict(),
                    'train': self.trainloss_total, 'test': self.testloss_total}, self.save_path + '/model_final.tar')

        plot_losses(self.trainloss_total, self.testloss_total, self.save_path)

        return self.model


    def train_cell(self, data_loader_train, data_loader_test, check_gradient=True, regularization=False):
        """
        """

        nb = len(data_loader_train)  # number of batches
        nw = self.pms.warmup_iters  # warmup iterations
        last_opt_step = -1

        record = time.time()
        correct = 0
        total = 0
        # note: I will still keep the iteration loop and no real epoch loop
        for i, ((inputs_train, targets_train), (inputs_test, targets_test)) in enumerate(
                zip(data_loader_train, data_loader_test)):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()  # turn on train mode!

            self.optimizer.zero_grad() # YOU HAVE TO KEEP THIS. Do not remove
            (input_train, lengths_train) = inputs_train
            input_train = input_train.to(self.device)
            lengths_train = lengths_train.to(self.device)
            targets_train= targets_train.to(self.device)

            # Warmup
            # ni = i + nb * epoch
            if i <= nw:
                xi = [0, nw]  # x interp
                self.accumulate = max(1, int(np.interp(i, xi, [1, self.pms.nbs / self.pms.batchsize]).round()))
                for j, x in enumerate(self.optimizer.param_groups):
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        i, xi, [self.pms.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(i)]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(i, xi, [self.pms.warmup_momentum, self.pms.momentum])

            # Forward
            with torch.cuda.amp.autocast(self.pms.amp):
                pred = self.model(input_train, lengths_train)

                # Calculate loss only on valid sequence parts
                mask = torch.arange(pred.size(1)).expand(len(lengths_train), pred.size(1)) < torch.tensor(lengths_train).unsqueeze(1)
                mask = mask.to(pred.device)
                lossfunc = WeightedFocalLoss(alpha = self.pms.loss_alpha , gamma = self.pms.loss_gamma)
                trainloss = lossfunc(pred[mask], targets_train[mask])

                self.trainloss_total.append(trainloss.item())


            # Backward
            self.scaler.scale(trainloss).backward() #######################
            for param_group in self.optimizer.param_groups:
                self.lr_history.append(param_group['lr'])
                if 'betas' in param_group:
                    self.momentum_history.append(param_group['betas'])
                else:
                    self.momentum_history.append(None)  # If momentum is not used

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if i - last_opt_step >= self.accumulate:
                self.optimizer_step() #########################
                last_opt_step = i

            if check_gradient:
                for p, n in self.model.named_parameters():
                    if n[-6:] == 'weight':
                        if (p.grad > 1e5).any() or (p.grad < 1e-5).any():
                            print('===========\ngradient:{}\n----------\n{}'.format(n, p.grad))
                            break


            ##########################################################################
            ###Test###

            (input_test, lengths_test) = inputs_train
            input_test = input_test.to(self.device)
            lengths_test = lengths_test.to(self.device)
            targets = targets_test.to(self.device)
            self.model.eval()
            with torch.no_grad():

                pred = self.model(input_test, lengths_test)

                testloss = lossfunc(pred, targets)

            self.testloss_total.append(testloss.item())

            del inputs_train, inputs_test, targets  # manually release GPU memory during training loop.

            if i % self.pms.print_freq == 0:
                print("Epoch{}\t".format(i), "Train Loss data {:.3f}".format(trainloss.item()))
                print("Epoch{}\t".format(i), "Test Loss data {:.3f}".format(testloss.item()),
                      'Cost: {}\t s.'.format(time.time() - record))
                gpu_usage = get_gpu_info(torch.cuda.current_device())
                print('GPU memory usage: {}/{}'.format(gpu_usage[0], gpu_usage[1]))
                record = time.time()

            stop = self.stopper(i, testloss.item())

            if not stop:
                if self.stopper.best_epoch == i:
                    torch.save(
                        {'ema': deepcopy(self.ema.ema),'state_dict': self.model.state_dict(),
                         'epoch': self.stopper.best_epoch,"date": datetime.now().isoformat(),
                         "loss_alpha": self.pms.loss_alpha, "loss_beta": self.pms.loss_gamma},
                        self.save_path + '/model_bestepoch.tar')
            else:
                break

            if i == (self.pms.epochs - 1):
                break



        # at finish
        self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
        self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

        # Validation
        # to be added
        print("on_fit_epoch_end")
        gc.collect()
        torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

    def build_optimizer(self, model, lr=0.001, momentum=0.9, decay=1e-5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations. Most importantly, it do not apply weight decay to

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        name = "AdamW"

        for module_name, module in model.named_modules():
            # the requires_grad info should be included
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        print(
            f"{'optimizer:'} {type(optimizer).__name__}(default lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer

    def setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.pms.cos_lr:
            self.lf = one_cycle(1, self.pms.lrf, self.pms.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.pms.epochs, 0) * (1.0 - self.pms.lrf) + self.pms.lrf  # linear

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)




