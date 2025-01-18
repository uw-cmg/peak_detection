import json

import numpy as np
import torch
import os
import torch.nn.functional as F
import itertools
import re
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
CHEMICAL_ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
                     'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
                     'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                     'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
                     'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
                     'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
                     'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
                     'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                     'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                     'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
                     'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

le = LabelEncoder()
le.fit(CHEMICAL_ELEMENTS)


class Dataset:
    """
    :argument
    threshold_c: filter out the peaks with very low relative counts (normalized by the maximum counts),
    """
    def __init__(self, data_dir, filestart=0, normalize_c=True,
                 subset=1, label_encoder=le, threshold_c=1e-7, **kwargs):

        self.data_dir = data_dir
        self.subset = subset
        self.normalize_c = normalize_c
        self.threshold_c = threshold_c
        # filenum = len(os.listdir(data_dir))
        datalist = []
        for data_dir_single in data_dir:
            datalist = datalist + [data_dir_single + it for it in sorted(os.listdir(data_dir))]
        if self.subset < 1:
            datalist = random.sample(datalist, int(len(datalist) * self.subset))

        self.ids = datalist
        self.label_encoder = label_encoder

    def __getitem__(self, i):
        img_id = self.ids[i]  # folder names
        file = pd.read_csv(img_id)  ###########
        mc = file.get(['mc']).to_numpy().squeeze()
        counts = file.get(['counts']).to_numpy().squeeze()
        target = {'ion': file.get(['ion']).to_numpy().squeeze(), 'charge': file.get(['charge']).to_numpy().squeeze(),
                  'ion2': file.get(['ion2']).to_numpy().squeeze(),
                  'charge2': file.get(['charge2']).to_numpy().squeeze()}
        if self.normalize_c:
            counts = (counts - counts.min()) / (counts.max() - counts.min())
        labels = np.array([re.findall('.[^A-Z]*', it)[0] for it in target['ion']]) # Not including the light elements in mole here
        # Initialize label encoder if not provided
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels.ravel())
        else:
            self.label_encoder = self.label_encoder
            encoded_labels = self.label_encoder.transform(labels.ravel())

        mc = torch.as_tensor(mc)
        counts = torch.as_tensor(counts)
        encoded_labels = torch.as_tensor(encoded_labels, dtype=torch.int64)
        indexes = counts > self.threshold_c
        return (mc[indexes], counts[indexes]), encoded_labels[indexes]

    def __len__(self):
        return len(self.ids)

