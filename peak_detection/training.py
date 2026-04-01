import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import simplify_label
from .rf_model import get_signature_features


def load_ion_training_data(path='peak_detection/Ionclassifier/training_data/NewData/Data0001',
                           element_list=list(),
                           elements_to_get_molecules=list(),
                           threshold_c=1e-8,
                           num_files=1000,
                           neighbor_threshold=0.0,
                           use_signature=False):
    """
    Load the evaluation files, get input and gt, normalized counts,
    including neighborhood features.
    """
    features_all = list()
    ions_all = list()

    if not os.path.exists(path):
        cwd = os.getcwd()
        if 'AI_example' in cwd:
            potential_path = os.path.join('..', path)
            if os.path.exists(potential_path):
                path = potential_path
            else:
                print(f"Warning: Training data path {path} not found.")
                return np.array([]), np.array([])

    if not os.path.exists(path):
        print(f"Warning: Training data path {path} not found.")
        return np.array([]), np.array([])

    files = [f for f in os.listdir(path) if f.endswith('.csv')][0:num_files]
    max_neigh = 0

    raw_data_per_file = []
    for file in tqdm(files, desc='Loading and parsing classifier training data'):
        df = pd.read_csv(os.path.join(path, file), keep_default_na=False)
        mc = df.get(['mc']).to_numpy().squeeze()
        counts = df.get(['counts']).to_numpy().squeeze()

        if counts.max() == counts.min():
            counts = np.zeros_like(counts)
        else:
            counts = (counts - counts.min()) / (counts.max() - counts.min())

        indexes = counts > threshold_c
        ions_raw = df.get(['ion']).to_numpy().squeeze()
        ions2_raw = df.get(['ion2']).to_numpy().squeeze()

        target_ions = []
        for i1, i2 in zip(ions_raw, ions2_raw):
            if i2 and i2 != "":
                target_ions.append(i2)
            else:
                target_ions.append(i1)
        ions = np.array(target_ions)

        mc_f = mc[indexes]
        ions_f = ions[indexes]

        # Filtering logic
        inds_keep = list()
        if element_list != 'all':
            for i, ion in enumerate(ions_f):
                ion_str = str(ion)
                atoms = re.findall(r'[A-Z][a-z]?', ion_str)
                is_molecule = len(atoms) > 1

                if is_molecule:
                    if elements_to_get_molecules and all(a in elements_to_get_molecules for a in atoms):
                        inds_keep.append(i)
                else:
                    if atoms[0] in element_list:
                        inds_keep.append(i)
        else:
            inds_keep = list(range(len(ions_f)))

        mc_k = mc_f[inds_keep]
        ions_k = [simplify_label(str(ion)) for ion in ions_f[inds_keep]]

        if len(mc_k) == 0:
            continue

        # Build features for the kept peaks
        file_features = []
        for target_mc in mc_k:
            # 1. Neighborhood part (positions)
            neighbors = mc_f[(np.abs(mc_f - target_mc) < neighbor_threshold) & (mc_f != target_mc)]
            neigh_part = [target_mc] + sorted(neighbors.tolist())
            max_neigh = max(max_neigh, len(neigh_part))

            # 2. Signature part (intensities)
            sigs_part = []
            if use_signature:
                sigs_part = get_signature_features(target_mc, mc_f, counts[indexes])

            file_features.append((neigh_part, sigs_part))

        raw_data_per_file.append((file_features, ions_k))

    if not raw_data_per_file:
        return np.array([]), np.array([])

    for features_pairs, labels in raw_data_per_file:
        padded_features = []
        for neigh, sigs in features_pairs:
            # Pad neighborhood part first, then append signatures
            f_padded = neigh + [0.0] * (max_neigh - len(neigh)) + sigs
            padded_features.append(f_padded)
        features_all.append(np.array(padded_features))
        ions_all.extend(labels)

    all_features = np.vstack(features_all)
    all_ions = np.array(ions_all)

    print(f"Loaded training data with feature vector length: {all_features.shape[1]} (Neighborhood: {max_neigh}, Signature: {10 if use_signature else 0})")
    return all_features, all_ions


def get_similar_elements(base_elements):
    """Returns a set of elements that belong to the same groups as the base elements."""
    from pymatgen.core import Element
    similar = set()
    for el_sym in base_elements:
        try:
            el = Element(el_sym)
            group = el.group
            for other_el in Element:
                if other_el.group == group:
                    similar.add(other_el.symbol)
        except Exception:
            pass
    return similar


def build_empirical_mc_distributions(path, num_files=500):
    """Builds an empirical mapping of [label]: {mean_mc, std_mc} from synthetic data."""
    files = [f for f in os.listdir(path) if f.endswith('.csv')][:num_files]
    mc_data = {}

    for file in files:
        df = pd.read_csv(os.path.join(path, file), keep_default_na=False)
        if 'ion' not in df.columns:
            continue

        ions = df['ion'].to_numpy()
        ions2 = df['ion2'].to_numpy() if 'ion2' in df.columns else np.array([''] * len(df))
        mc = df['mc'].to_numpy()

        for m, i1, i2 in zip(mc, ions, ions2):
            label = str(i2).strip() if str(i2).strip() else str(i1).strip()
            if not label:
                continue

            if label not in mc_data:
                mc_data[label] = []
            mc_data[label].append(m)

    stats = {}
    for label, vals in mc_data.items():
        if len(vals) > 5:
            stats[label] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'count': len(vals)
            }
    return stats
