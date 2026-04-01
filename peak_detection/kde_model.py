import os
import numpy as np
from sklearn.neighbors import KernelDensity
from pymatgen.core import Composition

from .training import load_ion_training_data


class KDECache:
    """Wraps the global KDE lookup model state to avoid module-level globals."""

    def __init__(self):
        self.lookup_model = None
        self.mc_lookup_data = None
        self._training_path = None
        self._include_molecules = None

    def get_or_build(self, training_path=None, include_molecules=False):
        """Return cached (lookup_model, mc_lookup_data), rebuilding if params changed."""
        if (training_path != self._training_path
                or include_molecules != self._include_molecules
                or self.lookup_model is None
                or self.mc_lookup_data is None):
            self.lookup_model, self.mc_lookup_data = make_lookup_model(
                training_path=training_path, include_molecules=include_molecules
            )
            self._training_path = training_path
            self._include_molecules = include_molecules
        return self.lookup_model, self.mc_lookup_data

    def reset(self):
        self.lookup_model = None
        self.mc_lookup_data = None
        self._training_path = None
        self._include_molecules = None


# Module-level default cache instance
_default_cache = KDECache()


def make_lookup_model(training_path=None, include_molecules=False, make_plot=False, xmin=0, xmax=200):
    """
    Trains KDE models for elements based on training data.
    Returns (lookup_dict_dens, lookup_dict).
    """
    element_list = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
        'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
        'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
        'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
        'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
        'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
        'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U'
    ]

    if training_path is None:
        training_path = 'Ionclassifier/training_data/NewData/Data0001'

    # Check if we are inside the research projects dir or a subdir
    if not os.path.exists(training_path):
        cwd = os.getcwd()
        if 'AI_example' in cwd:
            training_path_adj = os.path.join('..', training_path)
            if os.path.exists(training_path_adj):
                training_path = training_path_adj

    X, ions = load_ion_training_data(
        path=training_path,
        element_list=element_list,
        elements_to_get_molecules=element_list if include_molecules else [],
        threshold_c=1e-8,
        num_files=1000
    )

    # Use first column (main mc) for KDE distribution
    mc = X[:, 0]

    # Group the mc and counts by ion as a look-up mechanism
    lookup_dict = dict()
    for mc1, ion in zip(mc, ions):
        if ion not in lookup_dict:
            lookup_dict[ion] = list()
        lookup_dict[ion].append(mc1)

    lookup_dict_dens = dict()
    for k in lookup_dict.keys():
        data = np.array(lookup_dict[k])
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data.reshape(-1, 1))
        lookup_dict_dens[k] = kde

    return lookup_dict_dens, lookup_dict


def predict_lookup_model(lookup_model, x, num_top_elements=3):
    """
    Get ranking of possible elements for new data point.
    Returns (pred_ions, confs).
    """
    ranking_dict = dict()
    for k in lookup_model.keys():
        log_prob = lookup_model[k].score_samples(x)
        pdf_value = np.exp(log_prob)[0]
        ranking_dict[k] = pdf_value

    prediction_rankings = sorted(ranking_dict.items(), key=lambda x: x[1], reverse=True)
    preds = prediction_rankings[:num_top_elements]
    pred_ions = [i[0] for i in preds]
    confs = [i[1] for i in preds]
    return pred_ions, confs


def suggest_unknown_candidates(mc_center, empirical_stats, base_elements, similar_elements,
                               local_element=None, top_k=5):
    """Provides top-K molecular candidates for an unknown m/c peak based on mass diff, chemical similarity, and proximity."""
    candidates = []

    # Clean and parse the local element if present
    local_el_sym = None
    if local_element and local_element != 'Unknown':
        try:
            local_comp = Composition(local_element)
            if len(local_comp.elements) > 0:
                local_el_sym = max(local_comp.elements, key=lambda e: e.atomic_mass).symbol
        except Exception:
            local_el_sym = local_element

    for label, stat in empirical_stats.items():
        mc_diff = abs(stat['mean'] - mc_center)
        score = mc_diff

        try:
            comp = Composition(label)
            elements_in_cand = [e.symbol for e in comp.elements]
        except Exception:
            elements_in_cand = [label]

        penalty = 2.0  # Tier 4: Alien elements
        for el in elements_in_cand:
            if local_el_sym and el == local_el_sym:
                penalty = -1.0  # Tier 1: Has local element
                break
            elif el in base_elements:
                penalty = min(penalty, 0.0)  # Tier 2: Has RRNG base element
            elif el in similar_elements:
                penalty = min(penalty, 0.5)  # Tier 3: Has similar element

        final_score = score + penalty
        candidates.append((label, stat['mean'], final_score, mc_diff))

    candidates.sort(key=lambda x: x[2])
    return candidates[:top_k]
