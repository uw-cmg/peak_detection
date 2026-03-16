import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
import torch
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KernelDensity
from pymatgen.core import Composition

# Setup for YOLO model (RangingNN) imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# Hack to support 'peak_detection' namespace for YOLO code
if not os.path.exists(os.path.join(current_dir, "peak_detection")):
    try:
        os.makedirs(os.path.join(current_dir, "peak_detection"), exist_ok=True)
        # Touch __init__.py to make it a package
        with open(os.path.join(current_dir, "peak_detection", "__init__.py"), "w") as f:
            pass
        if not os.path.exists(os.path.join(current_dir, "peak_detection", "RangingNN")):
            os.symlink(os.path.join(current_dir, "RangingNN"), os.path.join(current_dir, "peak_detection", "RangingNN"))
    except: pass

try:
    import apav
except ImportError:
    apav = None

def map01(ar):
    return (ar-ar.min()) / (ar.max()-ar.min())

def load_apt(apt_file):
    """
    Load the .apt file (binary) or .csv file and get histogram.
    """
    if apt_file.lower().endswith('.csv'):
        import pandas as pd
        print(f"Loading CSV: {apt_file}")
        df = pd.read_csv(apt_file)
        # Removed 0.5 m/c cutoff
        x = df['x'].values
        spectrum = df['y'].values
        # For CSV, we assume it's already mapped/processed as per user function
        spectrum_log = torch.tensor(spectrum, dtype=torch.float32)
        return x, spectrum, spectrum_log

    if apav is None:
        '''
        filesize = os.path.getsize(apt_file)
        num_values = (filesize - 1024) // 4
        with open(apt_file, 'rb') as f:
            f.seek(1024)
            data = np.fromfile(f, dtype='<f4', count=num_values)
        bins = np.arange(0, 307.2, 0.01)
        spectrum, _ = np.histogram(data, bins=bins)
        x = bins[:-1]
        print('HERE', x, spectrum)
        '''
        print("Error: apav package not detected, cannot open .apt file")
    else:
        d = apav.load_apt(apt_file)
        x, spectrum = d.mass_histogram(bin_width=0.01, lower=0, upper=307.2, multiplicity='all', norm=False)
    
    spectrum_log = torch.tensor(map01(np.log(spectrum+1)), dtype=torch.float32)
    return x, spectrum, spectrum_log

def remove_peaks_and_patch(spectrum, detected_ranges, window=10):
    """
    Replaces detected peak ranges with the average of surrounding noise.
    """
    new_spectrum = spectrum.copy()
    for p in detected_ranges:
        # Convert m/c range to indices (assuming 0.01 binning)
        start_idx = int(np.round(p['start'] * 100))
        end_idx = int(np.round(p['end'] * 100))
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(spectrum) - 1, end_idx)
        
        # Define local noise windows
        left_window = spectrum[max(0, start_idx-window):max(0, start_idx)]
        right_window = spectrum[min(len(spectrum), end_idx+1):min(len(spectrum), end_idx+1+window)]
        
        noise_pool = np.concatenate([left_window, right_window])
        if len(noise_pool) > 0:
            avg_noise = np.mean(noise_pool)
        else:
            avg_noise = 0
            
        new_spectrum[start_idx:end_idx+1] = avg_noise
        
    return new_spectrum

def simplify_label(label):
    """
    Normalizes complex labels to standard chemical notation.
    e.g. "H:2" -> "H2", "HH" -> "H2", "Si:1 O:1" -> "SiO", "COO" -> "CO2"
    """
    if not label or label == "Unknown":
        return label
        
    # Handle RRNG format: "Si:1 O:1", "H:2"
    if ':' in label:
        parts = re.findall(r'([A-Z][a-z]?):(\d+)', label)
        if parts:
            res = ""
            for sym, count in parts:
                c = int(count)
                res += sym + (str(c) if c > 1 else "")
            return res
            
    # Handle Synthetic format or simple strings: "HH", "COO", "H2O"
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', label)
    if parts:
        counts = {}
        order = []
        for sym, count_str in parts:
            count = int(count_str) if count_str else 1
            if sym not in counts:
                counts[sym] = 0
                order.append(sym)
            counts[sym] += count
        res = ""
        # Sort alphabetically for consistency in identification
        for a in sorted(order):
            res += a + (str(counts[a]) if counts[a] > 1 else "")
        return res

    return label

def get_signature_features(target_mc, all_mc, all_counts, threshold=0.1):
    """
    Extract intensities at specific relative and absolute offsets from target_mc.
    Returns 10 features (ratio of intensity at offset to target intensity).
    Offsets: 0.5X, 2.0X (charge states), 
             X±1.0, X±0.5, X±0.33, X±2.0 (isotopes)
    """
    # Find intensity of target peak
    target_idx = np.argmin(np.abs(all_mc - target_mc))
    target_intensity = all_counts[target_idx]
    
    if target_intensity <= 0:
        return [0.0] * 10
        
    offsets_rel = [0.5, 2.0]
    offsets_abs = [-1.0, 1.0, -0.5, 0.5, -0.33, 0.33, -2.0, 2.0]
    
    sigs = []
    
    # Relative charge state shifts
    for r in offsets_rel:
        query_mc = target_mc * r
        idx = np.argmin(np.abs(all_mc - query_mc))
        if np.abs(all_mc[idx] - query_mc) < threshold:
            sigs.append(all_counts[idx] / target_intensity)
        else:
            sigs.append(0.0)
            
    # Absolute isotopic shifts
    for a in offsets_abs:
        query_mc = target_mc + a
        idx = np.argmin(np.abs(all_mc - query_mc))
        if np.abs(all_mc[idx] - query_mc) < threshold:
            sigs.append(all_counts[idx] / target_intensity)
        else:
            sigs.append(0.0)
            
    return sigs

def parse_rrng(filepath):
    """Parses a .RRNG file for benchmarking, including labels."""
    ranges = []
    if not os.path.exists(filepath): return ranges
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    # Pattern to capture start, end, and the label (everything between end mass and Color:)
    range_pattern = re.compile(r'Range(\d+)=([\d.]+) ([\d.]+) (.*) Color:([0-9A-Fa-f]{6})')
    for line in lines:
        match = range_pattern.match(line.strip())
        if match:
            # Clean up label: e.g. "Vol:0.02003 Si:1" -> "Si:1"
            raw_label = match.group(4).strip()
            # Remove "Vol:XXXX" part if present
            label = re.sub(r'Vol:[\d.]+', '', raw_label).strip()
            ranges.append({
                'start': float(match.group(2)), 
                'end': float(match.group(3)),
                'label': simplify_label(label)
            })
    return ranges

def extract_elements_from_rrng(rrng_file):
    """Extracts unique element symbols from an RRNG file."""
    elements = set()
    if not os.path.exists(rrng_file):
        return list(elements)
    
    truth = parse_rrng(rrng_file)
    for t in truth:
        # Label is like 'Si:1 O:1' or 'Au:1'
        # Elements are capital + lowercase/nothing
        found = re.findall(r'([A-Z][a-z]?)', t['label'])
        for f in found:
            elements.add(f)
    return sorted(list(elements))

def load_ion_training_data(path='peak_detection/Ionclassifier/training_data/NewData/Data0001',
                           element_list=list(),
                           elements_to_get_molecules=list(),
                           threshold_c=1e-8,
                           num_files=1000,
                           neighbor_threshold=2.0,
                           use_signature=True):
    """
    load the evaluation files, get input and gt, normalized counts,
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
            counts = ( counts - counts.min() ) / (counts.max() - counts.min()) 
            
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
        
        if len(mc_k) == 0: continue
        
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

# Cache for KDE lookup model
_KDE_LOOKUP_MODEL = None
_MC_LOOKUP_DATA = None
_CURRENT_TRAINING_PATH = None
_CURRENT_INCLUDE_MOLECULES = None

def make_lookup_model(training_path=None, include_molecules=False, make_plot=False, xmin=0, xmax=200):
    """
    Trains KDE models for elements based on training data.
    """
    element_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
                         'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                         'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
                         'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                         'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
                         'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
                         'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U']

    if training_path is None:
        training_path = 'Ionclassifier/training_data/NewData/Data0001'
    
    # Check if we are inside the research projects dir or a subdir
    if not os.path.exists(training_path):
        cwd = os.getcwd()
        if 'AI_example' in cwd:
            # We are likely in a prefix folder, go up
            training_path_adj = os.path.join('..', training_path)
            if os.path.exists(training_path_adj):
                training_path = training_path_adj

    X, ions = load_ion_training_data(path=training_path,
                                           element_list = element_list,
                                           elements_to_get_molecules=element_list if include_molecules else [], 
                                           threshold_c= 1e-8,
                                           num_files=1000)

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
        # Fit KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data.reshape(-1, 1))
        lookup_dict_dens[k] = kde

    return lookup_dict_dens, lookup_dict

def predict_lookup_model(lookup_model, x, num_top_elements=3):
    """
    Get ranking of possible elements for new data point.
    """
    ranking_dict = dict()
    for k in lookup_model.keys():
        log_prob = lookup_model[k].score_samples(x)     # log(f(x0))
        pdf_value = np.exp(log_prob)[0]
        ranking_dict[k] = pdf_value

    prediction_rankings = sorted(ranking_dict.items(), key=lambda x: x[1], reverse=True)
    preds = prediction_rankings[:num_top_elements]
    pred_ions = [i[0] for i in preds]
    confs = [i[1] for i in preds]
    return pred_ions, confs

def build_empirical_mc_distributions(path, num_files=500):
    """Builds an empirical mapping of [label]: {mean_mc, std_mc} from synthetic data"""
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
            if not label: continue
            
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

def suggest_unknown_candidates(mc_center, empirical_stats, base_elements, similar_elements, local_element=None, top_k=5):
    """Provides top-K molecular candidates for an unknown m/c peak based on mass diff, chemical similarity, and proximity."""
    candidates = []
    
    # Clean and parse the local element if present
    local_el_sym = None
    if local_element and local_element != 'Unknown':
        try:
            local_comp = Composition(local_element)
            # just take the heaviest element as the representative local element, or whatever is there
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
            
        penalty = 2.0 # Tier 4: Alien elements
        for el in elements_in_cand:
            if local_el_sym and el == local_el_sym:
                penalty = -1.0 # Tier 1: Has local element
                break
            elif el in base_elements:
                penalty = min(penalty, 0.0) # Tier 2: Has RRNG base element
            elif el in similar_elements:
                penalty = min(penalty, 0.5) # Tier 3: Has similar element
                
        final_score = score + penalty
        candidates.append((label, stat['mean'], final_score, mc_diff))
    
    # Sort primarily by the computed score
    candidates.sort(key=lambda x: x[2])
    return candidates[:top_k]

def make_RF_encoder(unique_ions):
    target_encoder = dict()
    target_decoder = dict()
    for i, val in enumerate(unique_ions):
        target_encoder[val] = i
        target_decoder[i] = val
    return target_encoder, target_decoder

def train_RF_model(target_encoder, X, ions, n_estimators=25, random_state=42):
    targets = np.array([target_encoder.get(i, -1) for i in ions])
    mask = targets != -1
    X_filtered = X[mask]
    y_filtered = targets[mask]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_scaled, y_filtered)
    
    return scaler, model

def create_RF_model(X, ions, n_estimators=25, random_state=42):
    target_encoder, target_decoder = make_RF_encoder(unique_ions=np.unique(ions))
    scaler, model = train_RF_model(target_encoder=target_encoder,
                                   X=X,
                                   ions=ions,
                                   n_estimators=n_estimators,
                                   random_state=random_state)
    return scaler, model, target_decoder

def run_RF_model(detected_ranges, x_exp, spectrum_log, scaler, model, target_decoder, neighbor_threshold=2.0, use_signature=True):
    """Adapted run_RF_model to work with neighborhood features."""
    y_exp = spectrum_log.numpy() if hasattr(spectrum_log, 'numpy') else spectrum_log
    
    # 1. Get exact peak positions and intensities for all detected ranges
    peak_mcs = []
    peak_ints = []
    for p in detected_ranges:
        mask = (x_exp >= p['start']) & (x_exp <= p['end'])
        if np.any(mask):
            peak_idx = np.argmax(y_exp[mask])
            peak_mc = x_exp[mask][peak_idx]
            peak_int = y_exp[mask][peak_idx]
        else:
            peak_mc = (p['start'] + p['end']) / 2
            peak_int = 0.0
        peak_mcs.append(peak_mc)
        peak_ints.append(peak_int)
    
    peak_mcs = np.array(peak_mcs)
    peak_ints = np.array(peak_ints)

    # Normalize intensities for signature extraction (mimic training data normalization)
    if peak_ints.max() > peak_ints.min():
        norm_ints = (peak_ints - peak_ints.min()) / (peak_ints.max() - peak_ints.min())
    else:
        norm_ints = np.zeros_like(peak_ints)
    
    # 2. Build feature vectors for each peak
    expected_dim = model.n_features_in_
    X_raw = []
    for i, target_mc in enumerate(peak_mcs):
        # Neighborhood features (positions)
        neighbors = peak_mcs[(np.abs(peak_mcs - target_mc) < neighbor_threshold) & (peak_mcs != target_mc)]
        neighborhood_part = [target_mc] + sorted(neighbors.tolist())
        
        if use_signature:
            # Signature features (intensities)
            sigs = get_signature_features(target_mc, peak_mcs, norm_ints)
            # Neighborhood part length is expected_dim - 10
            neigh_target_len = expected_dim - 10
            if len(neighborhood_part) < neigh_target_len:
                neighborhood_part = neighborhood_part + [0.0] * (neigh_target_len - len(neighborhood_part))
            else:
                neighborhood_part = neighborhood_part[:neigh_target_len]
            feat = neighborhood_part + sigs
        else:
            # Revert to positional features only
            if len(neighborhood_part) < expected_dim:
                feat = neighborhood_part + [0.0] * (expected_dim - len(neighborhood_part))
            else:
                feat = neighborhood_part[:expected_dim]
        
        X_raw.append(feat)

    if not X_raw:
        return [], [], []

    X = np.array(X_raw)
    X_norm = scaler.transform(X)
    preds = model.predict_proba(X_norm)

    elements = []
    confs = []
    detailed_info = []
    
    for pred in preds:
        top_indices = pred.argsort()[-2:][::-1]
        info = {'el1': '', 'conf1': 0.0, 'el2': '', 'conf2': 0.0}
        results_str = []
        for i, idx in enumerate(top_indices):
            conf = float(pred[idx])
            actual_class = model.classes_[idx]
            raw_label = target_decoder[actual_class]
            clean_el = re.split('[:+]', str(raw_label))[0].strip()
            if i == 0:
                info['el1'] = clean_el
                info['conf1'] = conf
                results_str.append(f"{clean_el} ({conf:.2f})")
            elif conf >= 0.05:
                info['el2'] = clean_el
                info['conf2'] = conf
                results_str.append(f"{clean_el} ({conf:.2f})")
        elements.append(", ".join(results_str))
        confs.append(info['conf1'])
        detailed_info.append(info)
    return elements, confs, detailed_info


def identify_peaks(detected_ranges, x, spectrum_log, allowed_elements=None, flag_unknowns=True):
    """
    Assigns chemical labels to detected ranges by matching them against
    theoretical isotopic 'fingerprints' (mass patterns and relative abundances).
    """
    results = []
    # Convert spectrum_log to numpy for calculations
    y_exp = spectrum_log.numpy() if hasattr(spectrum_log, 'numpy') else spectrum_log

    # We use a sub-window around the peak for pattern matching
    sigma_guess = 0.05

    for p in detected_ranges:
        # Skip if already identified by RF or other method (and not Unknown)
        if p.get('label') and p.get('label') != 'Unknown':
            results.append(p)
            continue

        '''
        mc_center = (p['start'] + p['end']) / 2
        best_score = -1.0
        best_label = "Unknown"

        # Define neighborhood for comparison (e.g., +/- 1 Da, or +/- 5 if not flagging)
        window = 1.0 if flag_unknowns else 5.0
        mask = (x >= mc_center - window) & (x <= mc_center + window)
        x_sub = x[mask]
        y_sub = y_exp[mask]

        if len(x_sub) < 5 or np.max(y_sub) < 0.01:
            p['label'] = "Unknown"
            results.append(p)
            continue

        # Standardize experimental window for dot product
        y_sub_norm = y_sub / (np.linalg.norm(y_sub) + 1e-9)

        # Iterate through possible element/charge candidates
        for element, isos in ISOTOPES.items():
            if allowed_elements is not None and element not in allowed_elements:
                continue
            for charge in [1, 2]:
                # Generate theoretical template for this element/charge
                template = np.zeros_like(x_sub)
                found_near = False
                for mass, abundance in isos:
                    mc_iso = mass / charge
                    # Higher weight if it's actually near the detected peak
                    if abs(mc_iso - mc_center) < 0.5:
                        found_near = True
                    # Add Gaussian at isotopic position
                    template += abundance * np.exp(-((x_sub - mc_iso) ** 2) / (2 * sigma_guess ** 2))

                if flag_unknowns and not found_near:
                    continue

                # Calculate overlap score (dot product)
                template_norm = template / (np.linalg.norm(template) + 1e-9)
                score = np.dot(y_sub_norm, template_norm)

                if score > best_score:
                    best_score = score
                    best_label = f"{element}"

        # Threshold for 'Unknown'
        if flag_unknowns and best_score < 0.4:
            best_label = "Unknown"

        p['label'] = best_label
        p['id_score'] = best_score
        p['detailed_id'] = {'el1': best_label, 'conf1': max(0.0, best_score), 'el2': '', 'conf2': 0.0}
        results.append(p)
        '''

    return results

def calculate_iou(range1, range2):
    """Calculates Intersection over Union for two ranges (start, end)."""
    s1, e1 = range1['start'], range1['end']
    s2, e2 = range2['start'], range2['end']
    
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    intersection = max(0, inter_end - inter_start)
    
    union = (e1 - s1) + (e2 - s2) - intersection
    return intersection / union if union > 0 else 0

def calculate_iou_1d(interval1, interval2):
    """Calculates IoU for two [start, end] intervals."""
    s1, e1 = interval1
    s2, e2 = interval2
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    intersection = max(0, inter_end - inter_start)
    union = (e1 - s1) + (e2 - s2) - intersection
    return intersection / union if union > 0 else 0

def predict_peak_ranges_yolo(apt_file, spectrum_log, x_exp, rrng_file, n_iter=0, prefix=None, flag_unknowns=True, kde_threshold=0.25, use_mc_distance=False, mc_threshold=0.2, training_path=None, include_molecules=False, yolo_weights='best.pt', iou=0.01, conf=0.05, max_det=2000, mc_min=0.0, mc_max=307.2, use_neighborhood=True, neighbor_threshold=2.0, use_signature=True):
    """
    RangingNN YOLO model prediction wrapper.
    """
    import yaml
    from peak_detection.RangingNN.predictor import DetectionPredictor
    from pymatgen.core import Composition # Added import for Composition

    # Local paths
    modelpath = os.path.join(os.path.dirname(__file__), 'peak_detection/RangingNN/modelweights', yolo_weights)
    cfg_path = os.path.join(os.path.dirname(__file__), 'peak_detection/RangingNN/cfg/prediction_args.yaml')
    
    if not os.path.exists(modelpath) or not os.path.exists(cfg_path):
        print(f"  [Error] YOLO model files not found at {modelpath}")
        return [], None

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Standard settings from user snippet
    cfg['iou'] = iou
    cfg['conf'] = conf
    cfg['max_det'] = max_det

    # Initial prediction
    # YOLO model expects length 30720 (divisible by 32). 
    # Data now starts at 0.0 m/c.
    if spectrum_log.shape[0] < 30720:
        pad_size = 30720 - spectrum_log.shape[0]
        sp_padded = torch.zeros(30720)
        sp_padded[:spectrum_log.shape[0]] = spectrum_log # Fill from start
    else:
        sp_padded = spectrum_log[:30720]

    predictor = DetectionPredictor(modelpath, sp_padded[None, None, ...], save_dir='test_results', cfg=cfg)
    result = predictor()[0]
    peak_range_pred = result[:, :2].cpu()
    
    # Recursive iteration as per user snippet
    peak_range_toadd = []
    multiplier = 0.01
    
    if n_iter > 0:
        for it in range(n_iter):
            # Mask current detections
            n = sp_padded.shape[0]
            x1 = np.arange(n) * multiplier
            ranges = np.asarray(peak_range_pred, dtype=float)
            starts = ranges[:, 0] * multiplier
            ends = ranges[:, 1] * multiplier

            in_any_range = np.logical_or.reduce(
                (x1[:, None] > starts[None, :]) &
                (x1[:, None] < ends[None, :]),
                axis=1
            )
            
            idx_delete = x1[in_any_range]
            spectrum_log_mod = sp_padded.clone().numpy()
            
            # Mask current detections AND data outside the specified m/c range
            mask_detections = np.isin(x1, idx_delete)
            mask_outside = (x1 < mc_min) | (x1 > mc_max)
            
            spectrum_log_mod[mask_detections | mask_outside] = 0.2
            spectrum_log_mod = torch.Tensor(spectrum_log_mod)

            predictor_mod = DetectionPredictor(modelpath, spectrum_log_mod[None, None, ...], save_dir='test_results', cfg=cfg)
            result_mod = predictor_mod()[0]
            peak_range_pred_mod = result_mod[:, :2].cpu()

            # Compare new peaks
            tol = 0.5
            for i in peak_range_pred_mod:
                start, end = float(i[0]), float(i[1])
                max_iou = 0.0
                min_dist = 1000
                for j in peak_range_pred.tolist():
                    start2, end2 = float(j[0]), float(j[1])
                    iou = calculate_iou_1d([start, end], [start2, end2])
                    if iou > max_iou: max_iou = iou
                    dist = multiplier * abs(start - start2)
                    if dist < min_dist: min_dist = dist
                
                if max_iou == 0.0 and min_dist > tol:
                    peak_range_toadd.append([start, end])

    # Combine results
    final_ranges = peak_range_pred.tolist() + peak_range_toadd
    formatted_results = []
    for r in final_ranges:
        s, e = r[0] * multiplier, r[1] * multiplier
        formatted_results.append({
            'pos': (s + e) / 2,
            'start': s,
            'end': e
        })
    
    # Filter by 3 width (m/c cutoff removed)
    formatted_results = [r for r in formatted_results] # removed width and m/c filter
    
    # --- RF ELEMENT IDENTIFICATION ---
    # Extract species from RRNG for training
    truth_file = rrng_file
    truth_data = parse_rrng(rrng_file)
    truth_species = sorted(list(set([t['label'] for t in truth_data if 'label' in t and t['label'] != 'Unknown'])))
    elements_for_molecules = extract_elements_from_rrng(rrng_file)
    if prefix:
        prefix_internal = prefix
    else:
        prefix_internal = os.path.basename(apt_file).split('.')[0].lower()
    
    os.makedirs(prefix_internal, exist_ok=True)
    print(f"Training RF model on base elements: {elements_for_molecules}")
    print(f"Explicit classes from RRNG: {truth_species}")

    # Load synthetic training data
    if training_path is None:
        training_data_path = os.path.join(current_dir, 'peak_detection', 'Ionclassifier', 'training_data', 'NewData', 'Data0001')
    else:
        training_data_path = training_path
        
    # Determine effective threshold based on toggle
    eff_neighbor_threshold = neighbor_threshold if use_neighborhood else 0.0

    X_train, ions_train = load_ion_training_data(
        path=training_data_path,
        element_list=truth_species,
        elements_to_get_molecules=elements_for_molecules if include_molecules else [],
        num_files=1000,
        neighbor_threshold=eff_neighbor_threshold,
        use_signature=use_signature
    )

    if len(X_train) > 0:
        try:
            scaler_rf, model_rf, target_decoder_rf = create_RF_model(X_train, ions_train)
            raw_elements, rf_confs, detailed_rf = run_RF_model(
                formatted_results, x_exp, spectrum_log, scaler_rf, model_rf, target_decoder_rf,
                neighbor_threshold=eff_neighbor_threshold,
                use_signature=use_signature
            )
            
            global _KDE_LOOKUP_MODEL, _MC_LOOKUP_DATA, _CURRENT_TRAINING_PATH, _CURRENT_INCLUDE_MOLECULES
            if flag_unknowns:
                # Reset cache if training path or molecular state changed
                if training_path != _CURRENT_TRAINING_PATH or include_molecules != _CURRENT_INCLUDE_MOLECULES:
                    _KDE_LOOKUP_MODEL = None
                    _MC_LOOKUP_DATA = None
                    _CURRENT_TRAINING_PATH = training_path
                    _CURRENT_INCLUDE_MOLECULES = include_molecules

                if _KDE_LOOKUP_MODEL is None or _MC_LOOKUP_DATA is None:
                    print(f"  Training KDE verification model using: {training_path if training_path else 'default'} (Molecules: {include_molecules})...")
                    _KDE_LOOKUP_MODEL, _MC_LOOKUP_DATA = make_lookup_model(training_path=training_path, include_molecules=include_molecules)

            suggestions = []
            for i, (el, conf, det) in enumerate(zip(raw_elements, rf_confs, detailed_rf)):
                # --- Unphysical Peak Filtering ---
                pred1_el = det.get('el1', '')
                is_physical = True # Default to physical, then check for suspected unknowns
                if pred1_el and pred1_el != 'Unknown':
                    try:
                        comp = Composition(pred1_el)
                        element_obj = max(comp.elements, key=lambda e: e.atomic_mass)
                        atomic_mass = float(element_obj.atomic_mass)
                        
                        mc_center = (formatted_results[i]['start'] + formatted_results[i]['end']) / 2.0
                        mc_val = formatted_results[i]['start'] 
                        
                        # 1. KDE Check (User-requested distance check)
                        if flag_unknowns and not use_mc_distance and _KDE_LOOKUP_MODEL is not None:
                            if pred1_el in _KDE_LOOKUP_MODEL:
                                conf_kde = np.exp(_KDE_LOOKUP_MODEL[pred1_el].score_samples(np.array([[mc_val]])))[0]
                                if conf_kde < kde_threshold:
                                    # Too far away, double check with predictions ranking
                                    pred_ions, confs_kde = predict_lookup_model(_KDE_LOOKUP_MODEL, np.array([[mc_val]]))
                                    is_physical = False
                                    suggestions.append({
                                                'mc': mc_val,
                                                'rf_pred': pred1_el,
                                                'kde_suggestions': pred_ions,
                                                'confs': confs_kde
                                            })
                        # 2. MC Distance Check
                        elif flag_unknowns and use_mc_distance and _MC_LOOKUP_DATA is not None:
                            if pred1_el in _MC_LOOKUP_DATA:
                                train_mcs = np.array(_MC_LOOKUP_DATA[pred1_el])
                                min_dist = np.min(np.abs(train_mcs - mc_val))
                                if min_dist > mc_threshold:
                                    is_physical = False
                                    # For suggestions, we'll still use the KDE ranker logic to find best alternatives
                                    pred_ions, confs_kde = predict_lookup_model(_KDE_LOOKUP_MODEL, np.array([[mc_val]]))
                                    suggestions.append({
                                                'mc': mc_val,
                                                'rf_pred': pred1_el,
                                                'kde_suggestions': pred_ions,
                                                'confs': confs_kde
                                            })
                    except Exception:
                        pass # If we can't parse it, tentatively trust the RF model
                
                if pred1_el == 'Unknown' or not pred1_el:
                    is_physical = True # Pass through native unknowns
                    
                if not is_physical and flag_unknowns:
                    formatted_results[i]['label'] = f'Unknown ({pred1_el})'
                    formatted_results[i]['id_score'] = 1.0
                    formatted_results[i]['method'] = 'RF (Filtered)'
                    formatted_results[i]['detailed_id'] = {'el1': 'Unknown', 'conf1': 1.0, 'el2': '', 'conf2': 0.0}
                else:
                    formatted_results[i]['label'] = el
                    formatted_results[i]['id_score'] = float(conf)
                    formatted_results[i]['method'] = 'RF'
                    formatted_results[i]['detailed_id'] = det
            
            # Write element suggestions to file
            if suggestions:
                sug_path = os.path.join(prefix_internal, f"{prefix_internal}_element_suggestions.txt")
                with open(sug_path, 'w') as f:
                    f.write("Mass-to-Charge\tRF_Prediction\tKDE_Suggestions\tKDE_Confidences\n")
                    for s in suggestions:
                        sug_str = ", ".join(s['kde_suggestions'])
                        conf_str = ", ".join([f"{c:.4f}" for c in s['confs']])
                        f.write(f"{s['mc']:.3f}\t{s['rf_pred']}\t{sug_str}\t{conf_str}\n")
                print(f"  Unlikely RF predictions flagged. Suggestions saved to {sug_path}")
        except Exception as e:
            print(f"RF training/running failed: {e}. Falling back to isotopic patterns.")
            formatted_results = identify_peaks(formatted_results, x_exp, spectrum_log, allowed_elements=elements_for_molecules, flag_unknowns=flag_unknowns)
    else:
        print("RF training failed or no data, falling back to isotopic pattern matching.")
        formatted_results = identify_peaks(formatted_results, x_exp, spectrum_log, allowed_elements=elements_for_molecules, flag_unknowns=flag_unknowns)

    # --- DETAILED CSV EXPORT ---
    detailed_rows = []
    for p in formatted_results:
        best_iou = 0
        best_truth = None
        for t in truth_data:
            iou = calculate_iou(p, t)
            if iou > best_iou:
                best_iou = iou
                best_truth = t
        
        row = {
            'predicted peak start': p['start'],
            'predicted peak end': p['end']
        }
        
        if best_iou > 0.1:
            row['true peak start'] = best_truth['start']
            row['true peak end'] = best_truth['end']
            row['true element label'] = best_truth['label']
        else:
            row['true peak start'] = ''
            row['true peak end'] = ''
            row['true element label'] = 'Unknown'
            
        det = p.get('detailed_id', {'el1': 'Unknown', 'conf1': 0.0, 'el2': '', 'conf2': 0.0})
        row['pred element label 1'] = det['el1']
        row['pred confidence 1'] = round(det['conf1'], 3)
        row['pred element label 2'] = det['el2']
        row['pred confidence 2'] = round(det['conf2'], 3)
        
        detailed_rows.append(row)
        
    # Sort detailed rows by predicted peak start
    detailed_rows = sorted(detailed_rows, key=lambda x: x['predicted peak start'])
    import csv
    with open(os.path.join(prefix_internal, f"{prefix_internal}_detailed_results.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'predicted peak start', 'predicted peak end', 'true peak start', 
            'true peak end', 'true element label', 'pred element label 1', 
            'pred confidence 1', 'pred element label 2', 'pred confidence 2'
        ])
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"  Detailed results saved to {prefix_internal}/{prefix_internal}_detailed_results.csv")

    # --- ACCURACY ASSESSMENT ---
    detailed_rows = []
    for p in formatted_results:
        best_iou = 0
        best_truth = None
        for t in truth_data:
            iou = calculate_iou(p, t)
            if iou > best_iou:
                best_iou = iou
                best_truth = t
        
        row = {
            'predicted peak start': p['start'],
            'predicted peak end': p['end']
        }
        
        if best_iou > 0.1:
            row['true peak start'] = best_truth['start']
            row['true peak end'] = best_truth['end']
            row['true element label'] = best_truth['label']
        else:
            row['true peak start'] = ''
            row['true peak end'] = ''
            row['true element label'] = 'Unknown'
            
        det = p.get('detailed_id', {'el1': 'Unknown', 'conf1': 0.0, 'el2': '', 'conf2': 0.0})
        row['pred element label 1'] = det['el1']
        row['pred confidence 1'] = round(det['conf1'], 3)
        row['pred element label 2'] = det['el2']
        row['pred confidence 2'] = round(det['conf2'], 3)
        
        detailed_rows.append(row)
        
    # Sort detailed rows by predicted peak start
    detailed_rows = sorted(detailed_rows, key=lambda x: x['predicted peak start'])
    import csv
    with open(os.path.join(prefix_internal, f"{prefix_internal}_detailed_results.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'predicted peak start', 'predicted peak end', 'true peak start', 
            'true peak end', 'true element label', 'pred element label 1', 
            'pred confidence 1', 'pred element label 2', 'pred confidence 2'
        ])
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"  Detailed results saved to {prefix_internal}/{prefix_internal}_detailed_results.csv")

    # --- ACCURACY ASSESSMENT ---
    correct_matches = 0
    total_matches = 0
    correct_matches_ele = 0
    total_matches_ele = 0
    accuracy_details = []
    
    for row in detailed_rows:
        true_label = row['true element label']
        if true_label and true_label != 'Unknown':
            pred1 = str(row['pred element label 1'])
            pred2 = str(row['pred element label 2'])
            
            if pred1 == 'Unknown':
                accuracy_details.append(f"True: {true_label} -> Pred1: Unknown, Pred2:  | Excluded from Accuracy (Unphysical)")
                continue

            total_matches += 1
            
            try:
                true_comp = Composition(true_label)
                # Elemental Only: Truth must be a single element
                is_pure_element = len(true_comp.elements) == 1 and list(true_comp.values())[0] == 1
                if is_pure_element:
                    total_matches_ele += 1

                true_heavy_sym = max(true_comp.elements, key=lambda e: e.atomic_mass).symbol
                
                p1_comp = Composition(pred1) if (pred1 and pred1 != 'Unknown') else None
                p1_heavy = max(p1_comp.elements, key=lambda e: e.atomic_mass).symbol if p1_comp and len(p1_comp.elements) > 0 else None
                p2_comp = Composition(pred2) if (pred2 and pred2 != 'Unknown') else None
                p2_heavy = max(p2_comp.elements, key=lambda e: e.atomic_mass).symbol if p2_comp and len(p2_comp.elements) > 0 else None
                
                is_correct = (true_heavy_sym == p1_heavy or true_heavy_sym == p2_heavy)
                if is_correct:
                    correct_matches += 1
                    if is_pure_element:
                        correct_matches_ele += 1
                
                accuracy_details.append(f"True: {true_label} (Heavy: {true_heavy_sym}) -> Pred1: {pred1}, Pred2: {pred2} | Correct: {is_correct}")
            except Exception as e:
                accuracy_details.append(f"Error parsing composition for True: {true_label}, Pred1: {pred1}, Pred2: {pred2} - {e}")

    accuracy_pct = (correct_matches / total_matches * 100) if total_matches > 0 else 0.0
    accuracy_pct_ele = (correct_matches_ele / total_matches_ele * 100) if total_matches_ele > 0 else 0.0
    
    with open(os.path.join(prefix_internal, f"{prefix_internal}_rf_accuracy.txt"), 'w') as f:
        f.write(f"RF Model Accuracy Assessment for {prefix_internal}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Matched Peaks: {total_matches}\n")
        f.write(f"Correct Identifications (Top 2): {correct_matches}\n")
        f.write(f"Overall Accuracy: {accuracy_pct:.2f}%\n")
        f.write(f"Elemental-Only Accuracy: {accuracy_pct_ele:.2f}% ({correct_matches_ele}/{total_matches_ele})\n")
        f.write("-" * 50 + "\n\n")
        f.write("Line-by-line Details:\n")
        for detail in accuracy_details:
            f.write(detail + "\n")
    print(f"  Accuracy results saved to {prefix_internal}/{prefix_internal}_rf_accuracy.txt")

    # --- UNKNOWN PEAK SUGGESTER ---
    unknown_suggestions = []
    synthetic_data_dir = os.path.join(current_dir, 'peak_detection', 'Ionclassifier', 'training_data', 'NewData', 'Data0001')
    try:
        print("  Building empirical m/c distributions for Unknown peak suggestions...")
        empirical_stats = build_empirical_mc_distributions(path=synthetic_data_dir, num_files=500)
        
        similar_els = get_similar_elements(elements_for_molecules)
        
        for i, r in enumerate(detailed_rows):
            pred_el = r.get('pred element label 1', '')
            if pred_el == 'Unknown' or not pred_el:
                mc_center = (r['predicted peak start'] + r['predicted peak end']) / 2.0
                
                # Find the closest valid geographic peak
                closest_valid_el = None
                min_geo_dist = float('inf')
                for j, other_r in enumerate(detailed_rows):
                    if i == j: continue
                    other_pred = other_r.get('pred element label 1', '')
                    if other_pred and other_pred != 'Unknown':
                        other_center = (other_r['predicted peak start'] + other_r['predicted peak end']) / 2.0
                        dist = abs(mc_center - other_center)
                        if dist < min_geo_dist:
                            min_geo_dist = dist
                            closest_valid_el = other_pred
                            
                candidates = suggest_unknown_candidates(mc_center, empirical_stats, elements_for_molecules, similar_els, local_element=closest_valid_el, top_k=5)
                cand_str = ", ".join([f"{c[0]} (mean: {c[1]:.3f}, diff: {c[3]:.3f})" for c in candidates])
                
                unknown_suggestions.append({
                    'true_label': r.get('true element label', 'Unknown'),
                    'peak_start': r['predicted peak start'],
                    'peak_end': r['predicted peak end'],
                    'mc_center': mc_center,
                    'local_el': closest_valid_el if closest_valid_el else 'None',
                    'suggestions': cand_str
                })
                
        if unknown_suggestions:
            with open(os.path.join(prefix_internal, f"{prefix_internal}_unknown_predictions.txt"), "w") as f:
                f.write(f"Unknown Peak Suggestions for {prefix_internal} (Base Elements: {', '.join(elements_for_molecules)})\n")
                f.write("-" * 50 + "\n")
                for sug in unknown_suggestions:
                    f.write(f"True Label: {sug['true_label']}\n")
                    f.write(f"Peak Range: {sug['peak_start']:.4f} - {sug['peak_end']:.4f} (Center: {sug['mc_center']:.4f})\n")
                    f.write(f"  Nearest Valid Peak Element: {sug['local_el']}\n")
                    f.write(f"  Top Candidates: {sug['suggestions']}\n\n")
            print(f"  Unknown Peak Suggestions saved to {prefix_internal}/{prefix_internal}_unknown_predictions.txt")
        else:
            print("  No Unknown peaks found; skipping suggestions export.")
    except Exception as e:
        print(f"  Failed to build unknown peak suggestions: {e}")

    # Calculate unknown count
    unknown_count = sum(1 for r in detailed_rows if r.get('pred element label 1', '') == 'Unknown')
    
    return formatted_results, result, accuracy_pct, accuracy_pct_ele, unknown_count

def calculate_metrics(truth, predicted, iou_threshold=0.1):
    """
    Calculates Precision, Recall, and F1 score based on IoU overlap.
    A true peak is 'found' if any predicted peak has IoU > threshold.
    """
    if not truth:
        return 0, 0, 0
    if not predicted:
        return 0, 0, 0
        
    tp = 0
    matched_truth = set()
    matched_pred = set()
    
    for i, t in enumerate(truth):
        for j, p in enumerate(predicted):
            if calculate_iou(t, p) > iou_threshold:
                tp += 1
                matched_truth.add(i)
                matched_pred.add(j)
                break # Count each true peak at most once
                
    precision = len(matched_pred) / len(predicted) if len(predicted) > 0 else 0
    recall = len(matched_truth) / len(truth) if len(truth) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def main_process(apt_file, rrng_file, prefix, xlim=None, use_yolo=True, config_name="default", show_predictions=True, n_iter=0, flag_unknowns=True, kde_threshold=0.25, use_mc_distance=False, mc_threshold=0.2, training_path=None, include_molecules=False, yolo_weights='best.pt', iou=0.01, conf=0.05, max_det=2000, mc_min=0.0, mc_max=307.2, use_neighborhood=True, neighbor_threshold=2.0, use_signature=True):
    rf_accuracy = 0.0
    rf_accuracy_ele = 0.0
    unknown_count = 0

    print(f"\nDetecting peaks for {prefix} (Zoom: {xlim})...")
    x, spectrum, spectrum_log = load_apt(apt_file)
    
    y_mapped = spectrum_log.numpy()
    is_mapped = apt_file.lower().endswith('.csv')
    
    truth = parse_rrng(rrng_file)
    
    # Save true species and RF elements to files
    truth_species = sorted(list(set([t['label'] for t in truth if 'label' in t and t['label'] != 'Unknown'])))
    elements_for_molecules = extract_elements_from_rrng(rrng_file)
    
    os.makedirs(prefix, exist_ok=True)
    with open(os.path.join(prefix, f"{prefix}_rf_elements.txt"), 'w') as f:
        # Write both full species and basic elements for clarity
        f.write("--- Suggested RF Classes (Species) ---\n")
        f.write("\n".join(truth_species))
        f.write("\n\n--- Base Elements for Permutations ---\n")
        f.write("\n".join(sorted(elements_for_molecules)))
    
    with open(os.path.join(prefix, f"{prefix}_true_species.txt"), 'w') as f:
        f.write("\n".join(truth_species))
    
    print(f"  Metadata saved: {prefix}/{prefix}_rf_elements.txt, {prefix}/{prefix}_true_species.txt")

    # --- DETECTION SELECTION ---
    if not show_predictions:
        all_predicted = []
        detected1 = []
        final_detected2 = []
        background1 = np.zeros_like(x)
        y_smooth1 = y_mapped
    elif use_yolo:
        all_predicted, _, rf_accuracy, rf_accuracy_ele, unknown_count = predict_peak_ranges_yolo(apt_file, spectrum_log, x, rrng_file, n_iter=n_iter, prefix=prefix, flag_unknowns=flag_unknowns, kde_threshold=kde_threshold, use_mc_distance=use_mc_distance, mc_threshold=mc_threshold, training_path=training_path, include_molecules=include_molecules, yolo_weights=yolo_weights, iou=iou, conf=conf, max_det=max_det, mc_min=mc_min, mc_max=mc_max, use_neighborhood=use_neighborhood, neighbor_threshold=neighbor_threshold, use_signature=use_signature)
        detected1 = all_predicted
        final_detected2 = []
        background1 = np.zeros_like(x) # Simplified for YOLO
        y_smooth1 = y_mapped # No smooth track for YOLO

    pc, rc, f1c = calculate_metrics(truth, all_predicted)
    print(f"  Total Combined Metrics: Precision={pc:.3f}, Recall={rc:.3f}, F1={f1c:.3f}")

    # Calculate final found peaks (TP) for the summary
    tp_count = 0
    if len(truth) > 0 and len(all_predicted) > 0:
        matched_truth = set()
        for p in all_predicted:
            for i, t in enumerate(truth):
                if calculate_iou(p, t) > 0.1:
                    matched_truth.add(i)
        tp_count = len(matched_truth)

    # Calculate min/max mass ranges
    true_min = min([t['start'] for t in truth]) if truth else 0
    true_max = max([t['end'] for t in truth]) if truth else 0
    pred_min = min([p['start'] for p in all_predicted]) if all_predicted else 0
    pred_max = max([p['end'] for p in all_predicted]) if all_predicted else 0
    plot_xmax = max(true_max, pred_max) + 5

    # --- IDENTIFICATION ---
    identified_peaks = identify_peaks(all_predicted, x, spectrum_log, allowed_elements=elements_for_molecules)
    
    # Consolidate labels for summary
    pred_labels = [p.get('label', 'Unknown') for p in identified_peaks]
    
    stats = {
        'dataset': prefix,
        'config': config_name,
        'true_peaks_count': len(truth),
        'predicted_peaks_count': len(all_predicted),
        'found_peaks_count': tp_count,
        'precision': pc,
        'recall': rc,
        'f1': f1c,
        'true_min_mc': true_min,
        'true_max_mc': true_max,
        'pred_min_mc': pred_min,
        'pred_max_mc': pred_max,
        'rf_accuracy': round(rf_accuracy, 2),
        'rf_accuracy_ele': round(rf_accuracy_ele, 2),
        'unknown_count': unknown_count,
        'identifications': identified_peaks
    }

    if xlim is None:
        # Save results to text file only once
        results_file = os.path.join(prefix, f"{prefix}_peak_ranges.txt")
        with open(results_file, 'w') as f:
            f.write("peak_start, peak_end, round, peak_pos\n")
            for p in detected1:
                f.write(f"{p['start']:.4f}, {p['end']:.4f}, 1, {p['pos']:.4f}\n")
            for p in final_detected2:
                f.write(f"{p['start']:.4f}, {p['end']:.4f}, 2, {p['pos']:.4f}\n")
        print(f"Ranges saved to {results_file}")

    # Benchmark
    truth = parse_rrng(rrng_file)
    if xlim is None:
        print(f"Manual RRNG ranges: {len(truth)}")

    # Visualize comparison using Mapped Data
    plt.figure(figsize=(15, 8))
    plt.plot(x, y_mapped, color='black', alpha=0.3, label='Mapped Spectrum (map01)')
    
    # Plot true ranges (blue)
    for i, r in enumerate(truth):
        if xlim:
            if r['end'] < xlim[0] or r['start'] > xlim[1]:
                continue
        plt.axvspan(r['start'], r['end'], color='blue', alpha=0.15)
        if i == 0: plt.axvspan(r['start'], r['end'], color='blue', alpha=0.15, label='Real (RRNG)')
        
        # Add labels for truth
        if 'label' in r:
            center = (r['start'] + r['end']) / 2
            plt.text(center, 0.85, r['label'], color='blue', fontsize=6, 
                     ha='center', va='bottom', rotation=90, alpha=0.7)
        
    # Plot predicted ranges Round 1 (red)
    if show_predictions:
        label1 = "YOLO Prediction" if use_yolo else "Predicted R1"
        for i, p in enumerate(detected1):
            if xlim:
                if p['end'] < xlim[0] or p['start'] > xlim[1]:
                    continue
            plt.axvspan(p['start'], p['end'], color='red', alpha=0.3, hatch='//')
            if i == 0: plt.axvspan(p['start'], p['end'], color='red', alpha=0.3, hatch='//', label=label1)
            
            # Add labels for predictions
            if 'label' in p:
                center = (p['start'] + p['end']) / 2
                plt.text(center, 0.95, p['label'], color='darkred', fontsize=6, 
                         ha='center', va='bottom', rotation=90, alpha=0.8)

    # Plot predicted ranges Round 2 (purple) - Only if not YOLO and showing predictions
    if show_predictions and not use_yolo:
        for i, p in enumerate(final_detected2):
            if xlim:
                if p['end'] < xlim[0] or p['start'] > xlim[1]:
                    continue
            plt.axvspan(p['start'], p['end'], color='purple', alpha=0.4, hatch='\\\\')
            if i == 0: plt.axvspan(p['start'], p['end'], color='purple', alpha=0.4, hatch='\\\\', label='Predicted R2')
            
            # Labels for Round 2 if available
            if 'label' in p:
                center = (p['start'] + p['end']) / 2
                plt.text(center, 0.98, p['label'], color='indigo', fontsize=5, 
                         ha='center', va='bottom', rotation=90, alpha=0.8)
                
    plt.xlabel('Mass/Charge Ratio (Da)')
    plt.ylabel('Mapped Intensity (0-1)')
    zoom_suffix = f" (Zoom {xlim[0]}-{xlim[1]})" if xlim else ""
    method_name = "YOLO" if use_yolo else "Recursive"
    plt.title(f'{method_name} Comparison: {prefix}{zoom_suffix}')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.2)
    
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(0, plot_xmax)
    
    # Config-specific filename
    safe_config = "".join([c if c.isalnum() else "_" for c in (stats.get('config', 'default') if 'config' in stats else "run")])
    zoom_str = f"_zoom_{xlim[0]}_{xlim[1]}" if xlim else ""
    comp_plot_path = os.path.join(prefix, f"{prefix}_{safe_config}_comparison{zoom_str}.png")
    plt.savefig(comp_plot_path, dpi=300)
    print(f"Saved comparison plot to {comp_plot_path}")
    
    plt.close('all')
    return stats

def match_datasets(csv_dir, rrng_dir):
    """
    Robustly matches CSV files to RRNG files based on common naming patterns.
    """
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') or f.endswith('.apt')]
    rrng_files = [f for f in os.listdir(rrng_dir) if f.endswith('.RRNG')]
    
    def normalize(name):
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def get_key(filename):
        # Extract Rxxxx_xxxxxxx or similar unique ID
        match = re.search(r'R\d+_\d+', filename)
        if match:
            k = match.group(0)
            return k
        # Fallback for PureSi or others
        norm = normalize(filename.split('.')[0])
        if 'puresi' in norm: return 'puresi'
        if 'tinsio' in norm: return 'tinsio'
        if 'uwpid' in norm: 
            # Handle UWPIDPb -> UW PID Pb
            return norm[:7] 
        return norm

    matches = []
    print(f"DEBUG: CSV files: {len(csv_files)}, RRNG files: {len(rrng_files)}")
    for cf in csv_files:
        ckey = get_key(cf)
        best_match = None
        for rf in rrng_files:
            rkey = get_key(rf)
            if ckey == rkey or (ckey in normalize(rf) and len(ckey) > 5) or (rkey in normalize(cf) and len(rkey) > 5):
                best_match = rf
                break
        
        if best_match:
            # prefix for folder naming (sanitize)
            prefix = re.sub(r'[^a-zA-Z0-9]', '_', cf.split('.')[0]).lower()
            # Clean up double underscores
            prefix = re.sub(r'_+', '_', prefix).strip('_')
            matches.append((os.path.join(csv_dir, cf), os.path.join(rrng_dir, best_match), prefix))
        else:
            pass
            
    return matches

def plot_rf_accuracy_summary(all_stats, output_path="rf_accuracy_vs_dataset.png"):
    """Generates a summary plot for RF accuracy across datasets (matching plot_batch_results style)."""
    if not all_stats: return
    # Sort by dataset name alphabetically
    all_stats = sorted(all_stats, key=lambda x: x['dataset'])
    datasets = [s['dataset'] for s in all_stats]
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in datasets]
    overall_acc = [s.get('rf_accuracy', 0) for s in all_stats]
    elemental_acc = [s.get('rf_accuracy_ele', 0) for s in all_stats]
    
    # Calculate unknown fraction
    # unknown_count / predicted_peaks_count
    unk_frac = []
    for s in all_stats:
        pred_count = s.get('predicted_peaks_count', 1)
        if pred_count == 0: pred_count = 1
        unk_frac.append(s.get('unknown_count', 0) / pred_count)
    
    avg_overall = np.mean(overall_acc)
    avg_elemental = np.mean(elemental_acc)
    avg_unk = np.mean(unk_frac)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.plot(display_names, overall_acc, marker='o', color='black', label=f'RF Accuracy Overall (Avg: {avg_overall:.1f}%)', linewidth=1.5)
    ax1.plot(display_names, elemental_acc, marker='o', color='blue', label=f'RF Accuracy Elemental (Avg: {avg_elemental:.1f}%)', linewidth=1.5)
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('RF Accuracy (%)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(-5, 105)
    plt.xticks(rotation=90, ha='center', fontsize=8)
    
    ax2 = ax1.twinx()
    ax2.plot(display_names, unk_frac, marker='o', color='lightgrey', label=f'Unknown Fraction (Avg: {avg_unk:.3f})', linewidth=1.5)
    ax2.set_ylabel('Fraction of Unknowns', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    ax2.set_ylim(-0.05, 1.05)
    
    plt.title('RF Identification Accuracy and Unknown Peak Fraction')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved RF accuracy summary plot to {output_path}")
    plt.close()

def plot_yolo_metrics_summary(all_stats, output_path="yolo_metrics_vs_dataset.png"):
    """Generates a summary plot for YOLO metrics across datasets (matching plot_batch_results style)."""
    if not all_stats: return
    # Sort by dataset name alphabetically
    all_stats = sorted(all_stats, key=lambda x: x['dataset'])
    datasets = [s['dataset'] for s in all_stats]
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in datasets]
    precision = [s.get('precision', 0) for s in all_stats]
    recall = [s.get('recall', 0) for s in all_stats]
    f1 = [s.get('f1', 0) for s in all_stats]
    
    avg_p = np.mean(precision)
    avg_r = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    plt.figure(figsize=(14, 7))
    plt.plot(display_names, precision, marker='o', color='red', label=f'Precision (Avg: {avg_p:.3f})', linewidth=1.5)
    plt.plot(display_names, recall, marker='o', color='green', label=f'Recall (Avg: {avg_r:.3f})', linewidth=1.5)
    plt.plot(display_names, f1, marker='o', color='blue', label=f'F1 Score (Avg: {avg_f1:.3f})', linewidth=1.5)
    
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.ylabel('Score')
    plt.title('YOLO Peak Detection Performance across Datasets')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved YOLO metrics summary plot to {output_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Peak Finder for APT data.")
    parser.add_argument("--apt_path", type=str, default='ALL_APT_processedCSV', help="Path to .apt or processed .csv files")
    parser.add_argument("--rrng_path", type=str, default='ALL_RRNG', help="Path to .rrng files")
    parser.add_argument("--iou", type=float, default=0.01, help="YOLO IoU threshold (default: 0.01)")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold (default: 0.05)")
    parser.add_argument("--max_det", type=int, default=2000, help="YOLO max detections (default: 2000)")
    parser.add_argument("--yolo_weights", type=str, default='best_v0_2025-11-12.pt', help="YOLO weights filename (default: best.pt)")
    parser.add_argument("--n_iter", type=int, default=0, help="Number of YOLO recursive iterations (default: 0)")
    parser.add_argument("--mc_min", type=float, default=0.0, help="Minimum m/c range for YOLO iterations (default: 0.0)")
    parser.add_argument("--mc_max", type=float, default=307.2, help="Maximum m/c range for YOLO iterations (default: 307.2)")
    parser.add_argument("--flag_unknowns", action="store_true", default=True, help="Enable KDE-based unknown peak flagging (default: True)")
    parser.add_argument("--no_flag_unknowns", action="store_false", dest="flag_unknowns", help="Disable KDE-based unknown peak flagging")
    parser.add_argument("--kde_threshold", type=float, default=0.25, help="KDE density threshold for flagging unknowns (default: 0.25)")
    parser.add_argument("--use_mc_distance", action="store_true", help="Use m/c distance to training data for flagging unknowns instead of KDE")
    parser.add_argument("--mc_threshold", type=float, default=0.2, help="m/c distance threshold for flagging unknowns (default: 0.2)")
    parser.add_argument("--training_path", type=str, default='peak_detection/Ionclassifier/training_data/NewData_peakshift0_noise0/Data0001', help="Path to synthetic training data for RF/KDE models")
    parser.add_argument("--include_molecules", action="store_false", default=False, help="Include molecular species in synthetic training data")
    parser.add_argument("--use_neighborhood", action="store_false", default=False, help="Use multi-peak neighborhood features for RF classification (default: True)")
    parser.add_argument("--no_neighborhood", action="store_false", dest="use_neighborhood", help="Disable neighborhood features (use single peak only)")
    parser.add_argument("--neighbor_threshold", type=float, default=2.0, help="m/c window for neighbor searching (default: 2.0)")
    parser.add_argument("--use_signature", action="store_false", default=False, help="Use expert-mimicry signature features (isotopic/charge states) (default: True)")
    parser.add_argument("--no_signature", action="store_false", dest="use_signature", help="Disable signature features")
    
    args = parser.parse_args()

    csv_directory = args.apt_path
    rrng_directory = args.rrng_path
    
    if not os.path.exists(csv_directory) or not os.path.exists(rrng_directory):
        print(f"Error: One or both directories not found:\n  CSV: {csv_directory}\n  RRNG: {rrng_directory}")
    else:
        print(f"Scanning for datasets in {csv_directory}...")
        items_to_process = match_datasets(csv_directory, rrng_directory)
        print(f"Found {len(items_to_process)} matched datasets.")

    # We define configurations for comparison
    # Restricted to YOLO 1D Model only as per user request
    configs = [
        {"name": "YOLO 1D Model", "use_yolo": True, "n_iter": args.n_iter, "iou": args.iou, "conf": args.conf, "max_det": args.max_det, "yolo_weights": args.yolo_weights, "mc_min": args.mc_min, "mc_max": args.mc_max, "flag_unknowns": args.flag_unknowns, "kde_threshold": args.kde_threshold, "use_mc_distance": args.use_mc_distance, "mc_threshold": args.mc_threshold, "training_path": args.training_path, "include_molecules": args.include_molecules, "use_neighborhood": args.use_neighborhood, "neighbor_threshold": args.neighbor_threshold, "use_signature": args.use_signature},
    ]
    
    print("Starting Comprehensive Peak Finder Batch Processing...")
    all_stats = []
    
    for conf in configs:
        print(f"\n>>> CONFIG: {conf['name']}")
        conf_name = conf["name"]
        conf_params = {k:v for k,v in conf.items() if k != "name"}

        for apt_file, rrng_file, base_prefix in items_to_process:
            print(f"\n==================== DATASET: {base_prefix.upper()} ====================")
            try:
                # Run main process (each dataset gets its own subdirectory inside main_process)
                stats = main_process(apt_file, rrng_file, base_prefix, 
                                                config_name=conf_name, 
                                                **conf_params)
                all_stats.append(stats)
            except Exception as e:
                print(f"  [Error] Failed to process {base_prefix}: {e}")
            
    # Save global summary statistics to CSV
    if all_stats:
        import csv
        summary_file = "peak_detection_summary.csv"
        with open(summary_file, 'w', newline='') as f:
            fieldnames = ['dataset', 'config', 'true_peaks_count', 'predicted_peaks_count', 'found_peaks_count', 'precision', 'recall', 'f1', 'true_min_mc', 'true_max_mc', 'pred_min_mc', 'pred_max_mc', 'rf_accuracy', 'rf_accuracy_ele', 'unknown_count']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_stats:
                # Filter results for CSV export
                csv_row = {k: v for k, v in row.items() if k in fieldnames}
                writer.writerow(csv_row)
                
        # Aggregate identifications for YOLO model and save to a separate global CSV
        yolo_export = []
        for s in all_stats:
            if s.get('config') == "YOLO 1D Model":
                for p in s.get('identifications', []):
                    yolo_export.append({
                        'dataset': s['dataset'],
                        'mass_center': p['pos'],
                        'mass_start': p['start'],
                        'mass_end': p['end'],
                        'identified_label': p['label']
                    })
        
        if yolo_export:
            id_file = "yolo_identifications.csv"
            with open(id_file, 'w', newline='') as f:
                fieldnames = ['dataset', 'mass_center', 'mass_start', 'mass_end', 'identified_label']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in yolo_export:
                    writer.writerow(row)
            print(f"Global YOLO Identifications saved to {id_file}")
            
        # Generate summary plots
        plot_rf_accuracy_summary(all_stats)
        plot_yolo_metrics_summary(all_stats)

        print(f"\nBatch Processing Complete. Summary saved to {summary_file}")
    else:
        print("\nNo datasets were successfully processed.")

if __name__ == "__main__":
    main()

