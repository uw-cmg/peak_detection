import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .models import DetailedId


def get_signature_features(target_mc, all_mc, all_counts, threshold=0.1):
    """
    Extract intensities at specific relative and absolute offsets from target_mc.
    Returns 10 features (ratio of intensity at offset to target intensity).
    Offsets: 0.5X, 2.0X (charge states),
             X+-1.0, X+-0.5, X+-0.33, X+-2.0 (isotopes)
    """
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


def make_RF_encoder(unique_ions):
    """Create encoder/decoder dicts for ion labels."""
    target_encoder = dict()
    target_decoder = dict()
    for i, val in enumerate(unique_ions):
        target_encoder[val] = i
        target_decoder[i] = val
    return target_encoder, target_decoder


def train_RF_model_from_arrays(target_encoder, X, ions, n_estimators=25, random_state=42):
    """Train RF model from raw feature arrays and ion labels."""
    targets = np.array([target_encoder.get(i, -1) for i in ions])
    mask = targets != -1
    X_filtered = X[mask]
    y_filtered = targets[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_scaled, y_filtered)

    return scaler, model


def train_RF_model_from_dataframe(df, n_estimators=25, random_state=42):
    """Train RF model from a dataframe with 'target' column and feature columns."""
    features = [col for col in df.columns if col not in ['ions', 'target']]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['target']
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return scaler, model


def create_RF_model(X, ions, n_estimators=25, random_state=42):
    """Convenience wrapper: make encoder + train RF model."""
    target_encoder, target_decoder = make_RF_encoder(unique_ions=np.unique(ions))
    scaler, model = train_RF_model_from_arrays(
        target_encoder=target_encoder,
        X=X,
        ions=ions,
        n_estimators=n_estimators,
        random_state=random_state
    )
    return scaler, model, target_decoder


def run_RF_model(detected_ranges, x_exp, spectrum_log, scaler, model, target_decoder,
                 neighbor_threshold=2.0, use_signature=True):
    """Run trained RF model on detected ranges to classify peaks."""
    y_exp = spectrum_log.numpy() if hasattr(spectrum_log, 'numpy') else spectrum_log

    # 1. Get exact peak positions and intensities for all detected ranges
    peak_mcs = []
    peak_ints = []
    for p in detected_ranges:
        p_start = p.start if hasattr(p, 'start') else p['start']
        p_end = p.end if hasattr(p, 'end') else p['end']
        mask = (x_exp >= p_start) & (x_exp <= p_end)
        if np.any(mask):
            peak_idx = np.argmax(y_exp[mask])
            peak_mc = x_exp[mask][peak_idx]
            peak_int = y_exp[mask][peak_idx]
        else:
            peak_mc = (p_start + p_end) / 2
            peak_int = 0.0
        peak_mcs.append(peak_mc)
        peak_ints.append(peak_int)

    peak_mcs = np.array(peak_mcs)
    peak_ints = np.array(peak_ints)

    # Normalize intensities for signature extraction
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
            sigs = get_signature_features(target_mc, peak_mcs, norm_ints)
            neigh_target_len = expected_dim - 10
            if len(neighborhood_part) < neigh_target_len:
                neighborhood_part = neighborhood_part + [0.0] * (neigh_target_len - len(neighborhood_part))
            else:
                neighborhood_part = neighborhood_part[:neigh_target_len]
            feat = neighborhood_part + sigs
        else:
            if len(neighborhood_part) < expected_dim:
                feat = neighborhood_part + [0.0] * (expected_dim - len(neighborhood_part))
            else:
                feat = neighborhood_part[:expected_dim]

        X_raw.append(feat)

    if not X_raw:
        return [], [], [], np.array([])

    X = np.array(X_raw)
    X_norm = scaler.transform(X)
    preds = model.predict_proba(X_norm)

    elements = []
    confs = []
    detailed_info = []

    for pred in preds:
        top_indices = pred.argsort()[-2:][::-1]
        info = DetailedId()
        results_str = []
        for i, idx in enumerate(top_indices):
            conf = float(pred[idx])
            actual_class = model.classes_[idx]
            raw_label = target_decoder[actual_class]
            clean_el = re.split('[:+]', str(raw_label))[0].strip()
            if i == 0:
                info.el1 = clean_el
                info.conf1 = conf
                results_str.append(f"{clean_el} ({conf:.2f})")
            elif conf >= 0.05:
                info.el2 = clean_el
                info.conf2 = conf
                results_str.append(f"{clean_el} ({conf:.2f})")
        elements.append(", ".join(results_str))
        confs.append(info.conf1)
        detailed_info.append(info)
    return elements, confs, detailed_info, peak_mcs
