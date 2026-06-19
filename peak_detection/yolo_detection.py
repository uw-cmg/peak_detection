from __future__ import annotations

import os
import csv
import re
import bisect
import numpy as np
import torch

from pymatgen.core import Composition

from .models import DetailedId, PeakRange
from .utils import calculate_iou, calculate_iou_1d, is_molecule, simplify_label
from .data_io import parse_rrng, extract_elements_from_rrng
from .training import load_ion_training_data, load_ion_training_data_mc_vector, build_empirical_mc_samples
from .rf_model import create_RF_model, run_RF_model


def remove_peaks_and_patch(spectrum: np.ndarray, detected_ranges: list[PeakRange], window: int = 10) -> np.ndarray:
    """
    Replaces detected peak ranges with the average of surrounding noise.
    """
    new_spectrum = spectrum.copy()
    for p in detected_ranges:
        start_idx = int(np.round(p.start * 100))
        end_idx = int(np.round(p.end * 100))

        start_idx = max(0, start_idx)
        end_idx = min(len(spectrum) - 1, end_idx)

        left_window = spectrum[max(0, start_idx - window):max(0, start_idx)]
        right_window = spectrum[min(len(spectrum), end_idx + 1):min(len(spectrum), end_idx + 1 + window)]

        noise_pool = np.concatenate([left_window, right_window])
        if len(noise_pool) > 0:
            avg_noise = np.mean(noise_pool)
        else:
            avg_noise = 0

        new_spectrum[start_idx:end_idx + 1] = avg_noise

    return new_spectrum


def _range_intensity_stat(spectrum: np.ndarray, start_idx: float, end_idx: float, *, quantile: float = 0.9) -> float:
    """Robust peak-intensity statistic for a YOLO range in index units."""
    n = len(spectrum)
    lo = max(0, int(np.floor(float(start_idx))))
    hi = min(n, int(np.ceil(float(end_idx))) + 1)
    if hi <= lo:
        return 0.0
    vals = np.asarray(spectrum[lo:hi], dtype=float)
    if vals.size == 0:
        return 0.0
    q = min(1.0, max(0.0, float(quantile)))
    return float(np.quantile(vals, q))


def identify_peaks(detected_ranges: list[PeakRange], x: np.ndarray, spectrum_log, allowed_elements: list[str] | None = None, flag_unknowns: bool = True) -> list[PeakRange]:
    """
    Assigns chemical labels to detected ranges. (Simplified wrapper)
    """
    return detected_ranges


def predict_peak_ranges_yolo(apt_file, spectrum_log, x_exp, rrng_file,
                             n_iter=0, prefix=None, flag_unknowns=False,
                             mc_threshold=0.2, training_path=None,
                             training_num_files: int = 10000,
                             augment_molecule_training_charge_ratios: bool = False,
                             molecule_rf_rescue_elements: bool = False,
                             molecule_rf_rescue_threshold: float = 0.8,
                             molecule_rf_rescue_margin: float = 0.15,
                             molecule_rf_rescue_score_margin: float = 0.05,
                             molecule_rf_rescue_dist_margin: float = 0.05,
                             include_molecules=False, yolo_weights='best.pt',
                             iou=0.01, conf=0.05, max_det=2000,
                             iter_min_intensity_quantile: float = 0.10,
                             iter_min_intensity_fraction: float = 0.50,
                             iter_intensity_stat_quantile: float = 0.90,
                             mc_min=0.0, mc_max=307.2,
                             use_neighborhood=False, neighbor_threshold=2.0,
                             use_signature=False,
                             separate_molecule_rf=False,
                             unknown_molecule_rf: bool = False,
                             molecule_rf_threshold=0.8,
                             unknown_confidence_threshold: float = 0.6,
                             rf_accuracy_top_n: int = 1,
                             context_rescore: bool = False,
                             context_window_da: float = 2.0,
                             context_strength: float = 0.35,
                             context_min_confidence: float = 0.75,
                             context_min_candidate_confidence: float = 0.05,
                             context_override_margin: float = 0.05,
                             context_distance_sigma: float = 0.75,
                             context_rescue_unknown_same_label: bool = True,
                             context_rescue_unknown_min_score: float = 0.7,
                             followon_mc_vector_rf: bool = False,
                             followon_mc_vector_round_decimals: int = 3,
                             return_accuracy_breakdown: bool = False):
    """
    RangingNN YOLO model prediction wrapper.
    Unknown-flagging (when enabled) uses mc-distance checks against empirical training samples
    and optionally low top-candidate RF confidence.
    Optionally, a second RF model can be trained on molecular species only and applied to
    peaks flagged as unknown, or to elemental IDs as molecule rescue/mixed top-2 candidates.
    """
    import yaml
    from peak_detection.RangingNN.predictor import DetectionPredictor

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modelpath = os.path.join(base_dir, 'peak_detection', 'RangingNN', 'modelweights', yolo_weights)
    cfg_path = os.path.join(base_dir, 'peak_detection', 'RangingNN', 'cfg', 'prediction_args.yaml')

    if not os.path.exists(modelpath) or not os.path.exists(cfg_path):
        print(f"  [Error] YOLO model files not found at {modelpath}")
        empty_breakdown = {
            'species_including_unknowns': 0.0,
            'species_excluding_unknowns': 0.0,
            'elemental_including_unknowns': 0.0,
            'elemental_excluding_unknowns': 0.0,
            'molecular_including_unknowns': 0.0,
            'molecular_excluding_unknowns': 0.0,
            'counts': {
                'species_correct_including_unknowns': 0,
                'species_total_including_unknowns': 0,
                'species_correct_excluding_unknowns': 0,
                'species_total_excluding_unknowns': 0,
                'elemental_correct_including_unknowns': 0,
                'elemental_total_including_unknowns': 0,
                'elemental_correct_excluding_unknowns': 0,
                'elemental_total_excluding_unknowns': 0,
                'molecular_correct_including_unknowns': 0,
                'molecular_total_including_unknowns': 0,
                'molecular_correct_excluding_unknowns': 0,
                'molecular_total_excluding_unknowns': 0,
                'unknown_with_truth': 0,
            },
        }
        if return_accuracy_breakdown:
            return [], None, 0.0, 0.0, 0, empty_breakdown
        return [], None, 0.0, 0.0, 0

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['iou'], cfg['conf'], cfg['max_det'] = iou, conf, max_det

    if spectrum_log.shape[0] < 30720:
        sp_padded = torch.zeros(30720)
        sp_padded[:spectrum_log.shape[0]] = spectrum_log
    else:
        sp_padded = spectrum_log[:30720]

    predictor = DetectionPredictor(modelpath, sp_padded[None, None, ...], save_dir='test_results', cfg=cfg)
    result = predictor()[0]
    peak_range_pred = result[:, :2].cpu()

    # Recursive iteration. Each pass suppresses every peak found so far, then
    # adds only non-overlapping new ranges to the aggregate list.
    current_peak_ranges = [list(map(float, r)) for r in peak_range_pred.tolist()]
    multiplier = 0.01

    iter_min_intensity_threshold = None
    if n_iter > 0:
        first_pass_intensities = [
            _range_intensity_stat(
                sp_padded.numpy(),
                float(r[0]),
                float(r[1]),
                quantile=iter_intensity_stat_quantile,
            )
            for r in current_peak_ranges
        ]
        first_pass_intensities = [v for v in first_pass_intensities if np.isfinite(v)]
        if first_pass_intensities:
            base_intensity = float(np.quantile(
                np.asarray(first_pass_intensities, dtype=float),
                min(1.0, max(0.0, float(iter_min_intensity_quantile))),
            ))
            iter_min_intensity_threshold = base_intensity * max(0.0, float(iter_min_intensity_fraction))
            print(
                "  YOLO iterative intensity threshold: "
                f"{iter_min_intensity_threshold:.4g} "
                f"(fraction {float(iter_min_intensity_fraction):.3g} of "
                f"first-pass q{float(iter_min_intensity_quantile):.3g}={base_intensity:.4g}; "
                f"range stat q{float(iter_intensity_stat_quantile):.3g})"
            )

        for it in range(n_iter):
            n = sp_padded.shape[0]
            x1 = np.arange(n) * multiplier
            if current_peak_ranges:
                ranges = np.asarray(current_peak_ranges, dtype=float)
                starts, ends = ranges[:, 0] * multiplier, ranges[:, 1] * multiplier
                in_any_range = np.logical_or.reduce(
                    (x1[:, None] > starts[None, :]) & (x1[:, None] < ends[None, :]),
                    axis=1,
                )
            else:
                in_any_range = np.zeros_like(x1, dtype=bool)
            spectrum_log_mod = sp_padded.clone().numpy()
            spectrum_log_mod[np.isin(x1, x1[in_any_range]) | (x1 < mc_min) | (x1 > mc_max)] = 0.2
            spectrum_log_mod = torch.Tensor(spectrum_log_mod)
            predictor_mod = DetectionPredictor(modelpath, spectrum_log_mod[None, None, ...], save_dir='test_results', cfg=cfg)
            result_mod = predictor_mod()[0]
            peak_range_pred_mod = result_mod[:, :2].cpu()
            tol = 0.5
            added_this_iter = 0
            for i in peak_range_pred_mod:
                start, end = float(i[0]), float(i[1])
                if iter_min_intensity_threshold is not None:
                    candidate_intensity = _range_intensity_stat(
                        sp_padded.numpy(),
                        start,
                        end,
                        quantile=iter_intensity_stat_quantile,
                    )
                    if candidate_intensity < iter_min_intensity_threshold:
                        continue
                max_iou_val, min_dist = 0.0, 1000
                for j in current_peak_ranges:
                    start2, end2 = float(j[0]), float(j[1])
                    iou_val = calculate_iou_1d([start, end], [start2, end2])
                    if iou_val > max_iou_val: max_iou_val = iou_val
                    dist = multiplier * abs(start - start2)
                    if dist < min_dist: min_dist = dist
                if max_iou_val == 0.0 and min_dist > tol:
                    current_peak_ranges.append([start, end])
                    added_this_iter += 1
            print(f"  YOLO iterative pass {it + 1}/{n_iter}: added {added_this_iter} new ranges")

    final_ranges = current_peak_ranges
    formatted_results = []
    for r in final_ranges:
        s, e = r[0] * multiplier, r[1] * multiplier
        formatted_results.append(PeakRange(start=s, end=e, pos=(s + e) / 2))

    # --- RF ELEMENT IDENTIFICATION ---
    truth_data = parse_rrng(rrng_file)
    label_map = {simplify_label(str(t.label)): t.label for t in truth_data if t.label and t.label != 'Unknown'}
    truth_species_all = sorted(list(label_map.keys()))
    truth_species_primary = truth_species_all
    if not include_molecules:
        truth_species_primary = [s for s in truth_species_all if not is_molecule(s)]

    truth_molecules = [s for s in truth_species_all if is_molecule(s)]
    
    elements_for_molecules = extract_elements_from_rrng(rrng_file)
    prefix_internal = prefix if prefix else os.path.basename(apt_file).split('.')[0].lower()
    os.makedirs(prefix_internal, exist_ok=True)

    def _is_elemental_label(label: str) -> bool:
        try:
            comp = Composition(str(label))
            if len(comp.elements) != 1:
                return False
            return list(comp.values())[0] == 1
        except Exception:
            return bool(re.fullmatch(r'[A-Z][a-z]?$', str(label)))
    
    def _format_class_list(classes: list[str], max_items: int = 200) -> str:
        if not classes:
            return ""
        if len(classes) <= max_items:
            return ", ".join(classes)
        head = ", ".join(classes[:max_items])
        return f"{head}, ... (+{len(classes) - max_items} more)"

    print(f"Training RF model on base elements: {elements_for_molecules}")
    if separate_molecule_rf:
        print(f"Element RF target classes ({len([s for s in truth_species_primary if not is_molecule(s)])}): {_format_class_list([s for s in truth_species_primary if not is_molecule(s)])}")
        print(f"Molecule RF target classes ({len(truth_molecules)}): {_format_class_list(truth_molecules)}")
    else:
        print(f"Target classes for model ({len(truth_species_primary)}): {_format_class_list(truth_species_primary)}")

    training_data_path = training_path if training_path else os.path.join(base_dir, 'peak_detection', 'Ionclassifier', 'training_data', 'NewData', 'Data0001')
    eff_neighbor_threshold = neighbor_threshold if use_neighborhood else 0.0

    try:
        before_rescue_breakdown = None
        after_rescue_breakdown = None
        rescue_stats = {'considered': 0, 'overrides': 0}

        raw_elements_ele, rf_confs_ele, detailed_rf_ele = [], [], []
        raw_elements_mol, rf_confs_mol, detailed_rf_mol = [], [], []
        peak_mcs = np.array([])
        
        if not separate_molecule_rf:
            X_train, ions_train = load_ion_training_data(
                path=training_data_path, element_list=truth_species_primary,
                elements_to_get_molecules=elements_for_molecules if include_molecules else [],
                num_files=int(training_num_files),
                neighbor_threshold=eff_neighbor_threshold,
                use_signature=use_signature,
                augment_molecule_charge_ratios=bool(augment_molecule_training_charge_ratios),
            )
            if len(X_train) > 0:
                scaler_rf, model_rf, target_decoder_rf = create_RF_model(X_train, ions_train)
                raw_elements_initial, rf_confs_initial, detailed_rf_initial, peak_mcs = run_RF_model(
                    formatted_results, x_exp, spectrum_log, scaler_rf, model_rf, target_decoder_rf,
                    neighbor_threshold=eff_neighbor_threshold, use_signature=use_signature
                )
        else:
            truth_elements = [s for s in truth_species_primary if not is_molecule(s)]
            if include_molecules or truth_molecules:
                X_train_mol, ions_train_mol = load_ion_training_data(
                    path=training_data_path, element_list=truth_molecules,
                    elements_to_get_molecules=elements_for_molecules if include_molecules else [],
                    num_files=int(training_num_files),
                    neighbor_threshold=eff_neighbor_threshold,
                    use_signature=use_signature,
                    augment_molecule_charge_ratios=bool(augment_molecule_training_charge_ratios),
                )
                if len(X_train_mol) > 0:
                    scaler_rf_mol, model_rf_mol, target_decoder_rf_mol = create_RF_model(X_train_mol, ions_train_mol)
                    raw_elements_mol, rf_confs_mol, detailed_rf_mol, _ = run_RF_model(
                        formatted_results, x_exp, spectrum_log, scaler_rf_mol, model_rf_mol, target_decoder_rf_mol,
                        neighbor_threshold=eff_neighbor_threshold, use_signature=use_signature
                    )
            if truth_elements:
                X_train_ele, ions_train_ele = load_ion_training_data(
                    path=training_data_path, element_list=truth_elements,
                    elements_to_get_molecules=[], num_files=int(training_num_files),
                    neighbor_threshold=eff_neighbor_threshold,
                    use_signature=use_signature,
                    augment_molecule_charge_ratios=bool(augment_molecule_training_charge_ratios),
                )
                if len(X_train_ele) > 0:
                    scaler_rf_ele, model_rf_ele, target_decoder_rf_ele = create_RF_model(X_train_ele, ions_train_ele)
                    raw_elements_ele, rf_confs_ele, detailed_rf_ele, peak_mcs = run_RF_model(
                        formatted_results, x_exp, spectrum_log, scaler_rf_ele, model_rf_ele, target_decoder_rf_ele,
                        neighbor_threshold=eff_neighbor_threshold, use_signature=use_signature
                    )

        def _min_abs_distance_to_samples(sorted_samples: np.ndarray | None, value: float) -> float:
            if sorted_samples is None or len(sorted_samples) == 0:
                return float('inf')
            idx = bisect.bisect_left(sorted_samples, value)
            best = float('inf')
            if idx < len(sorted_samples):
                best = min(best, abs(float(sorted_samples[idx]) - value))
            if idx > 0:
                best = min(best, abs(float(sorted_samples[idx - 1]) - value))
            return best

        def _min_abs_distance_to_species_samples(species_key: str, mc_val: float) -> float:
            samples = mc_samples_by_species.get(species_key)
            if samples is None or len(samples) == 0:
                return float('inf')

            # For molecules, allow charge-aware matching by checking common charge-state scalings
            # of mc_val against the training sample distribution (e.g., z=2 -> 0.5x, z=3 -> 0.333x).
            # against the training sample distribution. Elements already include many charge states.
            if not is_molecule(species_key):
                return _min_abs_distance_to_samples(samples, mc_val)

            min_s = float(samples[0])
            max_s = float(samples[-1])
            candidates = [mc_val]
            # Integer multiples (mc * z)
            for mult in (2, 3, 4):
                scaled = mc_val * mult
                if (min_s - mc_threshold) <= scaled <= (max_s + mc_threshold):
                    candidates.append(scaled)
            # Fractional multiples (mc / z)
            for div in (2, 3, 4):
                scaled = mc_val / float(div)
                if (min_s - mc_threshold) <= scaled <= (max_s + mc_threshold):
                    candidates.append(scaled)
            return min(_min_abs_distance_to_samples(samples, c) for c in candidates)

        def _nearest_sample_value(sorted_samples: np.ndarray | None, value: float) -> float | None:
            if sorted_samples is None or len(sorted_samples) == 0:
                return None
            idx = bisect.bisect_left(sorted_samples, value)
            best_val = None
            best_dist = float('inf')
            if idx < len(sorted_samples):
                v = float(sorted_samples[idx])
                d = abs(v - value)
                if d < best_dist:
                    best_dist, best_val = d, v
            if idx > 0:
                v = float(sorted_samples[idx - 1])
                d = abs(v - value)
                if d < best_dist:
                    best_dist, best_val = d, v
            return best_val

        def _best_match_to_species_samples(
            species_key: str,
            mc_val: float,
            *,
            allow_scaling_for_elements: bool,
        ) -> tuple[float, float, float, float | None]:
            """
            Returns (best_dist, best_scale, scaled_mc, nearest_training_mc).

            - For molecules: always considers scaling to allow charge-aware matching:
              {1, 2, 3, 4, 0.5, 1/3, 0.25}.
            - For elements: only considers scaling when allow_scaling_for_elements=True (same set).
            """
            samples = mc_samples_by_species.get(species_key)
            if samples is None or len(samples) == 0:
                return float('inf'), 1.0, mc_val, None

            min_s = float(samples[0])
            max_s = float(samples[-1])

            consider_scaling = is_molecule(species_key) or allow_scaling_for_elements
            multipliers = (1.0, 2.0, 3.0, 4.0, 0.5, 1.0 / 3.0, 0.25) if consider_scaling else (1.0,)

            best_dist = float('inf')
            best_mult = 1.0
            best_scaled = mc_val
            best_nearest = None

            for mult in multipliers:
                scaled = mc_val * mult
                if mult != 1 and not ((min_s - mc_threshold) <= scaled <= (max_s + mc_threshold)):
                    continue
                dist = _min_abs_distance_to_samples(samples, scaled)
                if dist < best_dist:
                    best_dist = dist
                    best_mult = mult
                    best_scaled = scaled
                    best_nearest = _nearest_sample_value(samples, scaled)

            return best_dist, best_mult, best_scaled, best_nearest

        def _format_confidence_unknown_label(det: DetailedId | None, fallback_label: str) -> str:
            parts = []
            if det is not None and det.el1:
                parts.append(f"{det.el1} {float(det.conf1) * 100:.0f}%")
            elif fallback_label:
                parts.append(str(fallback_label))
            if det is not None and det.el2 and float(det.conf2) > 0:
                parts.append(f"{det.el2} {float(det.conf2) * 100:.0f}%")
            return f"Unknown ({', '.join(parts)})" if parts else "Unknown"

        # 2. Build empirical mc sample lookup (FOR MC-DISTANCE UNKNOWN FLAGGING)
        print("  Building mc-distance lookup table...")
        mc_samples_by_species = build_empirical_mc_samples(path=training_data_path, num_files=int(training_num_files))

        # 3. Refined Winner Selection (KDE REMOVED)
        for i in range(len(formatted_results)):
            mc_val = peak_mcs[i]
            alt_candidates = []
            if separate_molecule_rf:
                if rf_confs_ele:
                    alt_candidates.append(
                        {'el': raw_elements_ele[i], 'conf': rf_confs_ele[i], 'det': detailed_rf_ele[i], 'model': 'ele'}
                    )
                if rf_confs_mol:
                    alt_candidates.append(
                        {'el': raw_elements_mol[i], 'conf': rf_confs_mol[i], 'det': detailed_rf_mol[i], 'model': 'mol'}
                    )
            else:
                alt_candidates.append({'el': raw_elements_initial[i], 'conf': rf_confs_initial[i], 'det': detailed_rf_initial[i], 'model': 'joint'})

            best_score, best_candidate = -1.0, (alt_candidates[0] if alt_candidates else {'el': 'Unknown', 'conf': 0.0, 'det': DetailedId(el1='Unknown'), 'model': 'none'})

            for cand in alt_candidates:
                full_label = cand['el']
                main_label = re.split(r'\(|\s', full_label)[0].strip()
                dist_weight = 1.0
                species_key = simplify_label(main_label)
                dist_val = _min_abs_distance_to_species_samples(species_key, float(mc_val))
                if dist_val > mc_threshold:
                    if flag_unknowns or dist_val > (mc_threshold * 10):
                        dist_weight = 0.05 if dist_val > (mc_threshold * 1.5) else 0.5
                if cand['model'] == 'mol' and dist_weight >= 0.5: dist_weight *= 1.2
                cand_score = cand['conf'] * dist_weight
                if cand_score > best_score: best_score, best_candidate = cand_score, cand

            # STRICT MC-DISTANCE FLAGGING
            p = formatted_results[i]
            winner_full = best_candidate['el']
            winner_main = re.split(r'\(|\s', winner_full)[0].strip()
            is_unphysical = False
            if flag_unknowns:
                winner_key = simplify_label(winner_main)
                winner_dist = _min_abs_distance_to_species_samples(winner_key, float(mc_val))
                if winner_dist > mc_threshold:
                    is_unphysical = True
            confidence_unknown = False
            best_det = best_candidate.get('det')
            if (
                flag_unknowns
                and unknown_confidence_threshold is not None
                and float(unknown_confidence_threshold) > 0
                and best_det is not None
                and getattr(best_det, 'el1', '')
                and str(best_det.el1) != 'Unknown'
                and float(getattr(best_det, 'conf1', 0.0) or 0.0) < float(unknown_confidence_threshold)
            ):
                confidence_unknown = True
            
            if (is_unphysical or winner_main == 'Unknown') and flag_unknowns:
                p.label = f'Unknown ({winner_main})'
                p.id_score, p.is_unknown = 1.0, True
                p.method = 'RF-Unknown'
                p.detailed_id = DetailedId(el1=p.label, conf1=best_candidate['conf'], el2='Unknown', conf2=0.0)
            elif confidence_unknown:
                p.label = _format_confidence_unknown_label(best_det, winner_main)
                p.id_score, p.is_unknown = float(best_candidate['conf']), True
                p.method = 'RF-Unknown-LowConf'
                p.detailed_id = DetailedId(
                    el1=p.label,
                    conf1=float(getattr(best_det, 'conf1', best_candidate['conf']) or 0.0),
                    el2=str(getattr(best_det, 'el2', '') or ''),
                    conf2=float(getattr(best_det, 'conf2', 0.0) or 0.0),
                )
            else:
                p.label = winner_full
                p.id_score, p.is_unknown = float(best_candidate['conf']), False
                if best_candidate.get('model') == 'mol':
                    p.method = 'RF-mol'
                elif best_candidate.get('model') == 'ele':
                    p.method = 'RF-ele'
                else:
                    p.method = 'RF'
                p.detailed_id = best_candidate['det']

        # 4. Optional: second-pass molecule-only RF on unknown peaks
        scaler_rf_mol2 = None
        model_rf_mol2 = None
        target_decoder_rf_mol2 = None

        def _train_molecule_only_rf_if_needed() -> bool:
            nonlocal scaler_rf_mol2, model_rf_mol2, target_decoder_rf_mol2
            if scaler_rf_mol2 is not None and model_rf_mol2 is not None and target_decoder_rf_mol2 is not None:
                return True
            X_train_mol2, ions_train_mol2 = load_ion_training_data(
                path=training_data_path,
                element_list=truth_molecules,
                elements_to_get_molecules=[],
                num_files=int(training_num_files),
                neighbor_threshold=eff_neighbor_threshold,
                use_signature=use_signature,
                augment_molecule_charge_ratios=bool(augment_molecule_training_charge_ratios),
            )
            if len(X_train_mol2) == 0:
                return False
            scaler_rf_mol2, model_rf_mol2, target_decoder_rf_mol2 = create_RF_model(X_train_mol2, ions_train_mol2)
            return True

        if truth_molecules and (unknown_molecule_rf or molecule_rf_rescue_elements) and flag_unknowns:
            print(f"  Molecule-only RF target classes ({len(truth_molecules)}): {_format_class_list(truth_molecules)}")

        if unknown_molecule_rf and flag_unknowns and truth_molecules:
            unknown_indices = [i for i, p in enumerate(formatted_results) if getattr(p, 'is_unknown', False)]
            if unknown_indices and _train_molecule_only_rf_if_needed():
                unknown_peaks = [formatted_results[i] for i in unknown_indices]
                mol_elements2, mol_confs2, mol_details2, mol_mcs2 = run_RF_model(
                    unknown_peaks,
                    x_exp,
                    spectrum_log,
                    scaler_rf_mol2,
                    model_rf_mol2,
                    target_decoder_rf_mol2,
                    neighbor_threshold=eff_neighbor_threshold,
                    use_signature=use_signature,
                )

                recovered = 0
                for local_i, global_i in enumerate(unknown_indices):
                    det2 = mol_details2[local_i]
                    pred2 = str(det2.el1) if det2 else ''
                    conf2 = float(mol_confs2[local_i]) if mol_confs2 else 0.0
                    if not pred2 or pred2 == 'Unknown':
                        continue
                    if conf2 < float(molecule_rf_threshold):
                        continue
                    dist2 = _min_abs_distance_to_species_samples(simplify_label(pred2), float(mol_mcs2[local_i]))
                    if dist2 > mc_threshold:
                        continue

                    p = formatted_results[global_i]
                    p.label = mol_elements2[local_i]
                    p.id_score = conf2
                    p.is_unknown = False
                    p.method = 'RF-mol2'
                    p.detailed_id = det2
                    recovered += 1

                if recovered:
                    print(f"  Molecule RF recovered {recovered}/{len(unknown_indices)} unknown peaks.")

        # 5. Optional: follow-on RF using mc-vector features per predicted species group
        if followon_mc_vector_rf:
            try:
                X_train_vec, ions_train_vec = load_ion_training_data_mc_vector(
                    path=training_data_path,
                    element_list=truth_species_primary,
                    elements_to_get_molecules=elements_for_molecules if include_molecules else [],
                    num_files=int(training_num_files),
                    mc_round_decimals=int(followon_mc_vector_round_decimals),
                    augment_molecule_charge_ratios=bool(augment_molecule_training_charge_ratios),
                )
                if len(X_train_vec) == 0:
                    print("  [Warn] Follow-on mc-vector RF: no training data; skipping.")
                else:
                    scaler_vec, model_vec, target_decoder_vec = create_RF_model(X_train_vec, ions_train_vec)
                    vec_len = int(model_vec.n_features_in_)
                    print(f"  Follow-on mc-vector RF enabled (vector length {vec_len}).")

                    # Build per-(predicted label) groups of detected m/c values using initial RF peak m/cs.
                    group_to_indices: dict[str, list[int]] = {}
                    group_to_mcs: dict[str, list[float]] = {}
                    for i, p in enumerate(formatted_results):
                        # Group by current assigned label (post unknown/molecule recovery)
                        raw = str(p.label) if getattr(p, 'label', None) is not None else ''
                        main = re.split(r'\(|,', raw)[0].strip()
                        key = simplify_label(main) if main else 'Unknown'
                        if not key or key == 'Unknown':
                            continue
                        group_to_indices.setdefault(key, []).append(i)
                        if len(peak_mcs) > i:
                            group_to_mcs.setdefault(key, []).append(float(peak_mcs[i]))

                    if not group_to_indices:
                        print("  Follow-on mc-vector RF: no non-unknown groups; skipping.")
                    else:
                        def _vectorize_mcs(mcs: list[float]) -> list[float]:
                            if not mcs:
                                return [0.0] * vec_len
                            uniq = np.unique(np.round(np.asarray(mcs, dtype=float), int(followon_mc_vector_round_decimals)))
                            vals = sorted(float(x) for x in uniq.tolist())
                            if len(vals) > vec_len:
                                # Downsample to cover the range (keep endpoints)
                                idxs = np.linspace(0, len(vals) - 1, vec_len)
                                idxs = np.round(idxs).astype(int).tolist()
                                vals = [vals[j] for j in idxs]
                            return vals + [0.0] * (vec_len - len(vals))

                        groups = sorted(group_to_indices.keys())
                        X_groups = np.asarray([_vectorize_mcs(group_to_mcs.get(g, [])) for g in groups], dtype=float)
                        Xg = scaler_vec.transform(X_groups)
                        probs = model_vec.predict_proba(Xg)

                        updated_groups = 0
                        for gi, g in enumerate(groups):
                            prob = probs[gi]
                            top_idx = int(np.argmax(prob))
                            conf = float(prob[top_idx])
                            pred_class = model_vec.classes_[top_idx]
                            pred_label = str(target_decoder_vec[pred_class])
                            pred_main = re.split(r'\(|,', pred_label)[0].strip()
                            pred_key = simplify_label(pred_main) if pred_main else ''
                            if not pred_key:
                                continue

                            # Apply per-peak physicality check (optional) for the proposed label
                            ok = True
                            if flag_unknowns:
                                for idx in group_to_indices[g]:
                                    mc_val = float(peak_mcs[idx]) if len(peak_mcs) > idx else float(formatted_results[idx].pos)
                                    dist = _min_abs_distance_to_species_samples(pred_key, mc_val)
                                    if dist > mc_threshold:
                                        ok = False
                                        break
                            if not ok:
                                continue

                            for idx in group_to_indices[g]:
                                p = formatted_results[idx]
                                p.label = pred_main
                                p.id_score = conf
                                p.is_unknown = False
                                p.method = 'RF-mcvec'
                                p.detailed_id = DetailedId(el1=pred_main, conf1=conf, el2=str(p.detailed_id.el1) if p.detailed_id else 'Unknown', conf2=0.0)
                            updated_groups += 1

                        if updated_groups:
                            print(f"  Follow-on mc-vector RF updated {updated_groups}/{len(groups)} species-groups.")
            except Exception as e:
                print(f"  [Warn] Follow-on mc-vector RF failed: {e}")

        # 6. Optional: context-aware rescoring of ambiguous RF candidates.
        # This is candidate-only: nearby peaks can change the winner only to another
        # label that RF already listed for the target peak.
        context_override_rows: list[dict] = []
        if context_rescore:
            def _format_context_candidates(cands: list[dict]) -> str:
                return "; ".join(f"{c['label']}:{float(c['conf']):.3f}" for c in cands)

            def _parse_unknown_confidence_candidates(label: str) -> list[tuple[str, float]]:
                raw = str(label or '').strip()
                if not raw.startswith('Unknown') or '(' not in raw or ')' not in raw:
                    return []
                inner = raw.split('(', 1)[1].rsplit(')', 1)[0].strip()
                parsed: list[tuple[str, float]] = []
                for part in inner.split(','):
                    text = part.strip()
                    m = re.match(r'(.+?)\s+([0-9]+(?:\.[0-9]+)?)%$', text)
                    if not m:
                        continue
                    parsed.append((m.group(1).strip(), float(m.group(2)) / 100.0))
                return parsed

            def _rf_candidates_for_peak(p: PeakRange) -> list[dict]:
                merged: dict[str, dict] = {}

                def add_candidate(label: str, conf_value: float, *, display_label: str | None = None):
                    raw = str(label or '').strip()
                    if not raw:
                        return
                    if raw.startswith('Unknown'):
                        for parsed_label, parsed_conf in _parse_unknown_confidence_candidates(raw):
                            add_candidate(parsed_label, parsed_conf)
                        return
                    key = simplify_label(re.split(r'\(|,', raw)[0].strip())
                    if not key or key == 'Unknown':
                        return
                    conf_f = max(0.0, float(conf_value or 0.0))
                    if key not in merged or conf_f > float(merged[key]['conf']):
                        merged[key] = {
                            'label': key,
                            'display_label': display_label or raw,
                            'conf': conf_f,
                        }

                det = getattr(p, 'detailed_id', None)
                if det is not None:
                    add_candidate(str(getattr(det, 'el1', '') or ''), float(getattr(det, 'conf1', 0.0) or 0.0))
                    add_candidate(str(getattr(det, 'el2', '') or ''), float(getattr(det, 'conf2', 0.0) or 0.0))
                if not merged:
                    add_candidate(str(getattr(p, 'label', '') or ''), float(getattr(p, 'id_score', 0.0) or 0.0))

                return sorted(merged.values(), key=lambda c: float(c['conf']), reverse=True)

            try:
                window = max(0.0, float(context_window_da))
                strength = max(0.0, float(context_strength))
                min_top_conf = float(context_min_confidence)
                min_cand_conf = max(0.0, float(context_min_candidate_confidence))
                margin = float(context_override_margin)
                sigma = max(1e-9, float(context_distance_sigma))
                rescue_min_score = float(context_rescue_unknown_min_score)

                peak_positions = [
                    float(peak_mcs[i]) if len(peak_mcs) > i else float(getattr(p, 'pos', 0.0) or 0.0)
                    for i, p in enumerate(formatted_results)
                ]
                all_candidates = [_rf_candidates_for_peak(p) for p in formatted_results]
                considered = 0

                for i, p in enumerate(formatted_results):
                    target_candidates = [
                        c for c in all_candidates[i]
                        if float(c['conf']) >= min_cand_conf
                    ]
                    if len(target_candidates) < 2:
                        continue

                    original = target_candidates[0]
                    original_label = str(original['label'])
                    original_conf = float(original['conf'])
                    if not getattr(p, 'is_unknown', False) and original_conf >= min_top_conf:
                        continue

                    considered += 1
                    target_labels = {str(c['label']) for c in target_candidates}
                    support = {label: 0.0 for label in target_labels}
                    neighbor_count = 0
                    target_mc = peak_positions[i]

                    for j, neighbor in enumerate(formatted_results):
                        if i == j:
                            continue
                        delta = abs(float(peak_positions[j]) - target_mc)
                        if delta > window:
                            continue
                        neighbor_candidates = all_candidates[j]
                        if not neighbor_candidates:
                            continue
                        distance_weight = float(np.exp(-0.5 * (delta / sigma) ** 2))
                        contributed = False
                        for cand in neighbor_candidates:
                            label = str(cand['label'])
                            if label not in support:
                                continue
                            support[label] += distance_weight * float(cand['conf'])
                            contributed = True
                        if contributed:
                            neighbor_count += 1

                    rescored = {
                        str(c['label']): float(c['conf']) + strength * support.get(str(c['label']), 0.0)
                        for c in target_candidates
                    }
                    best_label = max(rescored, key=rescored.get)
                    best_score = float(rescored[best_label])
                    original_score = float(rescored.get(original_label, original_conf))
                    same_label_unknown_rescue = (
                        bool(context_rescue_unknown_same_label)
                        and bool(getattr(p, 'is_unknown', False))
                        and best_label == original_label
                        and support.get(original_label, 0.0) > 0.0
                        and best_score >= rescue_min_score
                        and best_score >= (original_conf + margin)
                    )
                    candidate_switch = (
                        best_label != original_label
                        and best_score >= (original_score + margin)
                    )
                    if not candidate_switch and not same_label_unknown_rescue:
                        continue
                    override_reason = 'same_label_unknown_rescue' if same_label_unknown_rescue else 'candidate_switch'

                    old_label = str(getattr(p, 'label', '') or '')
                    old_method = str(getattr(p, 'method', '') or '')
                    old_is_unknown = bool(getattr(p, 'is_unknown', False))
                    old_det = getattr(p, 'detailed_id', None)
                    old_top2 = target_candidates[1] if len(target_candidates) > 1 else None

                    p.label = best_label
                    p.id_score = min(1.0, best_score)
                    p.is_unknown = False
                    p.method = f"{old_method}+context" if old_method else 'RF-context'
                    p.detailed_id = DetailedId(
                        el1=best_label,
                        conf1=min(1.0, best_score),
                        el2=original_label if original_label != best_label else (str(old_top2['label']) if old_top2 else ''),
                        conf2=original_conf if original_label != best_label else (float(old_top2['conf']) if old_top2 else 0.0),
                    )
                    all_candidates[i] = _rf_candidates_for_peak(p)

                    context_override_rows.append({
                        'peak_start': float(getattr(p, 'start', np.nan)),
                        'peak_end': float(getattr(p, 'end', np.nan)),
                        'peak_mc': target_mc,
                        'old_label': old_label,
                        'old_method': old_method,
                        'old_is_unknown': old_is_unknown,
                        'old_top1': original_label,
                        'old_top1_conf': original_conf,
                        'old_top2': str(getattr(old_det, 'el2', '') or (str(old_top2['label']) if old_top2 else '')) if old_det is not None else (str(old_top2['label']) if old_top2 else ''),
                        'old_top2_conf': float(getattr(old_det, 'conf2', 0.0) or 0.0) if old_det is not None else (float(old_top2['conf']) if old_top2 else 0.0),
                        'new_label': best_label,
                        'new_score': best_score,
                        'original_candidate': original_label,
                        'original_rescored_score': original_score,
                        'support_new': support.get(best_label, 0.0),
                        'support_original': support.get(original_label, 0.0),
                        'neighbor_count': neighbor_count,
                        'override_reason': override_reason,
                        'candidates_before': _format_context_candidates(target_candidates),
                        'scores_after': "; ".join(f"{k}:{v:.3f}" for k, v in sorted(rescored.items())),
                        'context_window_da': window,
                        'context_strength': strength,
                        'context_distance_sigma': sigma,
                        'context_override_margin': margin,
                        'context_rescue_unknown_min_score': rescue_min_score,
                    })

                if context_override_rows:
                    print(f"  Context RF rescoring overrides applied: {len(context_override_rows)}/{considered} candidates")
                    context_overrides_path = os.path.join(prefix_internal, f"{prefix_internal}_context_rescore_overrides.csv")
                    cols = [
                        'peak_start', 'peak_end', 'peak_mc',
                        'old_label', 'old_method', 'old_is_unknown',
                        'old_top1', 'old_top1_conf', 'old_top2', 'old_top2_conf',
                        'new_label', 'new_score',
                        'original_candidate', 'original_rescored_score',
                        'support_new', 'support_original', 'neighbor_count',
                        'override_reason',
                        'candidates_before', 'scores_after',
                        'context_window_da', 'context_strength',
                        'context_distance_sigma', 'context_override_margin',
                        'context_rescue_unknown_min_score',
                    ]
                    with open(context_overrides_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=cols)
                        writer.writeheader()
                        writer.writerows(context_override_rows)
                else:
                    print(f"  Context RF rescoring considered {considered} candidates; no overrides")
            except Exception as e:
                print(f"  [Warn] Context RF rescoring failed: {e}")

        def _compute_accuracy_counts(current_ranges: list[PeakRange]) -> dict:
            """
            Compute counts for truth-matched peaks:
              - overall, elemental-only, molecular-only
              - including unknowns (unknowns count against correctness)
              - excluding unknowns (unknown predictions removed from denominator)
            """
            truth_data_local = truth_data
            def _get_pred_labels(pr: PeakRange) -> list[str]:
                if getattr(pr, 'is_unknown', False):
                    return ['Unknown']
                top_n = max(1, int(rf_accuracy_top_n or 1))
                labels = []
                if pr.detailed_id is not None and pr.detailed_id.el1:
                    labels.append(str(pr.detailed_id.el1))
                    if top_n >= 2 and pr.detailed_id.el2 and float(getattr(pr.detailed_id, 'conf2', 0.0) or 0.0) > 0:
                        labels.append(str(pr.detailed_id.el2))
                    return labels[:top_n]
                raw = str(pr.label) if getattr(pr, 'label', None) is not None else ''
                return [re.split(r'\(|,', raw)[0].strip()] if raw else ['Unknown']

            total_inc = correct_inc = 0
            total_exc = correct_exc = 0
            total_ele_inc = correct_ele_inc = 0
            total_ele_exc = correct_ele_exc = 0
            total_mol_inc = correct_mol_inc = 0
            total_mol_exc = correct_mol_exc = 0
            unk_with_truth = 0

            for pr in current_ranges:
                best_iou, best_truth = 0.0, None
                for t in truth_data_local:
                    iou_val = calculate_iou(pr, t)
                    if iou_val > best_iou:
                        best_iou, best_truth = iou_val, t
                if best_iou <= 0.1 or best_truth is None:
                    continue
                true_label = str(best_truth.label)
                if not true_label or true_label == 'Unknown':
                    continue
                pred_labels = _get_pred_labels(pr)
                is_pred_unknown = (not pred_labels) or (pred_labels[0] == 'Unknown')
                if is_pred_unknown:
                    unk_with_truth += 1

                total_inc += 1
                is_ele_true = _is_elemental_label(true_label)
                if is_ele_true:
                    total_ele_inc += 1
                else:
                    total_mol_inc += 1

                is_correct = (not is_pred_unknown) and any(
                    simplify_label(true_label) == simplify_label(pred_label)
                    for pred_label in pred_labels[:max(1, int(rf_accuracy_top_n or 1))]
                    if pred_label and pred_label != 'Unknown'
                )
                if is_correct:
                    correct_inc += 1
                    if is_ele_true:
                        correct_ele_inc += 1
                    else:
                        correct_mol_inc += 1

                if not is_pred_unknown:
                    total_exc += 1
                    if is_ele_true:
                        total_ele_exc += 1
                    else:
                        total_mol_exc += 1
                    if is_correct:
                        correct_exc += 1
                        if is_ele_true:
                            correct_ele_exc += 1
                        else:
                            correct_mol_exc += 1

            def pct(c, t): return (c / t * 100.0) if t > 0 else 0.0

            return {
                'species_including_unknowns': pct(correct_inc, total_inc),
                'species_excluding_unknowns': pct(correct_exc, total_exc),
                'elemental_including_unknowns': pct(correct_ele_inc, total_ele_inc),
                'elemental_excluding_unknowns': pct(correct_ele_exc, total_ele_exc),
                'molecular_including_unknowns': pct(correct_mol_inc, total_mol_inc),
                'molecular_excluding_unknowns': pct(correct_mol_exc, total_mol_exc),
                'counts': {
                    'species_correct_including_unknowns': correct_inc,
                    'species_total_including_unknowns': total_inc,
                    'species_correct_excluding_unknowns': correct_exc,
                    'species_total_excluding_unknowns': total_exc,
                    'elemental_correct_including_unknowns': correct_ele_inc,
                    'elemental_total_including_unknowns': total_ele_inc,
                    'elemental_correct_excluding_unknowns': correct_ele_exc,
                    'elemental_total_excluding_unknowns': total_ele_exc,
                    'molecular_correct_including_unknowns': correct_mol_inc,
                    'molecular_total_including_unknowns': total_mol_inc,
                    'molecular_correct_excluding_unknowns': correct_mol_exc,
                    'molecular_total_excluding_unknowns': total_mol_exc,
                    'unknown_with_truth': unk_with_truth,
                },
            }

        before_rescue_breakdown = _compute_accuracy_counts(formatted_results)

        # 6. Optional: molecule RF "rescue" on peaks currently labeled as a single element (not unknown)
        rescue_stats = {'considered': 0, 'overrides': 0, 'mixed_candidates': 0}
        rescue_override_rows: list[dict] = []
        if molecule_rf_rescue_elements and truth_molecules and _train_molecule_only_rf_if_needed():
            candidate_indices = []
            candidate_peaks = []
            for i, p in enumerate(formatted_results):
                if getattr(p, 'is_unknown', False):
                    continue
                det = getattr(p, 'detailed_id', None)
                pred1 = str(det.el1) if det is not None and det.el1 else (re.split(r'\(|,', str(p.label))[0].strip() if p.label else '')
                if not pred1 or pred1 == 'Unknown':
                    continue
                if not _is_elemental_label(pred1):
                    continue
                candidate_indices.append(i)
                candidate_peaks.append(p)

            if candidate_peaks:
                mol_elements_r, mol_confs_r, mol_details_r, mol_mcs_r = run_RF_model(
                    candidate_peaks,
                    x_exp,
                    spectrum_log,
                    scaler_rf_mol2,
                    model_rf_mol2,
                    target_decoder_rf_mol2,
                    neighbor_threshold=eff_neighbor_threshold,
                    use_signature=use_signature,
                )

                for local_i, global_i in enumerate(candidate_indices):
                    rescue_stats['considered'] += 1
                    p = formatted_results[global_i]

                    det_ele = getattr(p, 'detailed_id', None)
                    ele_pred = str(det_ele.el1) if det_ele is not None and det_ele.el1 else ''
                    ele_conf = float(det_ele.conf1) if det_ele is not None else float(getattr(p, 'id_score', 0.0) or 0.0)
                    ele_key = simplify_label(ele_pred) if ele_pred else ''

                    det_m = mol_details_r[local_i]
                    mol_pred = str(det_m.el1) if det_m is not None and det_m.el1 else ''
                    mol_conf = float(mol_confs_r[local_i]) if mol_confs_r else 0.0
                    mol_key = simplify_label(mol_pred) if mol_pred else ''
                    if not mol_key or mol_key == 'Unknown' or not is_molecule(mol_key):
                        continue
                    if mol_conf < float(molecule_rf_rescue_threshold):
                        continue

                    mc_val = float(mol_mcs_r[local_i]) if len(mol_mcs_r) > local_i else float(getattr(p, 'pos', 0.0) or 0.0)
                    dist_m = _min_abs_distance_to_species_samples(mol_key, mc_val)
                    if dist_m > mc_threshold:
                        continue

                    dist_e = _min_abs_distance_to_species_samples(ele_key, mc_val) if ele_key else float('inf')
                    dist_margin = float(molecule_rf_rescue_dist_margin)
                    better_physical_fit = bool(np.isinf(dist_e) or (dist_m + dist_margin < dist_e))
                    comparable_physical_fit = bool(np.isinf(dist_e) or (dist_m <= dist_e + dist_margin))
                    q_m = max(0.0, 1.0 - (dist_m / mc_threshold)) if mc_threshold > 0 else 0.0
                    q_e = max(0.0, 1.0 - (dist_e / mc_threshold)) if (mc_threshold > 0 and not np.isinf(dist_e)) else 0.0
                    score_m = mol_conf * q_m
                    score_e = ele_conf * q_e

                    conf_margin = float(molecule_rf_rescue_margin)
                    score_margin = float(molecule_rf_rescue_score_margin)
                    rescue_action = ""
                    rescue_reason = ""
                    if better_physical_fit and mol_conf >= (ele_conf + conf_margin):
                        rescue_action = "override"
                        rescue_reason = "conf_margin"
                    elif better_physical_fit and score_m >= (score_e + score_margin):
                        rescue_action = "override"
                        rescue_reason = "score_margin"
                    elif comparable_physical_fit and mol_conf >= max(0.0, ele_conf - conf_margin):
                        rescue_action = "mixed_candidate"
                        rescue_reason = "conf_close"
                    elif comparable_physical_fit and score_m >= max(0.0, score_e - score_margin):
                        rescue_action = "mixed_candidate"
                        rescue_reason = "score_close"

                    if not rescue_action:
                        continue

                    if rescue_action == "override":
                        should_override = True
                    else:
                        should_override = False

                    # Capture a compact accepted-rescue record for audit/debugging.
                    try:
                        mol_best_dist, mol_best_scale, mol_scaled_mc, mol_nearest_train = _best_match_to_species_samples(
                            mol_key, mc_val, allow_scaling_for_elements=False
                        )
                    except Exception:
                        mol_best_dist, mol_best_scale, mol_scaled_mc, mol_nearest_train = dist_m, 1.0, mc_val, None
                    try:
                        ele_best_dist, ele_best_scale, ele_scaled_mc, ele_nearest_train = _best_match_to_species_samples(
                            ele_key, mc_val, allow_scaling_for_elements=False
                        ) if ele_key else (float('inf'), 1.0, mc_val, None)
                    except Exception:
                        ele_best_dist, ele_best_scale, ele_scaled_mc, ele_nearest_train = dist_e, 1.0, mc_val, None

                    rescue_override_rows.append({
                        'peak_start': float(getattr(p, 'start', np.nan)),
                        'peak_end': float(getattr(p, 'end', np.nan)),
                        'peak_mc': mc_val,
                        'element_pred_simple': ele_key if ele_key else ele_pred,
                        'element_conf': ele_conf,
                        'element_dist': float(dist_e) if not np.isinf(dist_e) else '',
                        'element_best_scale': ele_best_scale,
                        'element_scaled_mc': ele_scaled_mc,
                        'element_nearest_training_mc': ele_nearest_train if ele_nearest_train is not None else '',
                        'molecule_pred_simple': mol_key,
                        'molecule_conf': mol_conf,
                        'molecule_dist': float(dist_m),
                        'molecule_best_scale': mol_best_scale,
                        'molecule_scaled_mc': mol_scaled_mc,
                        'molecule_nearest_training_mc': mol_nearest_train if mol_nearest_train is not None else '',
                        'q_element': q_e,
                        'q_molecule': q_m,
                        'score_element': score_e,
                        'score_molecule': score_m,
                        'rescue_action': rescue_action,
                        'rescue_reason': rescue_reason,
                    })

                    if should_override:
                        p.label = mol_elements_r[local_i]
                        p.id_score = mol_conf
                        p.is_unknown = False
                        p.method = 'RF-mol-rescue'
                        p.detailed_id = DetailedId(el1=mol_pred, conf1=mol_conf, el2=ele_pred, conf2=ele_conf)
                        rescue_stats['overrides'] += 1
                    else:
                        p.is_unknown = False
                        p.method = f"{p.method}+mol-candidate" if p.method else "RF-mol-candidate"
                        p.detailed_id = DetailedId(el1=ele_pred, conf1=ele_conf, el2=mol_pred, conf2=mol_conf)
                        rescue_stats['mixed_candidates'] += 1

            if rescue_stats['overrides'] or rescue_stats['mixed_candidates']:
                print(
                    "  Molecule rescue accepted: "
                    f"{rescue_stats['overrides']} overrides, "
                    f"{rescue_stats['mixed_candidates']} mixed candidates / "
                    f"{rescue_stats['considered']} candidates"
                )
            else:
                print(
                    f"  Molecule rescue considered {rescue_stats['considered']} candidates; no accepted rescues"
                )
        after_rescue_breakdown = _compute_accuracy_counts(formatted_results)

        for p in formatted_results:
            orig = label_map.get(p.label)
            if orig: p.label = orig
            if p.detailed_id:
                p.detailed_id.el1 = label_map.get(p.detailed_id.el1, p.detailed_id.el1)
                p.detailed_id.el2 = label_map.get(p.detailed_id.el2, p.detailed_id.el2)

        if rescue_override_rows:
            mapped_rows = []
            for r in rescue_override_rows:
                r2 = dict(r)
                r2['element_pred'] = label_map.get(str(r2.get('element_pred_simple', '')), str(r2.get('element_pred_simple', '')))
                r2['molecule_pred'] = label_map.get(str(r2.get('molecule_pred_simple', '')), str(r2.get('molecule_pred_simple', '')))
                mapped_rows.append(r2)

            rescue_overrides_path = os.path.join(prefix_internal, f"{prefix_internal}_molecule_rescue_candidates.csv")
            cols = [
                'peak_start', 'peak_end', 'peak_mc',
                'element_pred', 'element_pred_simple', 'element_conf', 'element_dist',
                'element_best_scale', 'element_scaled_mc', 'element_nearest_training_mc',
                'molecule_pred', 'molecule_pred_simple', 'molecule_conf', 'molecule_dist',
                'molecule_best_scale', 'molecule_scaled_mc', 'molecule_nearest_training_mc',
                'q_element', 'q_molecule', 'score_element', 'score_molecule',
                'rescue_action', 'rescue_reason',
            ]
            try:
                with open(rescue_overrides_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=cols)
                    writer.writeheader()
                    writer.writerows(mapped_rows)
            except Exception as e:
                print(f"  [Warn] Failed writing rescue overrides CSV ({e})")

    except Exception as e:
        print(f"RF identification failed: {e}")
        import traceback
        traceback.print_exc()
        before_rescue_breakdown = None
        after_rescue_breakdown = None
        rescue_stats = {'considered': 0, 'overrides': 0, 'mixed_candidates': 0}

    # --- ACCURACY ASSESSMENT ---
    detailed_rows = []
    for p in formatted_results:
        best_iou, best_truth = 0, None
        for t in truth_data:
            iou_val = calculate_iou(p, t)
            if iou_val > best_iou: best_iou, best_truth = iou_val, t
        det = p.detailed_id if p.detailed_id is not None else DetailedId(el1='Unknown')
        row = {
            'predicted peak start': p.start, 'predicted peak end': p.end,
            'true peak start': best_truth.start if best_iou > 0.1 else '',
            'true peak end': best_truth.end if best_iou > 0.1 else '',
            'true element label': best_truth.label if best_iou > 0.1 else 'Unknown',
            'pred element label 1': det.el1, 'pred confidence 1': round(det.conf1, 3),
            'pred element label 2': det.el2, 'pred confidence 2': round(det.conf2, 3),
            'discarded': p.is_unknown
        }
        detailed_rows.append(row)

    detailed_results_path = os.path.join(prefix_internal, f"{prefix_internal}_detailed_results.csv")
    with open(detailed_results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['predicted peak start', 'predicted peak end', 'true peak start', 'true peak end', 'true element label', 'pred element label 1', 'pred confidence 1', 'pred element label 2', 'pred confidence 2', 'discarded'])
        writer.writeheader()
        writer.writerows(detailed_rows)

    # --- UNKNOWN PEAK ERROR REPORT (MC vs TRAINING) ---
    def _parse_unknown_reason(label1: str) -> str:
        s = str(label1)
        if not s.startswith('Unknown'):
            return ''
        if '(' in s and ')' in s:
            inner = s.split('(', 1)[1].rsplit(')', 1)[0].strip()
            return inner
        return ''

    if flag_unknowns and any(bool(r.get('discarded')) for r in detailed_rows):
        # If RF identification failed early, the mc-distance helpers may not exist.
        # In that case, rebuild the lookup here and define local helpers so the report still writes.
        if 'mc_samples_by_species' not in locals() or not isinstance(mc_samples_by_species, dict) or len(mc_samples_by_species) == 0:
            try:
                print("  Building mc-distance lookup table (for unknown error report)...")
                mc_samples_by_species = build_empirical_mc_samples(path=training_data_path, num_files=int(training_num_files))
            except Exception as e:
                print(f"  [Warn] Failed to build mc-distance lookup for unknown error report ({e}).")
                mc_samples_by_species = {}

        if '_best_match_to_species_samples' not in locals():
            def _min_abs_distance_to_samples(sorted_samples: np.ndarray | None, value: float) -> float:
                if sorted_samples is None or len(sorted_samples) == 0:
                    return float('inf')
                idx = bisect.bisect_left(sorted_samples, value)
                best = float('inf')
                if idx < len(sorted_samples):
                    best = min(best, abs(float(sorted_samples[idx]) - value))
                if idx > 0:
                    best = min(best, abs(float(sorted_samples[idx - 1]) - value))
                return best

            def _nearest_sample_value(sorted_samples: np.ndarray | None, value: float) -> float | None:
                if sorted_samples is None or len(sorted_samples) == 0:
                    return None
                idx = bisect.bisect_left(sorted_samples, value)
                best_val = None
                best_dist = float('inf')
                if idx < len(sorted_samples):
                    v = float(sorted_samples[idx])
                    d = abs(v - value)
                    if d < best_dist:
                        best_dist, best_val = d, v
                if idx > 0:
                    v = float(sorted_samples[idx - 1])
                    d = abs(v - value)
                    if d < best_dist:
                        best_dist, best_val = d, v
                return best_val

            def _best_match_to_species_samples(
                species_key: str,
                mc_val: float,
                *,
                allow_scaling_for_elements: bool,
            ) -> tuple[float, float, float, float | None]:
                samples = mc_samples_by_species.get(species_key)
                if samples is None or len(samples) == 0:
                    return float('inf'), 1.0, mc_val, None

                min_s = float(samples[0])
                max_s = float(samples[-1])

                consider_scaling = is_molecule(species_key) or allow_scaling_for_elements
                multipliers = (1.0, 2.0, 3.0, 4.0, 0.5, 1.0 / 3.0, 0.25) if consider_scaling else (1.0,)

                best_dist = float('inf')
                best_mult = 1.0
                best_scaled = mc_val
                best_nearest = None

                for mult in multipliers:
                    scaled = mc_val * mult
                    if mult != 1 and not ((min_s - mc_threshold) <= scaled <= (max_s + mc_threshold)):
                        continue
                    dist = _min_abs_distance_to_samples(samples, scaled)
                    if dist < best_dist:
                        best_dist = dist
                        best_mult = mult
                        best_scaled = scaled
                        best_nearest = _nearest_sample_value(samples, scaled)

                return best_dist, best_mult, best_scaled, best_nearest

        report_rows = []
        for row in detailed_rows:
            discarded = str(row.get('discarded', '')).lower() in ('true', '1', 'yes')
            if not discarded:
                continue

            true_label_raw = str(row.get('true element label', 'Unknown'))
            true_label_simple = simplify_label(true_label_raw) if true_label_raw else 'Unknown'

            reason_raw = _parse_unknown_reason(str(row.get('pred element label 1', '')))
            reason_simple = simplify_label(reason_raw) if reason_raw else ''

            def _safe_float(v):
                try:
                    if v is None:
                        return None
                    if isinstance(v, str) and v.strip() == '':
                        return None
                    return float(v)
                except Exception:
                    return None

            y_exp = spectrum_log.numpy() if hasattr(spectrum_log, 'numpy') else spectrum_log

            def _peak_max_mc(start: float | None, end: float | None) -> float | None:
                if start is None or end is None:
                    return None
                try:
                    mask = (x_exp >= float(start)) & (x_exp <= float(end))
                    if np.any(mask):
                        peak_idx = int(np.argmax(np.asarray(y_exp)[mask]))
                        return float(np.asarray(x_exp)[mask][peak_idx])
                    return float(start + end) / 2.0
                except Exception:
                    return None

            ts = _safe_float(row.get('true peak start'))
            te = _safe_float(row.get('true peak end'))
            ps = _safe_float(row.get('predicted peak start'))
            pe = _safe_float(row.get('predicted peak end'))

            # Use peak maximum m/c within the interval (not midpoint) because peaks can be asymmetric.
            true_peak_mc_max = _peak_max_mc(ts, te) if (ts is not None and te is not None) else None
            pred_peak_mc_max = _peak_max_mc(ps, pe) if (ps is not None and pe is not None) else None
            mc_used = true_peak_mc_max if true_peak_mc_max is not None else pred_peak_mc_max

            # Distances for true label (code behavior vs diagnostic scaling-for-elements)
            dist_true_code = float('inf')
            mult_true_code = 1
            scaled_true_code = mc_used if mc_used is not None else float('nan')
            nearest_true_code = None

            dist_true_scaled_any = float('inf')
            mult_true_scaled_any = 1
            scaled_true_scaled_any = mc_used if mc_used is not None else float('nan')
            nearest_true_scaled_any = None

            if mc_used is not None and true_label_simple and true_label_simple != 'Unknown':
                dist_true_code, mult_true_code, scaled_true_code, nearest_true_code = _best_match_to_species_samples(
                    true_label_simple,
                    float(mc_used),
                    allow_scaling_for_elements=False,
                )
                dist_true_scaled_any, mult_true_scaled_any, scaled_true_scaled_any, nearest_true_scaled_any = _best_match_to_species_samples(
                    true_label_simple,
                    float(mc_used),
                    allow_scaling_for_elements=True,
                )

            # Distances for the unknown "reason" (if present)
            dist_reason_code = float('inf')
            mult_reason_code = 1
            scaled_reason_code = mc_used if mc_used is not None else float('nan')
            nearest_reason_code = None

            dist_reason_scaled_any = float('inf')
            mult_reason_scaled_any = 1
            scaled_reason_scaled_any = mc_used if mc_used is not None else float('nan')
            nearest_reason_scaled_any = None

            if mc_used is not None and reason_simple and reason_simple != 'Unknown':
                dist_reason_code, mult_reason_code, scaled_reason_code, nearest_reason_code = _best_match_to_species_samples(
                    reason_simple,
                    float(mc_used),
                    allow_scaling_for_elements=False,
                )
                dist_reason_scaled_any, mult_reason_scaled_any, scaled_reason_scaled_any, nearest_reason_scaled_any = _best_match_to_species_samples(
                    reason_simple,
                    float(mc_used),
                    allow_scaling_for_elements=True,
                )

            report_rows.append({
                'predicted peak start': row.get('predicted peak start', ''),
                'predicted peak end': row.get('predicted peak end', ''),
                'true peak start': row.get('true peak start', ''),
                'true peak end': row.get('true peak end', ''),
                'true_peak_mc_max': true_peak_mc_max if true_peak_mc_max is not None else '',
                'pred_peak_mc_max': pred_peak_mc_max if pred_peak_mc_max is not None else '',
                'mc_used_for_distance': mc_used if mc_used is not None else '',
                'true element label': true_label_raw,
                'true label simplified': true_label_simple,
                'pred element label 1': row.get('pred element label 1', ''),
                'unknown reason raw': reason_raw,
                'unknown reason simplified': reason_simple,
                'true is molecule': bool(is_molecule(true_label_simple)) if true_label_simple else False,
                'training missing (true)': bool(np.isinf(dist_true_scaled_any)),
                'dist_true_code': dist_true_code,
                'mult_true_code': mult_true_code,
                'scaled_mc_true_code': scaled_true_code,
                'nearest_training_mc_true_code': nearest_true_code if nearest_true_code is not None else '',
                'within_threshold_true_code': (dist_true_code <= mc_threshold) if not np.isinf(dist_true_code) else False,
                'dist_true_scaled_any': dist_true_scaled_any,
                'mult_true_scaled_any': mult_true_scaled_any,
                'scaled_mc_true_scaled_any': scaled_true_scaled_any,
                'nearest_training_mc_true_scaled_any': nearest_true_scaled_any if nearest_true_scaled_any is not None else '',
                'within_threshold_true_scaled_any': (dist_true_scaled_any <= mc_threshold) if not np.isinf(dist_true_scaled_any) else False,
                'dist_reason_code': dist_reason_code,
                'mult_reason_code': mult_reason_code,
                'scaled_mc_reason_code': scaled_reason_code,
                'nearest_training_mc_reason_code': nearest_reason_code if nearest_reason_code is not None else '',
                'dist_reason_scaled_any': dist_reason_scaled_any,
                'mult_reason_scaled_any': mult_reason_scaled_any,
                'scaled_mc_reason_scaled_any': scaled_reason_scaled_any,
                'nearest_training_mc_reason_scaled_any': nearest_reason_scaled_any if nearest_reason_scaled_any is not None else '',
            })

        unknown_report_path = os.path.join(prefix_internal, f"{prefix_internal}_unknown_peak_error_report.csv")
        try:
            with open(unknown_report_path, 'w', newline='') as f:
                fieldnames = [
                    'predicted peak start', 'predicted peak end',
                    'true peak start', 'true peak end',
                    'true_peak_mc_max', 'pred_peak_mc_max', 'mc_used_for_distance',
                    'true element label', 'true label simplified',
                    'pred element label 1',
                    'unknown reason raw', 'unknown reason simplified',
                    'true is molecule',
                    'training missing (true)',
                    'dist_true_code', 'mult_true_code', 'scaled_mc_true_code', 'nearest_training_mc_true_code', 'within_threshold_true_code',
                    'dist_true_scaled_any', 'mult_true_scaled_any', 'scaled_mc_true_scaled_any', 'nearest_training_mc_true_scaled_any', 'within_threshold_true_scaled_any',
                    'dist_reason_code', 'mult_reason_code', 'scaled_mc_reason_code', 'nearest_training_mc_reason_code',
                    'dist_reason_scaled_any', 'mult_reason_scaled_any', 'scaled_mc_reason_scaled_any', 'nearest_training_mc_reason_scaled_any',
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report_rows)
            print(f"  Unknown peak error report saved: {unknown_report_path} ({len(report_rows)} rows)")
        except OSError as e:
            print(f"  [Warn] Failed to write unknown peak error report: {unknown_report_path} ({e})")

    # Use the computed before/after breakdowns from the in-pipeline evaluation.
    # If something failed before they were computed, fall back to a minimal empty breakdown.
    if after_rescue_breakdown is None:
        after_rescue_breakdown = {
            'species_including_unknowns': 0.0,
            'species_excluding_unknowns': 0.0,
            'elemental_including_unknowns': 0.0,
            'elemental_excluding_unknowns': 0.0,
            'molecular_including_unknowns': 0.0,
            'molecular_excluding_unknowns': 0.0,
            'counts': {
                'species_correct_including_unknowns': 0,
                'species_total_including_unknowns': 0,
                'species_correct_excluding_unknowns': 0,
                'species_total_excluding_unknowns': 0,
                'elemental_correct_including_unknowns': 0,
                'elemental_total_including_unknowns': 0,
                'elemental_correct_excluding_unknowns': 0,
                'elemental_total_excluding_unknowns': 0,
                'molecular_correct_including_unknowns': 0,
                'molecular_total_including_unknowns': 0,
                'molecular_correct_excluding_unknowns': 0,
                'molecular_total_excluding_unknowns': 0,
                'unknown_with_truth': 0,
            },
        }

    accuracy_breakdown = dict(after_rescue_breakdown)
    if molecule_rf_rescue_elements and before_rescue_breakdown is not None:
        accuracy_breakdown['before_rescue'] = before_rescue_breakdown
        accuracy_breakdown['after_rescue'] = after_rescue_breakdown
        accuracy_breakdown['rescue'] = rescue_stats

    accuracy_pct = float(after_rescue_breakdown.get('species_excluding_unknowns', 0.0) or 0.0)
    accuracy_pct_ele = float(after_rescue_breakdown.get('elemental_excluding_unknowns', 0.0) or 0.0)

    if return_accuracy_breakdown:
        return (
            formatted_results,
            result,
            accuracy_pct,
            accuracy_pct_ele,
            sum(1 for p in formatted_results if p.is_unknown),
            accuracy_breakdown,
        )

    return formatted_results, result, accuracy_pct, accuracy_pct_ele, sum(1 for p in formatted_results if p.is_unknown)
