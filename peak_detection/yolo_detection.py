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
                             include_molecules=False, yolo_weights='best.pt',
                             iou=0.01, conf=0.05, max_det=2000,
                             mc_min=0.0, mc_max=307.2,
                             use_neighborhood=False, neighbor_threshold=2.0,
                             use_signature=False,
                             separate_molecule_rf=False,
                             unknown_molecule_rf: bool = False,
                             molecule_rf_threshold=0.8,
                             followon_mc_vector_rf: bool = False,
                             followon_mc_vector_round_decimals: int = 3,
                             return_accuracy_breakdown: bool = False):
    """
    RangingNN YOLO model prediction wrapper.
    Unknown-flagging (when enabled) uses only mc-distance checks against empirical training samples.
    Optionally, a second RF model can be trained on molecular species only and applied only to
    peaks flagged as unknown.
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
            'counts': {
                'species_correct_including_unknowns': 0,
                'species_total_including_unknowns': 0,
                'species_correct_excluding_unknowns': 0,
                'species_total_excluding_unknowns': 0,
                'elemental_correct_including_unknowns': 0,
                'elemental_total_including_unknowns': 0,
                'elemental_correct_excluding_unknowns': 0,
                'elemental_total_excluding_unknowns': 0,
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

    # Recursive iteration
    peak_range_toadd = []
    multiplier = 0.01

    if n_iter > 0:
        for it in range(n_iter):
            n = sp_padded.shape[0]
            x1 = np.arange(n) * multiplier
            ranges = np.asarray(peak_range_pred, dtype=float)
            starts, ends = ranges[:, 0] * multiplier, ranges[:, 1] * multiplier
            in_any_range = np.logical_or.reduce((x1[:, None] > starts[None, :]) & (x1[:, None] < ends[None, :]), axis=1)
            spectrum_log_mod = sp_padded.clone().numpy()
            spectrum_log_mod[np.isin(x1, x1[in_any_range]) | (x1 < mc_min) | (x1 > mc_max)] = 0.2
            spectrum_log_mod = torch.Tensor(spectrum_log_mod)
            predictor_mod = DetectionPredictor(modelpath, spectrum_log_mod[None, None, ...], save_dir='test_results', cfg=cfg)
            result_mod = predictor_mod()[0]
            peak_range_pred_mod = result_mod[:, :2].cpu()
            tol = 0.5
            for i in peak_range_pred_mod:
                start, end = float(i[0]), float(i[1])
                max_iou_val, min_dist = 0.0, 1000
                for j in peak_range_pred.tolist():
                    start2, end2 = float(j[0]), float(j[1])
                    iou_val = calculate_iou_1d([start, end], [start2, end2])
                    if iou_val > max_iou_val: max_iou_val = iou_val
                    dist = multiplier * abs(start - start2)
                    if dist < min_dist: min_dist = dist
                if max_iou_val == 0.0 and min_dist > tol: peak_range_toadd.append([start, end])

    final_ranges = peak_range_pred.tolist() + peak_range_toadd
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
            
            if (is_unphysical or winner_main == 'Unknown') and flag_unknowns:
                p.label = f'Unknown ({winner_main})'
                p.id_score, p.is_unknown = 1.0, True
                p.method = 'RF-Unknown'
                p.detailed_id = DetailedId(el1=p.label, conf1=best_candidate['conf'], el2='Unknown', conf2=0.0)
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
        if unknown_molecule_rf and flag_unknowns and truth_molecules:
            print(f"  Molecule-only RF target classes ({len(truth_molecules)}): {_format_class_list(truth_molecules)}")

            unknown_indices = [i for i, p in enumerate(formatted_results) if getattr(p, 'is_unknown', False)]
            if unknown_indices:
                unknown_peaks = [formatted_results[i] for i in unknown_indices]
                X_train_mol2, ions_train_mol2 = load_ion_training_data(
                    path=training_data_path,
                    element_list=truth_molecules,
                    elements_to_get_molecules=[],
                    num_files=int(training_num_files),
                    neighbor_threshold=eff_neighbor_threshold,
                    use_signature=use_signature,
                    augment_molecule_charge_ratios=bool(augment_molecule_training_charge_ratios),
                )
                if len(X_train_mol2) > 0:
                    scaler_rf_mol2, model_rf_mol2, target_decoder_rf_mol2 = create_RF_model(X_train_mol2, ions_train_mol2)
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

        for p in formatted_results:
            orig = label_map.get(p.label)
            if orig: p.label = orig
            if p.detailed_id:
                p.detailed_id.el1 = label_map.get(p.detailed_id.el1, p.detailed_id.el1)
                p.detailed_id.el2 = label_map.get(p.detailed_id.el2, p.detailed_id.el2)

    except Exception as e:
        print(f"RF identification failed: {e}")
        import traceback
        traceback.print_exc()

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

    def _is_elemental_label(label: str) -> bool:
        try:
            comp = Composition(str(label))
            if len(comp.elements) != 1:
                return False
            return list(comp.values())[0] == 1
        except Exception:
            return bool(re.fullmatch(r'[A-Z][a-z]?$', str(label)))

    correct_matches_inc, total_matches_inc = 0, 0
    correct_matches_exc, total_matches_exc = 0, 0
    correct_matches_ele_inc, total_matches_ele_inc = 0, 0
    correct_matches_ele_exc, total_matches_ele_exc = 0, 0

    for row in detailed_rows:
        true_label = row['true element label']
        if true_label and true_label != 'Unknown':
            pred1 = str(row['pred element label 1'])
            is_pred_unknown = pred1.startswith('Unknown')
            is_elemental_true = _is_elemental_label(true_label)

            total_matches_inc += 1
            if is_elemental_true:
                total_matches_ele_inc += 1

            excluded = bool(flag_unknowns and is_pred_unknown)
            if not excluded:
                total_matches_exc += 1
                if is_elemental_true:
                    total_matches_ele_exc += 1

            try:
                is_correct = (not is_pred_unknown) and (simplify_label(true_label) == simplify_label(pred1))
                if is_correct:
                    correct_matches_inc += 1
                    if is_elemental_true:
                        correct_matches_ele_inc += 1
                    if not excluded:
                        correct_matches_exc += 1
                        if is_elemental_true:
                            correct_matches_ele_exc += 1
            except Exception:
                pass

    accuracy_pct = (correct_matches_exc / total_matches_exc * 100) if total_matches_exc > 0 else 0.0
    accuracy_pct_ele = (correct_matches_ele_exc / total_matches_ele_exc * 100) if total_matches_ele_exc > 0 else 0.0

    accuracy_breakdown = {
        'species_including_unknowns': (correct_matches_inc / total_matches_inc * 100) if total_matches_inc > 0 else 0.0,
        'species_excluding_unknowns': accuracy_pct,
        'elemental_including_unknowns': (correct_matches_ele_inc / total_matches_ele_inc * 100) if total_matches_ele_inc > 0 else 0.0,
        'elemental_excluding_unknowns': accuracy_pct_ele,
        'counts': {
            'species_correct_including_unknowns': correct_matches_inc,
            'species_total_including_unknowns': total_matches_inc,
            'species_correct_excluding_unknowns': correct_matches_exc,
            'species_total_excluding_unknowns': total_matches_exc,
            'elemental_correct_including_unknowns': correct_matches_ele_inc,
            'elemental_total_including_unknowns': total_matches_ele_inc,
            'elemental_correct_excluding_unknowns': correct_matches_ele_exc,
            'elemental_total_excluding_unknowns': total_matches_ele_exc,
        },
    }

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
