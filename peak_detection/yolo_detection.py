import os
import csv
import numpy as np
import torch

from pymatgen.core import Composition

from .utils import calculate_iou, calculate_iou_1d
from .data_io import parse_rrng, extract_elements_from_rrng
from .training import load_ion_training_data, get_similar_elements, build_empirical_mc_distributions
from .rf_model import create_RF_model, run_RF_model
from .kde_model import KDECache, make_lookup_model, predict_lookup_model, suggest_unknown_candidates


def remove_peaks_and_patch(spectrum, detected_ranges, window=10):
    """
    Replaces detected peak ranges with the average of surrounding noise.
    """
    new_spectrum = spectrum.copy()
    for p in detected_ranges:
        start_idx = int(np.round(p['start'] * 100))
        end_idx = int(np.round(p['end'] * 100))

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


def identify_peaks(detected_ranges, x, spectrum_log, allowed_elements=None, flag_unknowns=True):
    """
    Assigns chemical labels to detected ranges by matching them against
    theoretical isotopic 'fingerprints' (mass patterns and relative abundances).
    """
    results = []
    y_exp = spectrum_log.numpy() if hasattr(spectrum_log, 'numpy') else spectrum_log

    sigma_guess = 0.05

    for p in detected_ranges:
        # Skip if already identified by RF or other method (and not Unknown)
        if p.get('label') and p.get('label') != 'Unknown':
            results.append(p)
            continue

        # Isotopic fingerprint matching is currently disabled (commented out in original)
        # Pass through as-is
        results.append(p)

    return results


def predict_peak_ranges_yolo(apt_file, spectrum_log, x_exp, rrng_file,
                             n_iter=0, prefix=None, flag_unknowns=True,
                             kde_threshold=0.25, use_mc_distance=False,
                             mc_threshold=0.2, training_path=None,
                             include_molecules=False, yolo_weights='best.pt',
                             iou=0.01, conf=0.05, max_det=2000,
                             mc_min=0.0, mc_max=307.2,
                             use_neighborhood=True, neighbor_threshold=2.0,
                             use_signature=True, kde_cache=None):
    """
    RangingNN YOLO model prediction wrapper.

    Parameters
    ----------
    kde_cache : KDECache, optional
        Shared KDE cache instance. If None, a local one is created.
    """
    import yaml
    from peak_detection.RangingNN.predictor import DetectionPredictor

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Local paths
    modelpath = os.path.join(base_dir, 'peak_detection', 'RangingNN', 'modelweights', yolo_weights)
    cfg_path = os.path.join(base_dir, 'peak_detection', 'RangingNN', 'cfg', 'prediction_args.yaml')

    if not os.path.exists(modelpath) or not os.path.exists(cfg_path):
        print(f"  [Error] YOLO model files not found at {modelpath}")
        return [], None

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['iou'] = iou
    cfg['conf'] = conf
    cfg['max_det'] = max_det

    # Initial prediction
    if spectrum_log.shape[0] < 30720:
        pad_size = 30720 - spectrum_log.shape[0]
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
            starts = ranges[:, 0] * multiplier
            ends = ranges[:, 1] * multiplier

            in_any_range = np.logical_or.reduce(
                (x1[:, None] > starts[None, :]) &
                (x1[:, None] < ends[None, :]),
                axis=1
            )

            idx_delete = x1[in_any_range]
            spectrum_log_mod = sp_padded.clone().numpy()

            mask_detections = np.isin(x1, idx_delete)
            mask_outside = (x1 < mc_min) | (x1 > mc_max)

            spectrum_log_mod[mask_detections | mask_outside] = 0.2
            spectrum_log_mod = torch.Tensor(spectrum_log_mod)

            predictor_mod = DetectionPredictor(modelpath, spectrum_log_mod[None, None, ...], save_dir='test_results', cfg=cfg)
            result_mod = predictor_mod()[0]
            peak_range_pred_mod = result_mod[:, :2].cpu()

            tol = 0.5
            for i in peak_range_pred_mod:
                start, end = float(i[0]), float(i[1])
                max_iou_val = 0.0
                min_dist = 1000
                for j in peak_range_pred.tolist():
                    start2, end2 = float(j[0]), float(j[1])
                    iou_val = calculate_iou_1d([start, end], [start2, end2])
                    if iou_val > max_iou_val:
                        max_iou_val = iou_val
                    dist = multiplier * abs(start - start2)
                    if dist < min_dist:
                        min_dist = dist

                if max_iou_val == 0.0 and min_dist > tol:
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

    # --- RF ELEMENT IDENTIFICATION ---
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
        training_data_path = os.path.join(base_dir, 'peak_detection', 'Ionclassifier', 'training_data', 'NewData', 'Data0001')
    else:
        training_data_path = training_path

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
            raw_elements, rf_confs, detailed_rf, peak_mcs = run_RF_model(
                formatted_results, x_exp, spectrum_log, scaler_rf, model_rf, target_decoder_rf,
                neighbor_threshold=eff_neighbor_threshold,
                use_signature=use_signature
            )

            # KDE unknown flagging
            if kde_cache is None:
                kde_cache = KDECache()

            kde_lookup_model = None
            mc_lookup_data = None
            if flag_unknowns:
                print(f"  Training KDE verification model using: {training_path if training_path else 'default'} (Molecules: {include_molecules})...")
                kde_lookup_model, mc_lookup_data = kde_cache.get_or_build(
                    training_path=training_path, include_molecules=include_molecules
                )

            # --- Unphysical Peak Filtering ---
            suggestions = []
            for i, (el, conf_val, det) in enumerate(zip(raw_elements, rf_confs, detailed_rf)):
                pred1_el = det.get('el1', '')
                is_physical = True
                if pred1_el and pred1_el != 'Unknown':
                    try:
                        comp = Composition(pred1_el)
                        element_obj = max(comp.elements, key=lambda e: e.atomic_mass)

                        mc_val = peak_mcs[i]  # m/c of highest-intensity bin in range

                        # 1. KDE Check (User-requested distance check)
                        if flag_unknowns and not use_mc_distance and kde_lookup_model is not None:
                            if pred1_el in kde_lookup_model:
                                conf_kde = np.exp(kde_lookup_model[pred1_el].score_samples(np.array([[mc_val]])))[0]
                                if conf_kde < kde_threshold:
                                    # Too far away, double check with predictions ranking
                                    pred_ions, confs_kde = predict_lookup_model(kde_lookup_model, np.array([[mc_val]]))
                                    is_physical = False
                                    suggestions.append({
                                        'mc': mc_val,
                                        'rf_pred': pred1_el,
                                        'kde_suggestions': pred_ions,
                                        'confs': confs_kde
                                    })
                        # 2. MC Distance Check
                        elif flag_unknowns and use_mc_distance and mc_lookup_data is not None:
                            if pred1_el in mc_lookup_data:
                                train_mcs = np.array(mc_lookup_data[pred1_el])
                                min_dist = np.min(np.abs(train_mcs - mc_val))
                                if min_dist > mc_threshold:
                                    is_physical = False
                                    # For suggestions, we'll still use the KDE ranker logic to find best alternatives
                                    pred_ions, confs_kde = predict_lookup_model(kde_lookup_model, np.array([[mc_val]]))
                                    suggestions.append({
                                        'mc': mc_val,
                                        'rf_pred': pred1_el,
                                        'kde_suggestions': pred_ions,
                                        'confs': confs_kde
                                    })
                    except Exception:
                        pass

                if pred1_el == 'Unknown' or not pred1_el:
                    is_physical = True

                if not is_physical and flag_unknowns:
                    formatted_results[i]['label'] = f'Unknown ({pred1_el})'
                    formatted_results[i]['id_score'] = 1.0
                    formatted_results[i]['method'] = 'RF (Filtered)'
                    formatted_results[i]['detailed_id'] = {'el1': 'Unknown', 'conf1': 1.0, 'el2': '', 'conf2': 0.0}
                else:
                    formatted_results[i]['label'] = el
                    formatted_results[i]['id_score'] = float(conf_val)
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
            formatted_results = identify_peaks(formatted_results, x_exp, spectrum_log,
                                               allowed_elements=elements_for_molecules,
                                               flag_unknowns=flag_unknowns)
    else:
        print("RF training failed or no data, falling back to isotopic pattern matching.")
        formatted_results = identify_peaks(formatted_results, x_exp, spectrum_log,
                                           allowed_elements=elements_for_molecules,
                                           flag_unknowns=flag_unknowns)

    # --- DETAILED CSV EXPORT ---
    detailed_rows = []
    for p in formatted_results:
        best_iou = 0
        best_truth = None
        for t in truth_data:
            iou_val = calculate_iou(p, t)
            if iou_val > best_iou:
                best_iou = iou_val
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

    detailed_rows = sorted(detailed_rows, key=lambda x: x['predicted peak start'])
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
    synthetic_data_dir = os.path.join(base_dir, 'peak_detection', 'Ionclassifier', 'training_data', 'NewData', 'Data0001')
    try:
        print("  Building empirical m/c distributions for Unknown peak suggestions...")
        empirical_stats = build_empirical_mc_distributions(path=synthetic_data_dir, num_files=500)

        similar_els = get_similar_elements(elements_for_molecules)

        for i, r in enumerate(detailed_rows):
            pred_el = r.get('pred element label 1', '')
            if pred_el == 'Unknown' or not pred_el:
                mc_center = (r['predicted peak start'] + r['predicted peak end']) / 2.0

                closest_valid_el = None
                min_geo_dist = float('inf')
                for j, other_r in enumerate(detailed_rows):
                    if i == j:
                        continue
                    other_pred = other_r.get('pred element label 1', '')
                    if other_pred and other_pred != 'Unknown':
                        other_center = (other_r['predicted peak start'] + other_r['predicted peak end']) / 2.0
                        dist = abs(mc_center - other_center)
                        if dist < min_geo_dist:
                            min_geo_dist = dist
                            closest_valid_el = other_pred

                candidates = suggest_unknown_candidates(
                    mc_center, empirical_stats, elements_for_molecules,
                    similar_els, local_element=closest_valid_el, top_k=5
                )
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
