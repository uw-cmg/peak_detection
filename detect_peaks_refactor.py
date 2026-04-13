"""
detect_peaks_gg2.py — Clean CLI/notebook entry point for peak detection.

Usage:
    # Single dataset
    python detect_peaks_gg2.py --apt_path singletest --rrng_path RRNG_test

    # Callable from Python
    from detect_peaks_gg2 import process_dataset
    stats = process_dataset('data.csv', 'data.RRNG')
"""

import os
import sys
import re
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on path for peak_detection package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from peak_detection.models import DatasetStats
from peak_detection.data_io import load_apt_from_file, parse_rrng, extract_elements_from_rrng, save_rrng
from peak_detection.utils import calculate_iou, calculate_metrics
from peak_detection.yolo_detection import predict_peak_ranges_yolo, identify_peaks


def plot_yolo_comparison(stats, xlim=None, save_path=None, facecolor=None):
    """
    Plot YOLO prediction vs truth ranges on top of the spectrum.

    Parameters
    ----------
    stats : DatasetStats
        The DatasetStats returned by `process_dataset`.
    xlim : tuple, optional
        (xmin, xmax) to zoom into a region.
    save_path : str, optional
        If provided, save the figure to this path. Otherwise call plt.show()
        for interactive viewing.
    facecolor : str, optional
        Background color of the plot. If None, uses the matplotlib default.
    """
    x = stats.x
    y_mapped = stats.spectrum
    truth = stats.truth
    detected = stats.detected_ranges
    dataset = stats.dataset

    # Compute default x-axis upper bound
    true_max = max(t.end for t in truth) if truth else 0
    pred_max = max(p.end for p in detected) if detected else 0
    plot_xmax = max(true_max, pred_max) + 5

    fig, ax = plt.subplots(figsize=(15, 8))
    if facecolor is not None:
        fig.patch.set_facecolor(facecolor)
        ax.set_facecolor(facecolor)
        # Override dark-theme text colors so labels are visible
        text_color = 'black' if facecolor in ('white', 'w', '#ffffff', '#fff') else 'white'
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
    plt.plot(x, y_mapped, color='black', alpha=0.3, label='Mapped Spectrum (map01)')

    # Plot true ranges (blue)
    for i, r in enumerate(truth):
        if xlim and (r.end < xlim[0] or r.start > xlim[1]):
            continue
        plt.axvspan(r.start, r.end, color='blue', alpha=0.15)
        if i == 0:
            plt.axvspan(r.start, r.end, color='blue', alpha=0.15, label='Real (RRNG)')
        if r.label:
            center = (r.start + r.end) / 2
            plt.text(center, 0.85, r.label, color='blue', fontsize=6,
                     ha='center', va='bottom', rotation=90, alpha=0.7)

    # Plot predicted ranges (red)
    for i, p in enumerate(detected):
        if xlim and (p.end < xlim[0] or p.start > xlim[1]):
            continue
        plt.axvspan(p.start, p.end, color='red', alpha=0.3, hatch='//')
        if i == 0:
            plt.axvspan(p.start, p.end, color='red', alpha=0.3, hatch='//', label='YOLO Prediction')
        if p.label:
            center = (p.start + p.end) / 2
            plt.text(center, 0.95, p.label, color='darkred', fontsize=6,
                     ha='center', va='bottom', rotation=90, alpha=0.8)

    plt.xlabel('Mass/Charge Ratio (Da)')
    plt.ylabel('Mapped Intensity (0-1)')
    zoom_suffix = f" (Zoom {xlim[0]}-{xlim[1]})" if xlim else ""
    plt.title(f'YOLO Comparison: {dataset}{zoom_suffix}')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.2)

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(0, plot_xmax)

    if save_path:
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
        print(f"Saved comparison plot to {save_path}")
        plt.close('all')
    else:
        plt.show()


def process_dataset(
    apt_file: str,
    rrng_file: str,
    output_dir: str = None,
    *,
    # YOLO parameters
    yolo_weights: str = 'best_v0_2025-11-12.pt',
    n_iter: int = 0,
    iou: float = 0.01,
    conf: float = 0.05,
    max_det: int = 2000,
    mc_min: float = 0.0,
    mc_max: float = 307.2,
    # RF parameters
    training_path: str = None,
    training_num_files: int = 10000,
    augment_molecule_training_charge_ratios: bool = False,
    include_molecules: bool = False,
    use_neighborhood: bool = False,
    neighbor_threshold: float = 2.0,
    use_signature: bool = False,
    unknown_molecule_rf: bool = False,
    unknown_molecule_rf_threshold: float = 0.8,
    followon_mc_vector_rf: bool = False,
    followon_mc_vector_round_decimals: int = 3,
    # Unknown flagging
    flag_unknowns: bool = True,
    mc_threshold: float = 0.2,
    # Output control
    save_plots: bool = True,
    save_rrng_output: bool = False,
    save_csv: bool = True,
    xlim: tuple = None,
) -> DatasetStats:
    """
    Process a single APT dataset: detect peaks with YOLO, classify with RF, and evaluate.

    Returns a DatasetStats with metrics, detected_ranges, identifications, etc.
    """
    rf_accuracy = 0.0
    rf_accuracy_ele = 0.0
    unknown_count = 0

    if output_dir is None:
        output_dir = os.path.splitext(os.path.basename(apt_file))[0].lower()
        output_dir = re.sub(r'[^a-zA-Z0-9]', '_', output_dir).strip('_')
        output_dir = re.sub(r'_+', '_', output_dir)

    print(f"\nDetecting peaks for {output_dir} (Zoom: {xlim})...")
    x, spectrum, spectrum_log = load_apt_from_file(apt_file)

    y_mapped = spectrum_log.numpy()

    truth = parse_rrng(rrng_file)

    # Save true species and RF elements to files
    truth_species = sorted(list(set([t.label for t in truth if t.label and t.label != 'Unknown'])))
    elements_for_molecules = extract_elements_from_rrng(rrng_file)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{output_dir}_rf_elements.txt"), 'w') as f:
        f.write("--- Suggested RF Classes (Species) ---\n")
        f.write("\n".join(truth_species))
        f.write("\n\n--- Base Elements for Permutations ---\n")
        f.write("\n".join(sorted(elements_for_molecules)))

    with open(os.path.join(output_dir, f"{output_dir}_true_species.txt"), 'w') as f:
        f.write("\n".join(truth_species))

    print(f"  Metadata saved: {output_dir}/{output_dir}_rf_elements.txt, {output_dir}/{output_dir}_true_species.txt")

    # --- RF ELEMENT IDENTIFICATION ---
    all_predicted, _, rf_accuracy, rf_accuracy_ele, unknown_count, rf_accuracy_breakdown = predict_peak_ranges_yolo(
        apt_file, spectrum_log, x, rrng_file,
        n_iter=n_iter, prefix=output_dir,
        flag_unknowns=flag_unknowns,
        mc_threshold=mc_threshold,
        training_path=training_path, training_num_files=training_num_files, include_molecules=include_molecules,
        augment_molecule_training_charge_ratios=augment_molecule_training_charge_ratios,
        yolo_weights=yolo_weights, iou=iou, conf=conf, max_det=max_det,
        mc_min=mc_min, mc_max=mc_max,
        use_neighborhood=use_neighborhood, neighbor_threshold=neighbor_threshold,
        use_signature=use_signature,
        unknown_molecule_rf=unknown_molecule_rf,
        molecule_rf_threshold=unknown_molecule_rf_threshold,
        followon_mc_vector_rf=followon_mc_vector_rf,
        followon_mc_vector_round_decimals=followon_mc_vector_round_decimals,
        return_accuracy_breakdown=True,
    )

    # --- RF ACCURACY OUTPUT ---
    if rf_accuracy_breakdown and 'counts' in rf_accuracy_breakdown:
        c = rf_accuracy_breakdown['counts']
        print(
            "  RF Accuracy (All species): "
            f"{rf_accuracy_breakdown['species_including_unknowns']:.1f}% "
            f"({c.get('species_correct_including_unknowns', 0)}/{c.get('species_total_including_unknowns', 0)}) including unknowns, "
            f"{rf_accuracy_breakdown['species_excluding_unknowns']:.1f}% "
            f"({c.get('species_correct_excluding_unknowns', 0)}/{c.get('species_total_excluding_unknowns', 0)}) excluding unknowns"
        )
        print(
            "  RF Accuracy (Elemental only): "
            f"{rf_accuracy_breakdown['elemental_including_unknowns']:.1f}% "
            f"({c.get('elemental_correct_including_unknowns', 0)}/{c.get('elemental_total_including_unknowns', 0)}) including unknowns, "
            f"{rf_accuracy_breakdown['elemental_excluding_unknowns']:.1f}% "
            f"({c.get('elemental_correct_excluding_unknowns', 0)}/{c.get('elemental_total_excluding_unknowns', 0)}) excluding unknowns"
        )
    else:
        print(f"  RF Accuracy (All species): {rf_accuracy:.1f}%")
        print(f"  RF Accuracy (Elemental only): {rf_accuracy_ele:.1f}%")

    rf_species_total = 0
    rf_species_correct = 0
    rf_elemental_total = 0
    rf_elemental_correct = 0
    rf_molecular_total = 0
    rf_molecular_correct = 0
    rf_species_total_exc = 0
    rf_species_correct_exc = 0
    rf_elemental_total_exc = 0
    rf_elemental_correct_exc = 0
    rf_molecular_total_exc = 0
    rf_molecular_correct_exc = 0
    if rf_accuracy_breakdown and 'counts' in rf_accuracy_breakdown:
        c = rf_accuracy_breakdown.get('counts', {}) or {}
        rf_species_total = int(c.get('species_total_including_unknowns', 0) or 0)
        rf_species_correct = int(c.get('species_correct_including_unknowns', 0) or 0)
        rf_elemental_total = int(c.get('elemental_total_including_unknowns', 0) or 0)
        rf_elemental_correct = int(c.get('elemental_correct_including_unknowns', 0) or 0)
        rf_molecular_total = max(0, rf_species_total - rf_elemental_total)
        rf_molecular_correct = max(0, rf_species_correct - rf_elemental_correct)

        rf_species_total_exc = int(c.get('species_total_excluding_unknowns', 0) or 0)
        rf_species_correct_exc = int(c.get('species_correct_excluding_unknowns', 0) or 0)
        rf_elemental_total_exc = int(c.get('elemental_total_excluding_unknowns', 0) or 0)
        rf_elemental_correct_exc = int(c.get('elemental_correct_excluding_unknowns', 0) or 0)
        rf_molecular_total_exc = max(0, rf_species_total_exc - rf_elemental_total_exc)
        rf_molecular_correct_exc = max(0, rf_species_correct_exc - rf_elemental_correct_exc)

    detected1 = all_predicted

    # --- ACCURACY ASSESSMENT ---
    pc, rc, f1c = calculate_metrics(truth, all_predicted)
    print(f"  Total Combined Metrics: Precision={pc:.3f}, Recall={rc:.3f}, F1={f1c:.3f}")

    # Calculate final found peaks (TP)
    tp_count = 0
    if len(truth) > 0 and len(all_predicted) > 0:
        matched_truth = set()
        for p in all_predicted:
            for i, t in enumerate(truth):
                if calculate_iou(p, t) > 0.1:
                    matched_truth.add(i)
        tp_count = len(matched_truth)

    # Split unknowns by whether the predicted range matches any truth range.
    pred_with_truth = 0
    pred_no_truth = 0
    unknown_with_truth = 0
    unknown_no_truth = 0
    if len(all_predicted) > 0:
        for p in all_predicted:
            best_iou = 0.0
            for t in truth:
                iou_val = calculate_iou(p, t)
                if iou_val > best_iou:
                    best_iou = iou_val
            has_truth = best_iou > 0.1
            if has_truth:
                pred_with_truth += 1
                if getattr(p, 'is_unknown', False):
                    unknown_with_truth += 1
            else:
                pred_no_truth += 1
                if getattr(p, 'is_unknown', False):
                    unknown_no_truth += 1

    # Calculate min/max mass ranges
    true_min = min([t.start for t in truth]) if truth else 0
    true_max = max([t.end for t in truth]) if truth else 0
    pred_min = min([p.start for p in all_predicted]) if all_predicted else 0
    pred_max = max([p.end for p in all_predicted]) if all_predicted else 0
    plot_xmax = max(true_max, pred_max) + 5

    # --- IDENTIFICATION ---
    identified_peaks = identify_peaks(all_predicted, x, spectrum_log, allowed_elements=elements_for_molecules)

    stats = DatasetStats(
        dataset=output_dir,
        true_peaks_count=len(truth),
        predicted_peaks_count=len(all_predicted),
        found_peaks_count=tp_count,
        precision=pc,
        recall=rc,
        f1=f1c,
        true_min_mc=true_min,
        true_max_mc=true_max,
        pred_min_mc=pred_min,
        pred_max_mc=pred_max,
        rf_accuracy=round(rf_accuracy, 2),
        rf_accuracy_ele=round(rf_accuracy_ele, 2),
        rf_species_total=rf_species_total,
        rf_species_correct=rf_species_correct,
        rf_elemental_total=rf_elemental_total,
        rf_elemental_correct=rf_elemental_correct,
        rf_molecular_total=rf_molecular_total,
        rf_molecular_correct=rf_molecular_correct,
        rf_species_total_exc=rf_species_total_exc,
        rf_species_correct_exc=rf_species_correct_exc,
        rf_elemental_total_exc=rf_elemental_total_exc,
        rf_elemental_correct_exc=rf_elemental_correct_exc,
        rf_molecular_total_exc=rf_molecular_total_exc,
        rf_molecular_correct_exc=rf_molecular_correct_exc,
        unknown_count=unknown_count,
        unknown_count_with_truth=unknown_with_truth,
        unknown_count_no_truth=unknown_no_truth,
        predicted_peaks_with_truth=pred_with_truth,
        predicted_peaks_no_truth=pred_no_truth,
        identifications=identified_peaks,
        detected_ranges=all_predicted,
        x=x,
        spectrum=y_mapped,
        truth=truth,
    )

    # --- SAVE PEAK RANGES ---
    if xlim is None:
        results_file = os.path.join(output_dir, f"{output_dir}_peak_ranges.txt")
        with open(results_file, 'w') as f:
            f.write("peak_start, peak_end, round, peak_pos\n")
            for p in detected1:
                f.write(f"{p.start:.4f}, {p.end:.4f}, 1, {p.pos:.4f}\n")
        print(f"Ranges saved to {results_file}")

    # --- SAVE RRNG ---
    if save_rrng_output:
        rrng_out_path = os.path.join(output_dir, f"{output_dir}_predicted.RRNG")
        save_rrng(rrng_out_path, all_predicted)
        print(f"Predicted RRNG saved to {rrng_out_path}")

    # --- PLOT ---
    if save_plots:
        if xlim is None:
            print(f"Manual RRNG ranges: {len(truth)}")
        zoom_str = f"_zoom_{xlim[0]}_{xlim[1]}" if xlim else ""
        comp_plot_path = os.path.join(output_dir, f"{output_dir}_yolo_1d_model_comparison{zoom_str}.png")
        plot_yolo_comparison(stats, xlim=xlim, save_path=comp_plot_path)

        # Also save additional comparison plots sliced into fifths (0-25, 25-50, ...)
        # This runs only for the full-range plot to avoid generating slices of slices.
        if xlim is None:
            slice_edges = [0, 25, 50, 75, 100, 125]
            for lo, hi in zip(slice_edges, slice_edges[1:]):
                if float(lo) >= float(plot_xmax):
                    continue
                slice_path = os.path.join(output_dir, f"{output_dir}_yolo_1d_model_comparison_zoom_{lo}_{hi}.png")
                plot_yolo_comparison(stats, xlim=(float(lo), float(hi)), save_path=slice_path)

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
        match = re.search(r'R\d+_\d+', filename)
        if match:
            return match.group(0)
        norm = normalize(filename.split('.')[0])
        if 'puresi' in norm:
            return 'puresi'
        if 'tinsio' in norm:
            return 'tinsio'
        if 'uwpid' in norm:
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
            prefix = re.sub(r'[^a-zA-Z0-9]', '_', cf.split('.')[0]).lower()
            prefix = re.sub(r'_+', '_', prefix).strip('_')
            matches.append((os.path.join(csv_dir, cf), os.path.join(rrng_dir, best_match), prefix))

    return matches


def run_batch(csv_dir, rrng_dir, *, save_plots=True, save_csv=True, **kwargs):
    """
    Run process_dataset on all matched datasets in the given directories.
    Returns list of stats dicts.
    """
    items_to_process = match_datasets(csv_dir, rrng_dir)
    print(f"Found {len(items_to_process)} matched datasets.")

    all_stats = []

    for apt_file, rrng_file, base_prefix in items_to_process:
        print(f"\n==================== DATASET: {base_prefix.upper()} ====================")
        try:
            stats = process_dataset(
                apt_file, rrng_file, base_prefix,
                save_plots=save_plots,
                save_csv=save_csv,
                **kwargs
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"  [Error] Failed to process {base_prefix}: {e}")

    return all_stats


def plot_rf_accuracy_summary(all_stats, output_path="rf_accuracy_vs_dataset.png"):
    """Generates a summary plot for RF accuracy across datasets."""
    if not all_stats:
        return
    all_stats = sorted(all_stats, key=lambda x: x.dataset)
    datasets = [s.dataset for s in all_stats]
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in datasets]
    overall_acc = [s.rf_accuracy for s in all_stats]
    elemental_acc = [s.rf_accuracy_ele for s in all_stats]

    unk_frac_truth = []
    unk_frac_no_truth = []
    for s in all_stats:
        pred_count = s.predicted_peaks_count or 1
        unk_frac_truth.append((getattr(s, 'unknown_count_with_truth', 0) or 0) / pred_count)
        unk_frac_no_truth.append((getattr(s, 'unknown_count_no_truth', 0) or 0) / pred_count)

    avg_overall = np.mean(overall_acc)
    avg_elemental = np.mean(elemental_acc)
    avg_unk_truth = np.mean(unk_frac_truth)
    avg_unk_no_truth = np.mean(unk_frac_no_truth)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(display_names, overall_acc, marker='o', color='black', label=f'RF Accuracy Overall (Avg: {avg_overall:.1f}%)', linewidth=1.5)
    ax1.plot(display_names, elemental_acc, marker='o', color='blue', label=f'RF Accuracy Elemental (Avg: {avg_elemental:.1f}%)', linewidth=1.5)

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('RF Accuracy (%)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(-5, 105)
    plt.xticks(rotation=90, ha='center', fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(
        display_names, unk_frac_truth, marker='o', color='grey',
        label=f'Unknown Fraction (matched truth, Avg: {avg_unk_truth:.3f})',
        linewidth=1.5
    )
    ax2.plot(
        display_names, unk_frac_no_truth, marker='o', color='lightgrey',
        label=f'Unknown Fraction (no truth match, Avg: {avg_unk_no_truth:.3f})',
        linewidth=1.5, linestyle='--'
    )
    ax2.set_ylabel('Fraction of Unknowns', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    ax2.set_ylim(-0.05, 1.05)

    plt.title('RF Identification Accuracy and Unknown Peak Fraction (Truth-Matched vs Unmatched)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved RF accuracy summary plot to {output_path}")
    plt.close()


def plot_rf_counts_summary(all_stats, output_path="rf_counts_vs_dataset.png"):
    """Plot RF classification totals/correct counts across datasets (elemental-only and overall species)."""
    if not all_stats:
        return
    all_stats = sorted(all_stats, key=lambda x: x.dataset)
    datasets = [s.dataset for s in all_stats]
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in datasets]

    true_ele = [int(getattr(s, 'rf_elemental_total', 0) or 0) for s in all_stats]
    corr_ele = [int(getattr(s, 'rf_elemental_correct', 0) or 0) for s in all_stats]
    true_all = [int(getattr(s, 'rf_species_total', 0) or 0) for s in all_stats]
    corr_all = [int(getattr(s, 'rf_species_correct', 0) or 0) for s in all_stats]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(display_names, true_ele, marker='o', color='blue', label='True elemental (count)', linewidth=1.5)
    ax.plot(display_names, corr_ele, marker='o', color='navy', label='Correct elemental (count)', linewidth=1.5)
    ax.plot(display_names, true_all, marker='o', color='black', label='True overall (count)', linewidth=1.5)
    ax.plot(display_names, corr_all, marker='o', color='green', label='Correct overall (count)', linewidth=1.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Count')
    plt.xticks(rotation=90, ha='center', fontsize=8)
    ax.set_ylim(bottom=-1)
    ax.grid(True, alpha=0.2)
    plt.title('RF Classification Counts (True vs Correct)')
    ax.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved RF counts summary plot to {output_path}")
    plt.close()


def plot_rf_element_counts_summary(all_stats, output_path="rf_element_counts_vs_dataset.png"):
    """Plot RF elemental-only true/correct counts across datasets."""
    if not all_stats:
        return
    all_stats = sorted(all_stats, key=lambda x: x.dataset)
    datasets = [s.dataset for s in all_stats]
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in datasets]

    true_ele = [int(getattr(s, 'rf_elemental_total', 0) or 0) for s in all_stats]
    corr_ele = [int(getattr(s, 'rf_elemental_correct', 0) or 0) for s in all_stats]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(display_names, true_ele, marker='o', color='blue', label='True elemental (count)', linewidth=1.5)
    ax.plot(display_names, corr_ele, marker='o', color='navy', label='Correct elemental (count)', linewidth=1.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Count')
    plt.xticks(rotation=90, ha='center', fontsize=8)
    ax.set_ylim(bottom=-1)
    ax.grid(True, alpha=0.2)
    plt.title('RF Elemental Classification Counts (True vs Correct)')
    ax.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved RF elemental counts summary plot to {output_path}")
    plt.close()


def plot_rf_molecule_counts_summary(all_stats, output_path="rf_molecule_counts_vs_dataset.png"):
    """Plot RF molecule-only true/correct counts across datasets."""
    if not all_stats:
        return
    all_stats = sorted(all_stats, key=lambda x: x.dataset)
    datasets = [s.dataset for s in all_stats]
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in datasets]

    true_mol = [int(getattr(s, 'rf_molecular_total', 0) or 0) for s in all_stats]
    corr_mol = [int(getattr(s, 'rf_molecular_correct', 0) or 0) for s in all_stats]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(display_names, true_mol, marker='o', color='black', label='True molecules (count)', linewidth=1.5)
    ax.plot(display_names, corr_mol, marker='o', color='purple', label='Correct molecules (count)', linewidth=1.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Count')
    plt.xticks(rotation=90, ha='center', fontsize=8)
    ax.set_ylim(bottom=-1)
    ax.grid(True, alpha=0.2)
    plt.title('RF Molecular Classification Counts (True vs Correct)')
    ax.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved RF molecular counts summary plot to {output_path}")
    plt.close()


def plot_rf_element_accuracy_pct_summary(all_stats, output_path="rf_element_accuracy_pct_vs_dataset.png"):
    """
    Plot RF elemental-only correctness percentage across datasets (correct/true * 100),
    excluding unknown predictions from the denominator. Also overlays unknown fraction.
    """
    if not all_stats:
        return
    all_stats = sorted(all_stats, key=lambda x: x.dataset)
    datasets = [s.dataset for s in all_stats]
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in datasets]

    totals_inc = [int(getattr(s, 'rf_elemental_total', 0) or 0) for s in all_stats]
    totals_exc = [int(getattr(s, 'rf_elemental_total_exc', 0) or 0) for s in all_stats]
    corrects_exc = [int(getattr(s, 'rf_elemental_correct_exc', 0) or 0) for s in all_stats]

    pct = [(c / t * 100.0) if t > 0 else 0.0 for c, t in zip(corrects_exc, totals_exc)]
    pct_for_avg = [p for p, t in zip(pct, totals_exc) if t > 0]
    avg_pct = float(np.mean(pct_for_avg)) if pct_for_avg else 0.0
    unk_frac = [((ti - te) / ti) if ti > 0 else 0.0 for ti, te in zip(totals_inc, totals_exc)]
    unk_for_avg = [u for u, ti in zip(unk_frac, totals_inc) if ti > 0]
    avg_unk = float(np.mean(unk_for_avg)) if unk_for_avg else 0.0

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        display_names, pct, marker='o', color='navy',
        label=f'Elemental correct (%) (Avg: {avg_pct:.1f}%)', linewidth=1.5
    )

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Correct (%)')
    plt.xticks(rotation=90, ha='center', fontsize=8)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.2)
    plt.title('RF Elemental Classification Accuracy (Excluding Unknowns)')

    ax2 = ax.twinx()
    ax2.plot(
        display_names, unk_frac, marker='o', color='lightgrey',
        label=f'Unknown fraction (Avg: {avg_unk:.3f})', linewidth=1.5, linestyle='--'
    )
    ax2.set_ylabel('Unknown fraction')
    ax2.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved RF elemental accuracy percent plot to {output_path}")
    plt.close()


def plot_rf_molecule_accuracy_pct_summary(all_stats, output_path="rf_molecule_accuracy_pct_vs_dataset.png"):
    """
    Plot RF molecule-only correctness percentage across datasets (correct/true * 100),
    excluding unknown predictions from the denominator. Also overlays unknown fraction.
    """
    if not all_stats:
        return
    all_stats = sorted(all_stats, key=lambda x: x.dataset)
    datasets = [s.dataset for s in all_stats]
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in datasets]

    totals_inc = [int(getattr(s, 'rf_molecular_total', 0) or 0) for s in all_stats]
    totals_exc = [int(getattr(s, 'rf_molecular_total_exc', 0) or 0) for s in all_stats]
    corrects_exc = [int(getattr(s, 'rf_molecular_correct_exc', 0) or 0) for s in all_stats]

    pct = [(c / t * 100.0) if t > 0 else 0.0 for c, t in zip(corrects_exc, totals_exc)]
    pct_for_avg = [p for p, t in zip(pct, totals_exc) if t > 0]
    avg_pct = float(np.mean(pct_for_avg)) if pct_for_avg else 0.0
    unk_frac = [((ti - te) / ti) if ti > 0 else 0.0 for ti, te in zip(totals_inc, totals_exc)]
    unk_for_avg = [u for u, ti in zip(unk_frac, totals_inc) if ti > 0]
    avg_unk = float(np.mean(unk_for_avg)) if unk_for_avg else 0.0

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        display_names, pct, marker='o', color='purple',
        label=f'Molecular correct (%) (Avg: {avg_pct:.1f}%)', linewidth=1.5
    )

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Correct (%)')
    plt.xticks(rotation=90, ha='center', fontsize=8)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.2)
    plt.title('RF Molecular Classification Accuracy (Excluding Unknowns)')

    ax2 = ax.twinx()
    ax2.plot(
        display_names, unk_frac, marker='o', color='lightgrey',
        label=f'Unknown fraction (Avg: {avg_unk:.3f})', linewidth=1.5, linestyle='--'
    )
    ax2.set_ylabel('Unknown fraction')
    ax2.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved RF molecular accuracy percent plot to {output_path}")
    plt.close()


def plot_yolo_metrics_summary(all_stats, output_path="yolo_metrics_vs_dataset.png"):
    """Generates a summary plot for YOLO metrics across datasets."""
    if not all_stats:
        return
    all_stats = sorted(all_stats, key=lambda x: x.dataset)
    datasets = [s.dataset for s in all_stats]
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in datasets]
    precision = [s.precision for s in all_stats]
    recall = [s.recall for s in all_stats]
    f1 = [s.f1 for s in all_stats]

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
    parser = argparse.ArgumentParser(description="Peak detection for APT data (v2).")
    parser.add_argument("--apt_path", type=str, default='ALL_APT_processedCSV',
                        help="Path to .apt/.csv file or directory for batch mode")
    parser.add_argument("--rrng_path", type=str, default='ALL_RRNG',
                        help="Path to .rrng file or directory for batch mode")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (single mode only; defaults to name from apt_file)")

    # YOLO parameters
    parser.add_argument("--yolo_weights", type=str, default='best_v0_2025-11-12.pt')
    parser.add_argument("--n_iter", type=int, default=0)
    parser.add_argument("--iou", type=float, default=0.01)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max_det", type=int, default=2000)
    parser.add_argument("--mc_min", type=float, default=0.0)
    parser.add_argument("--mc_max", type=float, default=307.2)

    # RF parameters
    parser.add_argument("--training_path", type=str,
                        default='peak_detection/Ionclassifier/training_data/NewData_peakshift0_noise0/Data0001')
    parser.add_argument("--training_num_files", type=int, default=10000,
                        help="Number of synthetic training CSV files to scan (default loads all 10k when present)")
    parser.add_argument(
        "--augment_molecule_training_charge_ratios",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Augment molecule training m/c values with charge-state ratios (adds 1/2 and 1/3 m/c samples for molecular species)",
    )
    parser.add_argument("--include_molecules", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_neighborhood", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--neighbor_threshold", type=float, default=2.0)
    parser.add_argument("--use_signature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unknown_molecule_rf", action=argparse.BooleanOptionalAction, default=False,
                        help="Train a molecule-only RF and apply it only to peaks flagged as unknown")
    parser.add_argument("--unknown_molecule_rf_threshold", type=float, default=0.8,
                        help="Min confidence for molecule-only RF to un-flag an unknown peak")
    parser.add_argument("--followon_mc_vector_rf", action=argparse.BooleanOptionalAction, default=False,
                        help="Run a follow-on RF using a padded vector of unique m/c values per predicted species-group")
    parser.add_argument("--followon_mc_vector_round_decimals", type=int, default=3,
                        help="Rounding decimals used when determining unique m/c values for the follow-on mc-vector RF")

    # Unknown flagging
    parser.add_argument("--flag_unknowns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mc_threshold", type=float, default=0.2)

    # Output control
    parser.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_rrng_output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_csv", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    apt_path = args.apt_path
    rrng_path = args.rrng_path

    if not os.path.exists(apt_path) or not os.path.exists(rrng_path):
        print(f"Error: Path not found:\n  APT: {apt_path}\n  RRNG: {rrng_path}")
        sys.exit(1)

    # Build common kwargs from args
    common_kwargs = {
        'yolo_weights': args.yolo_weights,
        'n_iter': args.n_iter,
        'iou': args.iou,
        'conf': args.conf,
        'max_det': args.max_det,
        'mc_min': args.mc_min,
        'mc_max': args.mc_max,
        'training_path': args.training_path,
        'training_num_files': args.training_num_files,
        'augment_molecule_training_charge_ratios': args.augment_molecule_training_charge_ratios,
        'include_molecules': args.include_molecules,
        'use_neighborhood': args.use_neighborhood,
        'neighbor_threshold': args.neighbor_threshold,
        'use_signature': args.use_signature,
        'unknown_molecule_rf': args.unknown_molecule_rf,
        'unknown_molecule_rf_threshold': args.unknown_molecule_rf_threshold,
        'followon_mc_vector_rf': args.followon_mc_vector_rf,
        'followon_mc_vector_round_decimals': args.followon_mc_vector_round_decimals,
        'flag_unknowns': args.flag_unknowns,
        'mc_threshold': args.mc_threshold,
        'save_plots': args.save_plots,
        'save_rrng_output': args.save_rrng_output,
        'save_csv': args.save_csv,
    }

    # Detect single-file vs batch mode
    is_single = os.path.isfile(apt_path)

    if is_single:
        # Single file mode
        stats = process_dataset(apt_path, rrng_path, output_dir=args.output_dir, **common_kwargs)
        print(f"\nDone. Results in {stats.dataset}/")
    else:
        # Batch mode
        print(f"Scanning for datasets in {apt_path}...")
        all_stats = run_batch(apt_path, rrng_path, **common_kwargs)

        if all_stats:
            # Save global summary statistics to CSV
            summary_file = "peak_detection_summary.csv"
            fieldnames = [
                'dataset', 'config', 'true_peaks_count', 'predicted_peaks_count',
                'found_peaks_count', 'precision', 'recall', 'f1',
                'true_min_mc', 'true_max_mc', 'pred_min_mc', 'pred_max_mc',
                'rf_accuracy', 'rf_accuracy_ele',
                'rf_species_total', 'rf_species_correct',
                'rf_elemental_total', 'rf_elemental_correct',
                'rf_molecular_total', 'rf_molecular_correct',
                'rf_species_total_exc', 'rf_species_correct_exc',
                'rf_elemental_total_exc', 'rf_elemental_correct_exc',
                'rf_molecular_total_exc', 'rf_molecular_correct_exc',
                'unknown_count', 'unknown_count_with_truth', 'unknown_count_no_truth',
                'predicted_peaks_with_truth', 'predicted_peaks_no_truth',
            ]
            with open(summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_stats:
                    csv_row = {k: getattr(row, k) for k in fieldnames}
                    writer.writerow(csv_row)

            # Aggregate identifications for YOLO model
            yolo_export = []
            for s in all_stats:
                for p in s.identifications:
                    yolo_export.append({
                        'dataset': s.dataset,
                        'mass_center': p.pos,
                        'mass_start': p.start,
                        'mass_end': p.end,
                        'identified_label': p.label
                    })

            if yolo_export:
                id_file = "yolo_identifications.csv"
                with open(id_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['dataset', 'mass_center', 'mass_start', 'mass_end', 'identified_label'])
                    writer.writeheader()
                    for row in yolo_export:
                        writer.writerow(row)
                print(f"Global YOLO Identifications saved to {id_file}")

            # Generate summary plots
            plot_rf_accuracy_summary(all_stats)
            plot_rf_counts_summary(all_stats)
            plot_rf_element_counts_summary(all_stats)
            plot_rf_molecule_counts_summary(all_stats)
            plot_rf_element_accuracy_pct_summary(all_stats)
            plot_rf_molecule_accuracy_pct_summary(all_stats)
            plot_yolo_metrics_summary(all_stats)

            print(f"\nBatch Processing Complete. Summary saved to {summary_file}")
        else:
            print("\nNo datasets were successfully processed.")


if __name__ == "__main__":
    main()
