import re
import numpy as np


EXCLUDED_TRUTH_LABELS = {
    # Non-physical / placeholder RRNG entries that should never affect evaluation or plots
    "Da133Na",
    "Da13Na",
    "Da15Na",
    "Da18Na",
    "Da19Na",
    "Da20Na",
    "Da22Na",
    "Da27Na",
    "Da29Na",
    "Da2HuNa",
    "Da34Na",
    "Da35Na",
    "Da36Na",
    "Da37Na",
    "Da38Na",
    "Da39Na",
    "Da40Na",
    "Da41Na",
    "Da46Na",
    "Da47Na",
    "Da49Na",
    "Da64Na",
    "Da79Na",
    "DaHiNaNo",
    "HuNa",
    "HuNaPoSi",
}


def is_excluded_truth_label(label: str) -> bool:
    """Returns True if a RRNG truth label should be ignored for plotting/statistics."""
    if not label:
        return False
    try:
        return simplify_label(str(label)) in EXCLUDED_TRUTH_LABELS
    except Exception:
        return str(label) in EXCLUDED_TRUTH_LABELS


def map01(ar):
    """Min-max normalize an array to [0, 1]."""
    return (ar - ar.min()) / (ar.max() - ar.min())


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
            counts = {}
            for sym, count in parts:
                counts[sym] = counts.get(sym, 0) + int(count)
            res = ""
            for sym in sorted(counts.keys()):
                c = counts[sym]
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


def calculate_iou(range1, range2):
    """Calculates Intersection over Union for two PeakRange objects (or any object with .start/.end)."""
    s1, e1 = range1.start, range1.end
    s2, e2 = range2.start, range2.end

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
                break  # Count each true peak at most once

    precision = len(matched_pred) / len(predicted) if len(predicted) > 0 else 0
    recall = len(matched_truth) / len(truth) if len(truth) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def is_molecule(label):
    """
    Returns True if the label represents a molecular species (multiple atoms or multiple types).
    e.g. "Fe" -> False, "Fe2" -> True, "FeO" -> True
    """
    if not label or label == "Unknown":
        return False

    # Normalize to simple chemical notation
    clean = simplify_label(label)

    # Regex to find atom symbols and their counts
    # Example: "Fe2O3" -> [('Fe', '2'), ('O', '3')]
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', clean)

    if len(parts) > 1:
        return True
    if len(parts) == 1:
        _, count_str = parts[0]
        count = int(count_str) if count_str else 1
        return count > 1

    return False
