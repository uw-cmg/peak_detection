import re
import numpy as np


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


def calculate_iou(range1, range2):
    """Calculates Intersection over Union for two range dicts with 'start'/'end' keys."""
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
