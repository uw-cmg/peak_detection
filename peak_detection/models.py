"""Dataclass definitions for peak detection data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetailedId:
    """RF model's top-2 predictions for a peak."""
    el1: str = ''
    conf1: float = 0.0
    el2: str = ''
    conf2: float = 0.0


@dataclass
class PeakRange:
    """A detected or truth peak range."""
    start: float
    end: float
    pos: float = 0.0
    label: str = ''
    id_score: float = 0.0
    method: str = ''
    detailed_id: DetailedId | None = None
    is_unknown: bool = False


@dataclass
class DatasetStats:
    """Result of process_dataset()."""
    dataset: str
    config: str = 'YOLO 1D Model'
    true_peaks_count: int = 0
    predicted_peaks_count: int = 0
    found_peaks_count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_min_mc: float = 0.0
    true_max_mc: float = 0.0
    pred_min_mc: float = 0.0
    pred_max_mc: float = 0.0
    rf_accuracy: float = 0.0
    rf_accuracy_ele: float = 0.0
    rf_species_total: int = 0
    rf_species_correct: int = 0
    rf_elemental_total: int = 0
    rf_elemental_correct: int = 0
    rf_molecular_total: int = 0
    rf_molecular_correct: int = 0
    rf_species_total_exc: int = 0
    rf_species_correct_exc: int = 0
    rf_elemental_total_exc: int = 0
    rf_elemental_correct_exc: int = 0
    rf_molecular_total_exc: int = 0
    rf_molecular_correct_exc: int = 0
    # Optional: metrics before molecule-rescue overrides are applied
    rf_species_total_before: int = 0
    rf_species_correct_before: int = 0
    rf_elemental_total_before: int = 0
    rf_elemental_correct_before: int = 0
    rf_molecular_total_before: int = 0
    rf_molecular_correct_before: int = 0
    rf_species_total_before_exc: int = 0
    rf_species_correct_before_exc: int = 0
    rf_elemental_total_before_exc: int = 0
    rf_elemental_correct_before_exc: int = 0
    rf_molecular_total_before_exc: int = 0
    rf_molecular_correct_before_exc: int = 0
    molecule_rescue_considered: int = 0
    molecule_rescue_overrides: int = 0
    molecule_rescue_mixed_candidates: int = 0
    unknown_count: int = 0
    unknown_count_with_truth: int = 0
    unknown_count_no_truth: int = 0
    predicted_peaks_with_truth: int = 0
    predicted_peaks_no_truth: int = 0
    identifications: list = field(default_factory=list)
    detected_ranges: list = field(default_factory=list)
    x: np.ndarray | None = None
    spectrum: np.ndarray | None = None
    truth: list = field(default_factory=list)
