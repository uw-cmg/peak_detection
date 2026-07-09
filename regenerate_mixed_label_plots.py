"""Regenerate saved YOLO comparison PNGs with current mixed-label plotting.

This utility rebuilds plot inputs from the saved per-dataset
``*_detailed_results.csv`` files, plus the original APT/CSV and RRNG files.
It intentionally does not rerun YOLO or RF classification.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd

from detect_peaks_refactor import match_datasets, plot_yolo_comparison
from peak_detection.data_io import load_apt_from_file, parse_rrng
from peak_detection.models import DatasetStats, DetailedId, PeakRange


def _clean_label(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _clean_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _rebuild_detected_ranges(detailed_csv: Path) -> list[PeakRange]:
    df = pd.read_csv(detailed_csv)
    ranges: list[PeakRange] = []
    for _, row in df.iterrows():
        start = _clean_float(row.get("predicted peak start"))
        end = _clean_float(row.get("predicted peak end"))
        label1 = _clean_label(row.get("pred element label 1"))
        label2 = _clean_label(row.get("pred element label 2"))
        display_label = _clean_label(row.get("pred display label")) or label1
        conf1 = _clean_float(row.get("pred confidence 1"))
        conf2 = _clean_float(row.get("pred confidence 2"))
        is_unknown = str(row.get("discarded", "")).strip().lower() in {"true", "1", "yes"}
        ranges.append(
            PeakRange(
                start=start,
                end=end,
                pos=(start + end) / 2,
                label=display_label,
                detailed_id=DetailedId(el1=label1, conf1=conf1, el2=label2, conf2=conf2),
                is_unknown=is_unknown,
            )
        )
    return ranges


def regenerate_plots(results_dir: Path, apt_dir: Path, rrng_dir: Path) -> int:
    matches = {prefix: (Path(apt), Path(rrng)) for apt, rrng, prefix in match_datasets(apt_dir, rrng_dir)}
    regenerated = 0

    for dataset_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        detailed_csv = dataset_dir / f"{dataset}_detailed_results.csv"
        if not detailed_csv.exists():
            continue
        if dataset not in matches:
            print(f"[skip] No APT/RRNG match for {dataset}")
            continue

        apt_file, rrng_file = matches[dataset]
        x, _, spectrum_log = load_apt_from_file(str(apt_file))
        truth = parse_rrng(str(rrng_file))
        detected = _rebuild_detected_ranges(detailed_csv)
        stats = DatasetStats(
            dataset=dataset,
            x=x,
            spectrum=spectrum_log.numpy(),
            truth=truth,
            detected_ranges=detected,
        )

        full_path = dataset_dir / f"{dataset}_yolo_1d_model_comparison.png"
        plot_yolo_comparison(stats, save_path=str(full_path))
        regenerated += 1

        for lo, hi in zip([0, 25, 50, 75, 100], [25, 50, 75, 100, 125]):
            zoom_path = dataset_dir / f"{dataset}_yolo_1d_model_comparison_zoom_{lo}_{hi}.png"
            if zoom_path.exists():
                plot_yolo_comparison(stats, xlim=(float(lo), float(hi)), save_path=str(zoom_path))
                regenerated += 1

    return regenerated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--apt_dir", default="ALL_APT_processedCSV")
    parser.add_argument("--rrng_dir", default="ALL_RRNG_NEW")
    args = parser.parse_args()

    count = regenerate_plots(Path(args.results_dir), Path(args.apt_dir), Path(args.rrng_dir))
    print(f"Regenerated {count} comparison PNGs.")


if __name__ == "__main__":
    main()
