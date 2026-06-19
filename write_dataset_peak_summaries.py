"""Write per-dataset text summaries from a completed peak-detection run."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


def _ratio(num: float, den: float) -> str:
    if den == 0 or pd.isna(den):
        return "n/a"
    return f"{num / den:.3f} ({100.0 * num / den:.1f}%)"


def _count(value) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _is_molecule_label(label: str) -> bool:
    label = str(label or "").strip()
    if not label or label == "Unknown":
        return False
    return bool(re.search(r"\d", label)) or len(re.findall(r"[A-Z][a-z]?", label)) > 1


def _load_detailed_counts(detailed_csv: Path) -> dict[str, int]:
    df = pd.read_csv(detailed_csv)
    truth = df["true element label"].fillna("").astype(str).str.strip()
    matched = truth.ne("") & truth.ne("Unknown")
    discarded = df["discarded"].astype(str).str.lower().isin({"true", "1", "yes"})

    is_molecule = truth.map(_is_molecule_label)
    is_element = matched & ~is_molecule
    is_molecule = matched & is_molecule

    return {
        "found_truth_rows": int(matched.sum()),
        "found_true_elements": int(is_element.sum()),
        "found_true_molecules": int(is_molecule.sum()),
        "unknown_true_elements": int((matched & is_element & discarded).sum()),
        "unknown_true_molecules": int((matched & is_molecule & discarded).sum()),
        "unknown_truth_matched": int((matched & discarded).sum()),
        "unknown_unmatched": int((~matched & discarded).sum()),
    }


def _write_one_summary(dataset_dir: Path, row: pd.Series) -> Path | None:
    dataset = dataset_dir.name
    detailed_csv = dataset_dir / f"{dataset}_detailed_results.csv"
    if not detailed_csv.exists():
        return None

    detailed = _load_detailed_counts(detailed_csv)

    true_peaks = _count(row.get("true_peaks_count"))
    predicted_peaks = _count(row.get("predicted_peaks_count"))
    found_peaks = _count(row.get("found_peaks_count"))

    found_true_elements = _count(row.get("rf_elemental_total"))
    found_true_molecules = _count(row.get("rf_molecular_total"))
    correct_elements = _count(row.get("rf_elemental_correct"))
    correct_molecules = _count(row.get("rf_molecular_correct"))
    found_true_elements_exc = _count(row.get("rf_elemental_total_exc"))
    found_true_molecules_exc = _count(row.get("rf_molecular_total_exc"))
    correct_elements_exc = _count(row.get("rf_elemental_correct_exc"))
    correct_molecules_exc = _count(row.get("rf_molecular_correct_exc"))

    # Prefer the canonical summary counts for totals; use detailed_results for
    # the element/molecule split of unknown truth-matched peaks.
    if found_true_elements == 0 and found_true_molecules == 0:
        found_true_elements = detailed["found_true_elements"]
        found_true_molecules = detailed["found_true_molecules"]
    if found_true_elements_exc == 0 and found_true_molecules_exc == 0:
        found_true_elements_exc = max(0, found_true_elements - detailed["unknown_true_elements"])
        found_true_molecules_exc = max(0, found_true_molecules - detailed["unknown_true_molecules"])

    unknown_true_elements = detailed["unknown_true_elements"]
    unknown_true_molecules = detailed["unknown_true_molecules"]
    unknown_truth_matched = _count(row.get("unknown_count_with_truth"))
    unknown_unmatched = _count(row.get("unknown_count_no_truth"))
    if unknown_truth_matched == 0:
        unknown_truth_matched = detailed["unknown_truth_matched"]
    if unknown_unmatched == 0:
        unknown_unmatched = detailed["unknown_unmatched"]

    output_path = dataset_dir / f"{dataset}_peak_summary.txt"
    lines = [
        f"Dataset: {dataset}",
        "",
        "Peak detection",
        f"  True peaks: {true_peaks}",
        f"  Predicted peaks: {predicted_peaks}",
        f"    Predicted / true: {_ratio(predicted_peaks, true_peaks)}",
        f"  Found peaks: {found_peaks}",
        f"    Found / true: {_ratio(found_peaks, true_peaks)}",
        f"    Found / predicted: {_ratio(found_peaks, predicted_peaks)}",
        "",
        "Found peak classification, including unknowns as wrong",
        f"  Found true element peaks: {found_true_elements}",
        f"    Fraction of found peaks: {_ratio(found_true_elements, found_peaks)}",
        f"  Found true molecule peaks: {found_true_molecules}",
        f"    Fraction of found peaks: {_ratio(found_true_molecules, found_peaks)}",
        f"  Correct element classifications: {correct_elements}",
        f"    Correct / found true element peaks: {_ratio(correct_elements, found_true_elements)}",
        f"  Correct molecule classifications: {correct_molecules}",
        f"    Correct / found true molecule peaks: {_ratio(correct_molecules, found_true_molecules)}",
        "",
        "Found peak classification, excluding unknown predictions",
        f"  Non-unknown found true element peaks: {found_true_elements_exc}",
        f"    Fraction of found non-unknown peaks: {_ratio(found_true_elements_exc, found_true_elements_exc + found_true_molecules_exc)}",
        f"  Non-unknown found true molecule peaks: {found_true_molecules_exc}",
        f"    Fraction of found non-unknown peaks: {_ratio(found_true_molecules_exc, found_true_elements_exc + found_true_molecules_exc)}",
        f"  Correct element classifications: {correct_elements_exc}",
        f"    Correct / non-unknown found true element peaks: {_ratio(correct_elements_exc, found_true_elements_exc)}",
        f"  Correct molecule classifications: {correct_molecules_exc}",
        f"    Correct / non-unknown found true molecule peaks: {_ratio(correct_molecules_exc, found_true_molecules_exc)}",
        "",
        "Unknown found peaks",
        f"  Unknown peaks with matched truth: {unknown_truth_matched}",
        f"    Fraction of found peaks: {_ratio(unknown_truth_matched, found_peaks)}",
        f"    Fraction of predicted peaks: {_ratio(unknown_truth_matched, predicted_peaks)}",
        f"  Unknown true element peaks: {unknown_true_elements}",
        f"    Fraction of found true element peaks: {_ratio(unknown_true_elements, found_true_elements)}",
        f"  Unknown true molecule peaks: {unknown_true_molecules}",
        f"    Fraction of found true molecule peaks: {_ratio(unknown_true_molecules, found_true_molecules)}",
        f"  Unknown predicted peaks with no matched truth: {unknown_unmatched}",
        f"    Fraction of predicted peaks: {_ratio(unknown_unmatched, predicted_peaks)}",
        "",
        "Notes",
        "  Found peaks are predicted peaks matched to a true RRNG range at IoU > 0.1.",
        "  Classification counts use the RF top-N scoring from peak_detection_summary.csv.",
        "  The including-unknowns section counts unknown predictions as wrong classifications.",
        "  The excluding-unknowns section removes unknown predictions from the classification denominators.",
        "  Overall unknown fractions in rf_accuracy_vs_dataset.png use predicted peaks as the denominator.",
        "  Unknown element/molecule split is based on the matched true label in detailed_results.csv.",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def write_dataset_peak_summaries(results_dir: Path) -> list[Path]:
    summary_csv = results_dir / "peak_detection_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    summary = pd.read_csv(summary_csv).set_index("dataset")
    written: list[Path] = []
    for dataset, row in summary.iterrows():
        dataset_dir = results_dir / str(dataset)
        if not dataset_dir.is_dir():
            continue
        path = _write_one_summary(dataset_dir, row)
        if path is not None:
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    written = write_dataset_peak_summaries(Path(args.results_dir))
    print(f"Wrote {len(written)} per-dataset peak summary text files.")


if __name__ == "__main__":
    main()
