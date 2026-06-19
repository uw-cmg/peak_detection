"""Write aggregate audit summary for element/molecule prediction types."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip().lower() in {"", "nan"}


def clean_label(value) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def simplify_label(label: str) -> str:
    label = clean_label(label)
    if not label or label == "Unknown":
        return ""
    if label.startswith("Unknown"):
        return "Unknown"
    return re.split(r"\(|,|\s", label)[0].strip()


def is_molecule(label: str) -> bool:
    label = simplify_label(label)
    if not label or label == "Unknown":
        return False
    return bool(re.search(r"\d", label)) or len(re.findall(r"[A-Z][a-z]?", label)) > 1


def is_element(label: str) -> bool:
    label = simplify_label(label)
    return bool(label and label != "Unknown" and not is_molecule(label))


def ratio(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{num}/{den} = {num / den:.3f} ({100.0 * num / den:.1f}%)"


def labels_match(true_label: str, pred_label: str) -> bool:
    return bool(simplify_label(true_label) and simplify_label(true_label) == simplify_label(pred_label))


def prediction_labels(row) -> list[str]:
    if str(row.get("discarded", "")).strip().lower() in {"true", "1", "yes"}:
        return []
    labels = []
    for col, conf_col in (("pred element label 1", "pred confidence 1"), ("pred element label 2", "pred confidence 2")):
        lab = simplify_label(row.get(col, ""))
        if not lab or lab == "Unknown":
            continue
        if col.endswith("2") and float(row.get(conf_col, 0.0) or 0.0) <= 0:
            continue
        labels.append(lab)
    return labels


def prediction_kind(labels: list[str]) -> str:
    if not labels:
        return "unknown"
    has_element = any(is_element(label) for label in labels)
    has_molecule = any(is_molecule(label) for label in labels)
    if has_element and has_molecule:
        return "mixed_element_molecule"
    if has_element:
        return "pure_element"
    if has_molecule:
        return "pure_molecule"
    return "other"


def dataset_from_path(path: Path, suffix: str) -> str:
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else path.parent.name


def match_detailed_row(detailed_by_dataset: dict[str, pd.DataFrame], dataset: str, start: float, end: float):
    df = detailed_by_dataset.get(dataset)
    if df is None or df.empty:
        return None
    d = (df["predicted peak start"].astype(float) - float(start)).abs() + (df["predicted peak end"].astype(float) - float(end)).abs()
    idx = d.idxmin()
    if float(d.loc[idx]) > 1e-3:
        return None
    return df.loc[idx]


def build_summary(results_dir: Path) -> str:
    summary_csv = results_dir / "peak_detection_summary.csv"
    summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
    detailed_files = sorted(results_dir.glob("*/*_detailed_results.csv"))
    detailed_by_dataset = {
        dataset_from_path(path, "_detailed_results.csv"): pd.read_csv(path)
        for path in detailed_files
    }

    rows = []
    for dataset, df in detailed_by_dataset.items():
        for _, row in df.iterrows():
            true_label = simplify_label(row.get("true element label", ""))
            if not true_label or true_label == "Unknown":
                continue
            labels = prediction_labels(row)
            kind = prediction_kind(labels)
            rows.append(
                {
                    "dataset": dataset,
                    "true_label": true_label,
                    "true_kind": "molecule" if is_molecule(true_label) else "element",
                    "pred_kind": kind,
                    "top1": labels[0] if labels else "",
                    "top2": labels[1] if len(labels) > 1 else "",
                    "correct_top1": labels_match(true_label, labels[0]) if labels else False,
                    "correct_top2": any(labels_match(true_label, label) for label in labels[:2]),
                }
            )
    pred_df = pd.DataFrame(rows)

    def pred_type_stats(kind: str, true_kind: str) -> tuple[int, int]:
        part = pred_df[pred_df["pred_kind"].eq(kind)]
        denom = len(part)
        correct = int((part["true_kind"].eq(true_kind) & part["correct_top2"]).sum())
        return correct, denom

    pure_element_correct, pure_element_total = pred_type_stats("pure_element", "element")
    pure_molecule_correct, pure_molecule_total = pred_type_stats("pure_molecule", "molecule")

    mixed = pred_df[pred_df["pred_kind"].eq("mixed_element_molecule")]
    mixed_total = len(mixed)
    mixed_correct_top2 = int(mixed["correct_top2"].sum())
    mixed_new_top2_correct = int((~mixed["correct_top1"] & mixed["correct_top2"]).sum())
    mixed_new_true_elements = int((mixed["true_kind"].eq("element") & ~mixed["correct_top1"] & mixed["correct_top2"]).sum())
    mixed_new_true_molecules = int((mixed["true_kind"].eq("molecule") & ~mixed["correct_top1"] & mixed["correct_top2"]).sum())

    unknown_truth_element = 0
    unknown_truth_molecule = 0
    unknown_truth_total = 0
    unknown_no_truth = 0
    truth_matched_total = 0
    predicted_total = 0
    predicted_no_truth_total = 0
    predicted_with_truth_total = 0
    for df in detailed_by_dataset.values():
        predicted_total += len(df)
        truth = df["true element label"].fillna("").astype(str).str.strip()
        matched = truth.ne("") & truth.ne("Unknown")
        discarded = df["discarded"].astype(str).str.lower().isin({"true", "1", "yes"})
        truth_matched_total += int(matched.sum())
        predicted_with_truth_total += int(matched.sum())
        predicted_no_truth_total += int((~matched).sum())
        unknown_truth_total += int((matched & discarded).sum())
        unknown_no_truth += int((~matched & discarded).sum())
        for label in truth[matched & discarded]:
            if is_molecule(label):
                unknown_truth_molecule += 1
            else:
                unknown_truth_element += 1

    context_files = sorted(results_dir.glob("*/*_context_rescore_overrides.csv"))
    context_rows = []
    for path in context_files:
        dataset = dataset_from_path(path, "_context_rescore_overrides.csv")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            detailed = match_detailed_row(detailed_by_dataset, dataset, row["peak_start"], row["peak_end"])
            true_label = simplify_label(detailed.get("true element label", "")) if detailed is not None else ""
            old_unknown = str(row.get("old_is_unknown", "")).strip().lower() in {"true", "1", "yes"}
            old_labels = [] if old_unknown else [simplify_label(row.get("old_top1", "")), simplify_label(row.get("old_top2", ""))]
            old_labels = [label for label in old_labels if label and label != "Unknown"]
            new_label = simplify_label(row.get("new_label", ""))
            old_correct = bool(true_label and any(labels_match(true_label, label) for label in old_labels[:2]))
            new_correct = bool(true_label and labels_match(true_label, new_label))
            context_rows.append(
                {
                    "dataset": dataset,
                    "true_label": true_label,
                    "old_is_unknown": old_unknown,
                    "override_reason": clean_label(row.get("override_reason", "")),
                    "old_correct": old_correct,
                    "new_correct": new_correct,
                    "made_newly_correct": (not old_correct) and new_correct,
                }
            )
    context_df = pd.DataFrame(context_rows)

    rescue_files = sorted(results_dir.glob("*/*_molecule_rescue_candidates.csv"))
    rescue_rows = []
    for path in rescue_files:
        dataset = dataset_from_path(path, "_molecule_rescue_candidates.csv")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            detailed = match_detailed_row(detailed_by_dataset, dataset, row["peak_start"], row["peak_end"])
            true_label = simplify_label(detailed.get("true element label", "")) if detailed is not None else ""
            ele = simplify_label(row.get("element_pred_simple", ""))
            mol = simplify_label(row.get("molecule_pred_simple", ""))
            action = clean_label(row.get("rescue_action", ""))
            top1_before_correct = bool(true_label and labels_match(true_label, ele))
            rescue_correct = bool(true_label and (labels_match(true_label, mol) or (action == "mixed_candidate" and labels_match(true_label, ele))))
            molecule_added_correct = bool(action == "mixed_candidate" and true_label and not labels_match(true_label, ele) and labels_match(true_label, mol))
            override_made_correct = bool(action == "override" and true_label and not labels_match(true_label, ele) and labels_match(true_label, mol))
            rescue_rows.append(
                {
                    "dataset": dataset,
                    "true_label": true_label,
                    "action": action,
                    "element_before_correct": top1_before_correct,
                    "rescue_correct": rescue_correct,
                    "mixed_made_newly_correct": molecule_added_correct,
                    "override_made_newly_correct": override_made_correct,
                }
            )
    rescue_df = pd.DataFrame(rescue_rows)

    lines = [
        "Element/molecule classification audit summary",
        f"Results directory: {results_dir}",
        "",
        "Step-by-step analysis",
    ]
    if not summary_df.empty:
        def s(col: str) -> int:
            return int(summary_df[col].sum()) if col in summary_df.columns else 0

        true_peaks = s("true_peaks_count")
        predicted_peaks = s("predicted_peaks_count")
        found_unique = s("found_peaks_count")
        truth_matched_predictions = s("predicted_peaks_with_truth")
        no_truth_predictions = s("predicted_peaks_no_truth")

        species_before_correct = s("rf_species_correct_before")
        species_before_total = s("rf_species_total_before")
        species_before_correct_exc = s("rf_species_correct_before_exc")
        species_before_total_exc = s("rf_species_total_before_exc")
        element_before_correct = s("rf_elemental_correct_before")
        element_before_total = s("rf_elemental_total_before")
        element_before_correct_exc = s("rf_elemental_correct_before_exc")
        element_before_total_exc = s("rf_elemental_total_before_exc")
        molecule_before_correct = s("rf_molecular_correct_before")
        molecule_before_total = s("rf_molecular_total_before")
        molecule_before_correct_exc = s("rf_molecular_correct_before_exc")
        molecule_before_total_exc = s("rf_molecular_total_before_exc")

        species_after_correct = s("rf_species_correct")
        species_after_total = s("rf_species_total")
        species_after_correct_exc = s("rf_species_correct_exc")
        species_after_total_exc = s("rf_species_total_exc")
        element_after_correct = s("rf_elemental_correct")
        element_after_total = s("rf_elemental_total")
        element_after_correct_exc = s("rf_elemental_correct_exc")
        element_after_total_exc = s("rf_elemental_total_exc")
        molecule_after_correct = s("rf_molecular_correct")
        molecule_after_total = s("rf_molecular_total")
        molecule_after_correct_exc = s("rf_molecular_correct_exc")
        molecule_after_total_exc = s("rf_molecular_total_exc")

        lines.extend(
            [
                "  Step 0: YOLO peak detection.",
                f"    True RRNG peaks: {true_peaks}",
                f"    Predicted YOLO peaks: {predicted_peaks}",
                f"    Unique true peaks found by at least one prediction: {ratio(found_unique, true_peaks)}",
                f"    Predicted peaks matched to a truth range: {ratio(truth_matched_predictions, predicted_peaks)}",
                f"    Predicted peaks with no matched truth: {ratio(no_truth_predictions, predicted_peaks)}",
                "",
                "  Step 1: RF classification before molecule-rescue mixed assignments.",
                "    This stage includes the base RF assignment, unknown flagging, molecule RF recovery on unknowns, and context rescoring.",
                "    A completely raw pre-unknown RF snapshot was not saved in this run.",
                f"    All species, including unknowns as wrong: {ratio(species_before_correct, species_before_total)}",
                f"    All species, excluding unknown predictions: {ratio(species_before_correct_exc, species_before_total_exc)}",
                f"    Elements, including unknowns as wrong: {ratio(element_before_correct, element_before_total)}",
                f"    Elements, excluding unknown predictions: {ratio(element_before_correct_exc, element_before_total_exc)}",
                f"    Molecules, including unknowns as wrong: {ratio(molecule_before_correct, molecule_before_total)}",
                f"    Molecules, excluding unknown predictions: {ratio(molecule_before_correct_exc, molecule_before_total_exc)}",
                "",
                "  Step 2: Unknown predictions after RF unknown handling/recovery.",
                f"    Unknown predictions with matched truth: {unknown_truth_total}",
                f"      True elements among truth-matched unknowns: {unknown_truth_element}",
                f"      True molecules among truth-matched unknowns: {unknown_truth_molecule}",
                f"    Unknown predictions with no matched truth: {unknown_no_truth}",
                f"    Total unknown predictions: {unknown_truth_total + unknown_no_truth}",
                "",
            ]
        )
    else:
        lines.extend(["  peak_detection_summary.csv was not found, so stepwise aggregate RF counts could not be added.", ""])

    lines.extend(
        [
            "  Step 3: In-context RF rescoring.",
        ]
    )
    if context_df.empty:
        lines.append("    No context rescore override rows found.")
    else:
        lines.extend(
            [
                f"    Context override rows: {len(context_df)}",
                f"    Overrides originating from unknown predictions: {int(context_df['old_is_unknown'].sum())}",
                f"    Overrides that made a previously incorrect/unknown peak correct: {int(context_df['made_newly_correct'].sum())}",
                f"    Overrides that were correct after the override: {int(context_df['new_correct'].sum())}",
            ]
        )
    lines.append("")

    if not summary_df.empty:
        considered = s("molecule_rescue_considered")
        overrides = s("molecule_rescue_overrides")
        mixed_candidates = s("molecule_rescue_mixed_candidates")
        lines.extend(
            [
                "  Step 4: Molecule rescue on element-labeled peaks.",
                f"    Element-labeled peaks considered by molecule-only RF rescue: {considered}",
                f"    Accepted hard molecule overrides: {overrides}",
                f"    Accepted mixed element/molecule candidates: {mixed_candidates}",
                f"    All species, including unknowns as wrong: {ratio(species_before_correct, species_before_total)} -> {ratio(species_after_correct, species_after_total)}",
                f"    All species, excluding unknown predictions: {ratio(species_before_correct_exc, species_before_total_exc)} -> {ratio(species_after_correct_exc, species_after_total_exc)}",
                f"    Elements, including unknowns as wrong: {ratio(element_before_correct, element_before_total)} -> {ratio(element_after_correct, element_after_total)}",
                f"    Elements, excluding unknown predictions: {ratio(element_before_correct_exc, element_before_total_exc)} -> {ratio(element_after_correct_exc, element_after_total_exc)}",
                f"    Molecules, including unknowns as wrong: {ratio(molecule_before_correct, molecule_before_total)} -> {ratio(molecule_after_correct, molecule_after_total)}",
                f"    Molecules, excluding unknown predictions: {ratio(molecule_before_correct_exc, molecule_before_total_exc)} -> {ratio(molecule_after_correct_exc, molecule_after_total_exc)}",
                f"    Net molecule-correct gain: +{molecule_after_correct - molecule_before_correct}",
                f"    Net element-correct change: {element_after_correct - element_before_correct:+d}",
                "",
                "  Step 5: Final top-2 classification after mixed element/molecule assignments.",
                f"    All species, including unknowns as wrong: {ratio(species_after_correct, species_after_total)}",
                f"    All species, excluding unknown predictions: {ratio(species_after_correct_exc, species_after_total_exc)}",
                f"    Elements, including unknowns as wrong: {ratio(element_after_correct, element_after_total)}",
                f"    Elements, excluding unknown predictions: {ratio(element_after_correct_exc, element_after_total_exc)}",
                f"    Molecules, including unknowns as wrong: {ratio(molecule_after_correct, molecule_after_total)}",
                f"    Molecules, excluding unknown predictions: {ratio(molecule_after_correct_exc, molecule_after_total_exc)}",
                "",
            ]
        )

    lines.extend([
        "Pure prediction correctness",
        "  Pure element predictions are truth-matched, non-unknown peaks whose retained RF candidates are element-only.",
        "  Pure molecule predictions are truth-matched, non-unknown peaks whose retained RF candidates are molecule-only.",
        f"  Correct elements from pure element predictions: {ratio(pure_element_correct, pure_element_total)}",
        f"  Correct molecules from pure molecule predictions: {ratio(pure_molecule_correct, pure_molecule_total)}",
        "",
        "Mixed element/molecule assignments",
        "  Mixed predictions are truth-matched, non-unknown peaks with at least one element candidate and one molecule candidate.",
        f"  Mixed prediction rows: {mixed_total}",
        f"  Correct using top-2 mixed candidates: {ratio(mixed_correct_top2, mixed_total)}",
        f"  Newly correct because top-2 mixed assignment included the true label: {mixed_new_top2_correct}",
        f"    Newly correct true elements: {mixed_new_true_elements}",
        f"    Newly correct true molecules: {mixed_new_true_molecules}",
        "",
        "Unknown prediction statistics",
        f"  Total predicted peaks: {predicted_total}",
        f"  Predicted peaks with matched truth: {predicted_with_truth_total}",
        f"  Predicted peaks with no matched truth: {predicted_no_truth_total}",
        f"  Unknown predictions with matched truth: {unknown_truth_total}",
        f"    Fraction of all predicted peaks: {ratio(unknown_truth_total, predicted_total)}",
        f"    Fraction of truth-matched predicted peaks: {ratio(unknown_truth_total, truth_matched_total)}",
        f"    Unknown true element peaks: {unknown_truth_element}",
        f"      Fraction of truth-matched unknowns: {ratio(unknown_truth_element, unknown_truth_total)}",
        f"    Unknown true molecule peaks: {unknown_truth_molecule}",
        f"      Fraction of truth-matched unknowns: {ratio(unknown_truth_molecule, unknown_truth_total)}",
        f"  Unknown predictions with no matched truth: {unknown_no_truth}",
        f"    Fraction of all predicted peaks: {ratio(unknown_no_truth, predicted_total)}",
        f"    Fraction of no-truth predicted peaks: {ratio(unknown_no_truth, predicted_no_truth_total)}",
        "",
        "Molecule-rescue audit CSVs",
    ])
    if rescue_df.empty:
        lines.append("  No molecule rescue candidate rows found.")
    else:
        action_counts = rescue_df["action"].value_counts().to_dict()
        mixed_action_total = int(action_counts.get("mixed_candidate", 0))
        override_total = int(action_counts.get("override", 0))
        lines.extend(
            [
                f"  Accepted rescue rows: {len(rescue_df)}",
                f"    mixed_candidate rows: {mixed_action_total}",
                f"    override rows: {override_total}",
                f"  Mixed-candidate rows that newly made the peak correct via the molecule candidate: {int(rescue_df['mixed_made_newly_correct'].sum())}",
                f"  Override rows that newly made the peak correct via the molecule candidate: {int(rescue_df['override_made_newly_correct'].sum())}",
            ]
        )
    lines.extend(["", "In-context rescoring corrections"])
    if context_df.empty:
        lines.append("  No context rescore override rows found.")
    else:
        reason_counts = context_df["override_reason"].value_counts().to_dict()
        lines.extend(
            [
                f"  Context override rows: {len(context_df)}",
                f"  Context overrides that made a previously incorrect/unknown peak correct: {int(context_df['made_newly_correct'].sum())}",
                f"  Context overrides that were correct after the override: {int(context_df['new_correct'].sum())}",
                f"  Context overrides originating from unknown predictions: {int(context_df['old_is_unknown'].sum())}",
                "  Override reasons:",
            ]
        )
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"    {reason}: {count}")

    lines.extend(
        [
        "",
        "Notes",
        "  Correctness is evaluated only on predicted peaks matched to a true RRNG range.",
        "  Step 0 'unique true peaks found' can differ from truth-matched predicted peaks because multiple predictions can match the same true range.",
        "  The saved before/after RF counts bracket molecule rescue; they do not expose a fully raw pre-unknown RF stage.",
        "  Fractions above are prediction-centric precision-style fractions by prediction type.",
        "  'Newly correct' means the accepted top-1 prediction before the step was not correct, but the added/rescored candidate made the final top-2 or final label correct.",
    ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / "element_molecule_prediction_type_audit_summary.txt"
    output.write_text(build_summary(results_dir))
    print(output)


if __name__ == "__main__":
    main()
