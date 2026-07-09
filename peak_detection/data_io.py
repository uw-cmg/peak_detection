from __future__ import annotations

import os
import re
import numpy as np
import torch

from .models import PeakRange
from .utils import map01, simplify_label, is_excluded_truth_label

try:
    import apav
except ImportError:
    apav = None


def load_apt_from_file(apt_file):
    """
    Load a .apt/.pos binary file or processed .csv file and get histogram.
    Returns (x, spectrum, spectrum_log).
    """
    ext = os.path.splitext(apt_file)[1].lower()

    if ext == '.csv':
        import pandas as pd
        print(f"Loading CSV: {apt_file}")
        df = pd.read_csv(apt_file)
        x = df['x'].values
        spectrum = df['y'].values
        spectrum_log = torch.tensor(spectrum, dtype=torch.float32)
        return x, spectrum, spectrum_log

    if apav is None:
        print("Error: apav package not detected, cannot open .apt/.pos file")
        return None, None, None
    elif ext == '.apt':
        print(f"Loading APT: {apt_file}")
        d = apav.load_apt(apt_file)
    elif ext == '.pos':
        print(f"Loading POS: {apt_file}")
        d = apav.load_pos(apt_file)
    else:
        print(f"Error: unsupported APT input extension '{ext}' for {apt_file}")
        return None, None, None

    x, spectrum = d.mass_histogram(bin_width=0.01, lower=0, upper=307.2, multiplicity='all', norm=False)

    spectrum_log = torch.tensor(map01(np.log(spectrum + 1)), dtype=torch.float32)
    return x, spectrum, spectrum_log


def parse_rrng(filepath: str) -> list[PeakRange]:
    """Parses a .RRNG file for benchmarking, including labels."""
    ranges = []
    if not os.path.exists(filepath):
        return ranges
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    range_pattern = re.compile(r'Range(\d+)=([\d.]+) ([\d.]+) (.*) Color:([0-9A-Fa-f]{6})')
    for line in lines:
        match = range_pattern.match(line.strip())
        if match:
            raw_label = match.group(4).strip()
            label = re.sub(r'Vol:[\d.]+', '', raw_label).strip()
            species_parts = re.findall(r'\b([A-Z][a-z]?):(\d+)\b', label)
            if not species_parts:
                continue
            label = " ".join(f"{sym}:{count}" for sym, count in species_parts)
            start = float(match.group(2))
            end = float(match.group(3))
            label_simple = simplify_label(label)
            if is_excluded_truth_label(label_simple):
                continue
            ranges.append(PeakRange(
                start=start,
                end=end,
                pos=(start + end) / 2,
                label=label_simple
            ))
    return ranges


def extract_elements_from_rrng(rrng_file):
    """Extracts unique element symbols from an RRNG file."""
    elements = set()
    if not os.path.exists(rrng_file):
        return list(elements)

    truth = parse_rrng(rrng_file)
    for t in truth:
        found = re.findall(r'([A-Z][a-z]?)', t.label)
        for f in found:
            elements.add(f)
    return sorted(list(elements))


def _species_to_rrng_notation(species):
    """
    Convert a species label like 'ZrH', 'Ti', 'H2' to RRNG notation like 'Zr:1 H:1', 'Ti:1', 'H:2'.
    Returns (rrng_str, list_of_base_elements).
    """
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', species)
    elements = []
    rrng_parts = []
    for sym, count_str in parts:
        if not sym:
            continue
        count = int(count_str) if count_str else 1
        rrng_parts.append(f"{sym}:{count}")
        if sym not in elements:
            elements.append(sym)
    return " ".join(rrng_parts), elements


def _get_primary_species(r: PeakRange) -> tuple[str, bool]:
    """
    Extract the primary species name from a PeakRange.
    Uses detailed_id.el1 if available, otherwise parses the label.
    Returns (species_str, is_unknown).
    """
    label = r.label or 'Unknown'

    # Use is_unknown flag if available
    if r.is_unknown:
        unknown_match = re.match(r'Unknown\s*\((\w+)\)', label)
        if unknown_match:
            return unknown_match.group(1), True
        return 'Unknown', True

    if label == 'Unknown':
        return 'Unknown', True

    # Use detailed_id el1 if available (clean species name)
    if r.detailed_id is not None:
        el1 = r.detailed_id.el1
        if el1 and el1 != 'Unknown':
            return el1, False

    # Fallback: parse the label string "Ti (0.84), Al (0.12)" -> "Ti"
    match = re.match(r'([A-Za-z]\w*)', label)
    if match:
        return match.group(1), False

    return label, False


def save_rrng(filepath: str, detected_ranges: list[PeakRange], color_map: dict | None = None) -> None:
    """
    Write predicted ranges to a .rrng file.

    Parameters
    ----------
    filepath : str
        Output file path.
    detected_ranges : list[PeakRange]
        Each PeakRange has start, end, label fields,
        and optionally a DetailedId in detailed_id.
    color_map : dict, optional
        Mapping of {element: "RRGGBB"} hex color strings. If None, colors are omitted.
    """
    # Collect unique base elements and build per-range species info
    unique_elements = []
    seen_elements = set()
    range_info = []  # (species_rrng_str, is_unknown, species_label)

    for r in detected_ranges:
        species, is_unknown = _get_primary_species(r)

        if is_unknown:
            ion_name = f"Unknown_{species}" if species != 'Unknown' else 'Unknown'
            if ion_name not in seen_elements:
                unique_elements.append(ion_name)
                seen_elements.add(ion_name)
            rrng_str = f"{ion_name}:1"
            range_info.append((rrng_str, is_unknown, ion_name))
        else:
            rrng_str, base_elements = _species_to_rrng_notation(species)
            for elem in base_elements:
                if elem not in seen_elements:
                    unique_elements.append(elem)
                    seen_elements.add(elem)
            range_info.append((rrng_str, is_unknown, species))

    with open(filepath, 'w', encoding='utf-8') as f:
        # [Ions] section
        f.write("[Ions]\n")
        f.write(f"Number={len(unique_elements)}\n")
        for i, ion in enumerate(unique_elements, 1):
            f.write(f"Ion{i}={ion}\n")
        f.write("\n")

        # [Ranges] section
        f.write("[Ranges]\n")
        f.write(f"Number={len(detected_ranges)}\n")
        for i, (r, (rrng_str, is_unknown, species_label)) in enumerate(zip(detected_ranges, range_info), 1):
            start = f"{r.start:.5f}"
            end = f"{r.end:.5f}"
            color_part = ""
            if color_map is not None:
                color = color_map.get(species_label, "FF0000")
                color_part = f" Color:{color}"
            f.write(f"Range{i}={start} {end} Vol:0.00000 {rrng_str}{color_part}\n")
