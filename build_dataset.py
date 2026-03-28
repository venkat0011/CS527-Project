#!/usr/bin/env python3
"""
SV-COMP Dataset Builder for CS527 Project
Builds a balanced dataset of 100 true + 100 false per property.

Only includes folders applicable to both C and Python (transpiled):
    array-*, loops*, loop-*, recursive*, bitvector*, termination-*

Only includes plain .c files (skips .i preprocessed files and alloca variants).

Usage:
    python build_dataset.py --sv_benchmarks /path/to/sv-benchmarks --output dataset.csv

Output CSV columns:
    property, expected_verdict, c_file, c_file_abs, yml_path, folder
"""

import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter

import yaml  # pip install pyyaml

TARGET_PROPERTIES = {
    "unreach-call",
    "no-overflow",
    "def-behavior",
    "valid-memsafety",
    "valid-memcleanup",
    "no-data-race",
    "termination",
}

FOLDER_WHITELIST_PREFIXES = (
    "array-",
    "array_",
    "loops",
    "loop-",
    "loop_",
    "recursive",
    "bitvector",
    "termination-",
    "termination_",
)

SAMPLES_PER_VERDICT = 100
RANDOM_SEED = 42


def is_whitelisted(yml_path: Path, c_dir: Path) -> bool:
    try:
        rel = yml_path.relative_to(c_dir)
    except ValueError:
        return False
    folder = rel.parts[0].lower()
    return folder.startswith(FOLDER_WHITELIST_PREFIXES)


def property_name_from_path(prp_path: str) -> str:
    return Path(prp_path).stem


def parse_yml(yml_path: Path, sv_root: Path):
    try:
        with open(yml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    # Resolve input file — can be string or list
    input_files = data.get("input_files", "")
    if not input_files:
        return []
    if isinstance(input_files, list):
        if not input_files:
            return []
        input_files = input_files[0]
    input_files = str(input_files)

    c_file_abs = (yml_path.parent / input_files).resolve()

    # Only plain .c files — skip .i (preprocessed), alloca variants, and cil files
    if c_file_abs.suffix != ".c":
        return []
    name_lower = c_file_abs.name.lower()
    if "alloca" in name_lower or ".cil." in name_lower:
        return []

    sv_root_resolved = sv_root.resolve()
    try:
        c_file_rel = c_file_abs.relative_to(sv_root_resolved)
    except ValueError:
        c_file_rel = c_file_abs

    # Skip non-C tasks
    options = data.get("options", {}) or {}
    if options.get("language", "C") != "C":
        return []

    # Folder label for the CSV
    try:
        folder = yml_path.relative_to(sv_root_resolved / "c").parts[0]
    except (ValueError, IndexError):
        folder = yml_path.parent.name

    properties = data.get("properties", []) or []
    records = []
    for prop in properties:
        if not isinstance(prop, dict):
            continue
        prop_name = property_name_from_path(prop.get("property_file", ""))
        if prop_name not in TARGET_PROPERTIES:
            continue
        expected_verdict = prop.get("expected_verdict", None)
        if expected_verdict is None:
            continue

        records.append({
            "property":         prop_name,
            "expected_verdict": "true" if expected_verdict else "false",
            "c_file":           str(c_file_rel),
            "c_file_abs":       str(c_file_abs),
            "yml_path":         str(yml_path.relative_to(sv_root_resolved)
                                    if yml_path.resolve().is_relative_to(sv_root_resolved)
                                    else yml_path),
            "folder":           folder,
        })

    return records


def build_dataset(sv_benchmarks: Path, output: Path, seed: int = RANDOM_SEED):
    random.seed(seed)

    c_dir = sv_benchmarks / "c"
    if not c_dir.exists():
        raise FileNotFoundError(
            f"Could not find c/ directory under {sv_benchmarks}")

    # Show which folders will be included vs excluded
    all_folders = sorted(p.name for p in c_dir.iterdir() if p.is_dir())
    included = [f for f in all_folders if f.lower(
    ).startswith(FOLDER_WHITELIST_PREFIXES)]
    excluded = [f for f in all_folders if f not in included]
    print(f"Total folders under c/: {len(all_folders)}")
    print(f"Included ({len(included)}): {', '.join(included)}")
    print(f"Excluded ({len(excluded)}): {', '.join(excluded)}\n")

    print("Scanning whitelisted YML files ...")
    yml_files = [p for p in c_dir.rglob("*.yml") if is_whitelisted(p, c_dir)]
    print(f"Found {len(yml_files)} YML files in whitelisted folders\n")

    # Bucket records by (property, verdict)
    buckets = defaultdict(list)
    for i, yml_path in enumerate(yml_files):
        if i % 2000 == 0:
            print(f"  Parsed {i}/{len(yml_files)} ...")
        for rec in parse_yml(yml_path, sv_benchmarks):
            buckets[(rec["property"], rec["expected_verdict"])].append(rec)

    print("\n--- Available samples per (property, verdict) ---")
    for prop in sorted(TARGET_PROPERTIES):
        t = len(buckets.get((prop, "true"), []))
        f = len(buckets.get((prop, "false"), []))
        note = "  *** INSUFFICIENT ***" if (
            t < SAMPLES_PER_VERDICT or f < SAMPLES_PER_VERDICT) else ""
        print(f"  {prop:20s}  true={t:5d}  false={f:5d}{note}")

    # Sample and combine
    all_rows = []
    print("\n--- Sampling ---")
    for prop in sorted(TARGET_PROPERTIES):
        for verdict in ("true", "false"):
            pool = buckets.get((prop, verdict), [])
            n = min(SAMPLES_PER_VERDICT, len(pool))
            sampled = random.sample(pool, n) if n > 0 else []
            print(
                f"  {prop:20s}  verdict={verdict}  sampled={n:3d} / available={len(pool)}")
            all_rows.extend(sampled)

    # Write CSV
    fieldnames = ["property", "expected_verdict",
                  "c_file", "c_file_abs", "yml_path", "folder"]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {output}")

    print("\n--- Final dataset summary ---")
    counts = Counter((r["property"], r["expected_verdict"]) for r in all_rows)
    print(f"  {'Property':<20}  {'true':>5}  {'false':>5}  {'total':>5}")
    print(f"  {'-'*20}  {'-----':>5}  {'-----':>5}  {'-----':>5}")
    for prop in sorted(TARGET_PROPERTIES):
        t = counts.get((prop, "true"), 0)
        f = counts.get((prop, "false"), 0)
        print(f"  {prop:<20}  {t:>5}  {f:>5}  {t+f:>5}")
    print(f"\n  Grand total: {len(all_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build balanced SV-COMP dataset")
    parser.add_argument("--sv_benchmarks", type=Path, required=True,
                        help="Path to sv-benchmarks/ root directory")
    parser.add_argument("--output", type=Path, default=Path("dataset.csv"),
                        help="Output CSV path (default: dataset.csv)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    build_dataset(args.sv_benchmarks, args.output, args.seed)
