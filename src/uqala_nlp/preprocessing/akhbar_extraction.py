#!/usr/bin/env python3
"""
akhbar_extraction.py
──────────────────────────────────────────────────────────────────
Extract atomic narrative units (akhbars) from OpenITI corpus files.

OpenITI format:
  - Metadata: lines starting with #META# or #
  - Content marker: #META#Header#End#
  - Paragraphs: lines starting with ~~
  - Section breaks: lines starting with # (not # |)
  - Titles: lines starting with # |

An akhbar is a coherent narrative unit bounded by:
  1. Start: after #META#Header#End# (or after section break)
  2. End: when encountering a new section (# ) or EOF
  3. Quality: 80 ≤ Arabic chars ≤ 3000 (prevents metadata fragments & huge texts)

Usage:
  from src.uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file

  akhbars = extract_akhbars_from_file('path/to/openiti/file.txt')
  # Returns: list of str, each a coherent narrative unit
"""

import unicodedata
import pathlib
from typing import List, Tuple


def count_arabic_chars(text: str) -> int:
    """Count number of Arabic characters in text"""
    return sum(1 for c in text if unicodedata.category(c) == 'Lo' and '\u0600' <= c <= '\u06FF')


def extract_akhbars_from_file(filepath: str, min_len: int = 80, max_len: int = 3000) -> List[str]:
    """
    Extract akhbars (narrative units) from an OpenITI corpus file.

    Args:
        filepath: Path to OpenITI text file
        min_len: Minimum Arabic characters per akhbar (default 80)
        max_len: Maximum Arabic characters per akhbar (default 3000)

    Returns:
        List of akhbar texts (each is a coherent narrative unit)
    """
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return []

    content_started = False
    akhbars = []
    current_lines = []

    for line in lines:
        line = line.rstrip('\n\r')

        # Skip until we reach the metadata end marker
        if not content_started:
            if '#META#Header#End#' in line:
                content_started = True
            continue

        # Collect content lines (starting with ~~)
        if line.startswith('~~'):
            current_lines.append(line[2:].strip())

        # Section breaks (lines starting with # but not # |)
        elif line.startswith('# '):
            # Save previous akhbar if we have content
            if current_lines:
                text = ' '.join(current_lines)
                n_ar = count_arabic_chars(text)
                if min_len <= n_ar <= max_len:
                    akhbars.append(text)
            # Don't include title (# |) as content
            current_lines = [] if line.startswith('# |') else [line[2:].strip()]

        # Empty lines or other content
        else:
            if current_lines:
                text = ' '.join(current_lines)
                n_ar = count_arabic_chars(text)
                if min_len <= n_ar <= max_len:
                    akhbars.append(text)
            current_lines = []

    # Don't forget the last akhbar
    if current_lines:
        text = ' '.join(current_lines)
        n_ar = count_arabic_chars(text)
        if min_len <= n_ar <= max_len:
            akhbars.append(text)

    return akhbars


def extract_akhbars_from_corpus(
    corpus_dir: str,
    file_pattern: str = '*-ara1',
) -> Tuple[List[Tuple[str, int, str]], int]:
    """
    Extract all akhbars from entire corpus directory.

    Args:
        corpus_dir: Path to OpenITI corpus directory (e.g., 'openiti_corpus/data/')
        file_pattern: Glob pattern to match files (default '*-ara1' for OpenITI)

    Returns:
        Tuple of:
          - List of (source, khabar_num, text) tuples
          - Total count of akhbars

    Example:
        akhbars, total = extract_akhbars_from_corpus('openiti_corpus/data/')
        print(f"Extracted {total} akhbars from corpus")
    """
    corpus_path = pathlib.Path(corpus_dir)
    all_files = sorted(corpus_path.rglob(file_pattern))

    all_akhbars = []
    total_count = 0

    for filepath in all_files:
        source = filepath.parent.parent.name  # e.g., '0328IbnCabdRabbih'
        akhbars = extract_akhbars_from_file(str(filepath))

        for khabar_num, text in enumerate(akhbars):
            all_akhbars.append((source, khabar_num, text))
            total_count += 1

    return all_akhbars, total_count


if __name__ == '__main__':
    # Test on Ibn Abd Rabbih
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

    test_file = (
        pathlib.Path(__file__).parent.parent.parent.parent
        / 'openiti_corpus/data/0328IbnCabdRabbih/0328IbnCabdRabbih.CiqdFarid'
        / '0328IbnCabdRabbih.CiqdFarid.JK009200-ara1'
    )

    if test_file.exists():
        akhbars = extract_akhbars_from_file(str(test_file))
        print(f"[OK] Extracted {len(akhbars)} akhbars from {test_file.name}")
        for i, a in enumerate(akhbars[:3]):
            n_ar = count_arabic_chars(a)
            print(f"  Akhbar {i}: {n_ar} Arabic chars, {len(a)} total chars")
    else:
        print(f"Test file not found: {test_file}")
