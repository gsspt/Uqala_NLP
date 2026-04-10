#!/usr/bin/env python3
"""
Test script to verify OpenITI metadata cleaning.
Extracts akhbars from a sample file and displays before/after comparison.
"""

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from openiti_detection.detect_lr_xgboost import extract_akhbars_from_file, clean_openiti_metadata, count_arabic_chars

# ══════════════════════════════════════════════════════════════════════════════

test_texts = [
    # Exemple 1: ms#### + PageV##P### + |
    "قال الله عز وجل : ^ ( الذين إن مكناهم في الأرض أقاموا الصلاة وآتوا الزكاة ) ^ [ الحج : ] PageV01P023 | وقال النبي صلى الله عليه وسلم ms0001 : ( عدل ساعة في حكومة خير من عبادة ستين سنة )",

    # Exemple 2: % poésie %
    "قال الشاعر : % فكلكم راع ونحن رعية % وكل يلاقي ربه فيحاسبه % وهذا دليل على أهمية العدل في الحكم",

    # Exemple 3: mélange de marqueurs
    "وقال عمرو : إن السلطان عمود الدين ms0002 وقاموس العدل PageV01P024 | % والبيت لا يبتنى إلا له عمد % ولا عماد إذا لم ترس أوتاد",

    # Exemple 4: texte avec plusieurs références
    "قال الله تبارك وتعالى : ^ ( يأيها الذين آمنوا أطيعوا الله وأطيعوا الرسول ) ^ [ النساء : ] وقال أبو هريرة ms0003 PageV02P015 : لما نزلت هذه الآية أمرنا بطاعة الأئمة"
]

print("="*80)
print("OPENITI METADATA CLEANING TEST")
print("="*80)

for i, text in enumerate(test_texts, 1):
    print(f"\n{i}. ORIGINAL:")
    print(f"   {text[:120]}...")

    cleaned = clean_openiti_metadata(text)
    print(f"\n   CLEANED:")
    print(f"   {cleaned[:120]}...")

    ar_count = count_arabic_chars(cleaned)
    print(f"\n   Arabic chars: {ar_count}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# Test on actual file
# ══════════════════════════════════════════════════════════════════════════════

BASE = pathlib.Path(__file__).parent.parent
OPENITI_TARGETED = BASE / "openiti_targeted" / "0328IbnCabdRabbih"

if OPENITI_TARGETED.exists():
    print("="*80)
    print("EXTRACTING AKHBARS FROM 0328IbnCabdRabbih (SAMPLE)")
    print("="*80)

    files = list(OPENITI_TARGETED.rglob("*"))
    files = [f for f in files if f.is_file() and not f.name.startswith('.')]

    if files:
        sample_file = files[0]
        print(f"\nFile: {sample_file.name}\n")

        akhbars = extract_akhbars_from_file(sample_file)

        print(f"Total akhbars extracted: {len(akhbars)}\n")

        # Show first 5 akhbars
        print("First 5 akhbars:")
        for i, akhbar in enumerate(akhbars[:5], 1):
            ar_count = count_arabic_chars(akhbar)
            print(f"\n{i}. (Arabic chars: {ar_count})")
            print(f"   {akhbar[:150]}...")

            # Check for remaining metadata markers
            has_ms = "ms" in akhbar and any(c.isdigit() for c in akhbar[akhbar.find("ms"):akhbar.find("ms")+6])
            has_page = "PageV" in akhbar
            has_caret = "^" in akhbar
            has_percent = "%" in akhbar
            has_bracket = "[" in akhbar and "]" in akhbar

            issues = []
            if has_ms:
                issues.append("ms####")
            if has_page:
                issues.append("PageV##P###")
            if has_caret:
                issues.append("^ ^")
            if has_percent:
                issues.append("% %")
            if has_bracket:
                issues.append("[ ]")

            if issues:
                print(f"   ⚠️  Remaining markers: {', '.join(issues)}")
            else:
                print(f"   ✓ Clean (no metadata markers)")

print("\n" + "="*80)
