#!/usr/bin/env python3
"""Test if CAMeL Tools is available in the current environment"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer

    print("✅ CAMeL Tools imported successfully!")

    # Try loading database
    morpho_db = MorphologyDB.builtin_db()
    print("✅ Morphology database loaded!")

    analyzer = Analyzer(morpho_db)
    print("✅ Analyzer created!")

    # Test
    test_tokens = ['مجنون', 'قال', 'رأيت']
    for token in test_tokens:
        analyses = analyzer.analyze(token)
        if analyses:
            a = analyses[0]
            root = a.get('root', 'N/A')
            pos = a.get('pos', 'N/A')
            print(f"  {token:15} → root: {root:10} pos: {pos}")

    print("\n✅ CAMeL Tools is fully functional!")

except ImportError as e:
    print(f"❌ CAMeL Tools import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
