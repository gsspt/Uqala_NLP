#!/usr/bin/env python3
"""
smart_camel_loader.py
──────────────────────────────────────────────────────────────
Smart loader for CAMeL Tools that works in multiple environments:

1. If running from conda venv (local Claude Code) → use CAMeL directly
2. If running from Windows Store Python (web Claude Code) → import from venv
3. Fallback → disable morpho features (still works, just degraded)

This ensures the pipeline works EVERYWHERE.
"""

import sys
import pathlib
import importlib.util

HAS_CAMEL = False
analyzer = None

def load_camel_tools():
    """
    Smart loader for CAMeL Tools.

    Returns:
        tuple: (HAS_CAMEL: bool, analyzer: Analyzer or None)
    """
    global HAS_CAMEL, analyzer

    # ─── Method 1: Direct import (works in conda venv) ───────────────────
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        morpho_db = MorphologyDB.builtin_db()
        analyzer = Analyzer(morpho_db)
        HAS_CAMEL = True
        return True, analyzer
    except ImportError:
        pass  # Fall through to next method
    except Exception as e:
        print(f"Debug: Direct import failed: {e}", file=sys.stderr)
        pass

    # ─── Method 2: Import from conda venv (for Windows Store Python) ───────
    venv_path = pathlib.Path(r"C:\Users\augus\.conda\envs\uqala\Lib\site-packages")

    if venv_path.exists():
        # Add venv to path temporarily
        original_path = sys.path.copy()
        sys.path.insert(0, str(venv_path))

        try:
            from camel_tools.morphology.database import MorphologyDB
            from camel_tools.morphology.analyzer import Analyzer
            morpho_db = MorphologyDB.builtin_db()
            analyzer = Analyzer(morpho_db)
            HAS_CAMEL = True
            return True, analyzer
        except ImportError:
            pass
        except Exception as e:
            print(f"Debug: Venv import failed: {e}", file=sys.stderr)
            pass
        finally:
            sys.path = original_path

    # ─── Fallback: No CAMeL Tools ──────────────────────────────────────────
    return False, None


# Load CAMeL Tools on module import
HAS_CAMEL, analyzer = load_camel_tools()

if HAS_CAMEL:
    print("✅ CAMeL Tools loaded (morpho features will be extracted)")
else:
    print("⚠️  CAMeL Tools not available (morpho features will be 0.0 - degraded mode)")


# ══════════════════════════════════════════════════════════════════════════
# Utility functions for the pipeline

def extract_morpho_features_safe(text):
    """
    Safely extract morphological features.

    Returns dict with f65-f70 keys.
    If CAMeL not available, all values are 0.0.
    """
    features = {
        'f65_root_jnn_density': 0.0,
        'f66_root_aql_density': 0.0,
        'f67_root_hikma_density': 0.0,
        'f68_verb_density': 0.0,
        'f69_noun_density': 0.0,
        'f70_adj_density': 0.0,
    }

    if not HAS_CAMEL or analyzer is None:
        return features

    try:
        tokens = text.split()
        n_tokens = len(tokens) if tokens else 1

        jnn_count = aql_count = hikma_count = 0
        verb_count = noun_count = adj_count = 0

        for token in tokens:
            try:
                analyses = analyzer.analyze(token)
                if analyses:
                    a = analyses[0]
                    root = a.get('root', '')
                    pos = a.get('pos', '')

                    if root == 'ج.ن.ن':
                        jnn_count += 1
                    if root == 'ع.ق.ل':
                        aql_count += 1
                    if root == 'ح.ك.م':
                        hikma_count += 1

                    if pos == 'verb':
                        verb_count += 1
                    if pos == 'noun':
                        noun_count += 1
                    if pos == 'adj':
                        adj_count += 1
            except:
                pass

        features['f65_root_jnn_density'] = jnn_count / n_tokens
        features['f66_root_aql_density'] = aql_count / n_tokens
        features['f67_root_hikma_density'] = hikma_count / n_tokens
        features['f68_verb_density'] = verb_count / n_tokens
        features['f69_noun_density'] = noun_count / n_tokens
        features['f70_adj_density'] = adj_count / n_tokens

    except Exception as e:
        print(f"Warning: morpho extraction failed: {e}", file=sys.stderr)
        pass

    return features
