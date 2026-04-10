"""
isnad_filter.py
───────────────
Sépare l'isnad (chaîne de transmission) du matn (contenu narratif)
dans un akhbar arabe classique.

Algorithme :
  1. Calculer un score isnad par token (verbes de transmission + عن/بن + noms propres)
  2. Glisser une fenêtre pour trouver le point de rupture densité-haute → densité-basse
  3. Valider la coupure sur les marqueurs narratifs (قال / حكى / أن...)
  4. Retourner (isnad_part, matn_part)

Usage :
  from isnad_filter import split_isnad
  isnad, matn = split_isnad(text)
"""

import re
import unicodedata

# ── Verbes de transmission (أفعال الرواية) ───────────────────────────────────
TRANSMISSION_VERBS = {
    # Formes de أخبر
    'أخبرنا', 'أخبرني', 'أخبره', 'أخبرهم', 'أخبرها', 'أخبرك',
    # Formes de حدّث
    'حدثنا', 'حدثني', 'حدثه', 'حدثهم', 'حدثت', 'حدثك',
    # Formes de أنبأ
    'أنبأنا', 'أنبأني', 'أنبأه', 'أنبأت',
    # Formes de روى
    'روى', 'روينا', 'رواه', 'روت', 'رويت',
    # Formes de سمع
    'سمعت', 'سمعنا', 'سمعه', 'سمعتُ',
    # Formes de ذكر (faible signal)
    'ذكر', 'ذكره', 'ذكرنا', 'ذكرت',
    # Autres
    'أعلمنا', 'أعلمني', 'نقل', 'نقله', 'قرأت', 'قرأنا',
    'وصل', 'بلغنا', 'بلغني',
}

# ── Connecteurs d'isnad ───────────────────────────────────────────────────────
ISNAD_CONNECTORS = {'عن', 'عن', 'من', 'بن', 'ابن', 'بنت', 'بنو'}

# ── Kunyas et nisba fréquentes (préfixes de noms de transmetteurs) ────────────
RE_KUNYA  = re.compile(r'\bأب[وياُ]\b|\bأم\b|\bابن\b|\bبن\b|\bبنت\b')
RE_NISBA  = re.compile(r'ي[ةه]\b')   # ...ية / ...يه (nisba)

# ── Marqueurs de début du matn ────────────────────────────────────────────────
# On cherche قال / فقال / حكى / يُحكى / أنه / روي أن... suivi d'un contenu
RE_MATN_MARKERS = re.compile(
    r'(?<!\w)'
    r'(?:فقال|قالوا|قالت|قال|فقالت|'
    r'حكى|حُكي|يُحكى|حكي|'
    r'روي\s+أن|قيل\s+(?:إن|أن)|'
    r'أن(?:ه|ها|هم)\s+قال)'
    r'(?!\w)',
    re.UNICODE
)

# ── Normalisation légère ──────────────────────────────────────────────────────
RE_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')

def _normalize(token: str) -> str:
    t = RE_DIACRITICS.sub('', token)
    t = t.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    t = t.replace('ة', 'ه').replace('ى', 'ي')
    return t

def _tokenize(text: str) -> list[str]:
    return re.findall(r'[\u0600-\u06FF]+', text)

# ── Score isnad d'un token ────────────────────────────────────────────────────
def _token_isnad_score(token: str) -> float:
    norm = _normalize(token)
    if norm in TRANSMISSION_VERBS or token in TRANSMISSION_VERBS:
        return 1.5   # fort signal
    if norm in ISNAD_CONNECTORS or token in ISNAD_CONNECTORS:
        return 0.8
    if RE_KUNYA.match(token):
        return 0.7
    # Nom propre probable : commence par majuscule/hamza et >3 lettres
    if len(token) >= 4 and token[0] in 'اعمحسيطزخغفقبرتكدذ':
        # heuristique : les noms propres sont souvent sans article
        if not token.startswith('ال') and not token.startswith('وال'):
            return 0.3
    return 0.0

# ── Densité isnad sur une fenêtre de tokens ───────────────────────────────────
def _window_density(tokens: list[str], start: int, size: int) -> float:
    window = tokens[start: start + size]
    if not window:
        return 0.0
    total = sum(_token_isnad_score(t) for t in window)
    return total / len(window)

# ── Fonction principale ───────────────────────────────────────────────────────
def split_isnad(text: str, window: int = 8, threshold: float = 0.25) -> tuple[str, str]:
    """
    Retourne (isnad, matn).
    Si aucun isnad détecté, retourne ('', text).

    Paramètres :
      window    : taille de la fenêtre glissante (en tokens)
      threshold : densité minimale pour considérer un segment comme isnad
    """
    if not text or len(text) < 20:
        return ('', text)

    tokens = _tokenize(text)
    if len(tokens) < 4:
        return ('', text)

    # ── Étape 1 : calculer la densité isnad pour chaque position ─────────────
    densities = []
    for i in range(len(tokens)):
        densities.append(_window_density(tokens, i, window))

    # ── Étape 2 : trouver les candidats de coupure dans le texte brut ─────────
    # On liste tous les marqueurs narratifs et leurs positions en caractères
    candidates = [(m.start(), m.group()) for m in RE_MATN_MARKERS.finditer(text)]

    if not candidates:
        # Pas de marqueur narratif → tout est matn (ou text trop court pour avoir un isnad)
        return ('', text)

    # ── Étape 3 : trouver la DERNIÈRE coupure dans la zone isnad-dense ─────────
    # On cherche le dernier marqueur narratif tel que :
    #   - la densité de tout ce qui précède est ≥ threshold
    #   - la densité de ce qui suit (fenêtre) est < celle de ce qui précède
    # Cela gère les isnads imbriqués (plusieurs قال chainés).

    best_cut = None
    best_pos_in_text = -1  # on préfère les coupures les plus tardives dans la zone dense

    for char_pos, marker in candidates:
        ratio_before = char_pos / len(text)
        if ratio_before > 0.7:
            continue
        if ratio_before < 0.03:
            continue

        tokens_before = _tokenize(text[:char_pos])
        if len(tokens_before) < 2:
            continue

        density_before = sum(_token_isnad_score(t) for t in tokens_before) / len(tokens_before)
        tokens_after   = _tokenize(text[char_pos:])
        density_after  = (sum(_token_isnad_score(t) for t in tokens_after[:window]) / window
                          if len(tokens_after) >= window else 0.0)

        contrast = density_before - density_after
        # Valider : la zone avant est dense ET la zone après l'est moins
        if density_before >= threshold and contrast > 0.05:
            cut_end = char_pos + len(marker)
            # Préférer la coupure la plus TARDIVE dans la zone dense
            if char_pos > best_pos_in_text:
                best_pos_in_text = char_pos
                best_cut = cut_end

    if best_cut is None:
        return ('', text)

    isnad_part = text[:best_cut].strip()
    matn_part  = text[best_cut:].strip()

    # Sécurité : le matn doit être substantiel
    if len(matn_part) < 30:
        return ('', text)

    return (isnad_part, matn_part)


def get_matn(text: str) -> str:
    """Raccourci : retourne seulement le matn."""
    _, matn = split_isnad(text)
    return matn


# ── Diagnostic ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import json, pathlib, sys

    BASE = pathlib.Path(__file__).parent
    fp   = BASE / 'dataset_features.json'
    if not fp.exists():
        print("dataset_features.json introuvable")
        sys.exit(1)

    with open(fp, encoding='utf-8') as f:
        data = json.load(f)

    # Tester sur un échantillon
    n_split = 0
    n_total = 0
    avg_ratio_before = []

    samples = [e for e in data if e['label'] == 1][:20]
    for e in samples:
        text = e['text_ar']
        isnad, matn = split_isnad(text)
        n_total += 1
        if isnad:
            n_split += 1
            ratio = len(isnad) / len(text)
            avg_ratio_before.append(ratio)
            print(f"\n── {e['id']} ──")
            print(f"  ISNAD ({len(isnad)}c) : {isnad[:120]}…")
            print(f"  MATN  ({len(matn)}c)  : {matn[:120]}…")

    print(f"\n{n_split}/{n_total} textes avec isnad détecté")
    if avg_ratio_before:
        print(f"Ratio isnad moyen : {sum(avg_ratio_before)/len(avg_ratio_before):.1%}")
