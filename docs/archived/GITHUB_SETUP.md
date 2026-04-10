# GitHub Setup Instructions

Votre repo est maintenant organisé et commité localement. Voici comment le pousser sur GitHub:

## 1. Créer un nouveau repo sur GitHub

1. Allez à: https://github.com/new
2. **Repository name:** `Uqala-NLP` (ou `uqala-nlp`)
3. **Description:** "Detecting the Wise Fool (ʿāqil majnūn) in Classical Arabic Literature"
4. **Visibility:** Public
5. **Initialize this repository with:** (décochez tout, on va pousser notre code)
6. Click **Create repository**

## 2. Ajouter le remote GitHub

Après création du repo, GitHub vous affichera des commandes. Voici ce qu'il faut faire:

```bash
cd "c:\Users\augus\Desktop\Uqala NLP"

# Ajouter le remote (remplacer gsspt par votre username)
git remote add origin https://github.com/gsspt/Uqala-NLP.git

# Renommer la branche main (si nécessaire)
git branch -M main

# Pousser le code
git push -u origin main
```

## 3. Vérifier sur GitHub

Une fois le push fait:
1. Allez à: https://github.com/gsspt/Uqala-NLP
2. Vérifiez que tous les fichiers sont là
3. Vérifiez que le README s'affiche correctement

## 4. Configuration GitHub (optionnel mais recommandé)

### Ajouter des topics

Settings → About → Topics (recommandé):
- `nlp`
- `arabic-nlp`
- `machine-learning`
- `digital-humanities`
- `classical-arabic`

### Ajouter une description

Settings → About:
- Short description: "Detecting wise fools in Arabic texts using ML"
- Website: (votre site ou laissez vide)

### Protéger la branche main

Settings → Branches:
- Add rule for: `main`
- Require pull request reviews before merging ✓

## 5. Pour les prochains commits

```bash
# Faire des modifications
git add <fichiers>
git commit -m "Description du changement"
git push origin main
```

## Structure GitHub recommandée

```
Uqala-NLP/
├── README.md                    # Affichage sur GitHub
├── openiti_detection/           # Pipeline principal
├── scan/                        # Entraînement des modèles
├── comparison_ensemble/         # Comparaison des modèles
├── dataset_raw.json            # Données d'entraînement
├── isnad_filter.py             # Utilitaire
└── docs/                        # À créer pour docs avancées
    ├── ARCHITECTURE.md
    ├── FEATURES.md
    └── RESULTS.md
```

## Utiliser le repo en sessions web

Une fois sur GitHub, vous pouvez:

### 1. Cloner depuis n'importe où
```bash
git clone https://github.com/gsspt/Uqala-NLP.git
cd Uqala-NLP
pip install -r requirements.txt  # Si vous créez ce fichier
```

### 2. Travailler en sessions web avec GitHub Codespaces

GitHub > Code > Codespaces > Create codespace on main

Cela lance un VS Code dans le navigateur avec votre repo!

### 3. Utiliser Jupyter sur le web

Si vous voulez des notebooks interactifs, vous pouvez:
- Ajouter des `.ipynb` au repo
- Les visualiser directement sur GitHub
- Les exécuter avec Binder (mybinder.org)

## Prochaines étapes

1. **Créer `requirements.txt`:**
```bash
cd "c:\Users\augus\Desktop\Uqala NLP"
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add dependencies"
git push
```

2. **Créer un fichier LICENSE:**
```bash
echo "MIT License

Copyright (c) 2024 Augustin Pot

Permission is hereby granted..." > LICENSE
git add LICENSE
git commit -m "Add MIT license"
git push
```

3. **Ajouter un dossier `docs/`** pour documentation avancée

4. **Créer des GitHub Issues** pour tracker les améliorations futures

5. **Ajouter GitHub Actions** pour CI/CD (tests automatiques)

---

## Troubleshooting

### Si vous avez une erreur d'authentification:

Utilisez un Personal Access Token:

1. GitHub > Settings > Developer settings > Personal access tokens > Tokens (classic) > Generate new token
2. Cochez: `repo`, `workflow`, `gist`
3. Copiez le token
4. Dans terminal: `git config --global user.password <token>`

Ou configurez SSH:
```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
# Ajouter la clé pub à GitHub
git remote set-url origin git@github.com:gsspt/Uqala-NLP.git
```

---

## Commandes git essentielles

```bash
# Voir l'historique
git log --oneline

# Voir l'état
git status

# Voir les changements non-committes
git diff

# Annuler un changement
git checkout -- <fichier>

# Voir tous les remotes
git remote -v

# Changer de remote
git remote set-url origin <nouvelle-url>
```

---

**Vous êtes prêt! 🚀**
