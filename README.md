# Projet Data Science — Formule 1  (M2 Mathématiques Appliquées)

Analyse de données F1 avec **FastF1** — 4 modules couvrant l'apprentissage non supervisé, supervisé, le deep learning et le NLP.  
Les modèles sont entraînés sur **tous les GP 2025 + 2026 disponibles** et produisent des **prédictions pour le prochain Grand Prix 2026**.

## Structure

```
f1-datascience/
├── main.ipynb                 # Notebook principal — toute la pipeline + prédictions prochain GP
├── requirements.txt
├── data/cache/                # Cache FastF1 (auto-généré)
├── models/                    # Modèles entraînés (RF, XGB, MLP)
├── dashboard_f1.png           # Dashboard généré à la dernière exécution
├── src/
│   ├── data_loader.py         # Chargement FastF1 + get_next_gp()
│   ├── unsupervised.py        # IF + DBSCAN — anomalies + run_season_anomalies()
│   ├── supervised.py          # RF + XGBoost — pitstop + predict_next_gp_pitstops()
│   ├── deep_learning.py       # MLP PyTorch — vainqueur + predict_winner_next_gp()
│   └── nlp.py                 # HuggingFace — sentiment radio
└── notebooks/
    ├── 01_unsupervised.ipynb
    ├── 02_supervised.ipynb
    ├── 03_deep_learning.ipynb
    └── 04_nlp.ipynb
```

## Installation

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation — `main.ipynb`

Ouvrez le notebook principal. Il orchestre tout en 3 grandes phases :

```
Phase 1 — Calendrier & chargement
   • Détection automatique du prochain GP (get_next_gp)
   • Chargement de TOUS les GP 2025 + GP 2026 disponibles (~30 min premier lancement)

Phase 2 — Entraînement sur 2025+2026
   • Module 1 : anomalies IF + DBSCAN sur chaque GP → taux par circuit
   • Module 2 : RF + XGBoost sur tous les tours → fenêtre de pitstop
   • Module 3 : MLP PyTorch sur tous les résultats → prédiction vainqueur
   • Module 4 : NLP sur les 5 dernières sessions

Phase 3 — Dashboard prochain GP
   • Fenêtre de pitstop par compound (SOFT / MEDIUM / HARD)
   • Probabilités de victoire par pilote (avec IC MC Dropout)
   • Taux d'anomalies attendu (historique du circuit)
```

## Utilisation — CLI

```bash
# Un seul module
python main.py --module unsupervised --year 2023 --gp Monaco

# Prédiction pour le prochain GP (3 dernières sessions)
python main.py --predict-next --year 2026

# Tous les modules
python main.py --module all --year 2023
```

---

## Fondements mathématiques

### Module 1 — Détection d'anomalies (Isolation Forest + DBSCAN)

**Isolation Forest** (Liu et al., 2008) construit des arbres de partition aléatoire sur l'espace des features. La profondeur d'isolation d'un point x, notée h(x), est le nombre de coupures nécessaires pour l'isoler. Pour les points normaux, la théorie montre que E[h(x)] ≈ c(n) = 2H(n−1) − 2(n−1)/n (nombre harmonique), soit une profondeur d'ordre O(log n). Les anomalies, situées dans des régions peu denses, ont E[h(x)] ≪ c(n). Le score d'anomalie est s(x, n) = 2^{−E[h(x)]/c(n)} ∈ (0, 1), avec s proche de 1 pour les anomalies. La complexité est O(n log n) à l'entraînement et O(log n) par prédiction par arbre.

Pour **DBSCAN** (Ester et al., 1996), le paramètre eps est sélectionné automatiquement via la courbe k-distance : on trie les distances au k-ième voisin (k = min_samples) et on détecte le coude par le maximum de la dérivée seconde discrète Δ²d[i] = d[i+1] − 2d[i] + d[i−1], qui marque la transition entre la région dense et la région clairsemée. Une couche de validation statistique filtre les faux positifs par z-score sur LapTimeDelta : seules les anomalies avec |z| > 2.5 sont retenues (≈ 1.24% de la distribution normale). La qualité du clustering est évaluée par l'indice de Silhouette s(i) = (b(i) − a(i)) / max(a(i), b(i)) ∈ [−1, 1].

### Module 2 — Prédiction des pitstops (Random Forest + XGBoost)

**Random Forest** (Breiman, 2001) et **XGBoost** (Chen & Guestrin, 2016) sont entraînés avec `GridSearchCV` utilisant `TimeSeriesSplit(n_splits=5)`. Ce choix est fondamental : les données F1 ont une structure temporelle (les sessions sont ordonnées dans le temps), et un k-fold classique introduirait du data leakage en validant sur des sessions passées à partir d'un modèle entraîné sur des sessions futures. TimeSeriesSplit garantit la causalité : chaque fold entraîne uniquement sur les sessions passées.

La **courbe ROC** est accompagnée d'un intervalle de confiance Bootstrap à 95% (n=1000 itérations de rééchantillonnage avec remise), estimant AUC ± 1.96·σ(AUC). La **calibration** des probabilités est vérifiée via la reliability curve et corrigée par régression isotonique (CalibratedClassifierCV, method='isotonic'), qui apprend une transformation monotone g : p_brut → p_calibré. Le **test de McNemar** compare statistiquement RF et XGBoost : χ² = (|b−c| − 1)² / (b+c) avec b = #{RF correct, XGB incorrect}, rejeté si p < 0.05. L'interprétabilité est assurée par les valeurs **SHAP** (φᵢ) qui décomposent additivement chaque prédiction : f(x) = φ₀ + Σᵢ φᵢ(x).

### Module 3 — Prédiction du vainqueur (MLP PyTorch)

Le **MLP** (128→64→32→1) est initialisé par **He / Kaiming** (kaiming_normal_) pour les couches ReLU, qui garantit E[Var(aₗ)] = 1 à travers les couches en fixant Var(Wₗ) = 2/fan_in (He et al., 2015). Sans cette initialisation, les gradients s'éteignent ou explosent en profondeur. Le scheduler **CosineAnnealingLR** (Loshchilov & Hutter, 2017) fait varier le taux d'apprentissage selon ηₜ = η_min + ½(η_max − η_min)(1 + cos(πt/T_max)), favorisant les "flat minima" à faible courbure du Hessien qui généralisent mieux que les minima étroits (Hochreiter & Schmidhuber, 1997).

L'incertitude prédictive est quantifiée par **MC Dropout** (Gal & Ghahramani, 2016) : en gardant le dropout actif à l'inférence et en effectuant N=50 passes forward, on obtient une approximation de l'inférence variationnelle bayésienne. La moyenne μ et l'écart-type σ des probabilités capturent respectivement la prédiction et l'incertitude épistémique du modèle. Une **baseline Oracle** (le poleman gagne) établit la borne inférieure de performance attendue ; tout gain positif du MLP est un apport démontré.

### Module 4 — Analyse NLP des radios (Transformers + tests non-paramétriques)

Le modèle `cardiffnlp/twitter-roberta-base-sentiment` (fine-tuning de RoBERTa sur des tweets) classe chaque message en {négatif, neutre, positif}. La corrélation entre sentiment et performance est mesurée par le **coefficient de Spearman** ρ_s = 1 − 6Σd²ᵢ / n(n²−1), qui est robuste aux outliers et ne suppose pas la normalité des distributions (contrairement à Pearson). La significativité est testée par t = ρ_s√(n−2) / √(1−ρ²_s) ~ t(n−2).

Pour les comparaisons multiples (N pilotes testés simultanément), la **correction de Bonferroni** ajuste le seuil à α' = 0.05/N pour contrôler le taux d'erreur de famille (FWER). L'impact différentiel des types de messages (STRATEGIE vs ALERTE_TECHNIQUE) est testé par le **test de Mann-Whitney U**, statistique non-paramétrique qui compare les distributions de LapTimeDelta entre groupes sans hypothèse de normalité : U = #{(xᵢ, yⱼ) : xᵢ > yⱼ}.

---

## Modules

### 1. Non supervisé — Détection d'anomalies (`src/unsupervised.py`)

- **Par course** : Isolation Forest + DBSCAN avec eps auto-sélectionné, validation z-score, indice de Silhouette
- **Saison complète** : `run_season_anomalies(sessions)` → taux IF/DBSCAN par GP, heatmap saison
- **Prochain GP** : taux d'anomalies attendu basé sur l'historique du circuit

### 2. Supervisé — Prédiction du pitstop (`src/supervised.py`)

- RF + XGBoost avec `GridSearchCV` + `TimeSeriesSplit(n_splits=5)`
- Courbe ROC + AUC avec IC Bootstrap 95%, calibration isotonique, test de McNemar, SHAP
- `predict_next_gp_pitstops()` : courbes P(pitstop) vs âge des pneus par compound
- Sauvegarde → `models/rf_pitstop.joblib`, `models/xgb_pitstop.joblib`

### 3. Deep Learning — Prédiction du vainqueur (`src/deep_learning.py`)

- MLP PyTorch (He init, CosineAnnealingLR, MC Dropout N=50)
- Baseline Oracle poleman, détection automatique de l'overfitting, moving average
- `predict_winner_next_gp()` → probabilités + incertitude (WinProba ± WinProba_Std)
- Sauvegarde → `models/mlp_winner.pt`

### 4. NLP — Analyse des radios (`src/nlp.py`)

- Sentiment via RoBERTa (fallback mots-clés si absent)
- Corrélation de Spearman + correction de Bonferroni par pilote
- `test_message_type_impact()` → test de Mann-Whitney U par paire de types

---

## Compatibilité 2025–2026

FastF1 charge les sessions disponibles et ignore silencieusement les sessions futures.  
`get_next_gp(2026)` détecte automatiquement le prochain GP non encore couru.  
Cache dans `data/cache/` — les données déjà téléchargées ne sont pas re-téléchargées.
