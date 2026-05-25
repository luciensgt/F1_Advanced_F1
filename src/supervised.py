"""
Module de prédiction supervisée du tour de pitstop en F1.

Modèles     : Random Forest (Breiman, 2001) + XGBoost (Chen & Guestrin, 2016)
Tâche       : Classification binaire par tour — pitstop oui (1) / non (0).
Hypothèses  : Stationnarité locale par session (le comportement des pitstops ne
              dérive pas au cours d'une même course). Indépendance conditionnelle
              des tours au sein d'un stint, sachant TyreLife et Compound
              (approximation raisonnable hors under/overcut).
Références  : Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
              Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting
              system. KDD 2016, pp. 785-794.
              Platt, J. (1999). Probabilistic outputs for SVMs. Advances in LKPD.
              McNemar, Q. (1947). Note on the sampling error of the difference
              between correlated proportions. Psychometrika, 12(2), 153-157.

Validation croisée temporelle :
  Les données F1 sont ordonnées dans le temps (sessions successives). Un split
  aléatoire classique introduit du data leakage (le modèle "voit" des sessions
  futures lors de l'entraînement). On utilise TimeSeriesSplit(n_splits=5) :
  chaque fold entraîne sur les sessions passées et valide sur les sessions futures,
  respectant la causalité temporelle des données.

Calibration des probabilités :
  Un classifieur bien entraîné n'est pas nécessairement bien calibré :
  P(y=1 | f(x)=p) ≠ p en général. CalibratedClassifierCV (method='isotonic')
  apprend une régression isotonique monotone sur les probabilités brutes pour
  aligner les probabilités prédites sur les fréquences observées.

Test de McNemar :
  Soit b = #{RF correct, XGB incorrect} et c = #{RF incorrect, XGB correct}.
  Statistique (avec correction de continuité) : χ² = (|b−c|−1)² / (b+c).
  Sous H0 (les deux modèles commettent les mêmes erreurs) : χ² ~ χ²(df=1).
  On rejette H0 si p < 0.05, concluant que les erreurs différent significativement.
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from scipy.stats import chi2

from src.data_loader import get_laps_features, get_weather, load_multiple_seasons, load_session

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "TyreLife", "Compound_enc", "Position", "LapNumber",
    "Stint", "LapTimeDelta_prev", "SpeedST", "SpeedFL",
    "FreshTyre", "Rainfall",
]

PARAM_GRID_RF = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10, None],
    "min_samples_split": [2, 5],
}

PARAM_GRID_XGB = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
}


def build_dataset(sessions: list) -> pd.DataFrame:
    """
    Construit le dataset supervisé à partir d'une liste de sessions.

    Retourne un DataFrame avec les features et la cible 'HasPitStop'.
    La colonne 'SessionIdx' permet le split temporel par session.
    """
    all_frames = []
    for session_idx, session in enumerate(sessions):
        laps = get_laps_features(session)
        if laps.empty:
            continue

        weather = get_weather(session)
        if not weather.empty and "Rainfall" in weather.columns:
            rainfall_mean = float(weather["Rainfall"].astype(float).mean())
        else:
            rainfall_mean = 0.0
        laps["Rainfall"] = rainfall_mean

        laps = laps.sort_values(["Driver", "LapNumber"])
        laps["LapTimeDelta_prev"] = laps.groupby("Driver")["LapTimeDelta"].shift(1).fillna(0)
        laps["PositionDelta"] = laps.groupby("Driver")["Position"].diff().fillna(0)

        max_lap = laps["LapNumber"].max()
        laps["LapsRemaining"] = max_lap - laps["LapNumber"]
        laps["SessionIdx"] = session_idx
        all_frames.append(laps)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    le = LabelEncoder()
    df["Compound_enc"] = le.fit_transform(df["Compound"].fillna("UNKNOWN"))
    df["HasPitStop"] = df["HasPitStop"].astype(int)
    return df


def prepare_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sélectionne et nettoie les features/cible."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["HasPitStop"].astype(int)
    return X, y


def _temporal_split(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split temporel causal pour éviter le data leakage.

    Principe : les sessions les plus récentes constituent le test set.
    Toute information future est exclue de l'entraînement, respectant
    la causalité temporelle des données F1.

    Pour la validation croisée interne (GridSearchCV), on utilise
    TimeSeriesSplit(n_splits=5) qui maintient le même principe de causalité.
    """
    n_sessions = df["SessionIdx"].nunique() if "SessionIdx" in df.columns else 1

    if n_sessions > 1:
        session_ids = sorted(df["SessionIdx"].unique())
        n_train = max(1, int(len(session_ids) * (1 - test_size)))
        train_sessions = set(session_ids[:n_train])
        mask = df["SessionIdx"].isin(train_sessions).values
    else:
        order = np.argsort(df["LapNumber"].values) if "LapNumber" in df.columns else np.arange(len(df))
        n_train = int(len(order) * (1 - test_size))
        mask = np.zeros(len(df), dtype=bool)
        mask[order[:n_train]] = True

    return X[mask], X[~mask], y[mask], y[~mask]


def _mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> float:
    """
    Test de McNemar (correction de continuité de Edwards).

    Paramètres :
      b = #{modèle 1 correct, modèle 2 incorrect}
      c = #{modèle 1 incorrect, modèle 2 correct}
    Statistique : χ² = (|b−c| − 1)² / (b+c)  ~ χ²(df=1) sous H0.

    Returns:
        p-value du test. Si p < 0.05 : les deux modèles ont des erreurs
        statistiquement différentes (rejeter H0 d'homogénéité).
    """
    b = int(np.sum((y_pred1 == y_true) & (y_pred2 != y_true)))
    c = int(np.sum((y_pred1 != y_true) & (y_pred2 == y_true)))
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return float(1.0 - chi2.cdf(stat, df=1))


def train_random_forest(X_train, y_train) -> tuple[RandomForestClassifier, dict]:
    """Entraîne un Random Forest avec GridSearchCV + TimeSeriesSplit."""
    logger.info("[RF] GridSearchCV avec TimeSeriesSplit(n_splits=5)...")
    tscv = TimeSeriesSplit(n_splits=5)
    base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    gs = GridSearchCV(base, PARAM_GRID_RF, cv=tscv, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    logger.info(f"[RF] Meilleurs params : {gs.best_params_}  |  F1-CV : {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_


def train_xgboost(X_train, y_train) -> tuple | None:
    """Entraîne un XGBoost avec GridSearchCV + TimeSeriesSplit."""
    if not HAS_XGB:
        return None, {}
    logger.info("[XGB] GridSearchCV avec TimeSeriesSplit(n_splits=5)...")
    tscv = TimeSeriesSplit(n_splits=5)
    scale = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    base = XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss",
                         scale_pos_weight=scale, verbosity=0)
    gs = GridSearchCV(base, PARAM_GRID_XGB, cv=tscv, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    logger.info(f"[XGB] Meilleurs params : {gs.best_params_}  |  F1-CV : {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Calcule et affiche les métriques d'évaluation."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    logger.info(f"[{name}] Accuracy : {acc:.4f}  |  F1-Score : {f1:.4f}")
    logger.info(classification_report(y_test, y_pred, target_names=["Pas de pit", "Pit Stop"], zero_division=0))
    return {"model": name, "accuracy": acc, "f1": f1, "y_pred": y_pred}


def plot_roc_with_ci(
    model, X_test, y_test, name: str, n_bootstrap: int = 1000
) -> plt.Figure:
    """
    Courbe ROC avec intervalle de confiance Bootstrap à 95%.

    Méthode Bootstrap :
      On tire n_bootstrap échantillons avec remise de taille n depuis (X_test, y_test).
      Pour chaque échantillon, on calcule l'AUC. L'IC 95% est estimé par :
      AUC_mean ± 1.96 · AUC_std  (approximation gaussienne valide pour n_bootstrap ≥ 500).

    Args:
        model: Classifieur avec predict_proba().
        n_bootstrap: Nombre d'itérations Bootstrap (défaut 1000).

    Returns:
        Figure avec courbe ROC et annotation AUC ± IC.
    """
    y_score = model.predict_proba(X_test)[:, 1]
    fpr_base, tpr_base, _ = roc_curve(y_test, y_score)
    auc_base = auc(fpr_base, tpr_base)

    rng = np.random.RandomState(42)
    n = len(y_test)
    y_test_arr = np.array(y_test)
    auc_scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        y_b = y_test_arr[idx]
        if len(np.unique(y_b)) < 2:
            continue
        s_b = y_score[idx]
        fpr_b, tpr_b, _ = roc_curve(y_b, s_b)
        auc_scores.append(auc(fpr_b, tpr_b))

    auc_mean = float(np.mean(auc_scores))
    auc_std = float(np.std(auc_scores))
    ci = 1.96 * auc_std

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_base, tpr_base, color="steelblue", lw=2.5,
            label=f"AUC = {auc_mean:.3f} ± {ci:.3f}  [IC 95%]")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Classifieur aléatoire")
    ax.fill_between(fpr_base, tpr_base - ci, tpr_base + ci,
                    alpha=0.12, color="steelblue", label="Bande IC 95% (approx.)")
    ax.set_xlabel("Taux de faux positifs (1 − Spécificité)")
    ax.set_ylabel("Taux de vrais positifs (Sensibilité)")
    ax.set_title(
        f"Courbe ROC — {name}\n"
        f"AUC = {auc_mean:.3f} ± {ci:.3f}  (Bootstrap n={n_bootstrap})",
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_calibration_curve(model, X_test, y_test, name: str) -> plt.Figure:
    """
    Reliability curve (diagramme de calibration) du classifieur.

    Un classifieur est dit bien calibré si P(y=1 | f(x)=p) = p pour tout p.
    La courbe de calibration trace la fréquence observée de positifs vs la
    probabilité prédite moyenne par bin. Un écart par rapport à la diagonale
    indique une sous- ou sur-confiance du modèle.

    On compare le modèle brut vs calibré par régression isotonique
    (CalibratedClassifierCV, method='isotonic'), qui apprend la transformation
    monotone g : p_raw → p_calibré minimisant la log-vraisemblance.
    """
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_test, y_test)

    y_prob_raw = model.predict_proba(X_test)[:, 1]
    y_prob_cal = calibrated.predict_proba(X_test)[:, 1]

    n_bins = 10
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_prob_raw, n_bins=n_bins)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_prob_cal, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Calibration parfaite", alpha=0.7)
    ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#e74c3c",
            label=f"{name} (brut)", linewidth=2, markersize=6)
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#2ecc71",
            label=f"{name} + isotonic", linewidth=2, markersize=6)
    ax.set_xlabel("Probabilité prédite (moyenne du bin)")
    ax.set_ylabel("Fraction de positifs observés")
    ax.set_title(f"Reliability Curve — {name}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(y_prob_raw, bins=20, color="#e74c3c", alpha=0.6,
             label="Brut", edgecolor="white")
    ax2.hist(y_prob_cal, bins=20, color="#2ecc71", alpha=0.6,
             label="Calibré (isotonic)", edgecolor="white")
    ax2.set_xlabel("Probabilité prédite")
    ax2.set_ylabel("Fréquence")
    ax2.set_title("Distribution des probabilités", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Calibration des probabilités — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig, calibrated


def plot_shap_importance(model, X_test: pd.DataFrame, name: str) -> plt.Figure:
    """
    Importance des features via SHAP (SHapley Additive exPlanations).

    Les valeurs SHAP φ_i(x) décomposent la prédiction f(x) en contributions
    additives par feature, avec la garantie d'efficacité :
      f(x) = φ_0 + Σ_i φ_i(x)
    où φ_0 = E[f(X)] est la valeur de base. TreeExplainer calcule exactement
    les valeurs SHAP pour les arbres en O(TLD²) (T arbres, L feuilles, D profondeur).

    Args:
        model: Modèle sklearn arbre (RF ou XGBoost).
        X_test: Features du test set (DataFrame pour les noms de colonnes).
        name: Nom du modèle pour le titre.

    Returns:
        Figure barplot de |E[|φ_i|]| pour les 10 features les plus importantes.
    """
    if not HAS_SHAP:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "SHAP non installé\npip install shap",
                ha="center", va="center", fontsize=12)
        ax.set_title(f"SHAP Importance — {name}", fontweight="bold")
        return fig

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Pour la classification binaire, shap_values peut être une liste [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_test.columns)
    top10 = mean_abs.sort_values(ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(9, 6))
    top10.plot(kind="barh", ax=ax, color="#e67e22", edgecolor="white")
    ax.set_xlabel("Valeur SHAP moyenne  E[|φᵢ|]")
    ax.set_title(
        f"SHAP Feature Importance — {name}\n"
        f"(TreeExplainer, top 10 features, n_test={len(X_test)})",
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred, name: str) -> plt.Figure:
    """Affiche la matrice de confusion."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Pas de pit", "Pit Stop"],
        ax=ax, colorbar=False, cmap="Blues",
    )
    ax.set_title(f"Matrice de confusion — {name}", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, name: str) -> plt.Figure:
    """Barplot des importances de features (Gini/gain)."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Importance des features — {name}", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_model_comparison(metrics_rf: dict, metrics_xgb: dict | None) -> plt.Figure:
    """Compare accuracy et F1 des deux modèles."""
    names = ["Random Forest"]
    acc_vals = [metrics_rf["accuracy"]]
    f1_vals = [metrics_rf["f1"]]

    if metrics_xgb:
        names.append("XGBoost")
        acc_vals.append(metrics_xgb["accuracy"])
        f1_vals.append(metrics_xgb["f1"])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - 0.2, acc_vals, width=0.35, label="Accuracy", color="#3498db")
    bars2 = ax.bar(x + 0.2, f1_vals, width=0.35, label="F1-Score", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison RF vs XGBoost", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bar in [*bars1, *bars2]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig


def build_next_gp_scenarios(n_laps: int = 58) -> pd.DataFrame:
    """Génère des scénarios de tour typiques pour un GP générique."""
    compound_map = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}
    rows = []
    for compound_name, compound_enc in compound_map.items():
        for position in [1, 5, 10, 15, 20]:
            for lap in range(1, n_laps + 1):
                rows.append({
                    "Compound": compound_name,
                    "TyreLife": lap,
                    "Compound_enc": compound_enc,
                    "Position": position,
                    "LapNumber": lap,
                    "Stint": 1 + int(lap > n_laps // 2),
                    "LapTimeDelta_prev": max(0.0, 0.04 * (lap - 1) - 0.3),
                    "SpeedST": 288.0,
                    "SpeedFL": 260.0,
                    "FreshTyre": lap == 1,
                    "Rainfall": 0.0,
                })
    return pd.DataFrame(rows)


def predict_next_gp_pitstops(rf_model, xgb_model, next_gp: dict, n_laps: int = 58) -> pd.DataFrame:
    """Prédit les probabilités de pitstop pour le prochain GP."""
    df = build_next_gp_scenarios(n_laps=n_laps)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    df["PitProba_RF"] = rf_model.predict_proba(X)[:, 1]
    if xgb_model is not None:
        df["PitProba_XGB"] = xgb_model.predict_proba(X)[:, 1]

    # Probabilité cumulée : P(avoir pité avant le tour t) = 1 − Π_{k=1}^{t}(1−p_k)
    # Donne des courbes monotones croissantes interprétables comme "fenêtre de pitstop"
    for col_lap, col_cum in [("PitProba_RF", "CumPitProba_RF"), ("PitProba_XGB", "CumPitProba_XGB")]:
        if col_lap not in df.columns:
            continue
        cum_vals = np.zeros(len(df))
        for (compound, position), grp in df.groupby(["Compound", "Position"]):
            grp_s = grp.sort_values("TyreLife")
            probs = np.clip(grp_s[col_lap].values, 0.0, 1.0 - 1e-9)
            cum_p = 1.0 - np.cumprod(1.0 - probs)
            cum_vals[grp_s.index] = cum_p
        df[col_cum] = cum_vals

    return df


def plot_pitstop_forecast(df_forecast: pd.DataFrame, next_gp: dict) -> plt.Figure:
    """Courbes de probabilité cumulée de pitstop vs âge des pneus."""
    gp_name = next_gp.get("EventName", "Prochain GP")
    year = next_gp.get("Year", "")
    df_p10 = df_forecast[df_forecast["Position"] == 10].copy()
    compound_colors = {"SOFT": "#e74c3c", "MEDIUM": "#f39c12", "HARD": "#95a5a6"}

    # Préférer les colonnes cumulées si disponibles
    col_rf  = "CumPitProba_RF"  if "CumPitProba_RF"  in df_p10.columns else "PitProba_RF"
    col_xgb = "CumPitProba_XGB" if "CumPitProba_XGB" in df_p10.columns else "PitProba_XGB"
    ylabel  = "P(avoir pité avant ce tour)" if col_rf.startswith("Cum") else "P(pitstop ce tour)"
    has_xgb = col_xgb in df_p10.columns
    n_axes  = 2 if has_xgb else 1

    fig, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 5), squeeze=False)
    axes = axes[0]

    for ax, (col, label) in zip(axes, [(col_rf, "Random Forest"), (col_xgb, "XGBoost")]):
        if col not in df_p10.columns:
            break
        for compound, grp in df_p10.groupby("Compound"):
            grp_s = grp.sort_values("TyreLife")
            ax.plot(grp_s["TyreLife"], grp_s[col],
                    label=compound, color=compound_colors.get(compound, "gray"), linewidth=2.5)
        ax.axhline(0.5, color="black", linestyle=":", alpha=0.5, label="Seuil 50%")
        ax.set_xlabel("Âge des pneus (tours)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Fenêtre de pitstop — {label}\n{year} {gp_name}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run(
    year: int = 2023,
    gp_name: str = "Bahrain",
    session_type: str = "R",
    multi_season: bool = False,
) -> dict:
    """
    Point d'entrée principal du module supervisé.

    Returns:
        Dictionnaire contenant les modèles, métriques et figures.
    """
    logger.info("MODULE : Prédiction du tour de pitstop (Supervisé)")

    if multi_season:
        logger.info("Chargement multi-saisons (2021-2024)...")
        sessions = load_multiple_seasons(start=2021, end=2024, session_type=session_type)
    else:
        session = load_session(year, gp_name, session_type)
        sessions = [session] if session else []

    if not sessions:
        logger.error("Aucune session chargée.")
        return {}

    logger.info(f"{len(sessions)} session(s) chargée(s). Construction du dataset...")
    df = build_dataset(sessions)

    if df.empty:
        logger.error("Dataset vide.")
        return {}

    logger.info(
        f"Dataset : {len(df)} tours  |  Pitstops : {df['HasPitStop'].sum()} "
        f"({100*df['HasPitStop'].mean():.1f}%)"
    )

    X, y = prepare_Xy(df)
    X_train, X_test, y_train, y_test = _temporal_split(df, X, y, test_size=0.2)
    logger.info(f"Split temporel — Train: {len(X_train)}  Test: {len(X_test)}")

    # Random Forest
    rf_model, rf_params = train_random_forest(X_train, y_train)
    metrics_rf = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # XGBoost
    metrics_xgb = None
    xgb_model = None
    if HAS_XGB:
        xgb_model, _ = train_xgboost(X_train, y_train)
        if xgb_model:
            metrics_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Test de McNemar RF vs XGBoost
    if metrics_xgb is not None:
        p_mcnemar = _mcnemar_test(
            np.array(y_test),
            metrics_rf["y_pred"],
            metrics_xgb["y_pred"],
        )
        conclusion = "différentes (H0 rejetée)" if p_mcnemar < 0.05 else "non différentes (H0 non rejetée)"
        logger.info(
            f"[McNemar] RF vs XGBoost : p = {p_mcnemar:.4f}  → "
            f"Erreurs {conclusion} au seuil α=0.05"
        )
    else:
        p_mcnemar = None

    # Sauvegarde des modèles
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(rf_model, MODELS_DIR / "rf_pitstop.joblib")
    logger.info(f"[RF] Modèle sauvegardé : {MODELS_DIR / 'rf_pitstop.joblib'}")
    if xgb_model is not None:
        joblib.dump(xgb_model, MODELS_DIR / "xgb_pitstop.joblib")
        logger.info(f"[XGB] Modèle sauvegardé : {MODELS_DIR / 'xgb_pitstop.joblib'}")

    # Figures
    figs = {}
    figs["cm_rf"] = plot_confusion_matrix(y_test, metrics_rf["y_pred"], "Random Forest")
    figs["fi_rf"] = plot_feature_importance(rf_model, X.columns.tolist(), "Random Forest")
    figs["roc_rf"] = plot_roc_with_ci(rf_model, X_test, y_test, "Random Forest")
    cal_result = plot_calibration_curve(rf_model, X_test, y_test, "Random Forest")
    figs["calibration_rf"] = cal_result[0]
    figs["shap_rf"] = plot_shap_importance(rf_model, X_test, "Random Forest")

    if xgb_model and metrics_xgb:
        figs["cm_xgb"] = plot_confusion_matrix(y_test, metrics_xgb["y_pred"], "XGBoost")
        figs["fi_xgb"] = plot_feature_importance(xgb_model, X.columns.tolist(), "XGBoost")
        figs["roc_xgb"] = plot_roc_with_ci(xgb_model, X_test, y_test, "XGBoost")
        figs["shap_xgb"] = plot_shap_importance(xgb_model, X_test, "XGBoost")
        figs["comparison"] = plot_model_comparison(metrics_rf, metrics_xgb)
    else:
        figs["comparison"] = plot_model_comparison(metrics_rf, None)

    plt.show()

    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "data": df,
        "figures": figs,
        "metrics": {
            "rf": {"accuracy": metrics_rf["accuracy"], "f1": metrics_rf["f1"]},
            "xgb": {"accuracy": metrics_xgb["accuracy"], "f1": metrics_xgb["f1"]} if metrics_xgb else None,
            "mcnemar_p": p_mcnemar,
        },
    }
