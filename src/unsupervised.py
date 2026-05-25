"""
Module de détection d'anomalies non supervisée sur les données de course F1.

Modèles     : Isolation Forest (Liu et al., 2008) + DBSCAN (Ester et al., 1996)
Hypothèses  : Les anomalies sont rares (contamination ≤ 10%) et isolées dans l'espace
              des features normalisées. Les tours sont supposés i.i.d. conditionnellement
              au circuit et à la session (approximation raisonnable hors Safety Car).
Références  : Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). Isolation Forest. ICDM 2008.
              Ester, M. et al. (1996). A density-based algorithm for discovering clusters
              in large spatial databases with noise. KDD-96, pp. 226-231.

Isolation Forest — justification mathématique :
  Un arbre d'isolation partitionne récursivement l'espace en choisissant aléatoirement
  une feature et un seuil. La profondeur d'isolation h(x) d'un point x est le nombre
  de partitions nécessaires pour l'isoler. Pour les points normaux dans une région dense,
  E[h(x)] ≈ c(n) = 2·H(n-1) − 2(n-1)/n  (H = nombre harmonique, normalisation).
  Les anomalies ont E[h(x)] << c(n) car situées dans des régions peu denses.
  Score d'anomalie : s(x, n) = 2^{−E[h(x)]/c(n)} ∈ (0, 1), proche de 1 = anomalie.
  Complexité : O(n log n) à l'entraînement, O(log n) par prédiction par arbre.

DBSCAN — sélection automatique de eps :
  eps est sélectionné via la courbe k-distance (k = min_samples). On trie les distances
  au k-ième voisin en ordre croissant et on détecte le coude par le maximum de la
  dérivée seconde discrète Δ²d[i] = d[i+1] − 2·d[i] + d[i−1]. Ce point sépare la
  région dense (faibles distances) de la région clairsemée (fortes distances).
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data_loader import get_laps_features, load_session

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "LapTime_s", "LapTimeDelta", "TyreLife", "Position",
    "SpeedST", "SpeedFL", "SpeedI1", "SpeedI2",
]


def _prepare_features(laps_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Sélectionne, nettoie et normalise les features pour la détection d'anomalies."""
    available = [c for c in FEATURE_COLS if c in laps_df.columns]
    X = laps_df[available].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler


def auto_select_eps(X: np.ndarray, min_samples: int = 5) -> float:
    """
    Sélectionne automatiquement eps pour DBSCAN via la courbe k-distance.

    Algorithme :
      1. Calculer la distance au k-ième voisin (k = min_samples) pour chaque point.
      2. Trier ces distances en ordre croissant → courbe k-distance.
      3. Détecter le coude par maximum de la dérivée seconde discrète :
         Δ²d[i] = d[i+1] − 2·d[i] + d[i−1].
      Le coude sépare la région dense (faibles distances, points normaux)
      de la région clairsemée (fortes distances, bruit/anomalies).

    Args:
        X: Matrice de features normalisées (n_samples, n_features).
        min_samples: Paramètre min_samples de DBSCAN (= k pour k-NN).

    Returns:
        Valeur de eps correspondant au coude de la courbe k-distance.
    """
    k = min(min_samples, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])  # distance au k-ième voisin, ordre croissant

    if len(k_distances) < 3:
        return float(np.median(k_distances))

    # Dérivée seconde discrète : Δ²d[i] = d[i+1] − 2·d[i] + d[i−1]
    d2 = np.diff(k_distances, n=2)
    elbow_idx = int(np.argmax(d2)) + 1  # +1 : diff() réduit la longueur de 1
    eps_auto = float(k_distances[elbow_idx])
    logger.info(f"[DBSCAN] eps auto-sélectionné via coude k-distance = {eps_auto:.4f}")
    return eps_auto


def detect_isolation_forest(
    laps_df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Détecte les anomalies avec Isolation Forest.

    Args:
        laps_df: DataFrame issu de get_laps_features()
        contamination: Fraction attendue d'anomalies (0.0 à 0.5)
        random_state: Graine aléatoire pour la reproductibilité

    Returns:
        laps_df enrichi d'une colonne 'IF_Anomaly' (True = anomalie).
    """
    X_scaled, _ = _prepare_features(laps_df)
    if X_scaled.empty:
        laps_df["IF_Anomaly"] = False
        return laps_df

    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    preds = model.fit_predict(X_scaled)  # -1 = anomalie, 1 = normal
    scores = model.score_samples(X_scaled)

    result = laps_df.copy()
    result["IF_Anomaly"] = False
    result["IF_Score"] = np.nan
    result.loc[X_scaled.index, "IF_Anomaly"] = preds == -1
    result.loc[X_scaled.index, "IF_Score"] = scores
    return result


def detect_dbscan(
    laps_df: pd.DataFrame,
    eps: float | None = None,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Détecte les anomalies avec DBSCAN (points de bruit = anomalies).

    Si eps=None (défaut), eps est sélectionné automatiquement via auto_select_eps()
    qui détecte le coude de la courbe k-distance par dérivée seconde discrète.

    Args:
        laps_df: DataFrame issu de get_laps_features()
        eps: Rayon de voisinage DBSCAN. None = sélection automatique.
        min_samples: Nombre minimal de points pour former un cluster.

    Returns:
        laps_df enrichi d'une colonne 'DBSCAN_Anomaly' et 'DBSCAN_Label'.
    """
    X_scaled, _ = _prepare_features(laps_df)
    if X_scaled.empty:
        laps_df["DBSCAN_Anomaly"] = False
        laps_df["DBSCAN_Label"] = -1
        return laps_df

    if eps is None:
        eps = auto_select_eps(X_scaled.values, min_samples)

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X_scaled)

    result = laps_df.copy()
    result["DBSCAN_Label"] = -2
    result["DBSCAN_Anomaly"] = False
    result.loc[X_scaled.index, "DBSCAN_Label"] = labels
    result.loc[X_scaled.index, "DBSCAN_Anomaly"] = labels == -1  # bruit = anomalie
    return result


def compare_models(laps_df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Applique les deux modèles et valide les anomalies par z-score sur LapTimeDelta.

    Validation statistique :
      Pour chaque anomalie détectée, on calcule le z-score de LapTimeDelta :
      z = (LapTimeDelta − μ) / σ.  Une anomalie n'est confirmée que si |z| > z_threshold
      (seuil paramétrable, défaut 2.5 ≈ 1.24% des valeurs sous une loi normale).
      Cela réduit les faux positifs liés à des variations de performance normales.

    Args:
        laps_df: DataFrame issu de get_laps_features()
        z_threshold: Seuil sur |z-score| pour valider une anomalie (défaut 2.5).

    Returns:
        DataFrame avec IF_Anomaly, DBSCAN_Anomaly, Both_Anomaly,
        LapTimeDelta_Z (z-score) et Validated_Anomaly (consensus + z-score).
    """
    df = detect_isolation_forest(laps_df)
    df = detect_dbscan(df)
    df["Both_Anomaly"] = df["IF_Anomaly"] & df["DBSCAN_Anomaly"]

    # Validation statistique par z-score sur LapTimeDelta
    if "LapTimeDelta" in df.columns:
        mu = df["LapTimeDelta"].mean()
        sigma = df["LapTimeDelta"].std()
        if sigma > 1e-9:
            df["LapTimeDelta_Z"] = (df["LapTimeDelta"] - mu) / sigma
            # Anomalie validée = consensus des deux modèles ET |z| > seuil
            df["Validated_Anomaly"] = (
                df["Both_Anomaly"] & (df["LapTimeDelta_Z"].abs() > z_threshold)
            )
            n_raw = int(df["Both_Anomaly"].sum())
            n_val = int(df["Validated_Anomaly"].sum())
            logger.info(
                f"[compare_models] Consensus brut: {n_raw} → "
                f"validé (|z|>{z_threshold}): {n_val} anomalies"
            )
        else:
            df["LapTimeDelta_Z"] = 0.0
            df["Validated_Anomaly"] = df["Both_Anomaly"]
    else:
        df["Validated_Anomaly"] = df["Both_Anomaly"]

    return df


def plot_anomalies_timeline(df: pd.DataFrame, title: str = "Anomalies détectées") -> plt.Figure:
    """
    Visualise les anomalies sur la timeline de course (LapNumber vs LapTime_s).

    Args:
        df: DataFrame avec colonnes LapNumber, LapTime_s, IF_Anomaly, DBSCAN_Anomaly
        title: Titre du graphique

    Returns:
        Figure matplotlib.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, (col, label, color) in zip(
        axes,
        [("IF_Anomaly", "Isolation Forest", "#e74c3c"),
         ("DBSCAN_Anomaly", "DBSCAN", "#3498db")],
    ):
        normal = df[~df[col]]
        anomaly = df[df[col]]

        ax.scatter(normal["LapNumber"], normal["LapTime_s"], s=15, alpha=0.5,
                   color="steelblue", label="Normal")
        ax.scatter(anomaly["LapNumber"], anomaly["LapTime_s"], s=60, alpha=0.9,
                   color=color, marker="X", zorder=5, label=f"Anomalie ({col})")

        for lap in anomaly["LapNumber"].unique():
            ax.axvline(lap, color=color, alpha=0.15, linewidth=1.5)

        ax.set_ylabel("Temps au tour (s)")
        ax.set_title(f"Modèle : {label}  —  {anomaly.shape[0]} anomalies détectées")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Numéro de tour")
    plt.tight_layout()
    return fig


def plot_feature_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogramme des features clés coloré selon IF_Anomaly."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    n = len(available)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(available):
        for anomaly, grp in df.groupby("IF_Anomaly"):
            color = "#e74c3c" if anomaly else "steelblue"
            label = "Anomalie" if anomaly else "Normal"
            axes[i].hist(grp[col].dropna(), bins=40, alpha=0.6, color=color, label=label, density=True)
        axes[i].set_title(col, fontsize=10)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution des features — Normal vs Anomalie (IF)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def print_anomaly_report(df: pd.DataFrame) -> None:
    """
    Affiche un résumé textuel des anomalies détectées.

    Inclut l'indice de Silhouette pour DBSCAN :
      s(i) = (b(i) − a(i)) / max(a(i), b(i)),  où a(i) = distance intra-cluster
      et b(i) = distance au cluster le plus proche. s ∈ [−1, 1], proche de 1 = bon clustering.
    """
    total = len(df)
    for col, label in [
        ("IF_Anomaly", "Isolation Forest"),
        ("DBSCAN_Anomaly", "DBSCAN"),
        ("Both_Anomaly", "Consensus"),
        ("Validated_Anomaly", "Validé (z-score)"),
    ]:
        if col not in df.columns:
            continue
        n = df[col].sum()
        pct = 100 * n / total if total > 0 else 0
        logger.info(f"  {label:25s} : {n:4d} / {total} tours anormaux  ({pct:.1f}%)")

    # Indice de Silhouette pour DBSCAN
    if "DBSCAN_Label" in df.columns:
        X_scaled, _ = _prepare_features(df)
        if not X_scaled.empty:
            labels = df.loc[X_scaled.index, "DBSCAN_Label"].values
            mask = labels != -1
            unique_clusters = np.unique(labels[mask])
            if len(unique_clusters) >= 2 and mask.sum() >= 2:
                sil = silhouette_score(X_scaled.values[mask], labels[mask])
                logger.info(
                    f"  Silhouette DBSCAN         : {sil:.4f}  "
                    f"(1=clusters compacts/séparés, -1=mauvais clustering)"
                )
            else:
                logger.info("  Silhouette DBSCAN         : N/A (< 2 clusters non-bruit)")

    if "Validated_Anomaly" in df.columns and "Driver" in df.columns:
        logger.info("\n  Pilotes les plus touchés (anomalies validées) :")
        top = df[df["Validated_Anomaly"]].groupby("Driver").size().sort_values(ascending=False).head(5)
        for drv, cnt in top.items():
            logger.info(f"    {drv}: {cnt} tours")


def run_season_anomalies(sessions: list) -> pd.DataFrame:
    """
    Calcule les statistiques d'anomalies Isolation Forest + DBSCAN pour chaque session.

    Args:
        sessions: Liste de sessions FastF1 chargées.

    Returns:
        DataFrame avec une ligne par session : Circuit, Year, TotalLaps, IF_Rate, etc.
    """
    rows = []
    for session in sessions:
        try:
            laps = get_laps_features(session)
            if laps.empty:
                continue
            df_ann = compare_models(laps)
            try:
                year = int(session.date.year)
            except Exception:
                year = 0
            circuit = session.event.get("EventName", "?")
            rows.append({
                "Circuit": circuit,
                "Year": year,
                "TotalLaps": len(df_ann),
                "IF_Anomalies": int(df_ann["IF_Anomaly"].sum()),
                "DBSCAN_Anomalies": int(df_ann["DBSCAN_Anomaly"].sum()),
                "Consensus": int(df_ann["Both_Anomaly"].sum()),
                "Validated": int(df_ann.get("Validated_Anomaly", pd.Series(False)).sum()),
                "IF_Rate": round(100 * df_ann["IF_Anomaly"].mean(), 1),
                "DBSCAN_Rate": round(100 * df_ann["DBSCAN_Anomaly"].mean(), 1),
            })
        except Exception as e:
            logger.warning(f"[unsupervised] Session ignorée : {e}")
    return pd.DataFrame(rows)


def plot_season_anomalies(df_stats: pd.DataFrame) -> plt.Figure:
    """
    Visualisation des taux d'anomalies par Grand Prix sur la saison.
    Barplot horizontal (IF + DBSCAN) et ligne de tendance.
    """
    if df_stats.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Aucune donnée", ha="center", va="center")
        return fig

    df_s = df_stats.sort_values(["Year", "IF_Rate"], ascending=[True, True]).copy()
    label = df_s["Year"].astype(str) + " " + df_s["Circuit"]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(df_s) * 0.35)))

    ax = axes[0]
    colors = ["#e74c3c" if y >= 2026 else "#3498db" for y in df_s["Year"]]
    bars = ax.barh(label, df_s["IF_Rate"], color=colors, alpha=0.82)
    ax.axvline(5.0, color="black", linestyle=":", alpha=0.4, label="Seuil 5%")
    ax.set_xlabel("Taux d'anomalies IF (%)")
    ax.set_title("Isolation Forest — taux par GP", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%", va="center", fontsize=8)

    ax2 = axes[1]
    x = np.arange(len(df_s))
    ax2.bar(x - 0.2, df_s["IF_Rate"], width=0.35, label="Isolation Forest",
            color="#e74c3c", alpha=0.8)
    ax2.bar(x + 0.2, df_s["DBSCAN_Rate"], width=0.35, label="DBSCAN",
            color="#3498db", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_s["Circuit"], rotation=60, ha="right", fontsize=8)
    ax2.set_ylabel("Taux d'anomalies (%)")
    ax2.set_title("IF vs DBSCAN par GP", fontweight="bold")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Anomalies détectées — Saisons 2025–2026", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def run(year: int = 2023, gp_name: str = "Monaco", session_type: str = "R") -> dict:
    """
    Point d'entrée principal du module non supervisé.

    Args:
        year: Année de la saison
        gp_name: Grand Prix
        session_type: Type de session

    Returns:
        Dictionnaire contenant le DataFrame annoté et les figures.
    """
    logger.info(f"MODULE : Détection d'anomalies (Non supervisé)")
    logger.info(f"Session : {year} {gp_name} {session_type}")

    session = load_session(year, gp_name, session_type)
    if session is None:
        logger.error("Session non chargée.")
        return {}

    laps_df = get_laps_features(session)
    logger.info(f"Tours chargés : {len(laps_df)}")

    if laps_df.empty:
        logger.error("Aucune donnée de tour disponible.")
        return {}

    df_annotated = compare_models(laps_df)

    logger.info("Résultats :")
    print_anomaly_report(df_annotated)

    event_name = f"{year} GP {gp_name}"
    fig_timeline = plot_anomalies_timeline(df_annotated, title=f"Anomalies — {event_name}")
    fig_dist = plot_feature_distribution(df_annotated)

    plt.show()

    return {
        "data": df_annotated,
        "figures": {"timeline": fig_timeline, "distribution": fig_dist},
        "metrics": {
            "total_laps": len(df_annotated),
            "if_anomalies": int(df_annotated["IF_Anomaly"].sum()),
            "dbscan_anomalies": int(df_annotated["DBSCAN_Anomaly"].sum()),
            "consensus_anomalies": int(df_annotated.get("Both_Anomaly", pd.Series(False)).sum()),
            "validated_anomalies": int(df_annotated.get("Validated_Anomaly", pd.Series(False)).sum()),
        },
    }
