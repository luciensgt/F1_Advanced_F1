"""
Module NLP — Analyse des messages radio F1 pilote-ingénieur.

Modèle      : Transformers HuggingFace (RoBERTa) + analyse de corrélation non-paramétrique
Hypothèses  : Les messages radio encodent un signal observable de l'état émotionnel et
              stratégique du pilote/ingénieur. La relation sentiment/performance est
              supposée monotone (justifiant le test de Spearman plutôt que Pearson).
              Les erreurs de transcription sont supposées aléatoires (non biaisées).
Références  : Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional
              Transformers for Language Understanding. NAACL 2019.
              Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining
              Approach. arXiv:1907.11692.
              Spearman, C. (1904). The proof and measurement of association between
              two things. American Journal of Psychology, 15(1), 72-101.
              Mann, H. B. & Whitney, D. R. (1947). On a test of whether one of two
              random variables is stochastically larger than the other.
              Annals of Mathematical Statistics, 18(1), 50-60.

Corrélation de Spearman vs Pearson :
  ρ_s = 1 − 6Σd_i² / n(n²−1), où d_i = rang(x_i) − rang(y_i).
  Robuste aux outliers et ne suppose pas la normalité des variables.
  Test de significativité : t = ρ_s√(n−2)/√(1−ρ_s²) ~ t(n−2) sous H0.

Correction de Bonferroni (comparaisons multiples) :
  Lorsqu'on teste N pilotes simultanément, la probabilité d'une fausse découverte
  explose (problème de la multiplicité). La correction de Bonferroni contrôle le
  taux d'erreur de famille (FWER) : α' = α/N. C'est une correction conservatrice.

Test de Mann-Whitney U (non-paramétrique) :
  Teste H0 : P(X > Y) = 0.5, soit que les deux groupes (ex: STRATEGIE vs ALERTE)
  ont la même distribution de LapTimeDelta. Ne suppose pas la normalité.
  Statistique U = #{(x_i, y_j) : x_i > y_j}, normalisée par n₁·n₂.
"""

import logging
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")

try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from src.data_loader import get_laps_features, get_radio_messages, load_session

logger = logging.getLogger(__name__)

# ─── Règles de classification ─────────────────────────────────────────────────

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

CATEGORY_RULES = {
    "ALERTE_TECHNIQUE": [
        r"\b(engine|gearbox|brake|hydraulic|tyre|tire|puncture|damage|failure|issue|problem|overheating|mechanical|retired|retire)\b",
        r"\b(moteur|frein|boîte|pneu|crevaison|problème|surchauffe|mécanique)\b",
    ],
    "STRATEGIE": [
        r"\b(pit|box|undercut|overcut|lap\s*\d+|stay out|push|gap|delta|switch|medium|hard|soft|inter|wet|drs|tyre|compound)\b",
        r"\b(arrêt|compound|souple|dur|pluie|inter|rester dehors)\b",
    ],
    "ENCOURAGEMENT": [
        r"\b(well done|great|fantastic|brilliant|perfect|keep pushing|good work|amazing|excellent|safe|clear|go)\b",
        r"\b(super|excellent|parfait|continue|bravo|bien joué)\b",
    ],
    "INCIDENT": [
        r"\b(safety car|virtual safety car|vsc|sc|red flag|yellow|debris|crash|accident|collision|incident|investigate|penalty|unsafe)\b",
        r"\b(voiture de sécurité|drapeau rouge|jaune|accident|débris|pénalité)\b",
    ],
}

_NEG_RE = re.compile(
    r"\b(red flag|crash|accident|collision|retired|failure|damage|debris|unsafe|"
    r"incident|penalty|investigate|yellow|danger|problem|issue|mechanical|retire)\b",
    re.IGNORECASE,
)
_POS_RE = re.compile(
    r"\b(clear|green|safe|go|drs enabled|well done|brilliant|fantastic|great|excellent|"
    r"perfect|amazing|keep pushing|track clear|all clear)\b",
    re.IGNORECASE,
)

LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
SENTIMENT_NUM = {"positive": 1, "neutral": 0, "negative": -1}


# ─── Nettoyage texte ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise un message pour l'analyse NLP."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[^\w\s'.,!?/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:512]


def classify_message_type(text: str) -> str:
    """Classifie le type d'un message par règles regex."""
    text_lower = text.lower()
    for category, patterns in CATEGORY_RULES.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return category
    return "AUTRE"


# ─── Analyse de sentiment ─────────────────────────────────────────────────────

def _keyword_sentiment(text: str) -> dict:
    """Sentiment basé sur mots-clés — fallback quand le transformer n'est pas disponible."""
    if _NEG_RE.search(text):
        return {"label": "negative", "score": 0.72}
    if _POS_RE.search(text):
        return {"label": "positive", "score": 0.71}
    return {"label": "neutral", "score": 0.60}


def load_sentiment_pipeline():
    """Charge le pipeline HuggingFace, avec fallback sur distilbert puis mots-clés."""
    if not HAS_TRANSFORMERS:
        logger.warning("[NLP] Transformers absent — fallback mots-clés activé.")
        return None
    try:
        logger.info(f"[NLP] Chargement du modèle : {SENTIMENT_MODEL}")
        return hf_pipeline("sentiment-analysis", model=SENTIMENT_MODEL, top_k=1)
    except Exception:
        logger.warning("[NLP] Modèle principal indisponible — fallback distilbert.")
        try:
            return hf_pipeline("sentiment-analysis", top_k=1)
        except Exception as e:
            logger.error(f"[NLP] Impossible de charger un transformer : {e} — fallback mots-clés.")
            return None


def analyze_sentiment_batch(texts: list[str], pipe, batch_size: int = 32) -> list[dict]:
    """Analyse de sentiment sur un batch de textes."""
    if pipe is None:
        return [_keyword_sentiment(t) for t in texts]

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        try:
            raw = pipe(batch)
            for r in raw:
                item = r[0] if isinstance(r, list) else r
                label = LABEL_MAP.get(item["label"], item["label"].lower())
                results.append({"label": label, "score": float(item["score"])})
        except Exception as e:
            logger.warning(f"[NLP] Erreur batch {i // batch_size} : {e}")
            results += [_keyword_sentiment(t) for t in batch]

    return results


# ─── Construction du dataset ──────────────────────────────────────────────────

def _map_racing_number_to_driver(session, df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne 'Driver' à partir de 'RacingNumber'."""
    df = df.copy()
    if "Driver" in df.columns:
        return df

    num_to_drv = {}
    try:
        for _, row in session.results.iterrows():
            num = str(row.get("DriverNumber", "")).strip()
            abbr = str(row.get("Abbreviation", "")).strip()
            if num and abbr:
                num_to_drv[num] = abbr
    except Exception:
        pass

    if "RacingNumber" in df.columns:
        df["Driver"] = df["RacingNumber"].apply(
            lambda n: num_to_drv.get(str(n).strip(), "ALL") if pd.notna(n) else "ALL"
        )
    else:
        df["Driver"] = "ALL"

    return df


def build_radio_dataset(session, pipe) -> pd.DataFrame:
    """Construit un DataFrame annoté à partir des messages disponibles."""
    radio_df = get_radio_messages(session)

    if radio_df.empty:
        logger.info("[NLP] Aucun message disponible — données synthétiques pour démo.")
        return _generate_synthetic_radios(pipe)

    if "Message" not in radio_df.columns:
        if "Category" in radio_df.columns:
            radio_df["Message"] = radio_df["Category"].fillna("")
        else:
            radio_df["Message"] = ""

    radio_df["CleanMessage"] = radio_df["Message"].apply(clean_text)
    radio_df = radio_df[radio_df["CleanMessage"].str.len() > 3].copy()

    if radio_df.empty:
        logger.info("[NLP] Messages tous vides après nettoyage — données synthétiques.")
        return _generate_synthetic_radios(pipe)

    radio_df = _map_racing_number_to_driver(session, radio_df)
    radio_df["Type"] = radio_df["CleanMessage"].apply(classify_message_type)

    texts = radio_df["CleanMessage"].tolist()
    sentiments = analyze_sentiment_batch(texts, pipe)
    radio_df["Sentiment"] = [s["label"] for s in sentiments]
    radio_df["SentimentScore"] = [s["score"] for s in sentiments]
    radio_df["SentimentNum"] = radio_df["Sentiment"].map(SENTIMENT_NUM).fillna(0)

    logger.info(
        f"[NLP] {len(radio_df)} messages traités — "
        f"méthode: {'transformer' if pipe else 'mots-clés'}"
    )
    return radio_df.reset_index(drop=True)


def _generate_synthetic_radios(pipe=None) -> pd.DataFrame:
    """Données synthétiques avec sentiment calculé."""
    templates = [
        ("HAM", "Box this lap, we go to medium tyres"),
        ("VER", "The front left is gone, I can't push anymore, serious problem"),
        ("LEC", "Brilliant lap Charles, keep pushing, you are amazing"),
        ("SAI", "Safety car deployed, crash at turn 3, red flag possible"),
        ("ALO", "We have a hydraulic failure, check the dashboard, issue with the engine"),
        ("NOR", "Track clear, DRS enabled, gap to Hamilton is 1.2 seconds push push"),
        ("RUS", "Well done George, excellent job out there, fantastic lap"),
        ("PER", "Yellow flags sector 2, debris on track, danger ahead"),
        ("STR", "Accident in the pitlane, investigation underway, penalty likely"),
        ("GAS", "Copy, we will undercut on lap 32, medium compound ready"),
        ("HAM", "Red flag deployed, crash at turn 8, retire the car"),
        ("VER", "Great job, track is clear, push to the end, fantastic"),
        ("LEC", "Incident noted, unsafe release investigated, penalty possible"),
        ("SAI", "All clear, safety car ending, DRS enabled in three laps"),
        ("ALO", "Something is wrong with the brake, mechanical issue, problem"),
    ]

    rows = []
    for i, (drv, msg) in enumerate(templates):
        clean = clean_text(msg)
        sent = _keyword_sentiment(clean) if pipe is None else analyze_sentiment_batch([clean], pipe)[0]
        rows.append({
            "Time": pd.Timedelta(seconds=i * 180),
            "Driver": drv, "Message": msg, "CleanMessage": clean,
            "Type": classify_message_type(msg),
            "Sentiment": sent["label"],
            "SentimentScore": sent["score"],
            "SentimentNum": SENTIMENT_NUM.get(sent["label"], 0),
        })
    return pd.DataFrame(rows)


# ─── Corrélation sentiment / performance ──────────────────────────────────────

def correlate_sentiment_performance(radio_df: pd.DataFrame, laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrèle le sentiment radio avec la dégradation de performance par pilote.

    Utilise la corrélation de Spearman (robuste aux outliers, sans hypothèse de
    normalité) entre le sentiment moyen et l'écart au temps médian (LapTimeDelta).

    Correction de Bonferroni appliquée pour les comparaisons multiples :
    si N pilotes sont comparés simultanément, le seuil de significativité
    est ajusté à α' = 0.05 / N pour contrôler le FWER (Family-Wise Error Rate).

    Returns:
        DataFrame avec AvgSentiment, RadioCount, AvgTimeDelta, et attributs
        spearman_r, spearman_pval, bonferroni_alpha stockés dans df.attrs.
    """
    if radio_df.empty or laps_df.empty:
        return pd.DataFrame()
    if "Driver" not in radio_df.columns or "LapTimeDelta" not in laps_df.columns:
        return pd.DataFrame()

    drivers_in_race = laps_df["Driver"].unique().tolist()

    rows_expanded = []
    for _, row in radio_df.iterrows():
        if row["Driver"] == "ALL":
            for drv in drivers_in_race:
                r = row.copy()
                r["Driver"] = drv
                rows_expanded.append(r)
        else:
            rows_expanded.append(row)

    expanded = pd.DataFrame(rows_expanded)
    expanded = expanded[expanded["Driver"].isin(drivers_in_race)]
    if expanded.empty:
        return pd.DataFrame()

    sentiment_agg = (
        expanded.groupby("Driver")["SentimentNum"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "AvgSentiment", "count": "RadioCount"})
        .reset_index()
    )
    perf_agg = (
        laps_df.groupby("Driver")["LapTimeDelta"]
        .mean()
        .reset_index()
        .rename(columns={"LapTimeDelta": "AvgTimeDelta"})
    )

    result = sentiment_agg.merge(perf_agg, on="Driver", how="inner")

    if len(result) < 3:
        return result

    # Corrélation de Spearman (non-paramétrique, robuste aux outliers)
    rs, pval = spearmanr(result["AvgSentiment"], result["AvgTimeDelta"])

    # Correction de Bonferroni : α' = 0.05 / N (contrôle du FWER)
    n_drivers = len(result)
    alpha_bonferroni = 0.05 / n_drivers

    result.attrs["spearman_r"] = float(rs)
    result.attrs["spearman_pval"] = float(pval)
    result.attrs["bonferroni_alpha"] = float(alpha_bonferroni)

    sig_raw = "OUI" if pval < 0.05 else "NON"
    sig_bonf = "OUI *" if pval < alpha_bonferroni else "NON"

    logger.info(f"[NLP] Corrélation de Spearman : r_s = {rs:.3f}  |  p = {pval:.4f}")
    logger.info(
        f"[NLP] Significative (α=0.05) : {sig_raw}  |  "
        f"Après Bonferroni (α'={alpha_bonferroni:.4f}, N={n_drivers}) : {sig_bonf}"
    )
    if rs < 0:
        logger.info("[NLP] r_s < 0 : sentiment positif associé à de meilleures performances (LapTimeDelta < 0).")
    else:
        logger.info("[NLP] r_s > 0 : sentiment positif associé à des performances dégradées.")

    return result


def test_message_type_impact(radio_df: pd.DataFrame, laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test de Mann-Whitney U entre types de messages pour comparer leur impact
    sur la performance (LapTimeDelta).

    Teste H0 : la distribution de LapTimeDelta associée aux messages de type T1
    est identique à celle des messages de type T2.
    Le test est non-paramétrique (pas d'hypothèse de normalité) et robuste
    aux petits échantillons déséquilibrés.

    Statistique : U = #{(x_i, y_j) : x_i > y_j}, normalisée par n₁·n₂.
    Sous H0, U ~ N(n₁n₂/2, √(n₁n₂(n₁+n₂+1)/12)) pour n₁, n₂ suffisants.

    Returns:
        DataFrame avec les paires de types testées, U, p-value et significativité.
    """
    if radio_df.empty or laps_df.empty:
        return pd.DataFrame()

    perf = laps_df.groupby("Driver")["LapTimeDelta"].mean().reset_index()
    perf = perf.rename(columns={"LapTimeDelta": "_perf_delta"})
    # On ne garde que Driver + Type depuis radio_df pour éviter toute collision
    radio_slim = radio_df[["Driver", "Type"]].copy()
    merged = radio_slim.merge(perf, on="Driver", how="left")
    merged = merged.rename(columns={"_perf_delta": "LapTimeDelta"})
    merged = merged.dropna(subset=["LapTimeDelta", "Type"])

    if merged.empty:
        return pd.DataFrame()

    types = merged["Type"].unique()
    results = []

    for i, t1 in enumerate(types):
        for t2 in types[i + 1:]:
            g1 = merged[merged["Type"] == t1]["LapTimeDelta"].values
            g2 = merged[merged["Type"] == t2]["LapTimeDelta"].values
            if len(g1) < 3 or len(g2) < 3:
                continue

            u_stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
            sig = pval < 0.05
            results.append({
                "Type_1": t1, "Type_2": t2,
                "n1": len(g1), "n2": len(g2),
                "Median_1": round(float(np.median(g1)), 4),
                "Median_2": round(float(np.median(g2)), 4),
                "U_stat": round(float(u_stat), 2),
                "p_value": round(float(pval), 4),
                "significant_0.05": sig,
            })
            logger.info(
                f"[Mann-Whitney] {t1} vs {t2}: "
                f"U={u_stat:.1f}  p={pval:.4f}  {'* sig.' if sig else 'n.s.'}"
            )

    if not results:
        logger.warning("[NLP] Pas assez de données pour le test Mann-Whitney.")
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    logger.info(
        f"[NLP] Mann-Whitney : {df_res['significant_0.05'].sum()} paires significatives "
        f"sur {len(df_res)} testées."
    )
    return df_res


# ─── Visualisations ───────────────────────────────────────────────────────────

def plot_sentiment_distribution(radio_df: pd.DataFrame) -> plt.Figure:
    """Distribution des sentiments et des types de messages."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if "Sentiment" in radio_df.columns:
        counts = radio_df["Sentiment"].value_counts()
        colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
        bar_colors = [colors.get(s, "gray") for s in counts.index]
        axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor="white")
        axes[0].set_title("Distribution des sentiments", fontweight="bold")
        axes[0].set_ylabel("Nombre de messages")
        axes[0].grid(True, axis="y", alpha=0.3)
        for i, (idx, val) in enumerate(counts.items()):
            axes[0].text(i, val + 0.3, str(val), ha="center", fontsize=11, fontweight="bold")

    if "Type" in radio_df.columns:
        type_counts = radio_df["Type"].value_counts()
        type_colors = {
            "ALERTE_TECHNIQUE": "#e74c3c", "STRATEGIE": "#3498db",
            "ENCOURAGEMENT": "#2ecc71", "INCIDENT": "#f39c12", "AUTRE": "#95a5a6",
        }
        t_colors = [type_colors.get(t, "gray") for t in type_counts.index]
        axes[1].barh(type_counts.index, type_counts.values, color=t_colors)
        axes[1].set_title("Types de messages radio", fontweight="bold")
        axes[1].set_xlabel("Nombre de messages")
        axes[1].grid(True, axis="x", alpha=0.3)

    plt.suptitle("Analyse NLP des radios F1", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_driver_sentiment(radio_df: pd.DataFrame) -> plt.Figure:
    """Heatmap sentiment moyen par pilote et type de message.

    Les messages 'ALL' (direction de course : Safety Car, drapeaux…) sont diffusés
    à tous les pilotes individuels pour que la heatmap reflète le contexte réel de
    chaque pilote, pas uniquement ses messages directs.
    """
    if radio_df.empty or "Driver" not in radio_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Pas de données par pilote", ha="center", va="center")
        return fig

    # Broadcast messages "ALL" vers chaque pilote individuel
    individual_drivers = [d for d in radio_df["Driver"].unique() if d != "ALL"]
    all_msgs = radio_df[radio_df["Driver"] == "ALL"]
    driver_msgs = radio_df[radio_df["Driver"] != "ALL"]

    if not all_msgs.empty and individual_drivers:
        broadcast = pd.concat(
            [all_msgs.assign(Driver=drv) for drv in individual_drivers],
            ignore_index=True,
        )
        df_full = pd.concat([driver_msgs, broadcast], ignore_index=True)
    else:
        df_full = driver_msgs if not driver_msgs.empty else radio_df

    if df_full.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Pas de données par pilote", ha="center", va="center")
        return fig

    pivot = df_full.pivot_table(
        values="SentimentNum", index="Driver", columns="Type", aggfunc="mean"
    ).fillna(0)

    if pivot.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Pivot vide", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(12, max(5, len(pivot) * 0.55)))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={"label": "Sentiment moyen (−1=négatif, +1=positif)"})
    ax.set_title("Sentiment moyen par pilote et type de message\n"
                 "(messages direction de course inclus)", fontweight="bold")
    ax.set_xlabel("Type de message")
    ax.set_ylabel("Pilote")
    plt.tight_layout()
    return fig


def plot_sentiment_performance_correlation(corr_df: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot sentiment vs dégradation de performance.

    Affiche r_s (Spearman) et la significativité après correction de Bonferroni.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    if corr_df.empty:
        ax.text(0.5, 0.5, "Données de corrélation insuffisantes",
                ha="center", va="center", fontsize=12)
        ax.set_title("Corrélation : Sentiment radio ↔ Performance", fontweight="bold")
        return fig

    scatter = ax.scatter(
        corr_df["AvgSentiment"], corr_df["AvgTimeDelta"],
        s=corr_df.get("RadioCount", pd.Series([50] * len(corr_df))) * 10,
        alpha=0.75, c=corr_df["AvgSentiment"], cmap="RdYlGn", vmin=-1, vmax=1,
        edgecolors="white", linewidth=0.5,
    )

    for _, row in corr_df.iterrows():
        ax.annotate(row["Driver"], (row["AvgSentiment"], row["AvgTimeDelta"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8, alpha=0.85)

    x_std = corr_df["AvgSentiment"].std()
    if len(corr_df) > 2 and x_std > 1e-6:
        try:
            z = np.polyfit(corr_df["AvgSentiment"], corr_df["AvgTimeDelta"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(corr_df["AvgSentiment"].min(), corr_df["AvgSentiment"].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.6, label="Tendance linéaire")
        except np.linalg.LinAlgError:
            pass

    # Annotation statistique
    rs = corr_df.attrs.get("spearman_r")
    pval = corr_df.attrs.get("spearman_pval")
    bonf_alpha = corr_df.attrs.get("bonferroni_alpha")
    if rs is not None:
        sig_str = ""
        if pval is not None and bonf_alpha is not None:
            sig_str = " *Bonferroni" if pval < bonf_alpha else " (n.s. Bonferroni)"
        ax.text(0.03, 0.97,
                f"r_s = {rs:.3f}  p = {pval:.4f}{sig_str}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.colorbar(scatter, ax=ax, label="Sentiment moyen")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Sentiment moyen (−1 = négatif, +1 = positif)")
    ax.set_ylabel("Écart au temps médian (s)  — positif = plus lent")
    ax.set_title("Corrélation : Sentiment radio ↔ Performance\n"
                 "(Spearman ρ_s, correction de Bonferroni)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def run(year: int = 2023, gp_name: str = "Monaco", session_type: str = "R") -> dict:
    """Point d'entrée principal du module NLP."""
    logger.info(f"MODULE : Analyse NLP des radios F1")
    logger.info(f"Session : {year} {gp_name} {session_type}")

    session = load_session(year, gp_name, session_type)
    laps_df = get_laps_features(session) if session else pd.DataFrame()

    pipe = load_sentiment_pipeline()
    radio_df = build_radio_dataset(session, pipe)

    logger.info(f"{len(radio_df)} messages traités")
    if not radio_df.empty:
        logger.info(f"Types      : {radio_df['Type'].value_counts().to_dict()}")
        logger.info(f"Sentiments : {radio_df['Sentiment'].value_counts().to_dict()}")

    corr_df = correlate_sentiment_performance(radio_df, laps_df)

    # Test Mann-Whitney par type de message
    df_mw = test_message_type_impact(radio_df, laps_df)

    figs = {
        "distribution": plot_sentiment_distribution(radio_df),
        "driver_heatmap": plot_driver_sentiment(radio_df),
        "correlation": plot_sentiment_performance_correlation(corr_df),
    }
    plt.show()

    neg_pct = 100 * (radio_df["Sentiment"] == "negative").mean() if not radio_df.empty else 0
    rs = corr_df.attrs.get("spearman_r") if not corr_df.empty else None
    pval = corr_df.attrs.get("spearman_pval") if not corr_df.empty else None

    return {
        "radio_data": radio_df,
        "correlation_data": corr_df,
        "mann_whitney": df_mw,
        "figures": figs,
        "metrics": {
            "total_messages": len(radio_df),
            "negative_pct": round(neg_pct, 1),
            "types": radio_df["Type"].value_counts().to_dict() if not radio_df.empty else {},
            "spearman_r": round(rs, 3) if rs is not None else None,
            "spearman_pval": round(pval, 4) if pval is not None else None,
        },
    }
