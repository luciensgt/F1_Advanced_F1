"""
Module de deep learning pour la prédiction du vainqueur d'une course F1.

Modèle      : MLP (Multilayer Perceptron) avec Batch Normalization et MC Dropout
Hypothèses  : Les features pré-course (grille, historique pilote/circuit) sont
              disponibles avant le départ. Hypothèse d'échangeabilité des courses
              au sein d'un même circuit (les patterns de performance restent stables).
Références  : He, K. et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
              Performance on ImageNet Classification. ICCV 2015. [Initialisation He]
              Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural
              Networks from Overfitting. JMLR, 15(1), 1929-1958.
              Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
              ICML 2016. [Justification MC Dropout = inférence variationnelle]
              Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent
              with Warm Restarts. ICLR 2017. [CosineAnnealingLR]
              Hochreiter, S. & Schmidhuber, J. (1997). Flat Minima. Neural Computation,
              9(1), 1-42. [Avantage des minima larges pour la généralisation]

Initialisation He (kaiming_normal_) :
  Pour une activation ReLU, la variance de la sortie d'une couche linéaire est
  Var(a_l) = (fan_in/2) · Var(W_l) · Var(a_{l-1}).
  L'initialisation He fixe Var(W_l) = 2/fan_in, ce qui garantit E[Var(a_l)] = 1
  à travers les couches (preservation de la variance → pas de vanishing/exploding).

MC Dropout (Monte Carlo Dropout) :
  En maintenant le dropout actif à l'inférence et en effectuant N passages forward,
  on obtient une approximation de l'inférence variationnelle bayésienne (Gal & Ghahramani).
  mean = (1/N) Σ f_θ^(t)(x),  std = sqrt(Var_t[f_θ^(t)(x)])
  std capture l'incertitude épistémique (manque de données) et aléatoire (bruit).

CosineAnnealingLR :
  η_t = η_min + ½(η_max − η_min)(1 + cos(πt/T_max))
  Ce scheduler favorise les "flat minima" (minima à faible courbure de Hessien)
  qui généralisent mieux que les minima étroits (Hochreiter & Schmidhuber 1997),
  car la courbure détermine la sensibilité aux perturbations des données d'entraînement.
"""

import logging
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data_loader import get_weather, load_session

MODELS_DIR = Path(__file__).parent.parent / "models"

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ─── Architecture du réseau ──────────────────────────────────────────────────

class WinnerMLP(nn.Module if HAS_TORCH else object):
    """
    MLP pour la prédiction du vainqueur. 3 couches cachées + BatchNorm + Dropout.

    forward() retourne des logits bruts (compatible BCEWithLogitsLoss).
    predict_proba() retourne des probabilités (Sigmoid) pour l'inférence déterministe.
    predict_proba_mc() retourne (mean, std) via Monte Carlo Dropout (N passes forward).
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 64, 32], dropout: float = 0.3):
        if HAS_TORCH:
            super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        if HAS_TORCH:
            self.net = nn.Sequential(*layers)
            self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialisation He (kaiming_normal_) pour toutes les couches linéaires.

        Garantit E[Var(a_l)] = 1 pour les activations ReLU en fixant :
          std(W_l) = sqrt(2 / fan_in)
        ce qui préserve la magnitude du gradient à travers les couches.
        Référence : He et al. (2015). Delving Deep into Rectifiers. ICCV 2015.
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        """Inférence déterministe (dropout désactivé). Retourne P(victoire)."""
        return torch.sigmoid(self.net(x))

    def predict_proba_mc(self, x, n_samples: int = 50) -> tuple:
        """
        Monte Carlo Dropout : N passes forward avec dropout actif.

        Maintient self.train() pendant l'inférence pour activer le dropout,
        ce qui correspond à une approximation de l'inférence variationnelle
        bayésienne (Gal & Ghahramani, 2016).

        Args:
            x: Tenseur d'entrée (batch_size, input_dim).
            n_samples: Nombre de passes forward (défaut 50).

        Returns:
            (mean, std) — tenseurs de forme (batch_size, 1).
            mean = probabilité de victoire moyenne.
            std  = écart-type (incertitude épistémique + aléatoire).
        """
        self.train()  # active le dropout
        with torch.no_grad():
            preds = torch.stack(
                [torch.sigmoid(self.net(x)) for _ in range(n_samples)]
            )  # (n_samples, batch_size, 1)
        self.eval()
        return preds.mean(dim=0), preds.std(dim=0)


# ─── Préparation des données ──────────────────────────────────────────────────

def build_race_dataset(sessions: list) -> pd.DataFrame:
    """
    Construit un dataset de features pré-course à partir de plusieurs sessions.

    Chaque ligne correspond à un pilote dans une course.
    Cible : IsWinner (1 = premier, 0 = autres).
    """
    records = []
    driver_history: dict[str, list[float]] = defaultdict(list)
    circuit_history: dict[tuple, list[float]] = defaultdict(list)

    for session in sessions:
        if session is None:
            continue
        try:
            results = session.results
            weather = get_weather(session)
            rainfall = float(weather["Rainfall"].astype(float).mean()) if (
                not weather.empty and "Rainfall" in weather.columns
            ) else 0.0

            circuit = session.event.get("EventName", "Unknown")
            try:
                year = int(session.date.year)
            except Exception:
                year = 0

            for _, row in results.iterrows():
                drv = str(row.get("Abbreviation", row.get("DriverId", "UNK")))
                grid_pos = float(row.get("GridPosition", 10)) or 10.0
                finish_pos = float(row.get("Position", 20)) or 20.0
                points = float(row.get("Points", 0)) or 0.0
                team = str(row.get("TeamName", "Unknown"))

                hist = driver_history[drv]
                avg_recent = float(np.mean(hist[-5:])) if hist else 10.0
                std_recent = float(np.std(hist[-5:])) if len(hist) > 1 else 5.0

                circ_hist = circuit_history[(drv, circuit)]
                avg_circuit = float(np.mean(circ_hist)) if circ_hist else 10.0

                records.append({
                    "Driver": drv, "Year": year, "Circuit": circuit,
                    "GridPosition": grid_pos, "FinishPosition": finish_pos,
                    "Points": points, "Team": team, "Rainfall": rainfall,
                    "AvgRecentPos": avg_recent, "StdRecentPos": std_recent,
                    "AvgCircuitPos": avg_circuit, "IsWinner": int(finish_pos == 1),
                })

                driver_history[drv].append(finish_pos)
                circuit_history[(drv, circuit)].append(finish_pos)

        except Exception as e:
            logger.warning(f"[deep_learning] Session ignorée : {e}")
            continue

    return pd.DataFrame(records)


def prepare_Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode, normalise et retourne X, y et les noms de features."""
    le_team = LabelEncoder()
    df["Team_enc"] = le_team.fit_transform(df["Team"].fillna("Unknown"))

    feature_cols = [
        "GridPosition", "AvgRecentPos", "StdRecentPos", "AvgCircuitPos",
        "Rainfall", "Team_enc",
    ]
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["IsWinner"].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, y, feature_cols


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> tuple:
    """
    Entraîne le MLP et retourne le modèle + historique d'entraînement.

    Scheduler : CosineAnnealingLR favorise les "flat minima" (Hochreiter & Schmidhuber 1997)
    dont la courbure du Hessien est faible → meilleure généralisation.
    η_t = η_min + ½(η_max − η_min)(1 + cos(πt/T_max))

    Returns:
        (model, history) — history = dict avec 'train_loss', 'val_loss'.
    """
    if not HAS_TORCH:
        logger.warning("[DL] PyTorch absent — retour d'un modèle factice.")
        return None, {"train_loss": [], "val_loss": []}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[DL] Device : {device}")

    model = WinnerMLP(input_dim=X_train.shape[1]).to(device)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # CosineAnnealingLR : η suit un cosinus décroissant sur T_max epochs
    # → explore des régions plus larges de la loss landscape → meilleure généralisation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=False)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step()  # CosineAnnealingLR ne nécessite pas de métrique

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} — "
                f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return model, history


# ─── Évaluation ───────────────────────────────────────────────────────────────

def top_k_accuracy(model, X_test: np.ndarray, df_test: pd.DataFrame, k: int = 3) -> float:
    """
    Calcule le Top-K Accuracy : le vainqueur réel est-il dans les K pilotes
    ayant la plus haute probabilité prédite ?
    """
    if not HAS_TORCH or model is None:
        return 0.0

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        probs = np.array(model.predict_proba(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().tolist()).flatten()

    df_eval = df_test.copy()
    df_eval["Prob"] = probs
    df_eval["IsWinner"] = df_eval["IsWinner"].astype(int)

    correct = 0
    total_races = 0
    for (year, circuit), grp in df_eval.groupby(["Year", "Circuit"]):
        if grp["IsWinner"].sum() == 0:
            continue
        top_k = grp.nlargest(k, "Prob")["Driver"].tolist()
        winner = grp[grp["IsWinner"] == 1]["Driver"].iloc[0]
        correct += int(winner in top_k)
        total_races += 1

    return correct / total_races if total_races > 0 else 0.0


def _baseline_poleman_accuracy(df_test: pd.DataFrame) -> float:
    """
    Baseline mathématique : le vainqueur est le poleman (GridPosition == 1).

    Cette baseline naïve représente la borne inférieure de performance attendue.
    Un MLP qui ne surpasse pas cette baseline n'apporte pas de valeur ajoutée.
    """
    correct = 0
    total = 0
    for (_, _), grp in df_test.groupby(["Year", "Circuit"]):
        if grp["IsWinner"].sum() == 0:
            continue
        pole = grp[grp["GridPosition"] == grp["GridPosition"].min()]
        if pole.empty:
            continue
        if pole.iloc[0]["IsWinner"] == 1:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ─── Visualisations ───────────────────────────────────────────────────────────

def plot_learning_curve(history: dict) -> plt.Figure:
    """
    Courbe train/val loss avec moving average (fenêtre=5 epochs).

    Détecte automatiquement l'overfitting : première epoch où
    val_loss > 1.05 × min(val_loss), indiquant que la généralisation se dégrade.
    """
    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    epochs = np.arange(1, len(train_loss) + 1)

    window = 5

    def moving_avg(x, w):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    ma_train = moving_avg(train_loss, window)
    ma_val = moving_avg(val_loss, window)
    ma_epochs = epochs[window - 1:]

    # Détection de l'overfitting
    overfitting_epoch = None
    min_val = val_loss.min()
    for i, v in enumerate(val_loss):
        if i > len(val_loss) // 3 and v > 1.05 * min_val:
            overfitting_epoch = i + 1
            break

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(epochs, train_loss, color="steelblue", alpha=0.3, linewidth=1)
    ax.plot(epochs, val_loss, color="#e74c3c", alpha=0.3, linewidth=1)
    ax.plot(ma_epochs, ma_train, color="steelblue", linewidth=2.5,
            label=f"Train Loss (MA{window})")
    ax.plot(ma_epochs, ma_val, color="#e74c3c", linewidth=2.5, linestyle="--",
            label=f"Val Loss (MA{window})")

    if overfitting_epoch:
        ax.axvline(overfitting_epoch, color="orange", linestyle=":", linewidth=2,
                   label=f"Overfitting détecté (epoch {overfitting_epoch})")
        logger.info(f"[DL] Overfitting détecté à l'epoch {overfitting_epoch} "
                    f"(val_loss > 1.05 × min_val_loss={min_val:.4f})")

    ax.set_xlabel("Époque")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Courbe d'apprentissage — MLP Prédiction du vainqueur\n"
                 "(trait plein = signal brut, trait épais = moving average)",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_top_predictions(model, X_test: np.ndarray, df_test: pd.DataFrame, race_idx: int = 0) -> plt.Figure:
    """Barplot des probabilités de victoire pour une course donnée."""
    if not HAS_TORCH or model is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "PyTorch non disponible", ha="center", va="center")
        return fig

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        probs = np.array(model.predict_proba(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().tolist()).flatten()

    df_eval = df_test.copy()
    df_eval["Prob"] = probs

    races = list(df_eval.groupby(["Year", "Circuit"]).groups.keys())
    if race_idx >= len(races):
        race_idx = 0

    year, circuit = races[race_idx]
    race_data = df_eval[(df_eval["Year"] == year) & (df_eval["Circuit"] == circuit)].copy()
    race_data = race_data.sort_values("Prob", ascending=True)

    colors = ["#f39c12" if w else "steelblue" for w in race_data["IsWinner"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(race_data["Driver"], race_data["Prob"], color=colors)
    ax.set_xlabel("Probabilité de victoire")
    ax.set_title(f"Prédictions — {year} GP {circuit}", fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def predict_winner_next_gp(
    model,
    df_past: pd.DataFrame,
    next_gp: dict,
    grid_positions: dict | None = None,
    n_mc_samples: int = 50,
) -> pd.DataFrame:
    """
    Prédit les probabilités de victoire pour le prochain GP avec MC Dropout.

    Args:
        model: WinnerMLP entraîné.
        df_past: Dataset historique.
        next_gp: Info du prochain GP {EventName, Year}.
        grid_positions: {pilote: position_grille} optionnel.
        n_mc_samples: Nombre de passes MC Dropout pour l'incertitude.

    Returns:
        DataFrame avec WinProba (mean MC) et WinProba_Std (incertitude).
    """
    if not HAS_TORCH or model is None or df_past.empty:
        return pd.DataFrame()

    circuit = next_gp.get("EventName", "Unknown")
    year = next_gp.get("Year", 2026)
    drivers = df_past["Driver"].unique()
    records = []

    for drv in drivers:
        drv_df = df_past[df_past["Driver"] == drv]
        finish_positions = drv_df["FinishPosition"].tolist()
        recent = finish_positions[-5:] if len(finish_positions) >= 5 else finish_positions
        avg_recent = float(np.mean(recent)) if recent else 10.0
        std_recent = float(np.std(recent)) if len(recent) > 1 else 5.0
        circ_df = drv_df[drv_df["Circuit"] == circuit]
        avg_circuit = float(circ_df["FinishPosition"].mean()) if not circ_df.empty else avg_recent
        team = drv_df["Team"].iloc[-1] if not drv_df.empty else "Unknown"
        grid_pos = float(grid_positions.get(drv, avg_recent)) if grid_positions else avg_recent

        records.append({
            "Driver": drv, "Year": year, "Circuit": circuit,
            "GridPosition": grid_pos, "FinishPosition": 0.0, "Points": 0.0,
            "Team": team, "Rainfall": 0.0,
            "AvgRecentPos": avg_recent, "StdRecentPos": std_recent,
            "AvgCircuitPos": avg_circuit, "IsWinner": 0,
        })

    df_next = pd.DataFrame(records)
    if df_next.empty:
        return pd.DataFrame()

    le = LabelEncoder()
    all_teams = pd.concat([df_past["Team"], df_next["Team"]]).fillna("Unknown")
    le.fit(all_teams)
    df_next["Team_enc"] = le.transform(df_next["Team"].fillna("Unknown"))
    df_past_enc = df_past.copy()
    df_past_enc["Team_enc"] = le.transform(df_past_enc["Team"].fillna("Unknown"))

    feature_cols = ["GridPosition", "AvgRecentPos", "StdRecentPos", "AvgCircuitPos", "Rainfall", "Team_enc"]
    X_past = df_past_enc[feature_cols].fillna(0).values.astype(np.float32)
    X_next = df_next[feature_cols].fillna(0).values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(X_past)
    X_next_scaled = scaler.transform(X_next).astype(np.float32)

    device = next(model.parameters()).device
    x_tensor = torch.tensor(X_next_scaled, dtype=torch.float32).to(device)

    # MC Dropout : N passes forward avec dropout actif
    mean_probs, std_probs = model.predict_proba_mc(x_tensor, n_samples=n_mc_samples)
    mean_probs = np.array(mean_probs.cpu().tolist()).flatten()
    std_probs = np.array(std_probs.cpu().tolist()).flatten()

    # Normalisation pour que les probabilités somment à 1 (distribution sur les pilotes)
    total = mean_probs.sum()
    if total > 1e-9:
        std_probs = std_probs / total  # mise à l'échelle cohérente
        mean_probs = mean_probs / total

    df_next["WinProba"] = mean_probs
    df_next["WinProba_Std"] = std_probs

    return (
        df_next[["Driver", "Team", "GridPosition", "AvgRecentPos", "AvgCircuitPos",
                 "WinProba", "WinProba_Std"]]
        .sort_values("WinProba", ascending=False)
        .reset_index(drop=True)
    )


def plot_winner_forecast(df_pred: pd.DataFrame, next_gp: dict) -> plt.Figure:
    """
    Barplot horizontal des probabilités de victoire avec intervalles de confiance MC Dropout.

    Les barres d'erreur représentent ±1 std (IC ~68% sous approximation gaussienne),
    quantifiant l'incertitude épistémique du modèle sur chaque pilote.
    """
    gp_name = next_gp.get("EventName", "Prochain GP")
    year = next_gp.get("Year", "")

    if df_pred.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Données insuffisantes", ha="center", va="center")
        return fig

    top = df_pred.head(10).iloc[::-1].reset_index(drop=True)
    colors = ["#f39c12" if i == len(top) - 1 else "steelblue" for i in range(len(top))]
    has_std = "WinProba_Std" in top.columns

    fig, ax = plt.subplots(figsize=(10, 6))
    xerr = top["WinProba_Std"].values if has_std else None
    ax.barh(top["Driver"], top["WinProba"], color=colors, alpha=0.85,
            xerr=xerr, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "gray"})
    ax.set_xlabel("Probabilité de victoire (MC Dropout, N=50)")
    ax.set_title(
        f"Prédiction vainqueur — {year} {gp_name}\n"
        f"Barres d'erreur = ±1 std (incertitude épistémique, MC Dropout)",
        fontweight="bold",
    )
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.grid(True, axis="x", alpha=0.3)
    for i, row in top.iterrows():
        ax.text(row["WinProba"] + (xerr[i] if has_std else 0) + 0.005,
                i, f"{row['WinProba']:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def run(
    years: list[int] | None = None,
    gp_names: list[str] | None = None,
    epochs: int = 50,
) -> dict:
    """
    Point d'entrée principal du module deep learning.

    Returns:
        Dictionnaire contenant le modèle, les métriques et les figures.
    """
    logger.info("MODULE : Prédiction du vainqueur (Deep Learning — MLP)")

    if years is None:
        years = [2022, 2023]
    if gp_names is None:
        gp_names = ["Bahrain", "Monaco", "British", "Italian", "Abu Dhabi"]

    sessions = []
    for year in years:
        for gp in gp_names:
            session = load_session(year, gp, "R")
            if session is not None:
                sessions.append(session)

    logger.info(f"{len(sessions)} session(s) chargée(s).")
    if len(sessions) < 3:
        logger.warning("Peu de sessions disponibles — résultats peu fiables.")

    df = build_race_dataset(sessions)
    if df.empty:
        logger.error("Dataset vide.")
        return {}

    logger.info(f"Dataset : {len(df)} entrées pilote/course  |  Vainqueurs : {df['IsWinner'].sum()}")

    X, y, feature_cols = prepare_Xy(df)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)), test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    logger.info(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model, history = train_mlp(X_train, y_train, X_val, y_val, epochs=epochs)

    df_test = df.iloc[idx_test].reset_index(drop=True)
    top3_acc = top_k_accuracy(model, X_test, df_test, k=3)
    top1_acc = top_k_accuracy(model, X_test, df_test, k=1)

    # Baseline : modèle Oracle "le poleman gagne toujours"
    baseline_acc = _baseline_poleman_accuracy(df_test)
    gain = top1_acc - baseline_acc

    logger.info(f"Top-1 Accuracy (MLP)  : {top1_acc:.3f}")
    logger.info(f"Top-3 Accuracy (MLP)  : {top3_acc:.3f}")
    logger.info(f"Baseline (Poleman)    : {baseline_acc:.3f}")
    logger.info(f"Gain MLP vs baseline  : {gain:+.3f}  "
                f"({'MLP supérieur' if gain > 0 else 'Baseline meilleure'})")

    # Sauvegarde
    if HAS_TORCH and model is not None:
        MODELS_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / "mlp_winner.pt")
        logger.info(f"[DL] Modèle sauvegardé : {MODELS_DIR / 'mlp_winner.pt'}")

    figs = {}
    if history["train_loss"]:
        figs["learning_curve"] = plot_learning_curve(history)
    figs["predictions"] = plot_top_predictions(model, X_test, df_test, race_idx=0)
    plt.show()

    return {
        "model": model,
        "history": history,
        "data": df,
        "figures": figs,
        "metrics": {
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "baseline_poleman": baseline_acc,
            "gain_vs_baseline": gain,
            "epochs": epochs,
        },
    }
