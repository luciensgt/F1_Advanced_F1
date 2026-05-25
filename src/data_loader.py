"""
Module de chargement et préparation des données FastF1.
Utilisé par tous les autres modules comme source de données commune.

Hypothèses  : L'API FastF1 retourne des données cohérentes par session.
              Les temps au tour valides sont strictement positifs (on filtre
              les tours derrière voiture de sécurité via TrackStatus si besoin).
              LapTimeDelta = LapTime_s − médiane(pilote) est une mesure robuste
              de la dégradation de performance, résistante aux outliers de type
              Safety Car ou In/Out Lap (qui affectent la médiane de façon limitée).
              Stationnarité supposée au sein d'une même saison (les circuits ne
              changent pas drastiquement entre deux saisons consécutives).
Supporte    : Saisons 2018–2026.
"""

import logging
import warnings
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

try:
    from fastf1.core import DataNotLoadedError as _DataNotLoadedError
except ImportError:
    _DataNotLoadedError = Exception

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

CURRENT_SEASON = 2026


def list_events(year: int) -> pd.DataFrame:
    """
    Retourne le calendrier des événements pour une saison.

    Args:
        year: Année de la saison (2018–2026)

    Returns:
        DataFrame avec RoundNumber, EventName, Country, EventDate, EventFormat.
        DataFrame vide si la saison n'est pas disponible.
    """
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        cols = [c for c in ["RoundNumber", "EventName", "Country", "EventDate", "EventFormat"]
                if c in schedule.columns]
        return schedule[cols].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"[data_loader] Calendrier {year} indisponible : {e}")
        return pd.DataFrame()


def load_session(year: int, gp_name: str | int, session_type: str = "R") -> fastf1.core.Session | None:
    """
    Charge une session FastF1.

    Args:
        year: Année de la saison (2018–2026)
        gp_name: Nom partiel du Grand Prix (ex: "Monaco") ou numéro de manche (ex: 1)
        session_type: "R" Race | "Q" Qualifying | "FP1" "FP2" "FP3" | "S" Sprint

    Returns:
        Session chargée, ou None si indisponible.
    """
    try:
        session = fastf1.get_session(year, gp_name, session_type)
        session.load(telemetry=True, weather=True, messages=True, laps=True)
    except Exception as e:
        logger.warning(f"[data_loader] Chargement impossible : {year} {gp_name!r} {session_type} — {e}")
        return None

    try:
        _ = session.laps
    except (_DataNotLoadedError, Exception):
        logger.warning(f"[data_loader] Laps non chargés pour : {year} {gp_name!r} {session_type}")
        return None

    return session


def load_multiple_seasons(
    start: int = 2018,
    end: int = CURRENT_SEASON,
    session_type: str = "R",
    gp_filter: list[str] | None = None,
    max_rounds: int | None = None,
) -> list[fastf1.core.Session]:
    """
    Charge plusieurs saisons de sessions F1.

    Args:
        start: Première saison (incluse). Défaut : 2018.
        end: Dernière saison (incluse). Défaut : CURRENT_SEASON.
        session_type: Type de session à charger.
        gp_filter: Liste optionnelle de noms de GP (filtre partiel, insensible à la casse).
        max_rounds: Limite le nombre de manches par saison (utile pour les tests rapides).

    Returns:
        Liste de sessions chargées (sessions indisponibles silencieusement ignorées).
    """
    sessions = []
    for year in range(start, end + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            logger.warning(f"[data_loader] Calendrier {year} indisponible : {e}")
            continue

        rounds_loaded = 0
        for _, event in schedule.iterrows():
            if max_rounds and rounds_loaded >= max_rounds:
                break
            gp = event.get("EventName", "")
            if gp_filter and not any(f.lower() in gp.lower() for f in gp_filter):
                continue
            session = load_session(year, int(event["RoundNumber"]), session_type)
            if session is not None:
                sessions.append(session)
                rounds_loaded += 1

    return sessions


def get_laps_features(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extrait un DataFrame de features par tour à partir d'une session.

    LapTimeDelta = LapTime_s − médiane(pilote) mesure l'écart au comportement
    typique du pilote. Robuste aux tours anormaux (In/Out Lap, Safety Car) qui
    représentent moins de 10% des tours en conditions normales.

    Colonnes retournées :
        Driver, DriverNumber, LapNumber, LapTime_s, TyreLife, Compound, Stint,
        Position, SpeedST, SpeedFL, SpeedI1, SpeedI2,
        IsPersonalBest, TrackStatus, FreshTyre, HasPitStop, LapTimeDelta
    """
    if session is None:
        return pd.DataFrame()

    try:
        laps = session.laps.copy()
    except Exception as e:
        logger.warning(f"[data_loader] Données de tours non disponibles : {e}")
        return pd.DataFrame()

    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps["FreshTyre"] = laps["FreshTyre"].astype(bool)

    keep = [
        "Driver", "DriverNumber", "LapNumber", "LapTime_s",
        "TyreLife", "Compound", "Stint", "Position",
        "SpeedST", "SpeedFL", "SpeedI1", "SpeedI2",
        "IsPersonalBest", "TrackStatus", "FreshTyre",
        "PitInTime", "PitOutTime",
    ]
    keep = [c for c in keep if c in laps.columns]
    df = laps[keep].copy()

    df["HasPitStop"] = df["PitInTime"].notna() if "PitInTime" in df.columns else False

    median_by_driver = df.groupby("Driver")["LapTime_s"].transform("median")
    df["LapTimeDelta"] = df["LapTime_s"] - median_by_driver

    return df.reset_index(drop=True)


def get_radio_messages(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Retourne les messages de direction de course (race control messages).

    FastF1 expose session.race_control_messages qui contient les messages officiels
    (Safety Car, drapeaux, pénalités, DRS…).

    Returns:
        DataFrame avec au moins les colonnes Time, Message, Category.
        DataFrame vide si indisponible.
    """
    if session is None:
        return pd.DataFrame()

    try:
        rc = session.race_control_messages
        if rc is None or rc.empty:
            return pd.DataFrame()
        df = rc.copy()
        df["Source"] = "RaceControl"
        return df
    except Exception as e:
        logger.warning(f"[data_loader] Messages radio indisponibles : {e}")
        return pd.DataFrame()


def get_weather(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Retourne les données météo de la session.

    Returns:
        DataFrame avec Time, AirTemp, TrackTemp, Humidity, WindSpeed, Rainfall.
        DataFrame vide si indisponible.
    """
    if session is None:
        return pd.DataFrame()

    try:
        return session.weather_data.copy()
    except Exception as e:
        logger.warning(f"[data_loader] Météo indisponible : {e}")
        return pd.DataFrame()


def get_next_gp(year: int = CURRENT_SEASON) -> dict | None:
    """
    Retourne le prochain Grand Prix non encore disputé dans la saison.

    Args:
        year: Saison à interroger (défaut : CURRENT_SEASON).

    Returns:
        Dictionnaire {RoundNumber, EventName, Country, EventDate, Year}
        ou None si la saison est terminée ou le calendrier indisponible.
    """
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if schedule.empty:
            return None

        today = pd.Timestamp.now()
        dates = pd.to_datetime(schedule["EventDate"])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)

        future = schedule[dates > today]
        if future.empty:
            return None

        event = future.iloc[0]
        return {
            "RoundNumber": int(event.get("RoundNumber", 0)),
            "EventName": str(event.get("EventName", "")),
            "Country": str(event.get("Country", "")),
            "EventDate": event.get("EventDate"),
            "Year": year,
        }
    except Exception as e:
        logger.warning(f"[data_loader] Prochain GP indisponible : {e}")
        return None


def get_telemetry_features(
    session: fastf1.core.Session,
    driver: str,
    lap_number: int | None = None,
) -> pd.DataFrame:
    """
    Retourne la télémétrie haute-fréquence pour un pilote.

    Args:
        session: Session FastF1 chargée.
        driver: Code pilote (ex: "VER", "NOR").
        lap_number: Numéro de tour spécifique (None = tous les tours).

    Returns:
        DataFrame de télémétrie avec Speed, RPM, Throttle, Brake, DRS, X, Y.
    """
    if session is None:
        return pd.DataFrame()

    try:
        laps = session.laps.pick_driver(driver)
        if lap_number is not None:
            laps = laps[laps["LapNumber"] == lap_number]
        if laps.empty:
            return pd.DataFrame()
        tel = laps.get_telemetry()
        tel["Driver"] = driver
        return tel.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"[data_loader] Télémétrie indisponible pour {driver} : {e}")
        return pd.DataFrame()
