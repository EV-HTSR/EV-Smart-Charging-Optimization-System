#!/usr/bin/env python3
"""
Production-grade ML training pipeline (fixed & Streamlit-ready)

- Safe clustering pipeline: encoder/scaler + KMeans
- Per-station forecasting (RandomForest on lag features) when sufficient history
- Saves artifacts (timestamped):
    - clustering artifact (pkl)
    - cluster scatter (html + pickled fig)
    - forecasting models (pkl) (backend-friendly structure)
    - forecast metrics (json)
    - cluster insights (json) [JSON-safe]
- Bulk DB update for cluster_id
- Progress bars via tqdm and optional Streamlit progress callback

Notes:
- Requires: sklearn, pandas, numpy, sqlalchemy, plotly, tqdm
- Use Streamlit to call train_all(progress=callback) to render live UI progress.
"""
import os
import json
import pickle
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Callable

import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from sqlalchemy.orm import sessionmaker

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score

import plotly.express as px

# local imports - ensure your database.py provides `engine` and `Station`
from database import engine, Station

# -------------------------
# Config
# -------------------------
ARTIFACT_DIR = "ml_models"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
REQUESTED_N_CLUSTERS = 4
MIN_STATIONS_FOR_CLUSTERING = 4
MIN_HISTORY_DAYS_FOR_FORECAST = 30
FORECAST_HOLDOUT_DAYS = 14

BATCH_UPDATE_SIZE = 500

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_models")

# -------------------------
# Utilities
# -------------------------
def ts_name(prefix: str, ext: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(ARTIFACT_DIR, f"{prefix}_{ts}.{ext}")

def save_pickle(obj: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_json(obj: Dict, path: str):
    # Convert numpy types to native Python types recursively
    def _to_native(o):
        if isinstance(o, dict):
            return {str(k): _to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_native(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return _to_native(o.tolist())
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        return o

    safe_obj = _to_native(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_obj, f, indent=2, ensure_ascii=False)

def safe_sql_read(query: str):
    try:
        return pd.read_sql_query(query, engine)
    except Exception as e:
        logger.exception("SQL read error: %s", e)
        return pd.DataFrame()

# -------------------------
# Feature preparation
# -------------------------
def prepare_station_aggregate() -> pd.DataFrame:
    """Load station aggregates for clustering from DB."""
    query = """
    SELECT 
        s.station_id,
        s.latitude,
        s.longitude,
        s.charger_type,
        COALESCE(COUNT(cs.session_id), 0) AS total_sessions,
        COALESCE(AVG(cs.energy_consumed_kwh), 0) AS avg_energy,
        COALESCE(AVG(cs.duration_min), 0) AS avg_duration,
        COALESCE(SUM(cs.energy_consumed_kwh), 0) AS total_energy,
        COALESCE(SUM(cs.cost), 0) AS total_revenue
    FROM stations s
    LEFT JOIN charging_sessions cs ON s.station_id = cs.station_id
    GROUP BY s.station_id
    ORDER BY s.station_id
    """
    df = safe_sql_read(query)
    if df.empty:
        logger.warning("prepare_station_aggregate: no rows returned")
    else:
        df = df.reset_index(drop=True)
    return df

def build_preprocessor(df: pd.DataFrame):
    """Build ColumnTransformer: scale numeric features, one-hot encode charger_type."""
    numeric_cols = ["latitude", "longitude", "total_sessions", "avg_energy", "avg_duration", "total_energy", "total_revenue"]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0
    cat_cols = ["charger_type"]

    numeric_transformer = StandardScaler()
    # use new parameter name to silence warnings in newer sklearn
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # older sklearn uses 'sparse'
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, cat_cols

def safe_n_clusters(requested: int, n_samples: int) -> int:
    if n_samples < 2:
        return 1
    # choose at most requested, but not exceeding samples; choose at least 2
    return max(2, min(requested, n_samples))

# -------------------------
# Forecasting helpers
# -------------------------
def load_daily_history() -> pd.DataFrame:
    query = """
    SELECT 
        station_id,
        DATE(session_date) AS session_day,
        COUNT(session_id) AS daily_sessions,
        SUM(energy_consumed_kwh) AS daily_energy
    FROM charging_sessions
    GROUP BY station_id, DATE(session_date)
    ORDER BY station_id, DATE(session_date)
    """
    df = safe_sql_read(query)
    if df.empty:
        return df
    df["session_day"] = pd.to_datetime(df["session_day"])
    return df.sort_values(["station_id", "session_day"])

def make_lag_features(group: pd.DataFrame, lags=(1, 7, 14)) -> pd.DataFrame:
    """From per-station daily history create lag & rolling features."""
    g = group.set_index("session_day").asfreq("D", fill_value=0)
    g["daily_sessions"] = g["daily_sessions"].astype(float)
    g["daily_energy"] = g["daily_energy"].astype(float)

    for lag in lags:
        g[f"sessions_lag_{lag}"] = g["daily_sessions"].shift(lag).fillna(0)
        g[f"energy_lag_{lag}"] = g["daily_energy"].shift(lag).fillna(0)

    for w in (3, 7):
        g[f"sessions_roll_{w}"] = g["daily_sessions"].rolling(window=w, min_periods=1).mean().shift(1).fillna(0)
        g[f"energy_roll_{w}"] = g["daily_energy"].rolling(window=w, min_periods=1).mean().shift(1).fillna(0)

    g["dayofweek"] = g.index.dayofweek
    g["month"] = g.index.month

    g = g.dropna()
    g = g.reset_index().rename(columns={"index": "session_day"})
    return g

# -------------------------
# Main pipeline (with progress callback support)
# -------------------------
def train_all(progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Optional[str]]:
    """
    Run the complete training pipeline.

    progress: optional callback(progress_float_0_1, message) — useful for Streamlit UI
    Returns dict of artifact paths.
    """
    def _update(pct: float, msg: str):
        try:
            if progress:
                progress(pct, msg)
        except Exception:
            pass  # progress callback must never break pipeline

    logger.info("Starting ML training pipeline")
    _update(0.01, "Loading station aggregates")

    df_agg = prepare_station_aggregate()
    if df_agg.empty:
        logger.error("No station aggregate data available. Aborting.")
        return {}

    _update(0.05, "Building preprocessor")
    preprocessor, numeric_cols, cat_cols = build_preprocessor(df_agg)

    _update(0.08, "Fitting preprocessor")
    try:
        X_proc = preprocessor.fit_transform(df_agg[numeric_cols + cat_cols])
    except Exception as e:
        logger.exception("Preprocessor fit_transform failed: %s", e)
        return {}

    n_samples = len(df_agg)
    n_clusters = safe_n_clusters(REQUESTED_N_CLUSTERS, n_samples)
    _update(0.12, f"Training KMeans ({n_clusters} clusters)")

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_proc)
        df_agg["cluster_id"] = labels
    except Exception as e:
        logger.exception("KMeans training failed: %s", e)
        df_agg["cluster_id"] = -1
        kmeans = None

    # Save clustering artifact
    _update(0.18, "Saving clustering artifact")
    clustering_artifact = {
        "preprocessor": preprocessor,
        "kmeans": kmeans,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols
    }
    clustering_path = ts_name("clustering_artifact", "pkl")
    cluster_model_path = clustering_path  # canonical name for return
    try:
        save_pickle(clustering_artifact, clustering_path)
    except Exception as e:
        logger.exception("Failed to save clustering artifact: %s", e)
        clustering_path = None
        cluster_model_path = None

    # Plot cluster scatter and save html + pickled fig
    _update(0.22, "Creating cluster visual")
    try:
        fig = px.scatter(df_agg, x="total_energy", y="total_revenue",
                         color=df_agg["cluster_id"].astype(str),
                         hover_data=["station_id", "charger_type"],
                         title="Clusters: Total Energy vs Total Revenue")
        html_path = ts_name("cluster_scatter", "html")
        pkl_fig_path = ts_name("cluster_scatter", "pkl")
        fig.write_html(html_path)
        save_pickle(fig, pkl_fig_path)
    except Exception as e:
        logger.exception("Could not save cluster visual: %s", e)
        html_path = None
        pkl_fig_path = None

    # Compute silhouette score if possible
    sil_score = None
    try:
        if kmeans is not None and n_samples > n_clusters:
            sil_score = float(silhouette_score(X_proc, labels))
            logger.info("Silhouette score: %.4f", sil_score)
    except Exception:
        sil_score = None

    # Bulk update DB with cluster_id
    _update(0.30, "Updating station cluster_id in DB")
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        mappings = []
        for _, row in df_agg.iterrows():
            mappings.append({"station_id": int(row["station_id"]), "cluster_id": int(row["cluster_id"])})
            if len(mappings) >= BATCH_UPDATE_SIZE:
                session.bulk_update_mappings(Station, mappings)
                session.commit()
                mappings = []
        if mappings:
            session.bulk_update_mappings(Station, mappings)
            session.commit()
        session.close()
    except Exception as e:
        logger.exception("Error bulk-updating cluster IDs: %s", e)

    # Forecasting
    _update(0.35, "Loading daily history")
    df_daily = load_daily_history()
    forecast_artifacts = {}
    forecast_metrics = {}
    forecast_path = None
    metrics_path = None

    if df_daily.empty:
        logger.warning("No daily history found; skipping forecasting")
        forecast_path = None
        metrics_path = None
    else:
        station_ids = df_agg["station_id"].tolist()
        total = len(station_ids)
        logger.info("Training forecasts for %d stations (those with sufficient history)", total)
        # Use tqdm for terminal progress
        for idx, sid in enumerate(tqdm(station_ids, desc="Forecasting", ncols=100)):
            # update percent occasionally
            if idx % max(1, total // 20) == 0:
                _update(0.35 + 0.50 * idx / max(1, total), f"Forecasting station {idx+1}/{total}")

            grp = df_daily[df_daily["station_id"] == sid][["session_day", "daily_sessions", "daily_energy"]].copy()
            if len(grp) < MIN_HISTORY_DAYS_FOR_FORECAST:
                continue

            feats = make_lag_features(grp)
            if len(feats) < (FORECAST_HOLDOUT_DAYS + 7):
                continue

            for target in ("daily_sessions", "daily_energy"):
                try:
                    dfm = feats.dropna().copy()
                    X_cols = [c for c in dfm.columns if c not in ("session_day", "daily_sessions", "daily_energy")]
                    X = dfm[X_cols].values
                    y = dfm[target].values

                    train_X = X[:-FORECAST_HOLDOUT_DAYS]
                    train_y = y[:-FORECAST_HOLDOUT_DAYS]
                    hold_X = X[-FORECAST_HOLDOUT_DAYS:]
                    hold_y = y[-FORECAST_HOLDOUT_DAYS:]

                    # train RandomForest (parallel)
                    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
                    model.fit(train_X, train_y)
                    preds = model.predict(hold_X)

                    mae = float(mean_absolute_error(hold_y, preds))
                    rmse = float(math.sqrt(mean_squared_error(hold_y, preds)))

                    key = f"{int(sid)}_{target}"
                    forecast_artifacts[key] = {"model": model, "X_cols": X_cols}
                    forecast_metrics[key] = {"mae": mae, "rmse": rmse, "train_days": int(len(train_y)), "holdout_days": int(len(hold_y))}
                except Exception as e:
                    logger.warning("Forecast train failed for station %s target %s: %s", sid, target, e)
                    continue

        # Save forecast artifacts & metrics (unified structure for backend)
        _update(0.86, "Saving forecast artifacts")
        forecast_models_struct = {}
        for key, val in forecast_artifacts.items():
            try:
                sid_str, target = str(key).split("_", 1)
                sid = int(sid_str)
            except:
                continue

            if sid not in forecast_models_struct:
                forecast_models_struct[sid] = {}

            if "sessions" in target:
                forecast_models_struct[sid]["sessions_model"] = val.get("model")
                forecast_models_struct[sid]["sessions_X_cols"] = val.get("X_cols", [])
            elif "energy" in target:
                forecast_models_struct[sid]["energy_model"] = val.get("model")
                forecast_models_struct[sid]["energy_X_cols"] = val.get("X_cols", [])

            # Safe default
            forecast_models_struct[sid].setdefault("last_index", 0)

        forecast_path = os.path.join(ARTIFACT_DIR, "forecasting_models.pkl")
        try:
            save_pickle(forecast_models_struct, forecast_path)
            metrics_path = ts_name("forecast_metrics", "json")
            save_json(forecast_metrics, metrics_path)
            logger.info("Unified forecasting models saved: %s", forecast_path)
        except Exception as e:
            logger.exception("Failed to save unified forecast artifacts: %s", e)
            forecast_path = None
            metrics_path = None

        # ----------------------------
        # Compute & save weekly aggregated forecast (precompute at training time)
        # ----------------------------
        try:
            weekly_agg_sessions = np.zeros(7, dtype=float)
            weekly_agg_energy = np.zeros(7, dtype=float)

            # df_daily was loaded earlier in training: ensure it exists (if not, reload)
            try:
                _ = df_daily  # noqa
            except NameError:
                df_daily = load_daily_history()

            # For each station in the saved forecast models, build lag features from df_daily and
            # perform recursive 7-day forecast using the station's saved X_cols and models.
            for sid, mdata in forecast_models_struct.items():
                try:
                    sid_int = int(sid)
                except Exception:
                    continue

                sessions_model = mdata.get("sessions_model")
                energy_model = mdata.get("energy_model")
                sessions_X_cols = mdata.get("sessions_X_cols", [])
                energy_X_cols = mdata.get("energy_X_cols", [])

                if sessions_model is None and energy_model is None:
                    continue

                # extract station daily history
                grp = df_daily[df_daily["station_id"] == sid_int][["session_day", "daily_sessions", "daily_energy"]].copy()
                if grp.empty:
                    continue

                feats = make_lag_features(grp) if len(grp) > 0 else pd.DataFrame()
                if feats.empty:
                    continue

                # build historical arrays from last row of feats
                last_row = feats.iloc[-1:]
                # create value lists for recursive forecasting (ordered oldest->newest)
                hist_sessions = list(grp.set_index("session_day").asfreq("D", fill_value=0)["daily_sessions"].astype(float).tolist())
                hist_energy = list(grp.set_index("session_day").asfreq("D", fill_value=0)["daily_energy"].astype(float).tolist())

                # if X_cols available, use them; otherwise fallback to naive dayofweek only
                station_pred_sessions = []
                station_pred_energy = []
                last_hist_date = pd.to_datetime(grp["session_day"].iloc[-1]).date()
                for i in range(7):
                    pred_date = last_hist_date + timedelta(days=i + 1)

                    # helper to compute features matching X_cols; returns None if X_cols empty
                    def build_feat(cols, sess_hist, ener_hist, pd_date):
                        if not cols:
                            return None
                        vals = []
                        # helper to fetch k days back
                        def val_back(arr, k):
                            if k <= 0:
                                return 0.0
                            if len(arr) >= k:
                                return float(arr[-k])
                            return 0.0
                        for col in cols:
                            if col.startswith("sessions_lag_"):
                                k = int(col.split("_")[-1])
                                vals.append(val_back(sess_hist, k))
                            elif col.startswith("energy_lag_"):
                                k = int(col.split("_")[-1])
                                vals.append(val_back(ener_hist, k))
                            elif col.startswith("sessions_roll_"):
                                w = int(col.split("_")[-1])
                                window = sess_hist[-w:] if len(sess_hist) >= 1 else []
                                vals.append(float(np.mean(window)) if len(window) > 0 else 0.0)
                            elif col.startswith("energy_roll_"):
                                w = int(col.split("_")[-1])
                                window = ener_hist[-w:] if len(ener_hist) >= 1 else []
                                vals.append(float(np.mean(window)) if len(window) > 0 else 0.0)
                            elif col == "dayofweek":
                                vals.append(float(pd_date.weekday()))
                            elif col == "month":
                                vals.append(float(pd_date.month))
                            else:
                                vals.append(0.0)
                        return np.array(vals).reshape(1, -1)

                    s_feat = build_feat(sessions_X_cols, hist_sessions, hist_energy, pred_date)
                    e_feat = build_feat(energy_X_cols, hist_sessions, hist_energy, pred_date)

                    s_pred = None
                    e_pred = None
                    try:
                        if sessions_model is not None and s_feat is not None:
                            s_pred = float(sessions_model.predict(s_feat).ravel()[0])
                        if energy_model is not None and e_feat is not None:
                            e_pred = float(energy_model.predict(e_feat).ravel()[0])
                    except Exception:
                        # fallback to 0 if prediction fails for station
                        s_pred = 0.0 if s_pred is None else s_pred
                        e_pred = 0.0 if e_pred is None else e_pred

                    s_pred = max(0.0, s_pred if s_pred is not None else 0.0)
                    e_pred = max(0.0, e_pred if e_pred is not None else 0.0)

                    station_pred_sessions.append(s_pred)
                    station_pred_energy.append(e_pred)

                    # append predictions to history lists for recursive features
                    hist_sessions.append(s_pred)
                    hist_energy.append(e_pred)

                # align predictions to weekdays relative to today -> aggregate
                current_day = datetime.now().weekday()
                for j in range(7):
                    idx = (current_day + j + 1) % 7
                    weekly_agg_sessions[idx] += station_pred_sessions[j]
                    weekly_agg_energy[idx] += station_pred_energy[j]

            # Save aggregated weekly forecast JSON for fast serving
            weekly_forecast = {
                "days": ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                "predicted_sessions": [int(round(x)) for x in weekly_agg_sessions.tolist()],
                "predicted_energy": [round(float(x), 2) for x in weekly_agg_energy.tolist()],
                "generated_at": datetime.now().isoformat(),
                "note": f"Precomputed weekly aggregated forecast generated at training time using {len(forecast_models_struct)} station models."
            }
            weekly_path = os.path.join(ARTIFACT_DIR, "weekly_forecast.json")
            save_json(weekly_forecast, weekly_path)
            logger.info("Precomputed weekly forecast saved: %s", weekly_path)
        except Exception as e:
            logger.exception("Failed to compute/save weekly aggregated forecast: %s", e)

    # Insights (JSON-safe keys)
    _update(0.92, "Generating cluster insights")
    try:
        insights = {}
        for cid in sorted(df_agg["cluster_id"].unique()):
            cid_native = int(cid)  # ensure native int
            block = df_agg[df_agg["cluster_id"] == cid]
            insights[f"cluster_{cid_native}"] = {
                "cluster_id": cid_native,
                "station_count": int(len(block)),
                "avg_sessions": float(block["total_sessions"].mean()) if "total_sessions" in block else 0.0,
                "avg_revenue": float(block["total_revenue"].mean()) if "total_revenue" in block else 0.0,
                "common_charger": str(block["charger_type"].mode().iloc[0]) if not block["charger_type"].empty else "Unknown"
            }
        insights_path = ts_name("cluster_insights", "json")
        save_json(insights, insights_path)
    except Exception as e:
        logger.exception("Failed to build/save insights: %s", e)
        insights_path = None

    _update(1.0, "Training complete")
    logger.info("Training pipeline finished")

    return {
        "clustering_artifact": clustering_path if 'clustering_path' in locals() else None,
        "cluster_model": cluster_model_path if 'cluster_model_path' in locals() else None,
        "cluster_scatter_html": html_path if 'html_path' in locals() else None,
        "cluster_scatter_pkl": pkl_fig_path if 'pkl_fig_path' in locals() else None,
        "forecast_models": forecast_path if 'forecast_path' in locals() else None,
        "forecast_metrics": metrics_path if 'metrics_path' in locals() else None,
        "insights": insights_path if 'insights_path' in locals() else (insights_path if 'insights_path' in locals() else None),
        "silhouette": sil_score
    }

# -------------------------
# Streamlit helper
# -------------------------
def load_plotly_fig_for_streamlit(pkl_path: str):
    try:
        return load_pickle(pkl_path)
    except Exception as e:
        logger.error("Could not load pickled figure %s: %s", pkl_path, e)
        return None

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    artifacts = train_all()
    print("Artifacts produced:")
    print(json.dumps({k: (v if isinstance(v, str) or v is None else str(v)) for k, v in artifacts.items()}, indent=2))
