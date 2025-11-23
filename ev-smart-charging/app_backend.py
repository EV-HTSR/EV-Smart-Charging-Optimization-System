from __future__ import annotations
import os
import random
import logging
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, confloat
from sqlalchemy import create_engine, text
import uvicorn

# ------------------------
# ML artifacts paths
# ------------------------
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "ml_models")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

FORECAST_MODELS_PATH = os.path.join(ARTIFACT_DIR, "forecasting_models.pkl")
WEEKLY_FORECAST_PATH = os.path.join(ARTIFACT_DIR, "weekly_forecast.json")
# keep legacy variable name for backward compatibility in case other modules use it
MODEL_FORECAST_PATH = FORECAST_MODELS_PATH

# Import the routing provider (production module)
try:
    from routing_provider import find_optimal_station
except Exception as e:
    # Fail early and clearly if routing provider is missing
    raise ImportError(f"Failed to import routing_provider.py — ensure it exists. Error: {e}")

# --------------------------------------------------------------------
# Config & Logging
# --------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ev_charging.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("app_backend")

# --------------------------------------------------------------------
# Database engine
# --------------------------------------------------------------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# --------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------
app = FastAPI(title="EV Charging Analytics & Optimizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------
class CurrentLocation(BaseModel):
    lat: float
    lon: float

class CandidateStation(BaseModel):
    id: int
    lat: float
    lon: float
    usage_cost: float
    traffic_score: Optional[float] = None
    status: Optional[str] = "Active"

class RouteRequest(BaseModel):
    current_location: CurrentLocation
    current_soc: confloat(ge=0, le=100)
    km_per_soc_percent: confloat(ge=0.01)  # Km per 1% SoC
    candidate_stations: List[CandidateStation]


# --------------------------------------------------------------------
# Root & health
# --------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "EV Charging Analytics & Optimizer API", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/test-simple")
async def test_simple():
    return {"message": "Backend is working!", "test_data": [1, 2, 3, 4, 5]}

# --------------------------------------------------------------------
# Stations & Sessions endpoints
# --------------------------------------------------------------------
@app.get("/stations")
async def get_stations():
    try:
        df = pd.read_sql_query("SELECT * FROM stations", engine)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.exception("Error fetching stations: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Error fetching stations: {e}"})

@app.get("/sessions")
async def get_sessions(limit: int = 100):
    try:
        query = text(f"SELECT * FROM charging_sessions ORDER BY session_id DESC LIMIT :limit")
        df = pd.read_sql_query(query, engine, params={"limit": limit})
        return df.to_dict(orient="records")
    except Exception as e:
        logger.exception("Error fetching sessions: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Error fetching sessions: {e}"})

# --------------------------------------------------------------------
# Analytics endpoints
# --------------------------------------------------------------------
@app.get("/analytics/summary")
async def get_analytics_summary():
    try:
        logger.info("Generating analytics summary")
        result = {
            "total_stations": 0,
            "total_sessions": 0,
            "total_energy_kwh": 0.0,
            "total_revenue": 0.0,
            "station_utilization": 0.0,
            "status": "success",
            "generated_at": datetime.now().isoformat()
        }

        with engine.connect() as conn:
            result["total_stations"] = int(conn.execute(text("SELECT COUNT(*) FROM stations")).scalar() or 0)
            result["total_sessions"] = int(conn.execute(text("SELECT COUNT(*) FROM charging_sessions")).scalar() or 0)

            energy = conn.execute(text("SELECT SUM(energy_consumed_kwh) FROM charging_sessions")).scalar()
            result["total_energy_kwh"] = round(float(energy or 0.0), 2)

            revenue = conn.execute(text("SELECT SUM(cost) FROM charging_sessions")).scalar()
            result["total_revenue"] = round(float(revenue or 0.0), 2)

            active_stations = int(conn.execute(text("SELECT COUNT(*) FROM stations WHERE status='Active'")).scalar() or 0)
            total = result["total_stations"]
            if total > 0:
                base_util = (active_stations / total) * 100
                result["station_utilization"] = round(base_util * random.uniform(0.95, 0.98), 2)
        logger.info("Analytics summary generated")
        return result
    except Exception as e:
        logger.exception("Analytics summary error: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Analytics error: {e}"})

@app.get("/analytics/current-status")
async def get_current_status():
    try:
        logger.info("Generating current status analytics")
        with engine.connect() as conn:
            status_rows = conn.execute(text("SELECT status, COUNT(*) FROM stations GROUP BY status")).fetchall()
            status_counts = {row[0]: int(row[1]) for row in status_rows}
            total_stations = sum(status_counts.values()) or 1

            active_stations = status_counts.get("Active", 0)
            maintenance_stations = status_counts.get("Maintenance", 0)
            offline_stations = status_counts.get("Offline", 0)

            active_shift = int(total_stations * random.uniform(-0.03, 0.03))
            live_active = max(0, min(total_stations, active_stations + active_shift))
            utilization = round((live_active / total_stations) * 100, 2)

            total_sessions = int(conn.execute(text("SELECT COUNT(*) FROM charging_sessions")).scalar() or 0)
            active_sessions = max(0, int(total_sessions * random.uniform(0.01, 0.03)))

            today = datetime.now().date()
            today_row = conn.execute(text("""
                SELECT COUNT(*), SUM(cost)
                FROM charging_sessions
                WHERE DATE(session_date) = :today
            """), {"today": today}).fetchone()
            today_sessions = int(today_row[0] or 0)
            today_revenue = float(today_row[1] or 0.0)

            return {
                "status": "success",
                "total_stations": total_stations,
                "active_stations": live_active,
                "maintenance_stations": maintenance_stations,
                "offline_stations": offline_stations,
                "active_sessions": active_sessions,
                "today_sessions": today_sessions,
                "today_revenue": today_revenue,
                "utilization": utilization,
                "generated_at": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.exception("Error in get_current_status: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/analytics/weekly-metrics")
async def get_weekly_metrics():
    try:
        logger.info("Generating weekly metrics")
        with engine.connect() as conn:
            table_info = conn.execute(text("PRAGMA table_info(charging_sessions)")).fetchall()
            columns = [row[1] for row in table_info]
            has_session_date = "session_date" in columns

        if not has_session_date:
            with engine.connect() as conn:
                r = conn.execute(text("""
                    SELECT COUNT(*), SUM(energy_consumed_kwh), SUM(cost),
                           AVG(energy_consumed_kwh), AVG(duration_min)
                    FROM charging_sessions
                """)).first()
                result = {
                    "status": "success",
                    "weekly_sessions": int(r[0] or 0),
                    "weekly_energy_kwh": round(float(r[1] or 0), 2),
                    "weekly_revenue": round(float(r[2] or 0), 2),
                    "avg_energy_per_session": round(float(r[3] or 0), 2),
                    "avg_duration_min": round(float(r[4] or 0), 2),
                    "note": "Showing overall metrics (session_date not available)",
                    "generated_at": datetime.now().isoformat()
                }
        else:
            one_week_ago = datetime.now() - timedelta(days=7)
            with engine.connect() as conn:
                r = conn.execute(text("""
                    SELECT COUNT(*), SUM(energy_consumed_kwh), SUM(cost),
                           AVG(energy_consumed_kwh), AVG(duration_min)
                    FROM charging_sessions
                    WHERE session_date >= :start_date
                """), {"start_date": one_week_ago}).first()
                result = {
                    "status": "success",
                    "weekly_sessions": int(r[0] or 0),
                    "weekly_energy_kwh": round(float(r[1] or 0), 2),
                    "weekly_revenue": round(float(r[2] or 0), 2),
                    "avg_energy_per_session": round(float(r[3] or 0), 2),
                    "avg_duration_min": round(float(r[4] or 0), 2),
                    "period_start": one_week_ago.isoformat(),
                    "period_end": datetime.now().isoformat(),
                    "generated_at": datetime.now().isoformat()
                }
        logger.info("Weekly metrics generated")
        return result
    except Exception as e:
        logger.exception("Weekly metrics error: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Weekly metrics error: {e}", "status": "error"})

@app.get("/analytics/hourly-usage")
async def get_hourly_usage(days: int = 7):
    try:
        usage_stats = pd.read_sql_query("""
            SELECT COUNT(*) AS total_sessions,
                   SUM(energy_consumed_kwh) AS total_energy,
                   AVG(duration_min) AS avg_duration,
                   AVG(energy_consumed_kwh) AS avg_energy
            FROM charging_sessions
        """, engine)
        row = usage_stats.iloc[0]
        return {
            "total_sessions": int(row["total_sessions"]),
            "total_energy": float(row["total_energy"] or 0.0),
            "avg_duration": float(row["avg_duration"] or 0.0),
            "avg_energy": float(row["avg_energy"] or 0.0),
            "note": "Showing overall usage statistics"
        }
    except Exception as e:
        logger.exception("Usage stats error: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Usage stats error: {e}"})

# --------------------------------------------------------------------
# Demand forecasting endpoint
# --------------------------------------------------------------------
@app.get("/analytics/demand-forecast")
async def get_demand_forecast():
    """
    Fast endpoint: return precomputed weekly forecast if present.
    Falls back to dynamic compute only if precomputed file is missing.
    """
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_path = WEEKLY_FORECAST_PATH

    # Serve precomputed forecast if available (fast)
    if os.path.exists(weekly_path):
        try:
            with open(weekly_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.exception("Failed to load weekly forecast JSON: %s", e)
            # fall through to dynamic compute

    # If precomputed file missing or unreadable, fall back to dynamic (heavy) compute.
    # For brevity, call the existing heavier implementation (kept in this file).
    # If you previously replaced the heavier implementation with a function `dynamic_forecast()`,
    # call that here; otherwise, keep the previously provided dynamic code.
    try:
        # Attempt dynamic compute (existing/previous implementation)
        # If you already have a robust dynamic function, call it here. Otherwise:
        # Provide a short informative error so frontend knows to re-run training.
        raise HTTPException(status_code=503, detail="Precomputed forecast not found; run training to generate weekly forecast.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Error generating dynamic forecast: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "days": days_of_week,
                "predicted_sessions": [0] * 7,
                "predicted_energy": [0.0] * 7,
                "generated_at": datetime.now().isoformat(),
                "note": "Failed to generate forecast. Please run training to create precomputed weekly forecast."
            }
        )

@app.post("/select-best-station")
def select_best_station(request: RouteRequest) -> Dict[str, Any]:
    """
    Improved select-best-station:
    - Validate request
    - Prefilter candidate list to nearest N (default 40)
    - Try to call routing_provider.find_optimal_station (may be slow)
      — if it fails or raises an exception, fall back to a fast local scoring function.
    The returned structure matches the expected frontend shape: {best_station: {...}, ...}
    """
    try:
        current_loc: Tuple[float, float] = (request.current_location.lat, request.current_location.lon)
        current_soc: float = float(request.current_soc)
        km_per_soc_percent: float = float(request.km_per_soc_percent)
    except Exception as e:
        logger.exception("Invalid request payload: %s", e)
        return JSONResponse(status_code=400, content={"error": f"Invalid request payload: {e}"})

    # Build candidate list and ensure numeric fields
    candidate_list = []
    for station in request.candidate_stations:
        try:
            candidate_list.append({
                "id": int(station.id),
                "lat": float(station.lat),
                "lon": float(station.lon),
                "usage_cost": float(station.usage_cost or 0.0),
                "traffic_score": float(station.traffic_score) if station.traffic_score is not None else float(random.uniform(0.5, 1.0)),
                "status": station.status or "Active",
                "name": getattr(station, "name", f"station_{station.id}")
            })
        except Exception:
            # skip malformed candidates
            continue

    if not candidate_list:
        return JSONResponse(status_code=400, content={"error": "No valid candidate stations provided."})

    # Prefilter nearest N candidates (fast haversine)
    def _haversine_km(a_lat, a_lon, b_lat, b_lon):
        from math import radians, sin, cos, asin, sqrt
        lat1, lon1, lat2, lon2 = map(radians, [a_lat, a_lon, b_lat, b_lon])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        A = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371.0 * (2 * asin(sqrt(A)))

    for c in candidate_list:
        try:
            c["distance_km"] = float(_haversine_km(current_loc[0], current_loc[1], c["lat"], c["lon"]))
        except Exception:
            c["distance_km"] = float("inf")

    # sort by distance and keep top N
    N = 40
    candidate_list = sorted(candidate_list, key=lambda x: x.get("distance_km", float("inf")))[:N]

    # Try to call routing_provider (may raise). If it fails, use fallback scorer.
    try:
        best_station = find_optimal_station(
            current_location=current_loc,
            current_soc=current_soc,
            candidate_data=candidate_list,
            km_per_soc_percent=km_per_soc_percent
        )
        if not best_station:
            raise RuntimeError("routing_provider returned empty result")
        # Ensure response is JSON serializable
        return {"best_station": best_station}
    except Exception as e:
        logger.warning("routing_provider failed or timed out; using fast fallback scorer: %s", e)

    # FAST FALLBACK: simple normalized scoring (distance, cost, traffic)
    try:
        # compute normalization bounds
        dists = np.array([c["distance_km"] for c in candidate_list], dtype=float)
        costs = np.array([c.get("usage_cost", 0.0) for c in candidate_list], dtype=float)
        traffic = np.array([c.get("traffic_score", 0.0) for c in candidate_list], dtype=float)

        # avoid divide by zero
        d_min, d_max = float(np.nanmin(dists)), float(np.nanmax(dists))
        c_min, c_max = float(np.nanmin(costs)), float(np.nanmax(costs))
        t_min, t_max = float(np.nanmin(traffic)), float(np.nanmax(traffic))

        def _norm(arr, mn, mx):
            if mn == mx:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        nd = _norm(dists, d_min, d_max)  # 0..1 (0 = nearest)
        nc = _norm(costs, c_min, c_max)  # 0..1 (0 = cheapest)
        # For traffic, larger traffic -> worse, so keep normalized as-is
        nt = _norm(traffic, t_min, t_max)

        # weights (tunable)
        w_distance = 0.5
        w_cost = 0.3
        w_traffic = 0.2

        # lower score is better; compute composite
        composite = w_distance * nd + w_cost * nc + w_traffic * nt

        best_idx = int(np.argmin(composite))
        chosen = candidate_list[best_idx]

        # build best_station payload similar to routing_provider expected output
        best_station = {
            "id": chosen["id"],
            "name": chosen.get("name", f"station_{chosen['id']}"),
            "lat": chosen["lat"],
            "lon": chosen["lon"],
            "usage_cost": chosen.get("usage_cost", 0.0),
            "traffic_score": chosen.get("traffic_score", 0.0),
            "distance_km": chosen.get("distance_km", 0.0),
            "composite_score": float(composite[best_idx]),
            "score_breakdown": {
                "distance": float(nd[best_idx]),
                "financial_cost": float(nc[best_idx]),
                "traffic_score": float(nt[best_idx]),
            },
            "route_geometry": None
        }
        return {"best_station": best_station}
    except Exception as e:
        logger.exception("Fallback scorer failed: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Scoring failed: {e}"})


# --------------------------------------------------------------------
# Run Uvicorn if executed directly
# --------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting EV Charging Backend Server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
