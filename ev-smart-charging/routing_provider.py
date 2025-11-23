from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import math
import requests
import logging

# ============================================================
# Logger
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================
# Config
# ============================================================
OSRM_URL = (
    "http://router.project-osrm.org/route/v1/driving/"
    "{lon1},{lat1};{lon2},{lat2}?overview=false"
)

OSRM_TIMEOUT = 4.0
DEFAULT_SPEED_KMPH = 45.0
USE_OSRM = True   # toggle if OSRM is slow

# Weight constants
W_DISTANCE = 0.4
W_COST = 0.3
W_TRAFFIC = 0.2
W_RANGE = 0.1
RANGE_PENALTY = 1e6


# ============================================================
# Haversine Utilities
# ============================================================
def haversine_distance_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b

    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)

    a_hav = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a_hav), math.sqrt(1 - a_hav))
    return R * c


def travel_time_haversine_sec(
    a: Tuple[float, float],
    b: Tuple[float, float],
    speed_kmph: float = DEFAULT_SPEED_KMPH
) -> float:
    dist_km = haversine_distance_km(a, b)
    return (dist_km / max(speed_kmph, 1.0)) * 3600.0


# ============================================================
# OSRM Real Routing
# ============================================================
def osrm_route(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Optional[float], Optional[float]]:
    url = OSRM_URL.format(lat1=a[0], lon1=a[1], lat2=b[0], lon2=b[1])

    try:
        r = requests.get(url, timeout=OSRM_TIMEOUT)
        if r.status_code != 200:
            logger.warning(f"OSRM returned status {r.status_code}")
            return None, None

        data = r.json()
        route = data.get("routes", [{}])[0]

        return (
            route.get("distance", 0.0) / 1000.0,
            route.get("duration", 0.0),
        )

    except Exception as e:
        logger.warning(f"OSRM failed: {e}")
        return None, None


# ============================================================
# Scoring Logic
# ============================================================
# ============================================================
# Scoring Logic
# ============================================================
def compute_composite_score(
    distance_km: float,
    financial_cost: float,
    traffic_score: float,
    total_range_km: float,
    load_prediction: Optional[float] = None,
    price_prediction: Optional[float] = None
) -> Tuple[float, Dict[str, float]]:

    # Range penalty
    range_penalty_val = RANGE_PENALTY if distance_km > 0.8 * total_range_km else 0

    score_distance_raw = W_DISTANCE * distance_km
    score_cost = W_COST * financial_cost
    score_traffic = W_TRAFFIC * traffic_score
    score_range_penalty = W_RANGE * range_penalty_val
    
    score = (score_distance_raw + score_cost + score_traffic + score_range_penalty)

    if load_prediction is not None:
        score += 0.15 * load_prediction
    if price_prediction is not None:
        score += 0.1 * price_prediction

    # Return the final score and the breakdown of weighted components
    return score, {
        "distance": score_distance_raw,
        "financial_cost": score_cost,
        "traffic_score": score_traffic,
        "range_penalty": score_range_penalty
    }

# ============================================================
# Main Station Selector
# ============================================================
def find_optimal_station(
    current_location: Tuple[float, float],
    current_soc: float,
    candidate_data: List[Dict[str, Any]],
    km_per_soc_percent: float,
    load_model=None,
    price_model=None
) -> Optional[Dict[str, Any]]:

    total_range_km = current_soc * km_per_soc_percent
    best_station = None
    best_score = float("inf")

    for st in candidate_data:
        station_loc = (st["lat"], st["lon"])

        # Haversine always available
        hv_dist = haversine_distance_km(current_location, station_loc)
        hv_eta = travel_time_haversine_sec(current_location, station_loc)

        if hv_dist > total_range_km:
            continue

        # Try OSRM (Defines dist_km, eta_sec)
        used_osrm = False
        dist_km = hv_dist 
        eta_sec = hv_eta

        if USE_OSRM:
            d_osrm, t_osrm = osrm_route(current_location, station_loc)
            if d_osrm is not None:
                dist_km = d_osrm
                eta_sec = t_osrm
                used_osrm = True

        # ML Predictions (Defines load_pred, price_pred)
        load_pred = load_model.predict([[st["id"], dist_km]])[0] if load_model else None
        price_pred = price_model.predict([[st["id"], dist_km]])[0] if price_model else None

        # Composite score
        financial_cost = float(st.get("usage_cost", 0.0))
        traffic_score = float(st.get("traffic_score", 1.0))

        score, score_breakdown = compute_composite_score(
            distance_km=dist_km,
            financial_cost=financial_cost,
            traffic_score=traffic_score,
            total_range_km=total_range_km,
            load_prediction=load_pred,
            price_prediction=price_pred
        )

        # Select best
        if score < best_score:
            best_score = score
            best_station = {
                "id": st["id"], # Key consistent with frontend
                "lat": st["lat"],
                "lon": st["lon"],
                "distance_km": dist_km,
                "duration_sec": eta_sec,
                "used_osrm": used_osrm,
                "composite_score": score, # Key for main score
                "score_breakdown": score_breakdown, # Key for breakdown metrics
            }

    return best_station


# Backward compatible name
def find_optimal_station_sync(*args, **kwargs):
    logger.info("Using sync station finder…")
    return find_optimal_station(*args, **kwargs)
