#!/usr/bin/env python3
"""
Advanced EV Dataset Generator (fixed + industry-grade)

Produces stations and charging sessions, computes analytics, and optionally
writes CSVs aligned to your schema.
"""
import os
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from sqlalchemy.orm import sessionmaker
from sqlalchemy import func

# Import your DB models & engine (assumes these exist and match your schema)
from database import engine, Station, ChargingSession, Base

# -------------------------
# Configuration
# -------------------------
RANDOM_SEED = 42

# Per-city constants
STATIONS_PER_CITY = 12
MIN_SESSIONS_PER_STATION = 80
MAX_SESSIONS_PER_STATION = 220

# DB batching
BATCH_SIZE = 1000

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_data")
CSV_STATION_FILENAME = "ev_station.csv"
CSV_SESSION_FILENAME = "charging_sessions.csv"

# Shapefile (optional). If missing, generator uses jitter around city coords.
SHAPEFILE_PATH = r"E:\project\ev-smart-charging\California_City_Boundaries_and_Identifiers_with_Coastal_Buffers_8985027092893607450\California_City_Boundaries_and_Identifiers.shp"

# Behavior flags
RECREATE_DB = False   # If True, drop_all() and create_all() — useful in dev only
WRITE_CSV = True      # If True, writes station + session CSVs to OUTPUT_DIR

# Seed RNGs
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------
# Utilities
# -------------------------
def safe_get_city_column(df, city_name):
    if df is None or df.empty:
        return gpd.GeoDataFrame()
    for col in ['CDTFA_CITY', 'CITYNAME', 'CITY', 'NAME', 'NAME_2', 'CITY_NAME']:
        if col in df.columns:
            try:
                matches = df[df[col].astype(str).str.lower().str.strip() == city_name.lower().strip()]
                if not matches.empty:
                    return matches
            except Exception:
                continue
    string_cols = [c for c in df.columns if df[c].dtype == object]
    for col in string_cols:
        try:
            matches = df[df[col].astype(str).str.contains(city_name, case=False, na=False)]
            if not matches.empty:
                return matches
        except Exception:
            continue
    return gpd.GeoDataFrame()


def sample_point_within_polygon(polygon, max_attempts=500):
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(max_attempts):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            return y, x  # (lat, lon)
    rp = polygon.representative_point()
    return rp.y, rp.x


def parse_usage_cost_string(cost_str):
    if cost_str is None:
        return 0.0, 0.0, ""
    s = str(cost_str).lower()
    if 'free' in s:
        return 0.0, 0.0, 'Free'
    cleaned = s.replace('$', '').replace('/kwh', '').replace('per kwh', '').replace('kwh', '')
    nums = []
    for token in cleaned.replace('+', ' ').replace(',', ' ').split():
        try:
            nums.append(float(token))
        except Exception:
            tok = ''.join(ch for ch in token if (ch.isdigit() or ch == '.'))
            if tok:
                try:
                    nums.append(float(tok))
                except Exception:
                    continue
    price = nums[0] if len(nums) >= 1 else 0.0
    fee = nums[1] if len(nums) >= 2 else 0.0
    return price, fee, cost_str


# -------------------------
# City list & templates
# -------------------------
bay_area_cities = {
    'San Francisco': (37.7749, -122.4194),
    'San Jose': (37.3382, -121.8863),
    'Oakland': (37.8044, -122.2712),
    'Palo Alto': (37.4419, -122.1430),
    'Mountain View': (37.3861, -122.0839),
    'Berkeley': (37.8715, -122.2730),
    'Fremont': (37.5483, -121.9886),
    'Sunnyvale': (37.3688, -122.0363),
    'Redwood City': (37.4852, -122.2364),
    'San Mateo': (37.5470, -122.3145),
    'Santa Clara': (37.3541, -121.9552),
    'Cupertino': (37.3230, -122.0322),
    'Menlo Park': (37.4530, -122.1817),
    'Hayward': (37.6688, -122.0808),
    'Concord': (37.9780, -122.0311)
}

street_names = [
    'Main', 'Oak', 'Pine', 'Cedar', 'Elm', 'Maple', 'Birch', 'Willow',
    'Chestnut', 'Walnut', 'University', 'Market', 'Broadway', 'California',
    'Geary', 'Mission', 'Sunset', 'Park', 'Lake', 'Hill'
]

charger_types = ['Level 2', 'DC Fast', 'Tesla Supercharger', 'CCS', 'CHAdeMO']

usage_cost_templates = [
    'Free',
    '$0.10/kWh',
    '$0.15/kWh',
    '$0.20/kWh',
    '$0.25/kWh + $1 session fee'
]


# -------------------------
# Generator
# -------------------------
def generate_smart_data(recreate_db=False, write_csv=True, output_dir=OUTPUT_DIR):
    print("🔄 Starting EV dataset generation...")

    os.makedirs(output_dir, exist_ok=True)

    Session = sessionmaker(bind=engine)
    db = Session()

    # optional DB recreate (use carefully)
    if recreate_db:
        print("⚠ Recreating DB (drop_all/create_all) as requested.")
        try:
            Base.metadata.drop_all(engine)
            Base.metadata.create_all(engine)
            print("✅ DB recreated.")
        except Exception as e:
            print("❌ Error recreating DB:", e)
            db.close()
            return

    # load shapefile robustly
    ca_boundary = gpd.GeoDataFrame()
    try:
        ca_boundary = gpd.read_file(SHAPEFILE_PATH)
        ca_boundary = ca_boundary.to_crs(epsg=4326)
        print("📍 Shapefile loaded and reprojected to EPSG:4326")
    except Exception as e:
        print("⚠️ Could not load shapefile (will sample around city centers):", e)
        ca_boundary = gpd.GeoDataFrame()

    # create station rows and commit them so they have station_id
    station_meta = []
    print("🏗️ Creating stations...")
    for city, (lat_center, lon_center) in bay_area_cities.items():
        city_shape = gpd.GeoDataFrame()
        if not ca_boundary.empty:
            city_shape = safe_get_city_column(ca_boundary, city)

        if city_shape.empty:
            print(f"⚠ Polygon not found for {city} — sampling with jitter around center.")

        for i in range(STATIONS_PER_CITY):
            if not city_shape.empty:
                poly = city_shape.geometry.values[0]
                lat, lon = sample_point_within_polygon(poly)
            else:
                lat = lat_center + random.uniform(-0.03, 0.03)
                lon = lon_center + random.uniform(-0.03, 0.03)

            charger = random.choices(
                population=charger_types,
                weights=[0.6, 0.15, 0.05, 0.12, 0.08],
                k=1
            )[0]

            usage_str = random.choice(usage_cost_templates)
            price_per_kwh, session_fee, _ = parse_usage_cost_string(usage_str)

            status = random.choices(['Active', 'Maintenance', 'Offline'], weights=[85, 10, 5], k=1)[0]

            # Create station record (NO analytics fields yet)
            st = Station(
                name=f"{city} Charging Hub {i+1}",
                latitude=round(lat, 6),
                longitude=round(lon, 6),
                street_address=f"{random.randint(100,999)} {random.choice(street_names)} {random.choice(['St','Ave','Blvd','Dr'])}",
                city=city,
                state="CA",
                zip_code=f"94{random.randint(100,999)}",
                charger_type=charger,
                usage_cost=usage_str,
                status=status
            )

            db.add(st)

            # Store meta for session generation (use same keys later)
            station_meta.append({
                "station_obj": st,
                "price_per_kwh": price_per_kwh,
                "session_fee": session_fee
            })

    # commit stations so station_id PKs are stable
    try:
        db.commit()
        print(f"✅ Committed {len(station_meta)} stations to DB.")
    except Exception as e:
        print("❌ Error committing stations:", e)
        db.rollback()
        db.close()
        return

    # refresh station objects from DB to ensure station_id is available
    for m in station_meta:
        try:
            db.refresh(m["station_obj"])
        except Exception:
            # if refresh fails, continue; station_id may still be accessible after commit
            pass

    # session generation
    print("⚡ Generating sessions...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    session_objects_buffer = []
    created_session_count = 0

    for meta in station_meta:
        st = meta["station_obj"]
        price = meta["price_per_kwh"]
        fee = meta["session_fee"]

        num_sessions = random.randint(MIN_SESSIONS_PER_STATION, MAX_SESSIONS_PER_STATION)

        for _ in range(num_sessions):
            # realistic distributions by charger type
            if st.charger_type == 'Level 2':
                energy = float(np.random.normal(loc=22, scale=6))
                energy = max(4.0, min(45.0, energy))
                duration = float(np.random.normal(loc=90, scale=30))
                duration = max(20.0, min(480.0, duration))
            elif st.charger_type in ['DC Fast', 'CCS', 'CHAdeMO']:
                energy = float(np.random.normal(loc=40, scale=20))
                energy = max(8.0, min(120.0, energy))
                duration = float(np.random.normal(loc=25, scale=12))
                duration = max(5.0, min(180.0, duration))
            else:  # Tesla Supercharger etc
                energy = float(np.random.normal(loc=60, scale=18))
                energy = max(10.0, min(150.0, energy))
                duration = float(np.random.normal(loc=30, scale=10))
                duration = max(5.0, min(120.0, duration))

            cost = round(energy * price + fee, 2)
            session_date = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

            cs = ChargingSession(
                station_id=st.station_id,
                session_date=session_date,
                energy_consumed_kwh=round(energy, 2),
                duration_min=round(duration, 2),
                cost=cost
            )

            session_objects_buffer.append(cs)
            created_session_count += 1

            # commit in batches
            if len(session_objects_buffer) >= BATCH_SIZE:
                try:
                    db.bulk_save_objects(session_objects_buffer)
                    db.commit()
                    session_objects_buffer = []
                    print(f"✅ Committed a batch of {BATCH_SIZE} sessions (total so far: {created_session_count})")
                except Exception as e:
                    print("❌ Error committing session batch:", e)
                    db.rollback()
                    session_objects_buffer = []

    # commit remaining sessions
    try:
        if session_objects_buffer:
            db.bulk_save_objects(session_objects_buffer)
        db.commit()
        print(f"✅ Final sessions committed (total sessions: {created_session_count})")
    except Exception as e:
        print("❌ Error in final session commit:", e)
        db.rollback()

    # -------------------------
    # Recompute station analytics from DB authoritative source
    # -------------------------
    print("📊 Computing station analytics from committed sessions...")

    try:
        aggregates = db.query(
            ChargingSession.station_id,
            func.count(ChargingSession.session_id).label("cnt"),
            func.sum(ChargingSession.energy_consumed_kwh).label("sum_energy"),
            func.sum(ChargingSession.cost).label("sum_cost")
        ).group_by(ChargingSession.station_id).all()

        agg_map = {
            row.station_id: {
                "cnt": int(row.cnt or 0),
                "sum_energy": float(row.sum_energy or 0.0),
                "sum_cost": float(row.sum_cost or 0.0)
            } for row in aggregates
        }

        stations_db = db.query(Station).all()
        for st in stations_db:
            agg = agg_map.get(st.station_id, {"cnt": 0, "sum_energy": 0.0, "sum_cost": 0.0})
            st.session_count = agg["cnt"]
            # DB field is `total_energy`; CSV field will be `total_energy_kwh`
            if hasattr(st, "total_energy"):
                st.total_energy = round(agg["sum_energy"], 2)
            else:
                # fallback if your model uses different name
                try:
                    setattr(st, "total_energy", round(agg["sum_energy"], 2))
                except Exception:
                    pass
            st.total_revenue = round(agg["sum_cost"], 2)
            db.add(st)
        db.commit()
        print("✅ Station analytics updated from session aggregates.")
    except Exception as e:
        print("❌ Error computing/updating station analytics:", e)
        db.rollback()

    # -------------------------
    # Optionally export CSVs (canonical, from DB)
    # -------------------------
    if write_csv:
        print("💾 Exporting CSVs...")
        try:
            stations_q = db.query(Station).all()
            stations_out = []
            for st in stations_q:
                # map DB field total_energy -> csv total_energy_kwh
                total_energy_kwh = None
                if hasattr(st, "total_energy"):
                    total_energy_kwh = float(st.total_energy or 0.0)
                elif hasattr(st, "total_energy_kwh"):
                    total_energy_kwh = float(st.total_energy_kwh or 0.0)
                else:
                    total_energy_kwh = 0.0

                stations_out.append({
                    "station_id": st.station_id,
                    "name": st.name,
                    "latitude": st.latitude,
                    "longitude": st.longitude,
                    "street_address": st.street_address,
                    "city": st.city,
                    "state": st.state,
                    "zip_code": st.zip_code,
                    "charger_type": st.charger_type,
                    "usage_cost": st.usage_cost,
                    "status": st.status,
                    "session_count": st.session_count,
                    "total_energy_kwh": total_energy_kwh,
                    "total_revenue": float(st.total_revenue or 0.0)
                })
            station_df = pd.DataFrame(stations_out)
            station_csv_path = os.path.join(output_dir, CSV_STATION_FILENAME)
            station_df.to_csv(station_csv_path, index=False)
            print(f"✅ Wrote stations CSV: {station_csv_path}")
        except Exception as e:
            print("❌ Error writing stations CSV:", e)

        try:
            sessions_q = db.query(ChargingSession).order_by(ChargingSession.session_id).all()
            sessions_out = []
            for ss in sessions_q:
                sessions_out.append({
                    "session_id": ss.session_id,
                    "station_id": ss.station_id,
                    "session_date": ss.session_date.strftime("%Y-%m-%d %H:%M:%S") if ss.session_date else "",
                    "energy_consumed_kwh": float(ss.energy_consumed_kwh),
                    "duration_min": float(ss.duration_min),
                    "cost": float(ss.cost)
                })
            session_df = pd.DataFrame(sessions_out)
            session_csv_path = os.path.join(output_dir, CSV_SESSION_FILENAME)
            session_df.to_csv(session_csv_path, index=False)
            print(f"✅ Wrote sessions CSV: {session_csv_path}")
        except Exception as e:
            print("❌ Error writing sessions CSV:", e)

    # -------------------------
    # Final summary
    # -------------------------
    try:
        total_stations = db.query(Station).count()
        total_sessions = db.query(ChargingSession).count()
        active_count = db.query(Station).filter(Station.status == 'Active').count()
        maintenance_count = db.query(Station).filter(Station.status == 'Maintenance').count()
        offline_count = db.query(Station).filter(Station.status == 'Offline').count()

        print("\n🎯 Generation complete")
        print(f"📊 Stations: {total_stations}")
        print(f"📊 Sessions: {total_sessions}")
        if total_stations:
            print(f"📊 Avg sessions per station: {total_sessions/total_stations:.1f}")
        print(f"📊 Active: {active_count} | Maintenance: {maintenance_count} | Offline: {offline_count}")
        print(f"💾 Output dir: {os.path.abspath(output_dir)}")
    except Exception:
        pass

    db.close()


# -------------------------
# CLI / entrypoint
# -------------------------
if __name__ == "__main__":
    generate_smart_data(recreate_db=RECREATE_DB, write_csv=WRITE_CSV, output_dir=OUTPUT_DIR)
