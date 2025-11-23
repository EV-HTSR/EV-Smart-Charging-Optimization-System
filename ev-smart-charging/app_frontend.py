from __future__ import annotations
import os
import json
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from math import radians, sin, cos, asin, sqrt
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic

# -------------------------
# Configuration
# -------------------------
API_BASE = os.environ.get("API_BASE", "http://localhost:8001")
STATIONS_URL = f"{API_BASE}/stations"
SESSIONS_URL = f"{API_BASE}/sessions"
ANALYTICS_SUM_URL = f"{API_BASE}/analytics/summary"
ANALYTICS_STATUS_URL = f"{API_BASE}/analytics/current-status"
ANALYTICS_WEEKLY_URL = f"{API_BASE}/analytics/weekly-metrics"
ANALYTICS_HOURLY_URL = f"{API_BASE}/analytics/hourly-usage"
DEMAND_FORECAST_URL = f"{API_BASE}/analytics/demand-forecast"
SELECT_STATION_URL = f"{API_BASE}/select-best-station"
HEALTH_URL = f"{API_BASE}/health"

# ML artifact paths (if used locally)
CLUSTER_INSIGHTS_PATH = os.path.join("ml_models", "cluster_insights.json")
CLUSTER_VISUAL_PATH = os.path.join("ml_models", "cluster_scatter.html")

# Streamlit page config
st.set_page_config(page_title="EV Charging Analytics", layout="wide", page_icon="⚡")

# -------------------------
# Styling (unchanged UI)
# -------------------------
st.markdown(
    """
    <style>
        .main-header {font-size: 3rem; color: #1f77b4; margin-bottom: 0;}
        .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px;}
        .cluster-card {border-left: 4px solid #1f77b4; padding-left: 15px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# HTTP helpers (no caching of errors)
# -------------------------

def get_json(url: str, params: dict | None = None, timeout: int = 30) -> Any:
    """GET wrapper that returns JSON or an error dict. Does not cache errors.
    Keeps behavior simple so UI sees fresh backend state.
    """
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"error": "invalid_json_response", "text": r.text}
    except requests.exceptions.ConnectionError as e:
        return {"error": f"connection_error: {str(e)}"}
    except requests.exceptions.ReadTimeout as e:
        return {"error": f"timeout: {str(e)}"}
    except requests.HTTPError as e:
        # attempt to surface backend JSON error if present
        try:
            return {"error": r.json()}
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": f"frontend_exception: {str(e)}"}


def post_json(url: str, payload: dict, timeout: int = 60, retries: int = 1) -> Any:
    """
    POST wrapper with higher default timeout and optional retries.
    Returns parsed JSON or an error dict.
    """
    attempt = 0
    last_err = None
    while attempt <= retries:
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                return {"result": r.text}
        except requests.exceptions.ConnectionError as e:
            last_err = f"connection_error: {str(e)}"
            attempt += 1
        except requests.exceptions.ReadTimeout as e:
            last_err = f"timeout: {str(e)}"
            attempt += 1
        except requests.HTTPError as e:
            try:
                return {"error": r.json()}
            except Exception:
                return {"error": str(e)}
        except Exception as e:
            last_err = f"frontend_exception: {str(e)}"
            attempt += 1
    return {"error": last_err or "unknown_error"}



def safe_get(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def km_haversine(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1 = a; lat2, lon2 = b
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    A = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    C = 2 * asin(sqrt(A))
    return 6371.0 * C

# -------------------------
# Data loaders (simple, non-cached)
# -------------------------

def load_stations_df() -> pd.DataFrame:
    data = get_json(STATIONS_URL)
    if not data or (isinstance(data, dict) and data.get("error")):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "station_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "station_id"})
    for c in ("latitude", "longitude", "session_count", "total_energy_kwh", "total_revenue"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_sessions_df(limit: int = 500) -> pd.DataFrame:
    data = get_json(SESSIONS_URL, params={"limit": limit})
    if not data or (isinstance(data, dict) and data.get("error")):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "session_date" in df.columns:
        df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce")
    return df


def load_summary() -> dict:
    data = get_json(ANALYTICS_SUM_URL)
    return data if isinstance(data, dict) else {}


def load_status() -> dict:
    data = get_json(ANALYTICS_STATUS_URL)
    return data if isinstance(data, dict) else {}


def load_weekly() -> dict:
    data = get_json(ANALYTICS_WEEKLY_URL)
    return data if isinstance(data, dict) else {}


def load_forecast() -> dict:
    data = get_json(DEMAND_FORECAST_URL)
    return data if isinstance(data, dict) else {}

# -------------------------
# Helpers
# -------------------------

def _load_cluster_insights(path: Optional[str] = None):
    try:
        if path is None:
            path = CLUSTER_INSIGHTS_PATH
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _prepare_cluster_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cluster_id" not in df.columns:
        return pd.DataFrame()
    agg = {"session_count": "mean", "total_energy": "mean", "total_revenue": "mean"}
    grouped = df.groupby("cluster_id").agg(agg).reset_index()
    grouped = grouped.rename(columns={"session_count": "avg_session_count", "total_energy": "avg_total_energy", "total_revenue": "avg_total_revenue"})
    grouped["cluster_id"] = grouped["cluster_id"].astype(str)
    if "avg_session_duration_min" not in grouped.columns:
        grouped["avg_session_duration_min"] = np.nan
    return grouped

# -------------------------
# UI Pages
# -------------------------

def show_overview():
    st.markdown("<h1 class='main-header'>EV Charging Analytics Overview</h1>", unsafe_allow_html=True)

    analytics = load_summary()
    if not analytics or (isinstance(analytics, dict) and analytics.get("error")):
        st.warning("Could not load analytics data. Please check your database and backend.")
        return

    total_stations = analytics.get("total_stations", 0)
    total_sessions = analytics.get("total_sessions", 0)
    total_energy = analytics.get("total_energy_kwh", 0)
    total_revenue = analytics.get("total_revenue", 0)

    stations = load_stations_df()
    total_stations_count = len(stations)
    active_stations = len(stations[stations.get("status") == "Active"]) if not stations.empty and "status" in stations.columns else 0

    utilization_rate = (active_stations / total_stations_count * 100) if total_stations_count > 0 else 0
    avg_daily_sessions = total_sessions / 30 if total_sessions else 0
    revenue_per_kwh = (total_revenue / total_energy) if total_energy > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", f"{total_sessions:,}")
    with col2:
        st.metric("Energy Delivered", f"{total_energy:,.0f} kWh")
    with col3:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    with col4:
        st.metric("Station Utilization", f"{utilization_rate:.1f}%")

    st.subheader("📊 Performance Efficiency")
    eff_col1, eff_col2, eff_col3, eff_col4 = st.columns(4)
    avg_session_energy = (total_energy / total_sessions) if total_sessions else 0
    avg_session_revenue = (total_revenue / total_sessions) if total_sessions else 0

    with eff_col1:
        st.metric("Avg Daily Sessions", f"{avg_daily_sessions:.1f}")
    with eff_col2:
        st.metric("Avg Session Energy", f"{avg_session_energy:.1f} kWh")
    with eff_col3:
        st.metric("Avg Session Revenue", f"${avg_session_revenue:.2f}")
    with eff_col4:
        st.metric("Revenue per kWh", f"${revenue_per_kwh:.3f}")

    st.subheader("🔍 Operational Snapshot")
    current_status = load_status()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Sessions Now", current_status.get("active_sessions", 0))
    with col2:
        st.metric("Today's Sessions", current_status.get("today_sessions", 0))
    with col3:
        st.metric("Offline Stations", current_status.get("offline_stations", 0))

    st.subheader("🎯 Key Insights")
    insights = []
    revenue_per_station = total_revenue / total_stations_count if total_stations_count > 0 else 0
    sessions_per_station = total_sessions / total_stations_count if total_stations_count > 0 else 0
    insights.append(f"**💰 Revenue per Station**: ${revenue_per_station:,.2f}")
    insights.append(f"**⚡ Sessions per Station**: {sessions_per_station:.1f}")
    if utilization_rate < 50:
        insights.append(f"**⚠️ Low Utilization**: Only {utilization_rate:.1f}% of stations active")
    else:
        insights.append(f"**✅ Good Utilization**: {utilization_rate:.1f}% of stations active")

    cols = st.columns(len(insights))
    for i, insight in enumerate(insights):
        with cols[i]:
            st.info(insight)

    # st.subheader("📈 7-Day Performance")

    # weekly = load_weekly()
    # if weekly["weekly_sessions"] == 0:
    #     st.warning("No sessions recorded in the last week.")
    # else:
    #     base_sessions = weekly["weekly_sessions"] / 7
    #     sessions_data = [round(base_sessions * m) for m in [1.1,1.2,1.15,1.25,1.3,0.9,0.7]]
    #     fig = px.bar(x=days, y=sessions_data, title="Simulated Sessions (Last Week)")
    #     st.plotly_chart(fig, use_container_width=True)
    st.subheader("📈 7-Day Performance")

    weekly = load_weekly()  # API call
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    # 1️⃣ Try to get current week sessions
    weekly_sessions = weekly.get("weekly_sessions", 0)

    # 2️⃣ If current week has no sessions, try previous week
    if weekly_sessions == 0:
        weekly_sessions = weekly.get("previous_week_sessions", 0)
        if weekly_sessions > 0:
            chart_title = "Sessions (Last Available Week)"
        else:
            chart_title = "Simulated Sessions (No Data)"
    else:
        chart_title = "Sessions (Last Week)"

    # 3️⃣ Compute sessions_data
    if "daily_sessions" in weekly and isinstance(weekly["daily_sessions"], list) and any(weekly["daily_sessions"]):
        # Use actual daily breakdown if available and non-zero
        sessions_data = weekly["daily_sessions"]
    elif weekly_sessions > 0:
        # Distribute weekly total into a pattern
        base_sessions = weekly_sessions / 7
        pattern = [1.1, 1.2, 1.15, 1.25, 1.3, 0.9, 0.7]  # Mon → Sun
        sessions_data = [max(1, round(base_sessions * m)) for m in pattern]  # ensure >=1
    else:
        # Fallback simulated chart
        sessions_data = [3, 4, 2, 5, 3, 2, 1]  # minimal illustrative values
    # 5️⃣ Plot chart with proper labels
    fig = px.bar(
        x=days,
        y=sessions_data,
        labels={"x": "Day", "y": "Sessions"},
        title=chart_title
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("🚀 Recommended Actions")
    actions = []
    if utilization_rate < 60:
        actions.append("**📍 Expand marketing** in low-utilization areas")
    if revenue_per_kwh < 0.25:
        actions.append("**💰 Review pricing strategy** for better margins")
    if avg_session_energy < 15:
        actions.append("**🔌 Consider promoting** longer charging sessions")
    if not actions:
        actions.append("**✅ Operations are optimal** - maintain current strategy")
    for action in actions:
        st.success(action)


def show_station_map():
    st.header("📍 Interactive Station Map")
    stations = load_stations_df()
    if stations.empty:
        st.warning("Could not load station data.")
        return

    df = stations.copy()
    required = ["latitude","longitude","charger_type"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return

    palette = { 'Level 2':'#0072B2', 'DC Fast':'#009E73', 'Tesla Supercharger':'#D55E00', 'Unknown':'#999999' }

    col1, col2, col3 = st.columns(3)
    with col1:
        cities = sorted(df['city'].dropna().unique()) if 'city' in df.columns else []
        selected_city = st.selectbox('🌆 Filter by City', ['All'] + cities)
    with col2:
        chargers = sorted(df['charger_type'].dropna().unique()) if 'charger_type' in df.columns else []
        selected_charger = st.selectbox('🔌 Filter by Charger Type', ['All'] + chargers)
    with col3:
        statuses = sorted(df['status'].dropna().unique()) if 'status' in df.columns else []
        selected_status = st.selectbox('🟢 Filter by Status', ['All'] + statuses) if statuses else 'All'

    if selected_city != 'All': df = df[df['city']==selected_city]
    if selected_charger != 'All': df = df[df['charger_type']==selected_charger]
    if selected_status != 'All' and 'status' in df.columns: df = df[df['status']==selected_status]

    if df.empty:
        st.info('No stations to show')
        return

    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', hover_name='name', hover_data={'city':True,'charger_type':True,'street_address':True}, color='charger_type', color_discrete_map=palette, size_max=20, zoom=11, title='EV Charging Stations Network')
    fig.update_layout(mapbox_style='carto-positron', margin={'r':0,'t':40,'l':0,'b':0})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Stations Shown', len(df))
    with col2:
        st.metric('Cities', len(df['city'].unique()) if 'city' in df.columns else 0)
    with col3:
        st.metric('Charger Types', len(df['charger_type'].unique()))
    with col4:
        if 'status' in df.columns:
            st.metric('Available', int((df['status']=='Active').sum()))

    with st.expander('💡 Map Usage Tips'):
        st.markdown('''
        **🎨 Color Legend:**
        - 🔵 Level 2 (Standard)
        - 🟢 DC Fast (Quick)
        - 🟠 Tesla Supercharger (Premium)
        ''')


def show_data_explorer():
    st.header('📋 Data Explorer')
    tab1, tab2 = st.tabs(['Stations','Sessions'])
    with tab1:
        stations = get_json(STATIONS_URL)
        if stations and not (isinstance(stations, dict) and stations.get('error')):
            st.dataframe(pd.DataFrame(stations), use_container_width=True)
        else:
            st.info('No station data available')
    with tab2:
        sess = get_json(SESSIONS_URL, params={'limit':200})
        if sess and not (isinstance(sess, dict) and sess.get('error')):
            st.dataframe(pd.DataFrame(sess), use_container_width=True)
        else:
            st.info('No session data available')


def show_cluster_analysis():
    st.header('🤖 Cluster Analysis (Interactive)')
    stations = load_stations_df()
    if stations.empty:
        st.warning('No station data returned from backend. Cluster analysis unavailable.')
        return
    df = stations.copy()
    if 'cluster_id' not in df.columns:
        df['cluster_id'] = 0
    if 'total_energy' not in df.columns and 'total_energy_kwh' in df.columns:
        df['total_energy'] = df['total_energy_kwh']
    df['cluster_id'] = df['cluster_id'].astype(int)
    df['session_count'] = pd.to_numeric(df.get('session_count', pd.Series([0]*len(df))), errors='coerce').fillna(0)
    df['total_energy'] = pd.to_numeric(df.get('total_energy', pd.Series([0]*len(df))), errors='coerce').fillna(0.0)
    df['total_revenue'] = pd.to_numeric(df.get('total_revenue', pd.Series([0]*len(df))), errors='coerce').fillna(0.0)

    st.markdown('### Filters')
    filter_col1, filter_col2 = st.columns([2,3])
    with filter_col1:
        clusters = sorted(df['cluster_id'].unique().tolist())
        selected_clusters = st.multiselect('Select Cluster(s)', options=[str(x) for x in clusters], default=[str(x) for x in clusters])
        selected_clusters_int = [int(x) for x in selected_clusters]
    with filter_col2:
        chargers = sorted(df['charger_type'].fillna('Unknown').unique().tolist()) if 'charger_type' in df.columns else ['Unknown']
        selected_chargers = st.multiselect('Charger Type', options=chargers, default=chargers)

    filtered_df = df.copy()
    if selected_clusters:
        filtered_df = filtered_df[filtered_df['cluster_id'].isin(selected_clusters_int)]
    if selected_chargers:
        filtered_df = filtered_df[filtered_df['charger_type'].isin(selected_chargers)]

    cluster_agg = _prepare_cluster_aggregates(df)

    tab_map, tab_dist, tab_fin, tab_insights = st.tabs(['🗺️ Cluster Map','📊 Cluster Distribution','📈 Revenue & Cost','📝 Actionable Insights'])
    with tab_map:
        st.subheader('🗺️ Cluster Map')
        if filtered_df.empty:
            st.info('No stations match selected filters')
        else:
            fig_map = px.scatter_mapbox(filtered_df, lat='latitude', lon='longitude', hover_name='name', hover_data={'cluster_id':True,'charger_type':True,'session_count':True,'total_energy':True,'total_revenue':True}, color=filtered_df['cluster_id'].astype(str), size='session_count', size_max=25, zoom=10, mapbox_style='open-street-map')
            st.plotly_chart(fig_map, use_container_width=True, height=650)
            with st.expander('🔎 Stations in view (first 200)'):
                cols = ['name','cluster_id','charger_type','session_count','total_energy','total_revenue']
                st.dataframe(filtered_df[cols].sort_values('session_count', ascending=False).head(200), use_container_width=True)

    with tab_dist:
        st.subheader('📊 Cluster Distribution')
        if cluster_agg.empty:
            st.info('Cluster aggregation not available.')
        else:
            pivot = filtered_df.groupby(['cluster_id','charger_type']).agg({'session_count':'sum'}).reset_index()
            if pivot.empty:
                st.info('No data to show for selected filters.')
            else:
                fig1 = px.bar(pivot, x='charger_type', y='session_count', color=pivot['cluster_id'].astype(str), barmode='group', title='Sessions by Charger Type (grouped by cluster)')
                st.plotly_chart(fig1, use_container_width=True)
            usage_df = cluster_agg.copy()
            usage_long = usage_df.melt(id_vars=['cluster_id'], value_vars=['avg_session_count','avg_total_energy','avg_session_duration_min'], var_name='metric', value_name='value')
            usage_long['metric_label'] = usage_long['metric'].map({'avg_session_count':'Avg Sessions','avg_total_energy':'Avg Energy (kWh)','avg_session_duration_min':'Avg Duration (min)'}).fillna(usage_long['metric'])
            if usage_long[usage_long['metric']=='avg_session_duration_min']['value'].isna().all():
                usage_long = usage_long[usage_long['metric']!='avg_session_duration_min']
            fig2 = px.bar(usage_long, x='cluster_id', y='value', color='metric_label', barmode='group', title='Cluster-level Average Metrics')
            st.plotly_chart(fig2, use_container_width=True)

    with tab_fin:
        st.subheader('📈 Revenue & Cost')
        if filtered_df.empty:
            st.info('No stations to show for selected filters.')
        else:
            min_rev = float(filtered_df['total_revenue'].min())
            max_rev = float(filtered_df['total_revenue'].max())
            if min_rev == max_rev:
                rev_min, rev_max = min_rev, max_rev
            else:
                rev_min, rev_max = st.slider('Filter by Total Revenue ($)', min_value=float(np.floor(min_rev)), max_value=float(np.ceil(max_rev)), value=(float(np.floor(min_rev)), float(np.ceil(max_rev))))
            scatter_df = filtered_df[(filtered_df['total_revenue']>=rev_min) & (filtered_df['total_revenue']<=rev_max)]
            if scatter_df.empty:
                st.info('No stations match the revenue filter.')
            else:
                fig_scatter = px.scatter(scatter_df, x='total_energy', y='total_revenue', size='session_count', color=scatter_df['cluster_id'].astype(str), hover_name='name', title='Station Financial Performance: Energy vs Revenue (size = sessions)')
                st.plotly_chart(fig_scatter, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.write('**Top revenue stations (in selection)**')
                    top_rev = scatter_df.nlargest(5, 'total_revenue')[['name','cluster_id','total_revenue']]
                    st.table(top_rev.head(5).assign(total_revenue=lambda d: d['total_revenue'].map(lambda x: f"${x:,.2f}")))
                with col2:
                    st.write('**High energy stations (in selection)**')
                    top_energy = scatter_df.nlargest(5, 'total_energy')[['name','cluster_id','total_energy']]
                    st.table(top_energy.head(5).assign(total_energy=lambda d: d['total_energy'].map(lambda x: f"{x:,.0f} kWh")))

    with tab_insights:
        st.subheader('📝 Actionable Insights')
        insights = _load_cluster_insights()
        if not insights:
            st.warning("Cluster insights file not found. Please run the ML pipeline to generate 'ml_models/cluster_insights.json'.")
            st.info("You can run: `python train_models.py` in your project's root to produce insights.")
        else:
            for cluster_key, insight in insights.items():
                display_title = cluster_key
                try:
                    cluster_num = cluster_key.split('_')[-1]
                    display_title = f"Cluster {cluster_num} - {insight.get('characteristics','')}"
                except Exception:
                    pass
                with st.expander(display_title, expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric('Stations', insight.get('station_count', 'N/A'))
                    with col2:
                        avg_sessions = insight.get('avg_sessions', None)
                        st.metric('Avg Sessions', f"{avg_sessions:.0f}" if isinstance(avg_sessions, (int,float)) else 'N/A')
                    with col3:
                        avg_revenue = insight.get('avg_revenue', None)
                        st.metric('Avg Revenue', f"${avg_revenue:.0f}" if isinstance(avg_revenue, (int,float)) else 'N/A')
                    st.write(f"**Most common charger:** {insight.get('common_charger', 'Unknown')}")
                    st.write(f"**Avg energy delivered:** {insight.get('avg_energy', 'N/A')} kWh")
                    extra_text = insight.get('notes') or insight.get('recommendations') or insight.get('description')
                    if extra_text:
                        st.markdown(f"**Notes:** {extra_text}")


def show_usage_analytics():
    st.header('📈 Advanced Usage Analytics')
    stations = load_stations_df()
    sessions = load_sessions_df(limit=1000)
    if stations.empty or sessions.empty:
        st.warning('Could not load data for advanced analytics')
        return
    station_df = stations
    session_df = sessions
    total_revenue = station_df['total_revenue'].sum() if 'total_revenue' in station_df.columns else 0
    total_energy = station_df['total_energy_kwh'].sum() if 'total_energy_kwh' in station_df.columns else 0
    total_sessions = station_df['session_count'].sum() if 'session_count' in station_df.columns else len(session_df)

    tab1, tab2, tab3, tab4 = st.tabs(['📊 Overview','🔍 Patterns','🏆 Top Performers','📋 Insights'])
    with tab1:
        st.subheader('Performance Overview')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Revenue', f"${total_revenue:,.2f}")
        with col2:
            st.metric('Energy Delivered', f"{total_energy:,.0f} kWh")
        with col3:
            avg_session_energy = total_energy / total_sessions if total_sessions else 0
            st.metric('Avg Session', f"{avg_session_energy:.1f} kWh")
        with col4:
            avg_session_cost = total_revenue / total_sessions if total_sessions else 0
            st.metric('Avg Cost', f"${avg_session_cost:.2f}")
        if not session_df.empty and 'energy_consumed_kwh' in session_df.columns and 'cost' in session_df.columns:
            fig = px.scatter(session_df, x='energy_consumed_kwh', y='cost', title='Energy vs Cost Relationship', trendline='ols')
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader('Usage Patterns Analysis')
        for col in ['session_count','total_energy_kwh','total_revenue']:
            if col not in station_df.columns:
                station_df[col] = 0
        if not station_df.empty:
            charger_summary = station_df.groupby('charger_type').agg({'session_count':'sum','total_energy_kwh':'sum','total_revenue':'sum'}).reset_index()
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(charger_summary, values='session_count', names='charger_type', title='Sessions by Charger Type')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(charger_summary, x='charger_type', y='total_revenue', title='Revenue by Charger Type')
                st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.subheader('Top Performing Stations')
        if not station_df.empty:
            top_revenue = station_df.nlargest(5, 'total_revenue')
            st.write('**🏆 Top Stations by Revenue:**')
            for _, station in top_revenue.iterrows():
                st.write(f"• {station['name']} ({station.get('city','Unknown')}): ${station['total_revenue']:.2f}")
            top_energy = station_df.nlargest(5, 'total_energy_kwh')
            st.write('**⚡ Top Stations by Energy Delivered:**')
            for _, station in top_energy.iterrows():
                st.write(f"• {station['name']} ({station.get('city','Unknown')}): {station['total_energy_kwh']:.0f} kWh")
    with tab4:
        st.subheader('Business Insights')
        insights = []
        if total_energy > 0:
            revenue_per_kwh = total_revenue / total_energy
            insights.append(f"**💰 Revenue Efficiency**: ${revenue_per_kwh:.3f} per kWh delivered")
        insights.append(f"**📊 Session Stats**: {total_sessions} total sessions")
        if 'charger_type' in station_df.columns:
            charger_summary = station_df.groupby('charger_type').agg({'total_revenue':'sum'}).reset_index()
            if not charger_summary.empty:
                most_profitable = charger_summary.loc[charger_summary['total_revenue'].idxmax()]
                insights.append(f"**🏆 Best Performer**: {most_profitable['charger_type']} chargers generated ${most_profitable['total_revenue']:.2f}")
        for insight in insights:
            st.info(insight)
        st.success('''
        **💡 Strategic Recommendations:**
        - Expand high-performing charger types in top revenue cities
        - Optimize pricing strategy based on energy delivery costs
        - Focus maintenance on high-utilization stations
        - Consider demand-based pricing during peak periods
        ''')


def show_demand_forecast():
    st.header('📈 Demand Forecast')
    st.info('Next 7 days of charging demand prediction based on trained models.')

    forecast_data = get_json(DEMAND_FORECAST_URL, timeout=60)

    if not forecast_data or (isinstance(forecast_data, dict) and 'error' in forecast_data):
        err = forecast_data.get('error') if isinstance(forecast_data, dict) else 'No response'
        st.error(f'Could not fetch forecast data. Error: {err}')
        st.info('Hint: run `python train_models.py` and restart backend to generate `ml_models/weekly_forecast.json`.')
        return

    if not isinstance(forecast_data, dict) or 'days' not in forecast_data or 'predicted_sessions' not in forecast_data:
        st.error('Invalid forecast format received from backend.')
        st.json(forecast_data)
        return

    days = forecast_data.get('days', [])
    predicted_sessions = forecast_data.get('predicted_sessions', [])
    predicted_energy = forecast_data.get('predicted_energy', [])

    # Normalize types
    try:
        predicted_sessions = [int(round(float(x))) for x in predicted_sessions]
    except Exception:
        predicted_sessions = [int(x) if isinstance(x, (int, np.integer)) else 0 for x in predicted_sessions]
    try:
        predicted_energy = [float(x) for x in predicted_energy]
    except Exception:
        predicted_energy = [0.0 for _ in predicted_energy]

    if len(days) != len(predicted_sessions) or (predicted_energy and len(days) != len(predicted_energy)):
        st.error('Forecast arrays length mismatch (days vs predictions).')
        st.json(forecast_data)
        return

    tab1, tab2, tab3 = st.tabs(['Sessions Forecast','Energy Forecast','Analysis'])
    with tab1:
        st.subheader('📊 Predicted Daily Charging Sessions')
        if predicted_sessions:
            fig = px.bar(x=days, y=predicted_sessions, title='Predicted Daily Charging Sessions', labels={'x':'Day','y':'Number of Sessions'})
            st.plotly_chart(fig, use_container_width=True)
            peak_idx = int(np.argmax(predicted_sessions)) if predicted_sessions else 0
            peak_day = days[peak_idx] if days else 'N/A'
            st.metric('📆 Peak Demand Day', peak_day, f"{predicted_sessions[peak_idx]} sessions")
        else:
            st.info('No session data available for plotting.')
    with tab2:
        st.subheader('⚡ Predicted Daily Energy Demand')
        if predicted_energy:
            fig = px.line(x=days, y=predicted_energy, title='Predicted Daily Energy Demand (kWh)', labels={'x':'Day','y':'Energy (kWh)'})
            st.plotly_chart(fig, use_container_width=True)
            total_energy_fore = float(np.sum(predicted_energy))
            st.metric('🔋 Total Weekly Energy Forecast', f"{total_energy_fore:,.0f} kWh")
        else:
            st.info('No energy data available for plotting.')
    with tab3:
        st.subheader('📋 Forecast Analysis')
        col1, col2 = st.columns(2)
        with col1:
            st.write('**📅 Daily Predictions:**')
            if days and predicted_sessions and predicted_energy:
                for day, sessions, energy in zip(days, predicted_sessions, predicted_energy):
                    st.write(f"**{day}**: {sessions} sessions, {energy:,.0f} kWh")
            else:
                st.write('No predictions available.')
        with col2:
            st.write('**🎯 Insights:**')
            st.write('• **Weekdays**: Typically show higher demand due to commuter traffic.')
            st.write('• **Weekends**: Lower demand periods, ideal for scheduled maintenance.')
            st.write('• **Energy**: Total weekly energy requirement to inform grid load management.')
        if any(s > 0 for s in predicted_sessions):
            st.success('''
            **💡 Recommended Actions:**
            - **Optimize Pricing** during predicted peak session days/times.
            - **Staff Scheduling** should prioritize days with high session counts.
            - **Grid Load Management** should use the energy forecast (kWh) to prepare.
            ''')
        else:
            st.warning('Forecasted demand is zero or unavailable. Please review training data or rerun `train_models.py`.')

# -------------------------
# TSP optimizer (uses backend /select-best-station)
# -------------------------
@st.cache_data
def load_stations_tsp(csv_file: str):
    try:
        df = pd.read_csv(csv_file)
        # Add simulated traffic score if missing
        if "current_traffic_score" not in df.columns:
            random.seed(42) 
            df["current_traffic_score"] = [random.randint(1, 10) for _ in range(len(df))]
        return df
    except Exception as e:
        st.error(f"Could not load station data from CSV: {e}. Please ensure the file path is correct: {csv_file}")
        return pd.DataFrame()

def show_tsp_optimizer():
    st.header("🚗 EV Charging — Best Station Selector (TSP)")
    
    # Configuration - using global API_BASE
    BACKEND_ENDPOINT = f"{API_BASE}/select-best-station"
    OSRM_ROUTE_URL = "http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    # IMPORTANT: Keep the original file path as requested
    CSV_FILE = "E:\\project\\ev-smart-charging\\exported_data\\ev_stations.csv"
    MAX_NEARBY_DISTANCE_KM = 10 

    # -------------------------------
    # Initialize session state (scoped to this functionality)
    # -------------------------------
    if "best_station_result" not in st.session_state:
        st.session_state.best_station_result = None
    if "last_input" not in st.session_state:
        st.session_state.last_input = None

    # -------------------------------
    # Input panel
    # -------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        # Use unique keys
        current_lat = st.number_input("Current Latitude", value=37.7749, key="tsp_lat")
        current_lon = st.number_input("Current Longitude", value=-122.4194, key="tsp_lon")
        current_soc = st.slider("Current SoC (%)", 0, 100, 50, key="tsp_soc")
        km_per_soc_percent = st.number_input("Km per 1% SoC", min_value=0.1, value=3.0, step=0.1, key="tsp_kmpersoc")
        total_range_km = current_soc * km_per_soc_percent
        st.metric("🚗 Calculated Driving Range (km)", f"{total_range_km:.2f}")
        submit = st.button("Find Best Station", key="tsp_submit")

    with col2:
        st.markdown("### ℹ️ Info")
        st.info(
            "Enter your current location, battery SoC, and vehicle efficiency (km per 1% SoC). "
            "The map will show your driving range and candidate stations. This feature uses a Travelling Salesperson Problem (TSP) based algorithm to find the optimal charging stop."
        )

    # Load stations from CSV
    # Load stations from DATABASE instead of CSV (minimal-change)
    try:
        import sqlite3
        conn = sqlite3.connect("ev_charging.db")
        stations_df = pd.read_sql_query("SELECT * FROM stations", conn)
        conn.close()

        # Add simulated traffic score if missing
        if "current_traffic_score" not in stations_df.columns:
            random.seed(42)
            stations_df["current_traffic_score"] = [
                random.randint(1, 10) for _ in range(len(stations_df))
            ]

    except Exception as e:
        st.error(f"Could not load station data from database: {e}")
        return

    # Convert to list of dicts for backend
    # --------------------------
    # FAST FIX: Reduce station list BEFORE sending to backend
    # -------------------------
    def distance_km(lat1, lon1, lat2, lon2):
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return 6371 * (2 * asin(sqrt(a)))

    # Add distance column
    stations_df["dist"] = stations_df.apply(
        lambda r: distance_km(current_lat, current_lon, r["latitude"], r["longitude"]),
        axis=1
    )

    # Keep only **closest 30 stations**
    filtered = stations_df.nsmallest(30, "dist")

    # Build candidate list from filtered list (NOT entire dataset)
    candidate_stations = []
    for _, row in filtered.iterrows():
        # Convert usage cost safely
        raw_cost = str(row.get("usage_cost", "")).replace("$", "").replace("/kWh", "").strip()
        usage_cost = float(raw_cost) if raw_cost.replace(".", "", 1).isdigit() else 0.0

        candidate_stations.append({
            "id": int(row["station_id"]),

            # Backend-required fields
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "usage_cost": usage_cost,            # <-- REQUIRED field name
            "traffic_score": float(row.get("traffic", row.get("traffic_score", 1))),  # <-- REQUIRED
            "status": row.get("status", "Active"),
        })

   # -------------------------------
    # Submit to backend
    # -------------------------------
    if submit:
        payload = {
            "current_location": {"lat": current_lat, "lon": current_lon},
            "current_soc": current_soc,
            "km_per_soc_percent": km_per_soc_percent,
            "candidate_stations": candidate_stations
        }
        try:
            with st.spinner(f"Finding best station using {BACKEND_ENDPOINT}..."):
                resp = requests.post(BACKEND_ENDPOINT, json=payload, timeout=60)

            if resp.status_code != 200:
                st.error(f"Backend error: {resp.status_code} - {resp.text}")
                st.session_state.best_station_result = None
            else:
                st.session_state.best_station_result = resp.json()
                st.session_state.last_input = payload
        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Could not reach backend at {API_BASE}. Ensure the backend is running.")
            st.session_state.best_station_result = None
        except Exception as e:
            st.exception(e)
            st.session_state.best_station_result = None

    # -------------------------------
    # Display result and map
    # -------------------------------
    if st.session_state.best_station_result:
        result = st.session_state.best_station_result
        best = result.get("best_station")
        if not best:
            st.warning("No valid station found matching criteria (e.g., within range, or active).")
        else:
            best_name = best.get("name", "Unknown")
            st.markdown(f"## ✅ Best Station Selected: {best_name} (ID {best.get('id', 'N/A')})")

            # Composite Score Breakdown as KPI-style metrics
            st.subheader("Composite Score Breakdown")

            scores = best.get("score_breakdown", {})

                # Ranges and definitions (EDIT these as needed)
            range_info = {
                    "distance": (0, 100),
                    "financial_cost": (0, 10),
                    "traffic_score": (0, 5),
                    "range_penalty": (0, 50),
                    "composite_score": (0, 1)
                }

            definitions = {
                    "distance": "Lower is better; distance between current point and station.",
                    "financial_cost": "Estimated cost impact of charging at this station.",
                    "traffic_score": "Traffic congestion impact near the station.",
                    "range_penalty": "Penalty applied if EV range is insufficient.",
                    "composite_score": "Final weighted score combining all metrics."
                }

            col1, col2, col3, col4, col5 = st.columns(5)

                # Distance Score
            with col1:
                r = range_info["distance"]
                st.metric("Distance Score", f"{scores.get('distance', 0):.2f}", f"Range: {r[0]}–{r[1]}")
                st.caption(definitions["distance"])

            # Financial Cost
            with col2:
                r = range_info["financial_cost"]
                st.metric("Financial Cost", f"{scores.get('financial_cost', 0):.2f}", f"Range: {r[0]}–{r[1]}")
                st.caption(definitions["financial_cost"])

            # Traffic Score
            with col3:
                r = range_info["traffic_score"]
                st.metric("Traffic Score", f"{scores.get('traffic_score', 0):.2f}", f"Range: {r[0]}–{r[1]}")
                st.caption(definitions["traffic_score"])

            # Range Penalty
            with col4:
                r = range_info["range_penalty"]
                st.metric("Range Penalty", f"{scores.get('range_penalty', 0):.2f}", f"Range: {r[0]}–{r[1]}")
                st.caption(definitions["range_penalty"])

            # Composite Score
            with col5:
                r = range_info["composite_score"]
                st.metric("Composite Score", f"{best.get('composite_score', 0):.2f}", f"Range: {r[0]}–{r[1]}")
                st.caption(definitions["composite_score"])
            st.metric("Distance to Station (km)", f"{best.get('distance_km', 0):.2f}")


            # -------------------------------
            # Map visualization
            # -------------------------------
            m = folium.Map(location=[current_lat, current_lon], zoom_start=12)

            # Driving range circle
            folium.Circle(
                location=[current_lat, current_lon],
                radius=total_range_km*1000,
                color="blue",
                fill=True,
                fill_opacity=0.1,
                tooltip="Driving Range"
            ).add_to(m)

            # Current location
            folium.Marker([current_lat, current_lon], popup="Current Location", icon=folium.Icon(color="blue")).add_to(m)

            # Best station marker
            folium.Marker(
                [best["lat"], best["lon"]],
                popup=f"Best Station {best.get('id','N/A')} — {best_name}",
                icon=folium.Icon(color="green", icon="star")
            ).add_to(m)

            # Nearby stations
            for station in candidate_stations:
                if station["id"] == best.get("id"):
                    continue
                dist = geodesic((current_lat, current_lon), (station["lat"], station["lon"])).km
                
                if dist <= MAX_NEARBY_DISTANCE_KM:
                    reason = []
                    if station.get("status") != "Active":
                        reason.append(station.get("status"))
                    if station["usage_cost"] > best.get("usage_cost", 0):
                        reason.append("High usage cost")
                    if dist > total_range_km:
                        reason.append("Out of range")
                    tooltip_text = " / ".join(reason) if reason else "Nearby station"
                    folium.Marker(
                        [station["lat"], station["lon"]],
                        popup=f"{station.get('name','Unknown')} ({dist:.2f} km)",
                        tooltip=tooltip_text,
                        icon=folium.Icon(color="orange")
                    ).add_to(m)

            # Route to best station
            try:
                route_url = OSRM_ROUTE_URL.format(
                    lon1=current_lon, lat1=current_lat,
                    lon2=best["lon"], lat2=best["lat"]
                )
                route_resp = requests.get(route_url, timeout=10)
                if route_resp.status_code == 200:
                    data = route_resp.json()
                    coords = data["routes"][0]["geometry"]["coordinates"]
                    route_latlon = [[lat, lon] for lon, lat in coords]
                    folium.PolyLine(route_latlon, color="red", weight=4, opacity=0.8).add_to(m)
                else:
                    folium.PolyLine([[current_lat, current_lon], [best["lat"], best["lon"]]], color="red", weight=3, opacity=0.8).add_to(m)
            except Exception:
                folium.PolyLine([[current_lat, current_lon], [best["lat"], best["lon"]]], color="red", weight=3, opacity=0.8).add_to(m)

            st_folium(m, width=900, height=600)

# -------------------------
# Sidebar & Routing
# -------------------------
st.sidebar.title('⚡ EV Analytics Dashboard')
page = st.sidebar.radio('Navigation', ['Overview','Station Map','Data Explorer','Cluster Analysis','Usage Analytics','Demand Forecast','Smart Optimizer (TSP)'])
st.sidebar.markdown('---')
st.sidebar.info('**Team Members:**\n- Backend Engineer\n- ML Engineer\n- Frontend Developer\n- DevOps & QA')

# Health check
health = get_json(HEALTH_URL)
if health and not (isinstance(health, dict) and health.get('error')):
    st.sidebar.success('✅ Backend Connected')
else:
    st.sidebar.error('❌ Backend Offline')
    st.warning('Backend server may be offline — some pages will not function correctly.')

# Page router
if page == 'Overview':
    show_overview()
elif page == 'Station Map':
    show_station_map()
elif page == 'Data Explorer':
    show_data_explorer()
elif page == 'Cluster Analysis':
    show_cluster_analysis()
elif page == 'Usage Analytics':
    show_usage_analytics()
elif page == 'Demand Forecast':
    show_demand_forecast()
elif page == 'Smart Optimizer (TSP)':
    show_tsp_optimizer()

st.markdown('---')
st.caption('Merged frontend — UI preserved; backend integration improved.')
