import json
import time

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import FastMarkerCluster
from sklearn.neighbors import BallTree
from streamlit_folium import st_folium


class _Timer:
    """Prints elapsed time for a block to the terminal (not the browser) —
    run `streamlit run app.py` from a terminal and watch its output while
    you interact with the app to see which stage is actually slow."""
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        print(f"[TIMER] {self.label}: {time.perf_counter() - self.t0:.3f}s")

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(page_title="Coastwide AEP Comparison Dashboard", layout="wide")

DATA_PATH = "https://github.com/akhalid-twi/dashboards/raw/refs/heads/main/lwi-aep-comparison-dashboard/assets/dashboard_data_lw.parquet"
FEMA_PATH = "https://github.com/akhalid-twi/dashboards/raw/refs/heads/main/lwi-aep-comparison-dashboard/assets/fema_zones.parquet"  # lightweight, merged (A/V) vector layer — set to None to skip

DEFAULT_MAP_CENTER = [29.95, -89.90]
DEFAULT_MAP_ZOOM = 10

# -----------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "map_center" not in st.session_state:
    st.session_state.map_center = DEFAULT_MAP_CENTER
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = DEFAULT_MAP_ZOOM

# -----------------------------------------------------------------------
# Load data (cached — this part is genuinely safe to cache, since it's
# read-only data with no mutation risk)
# -----------------------------------------------------------------------
@st.cache_data
def load_data(path):
    gdf = gpd.read_parquet(path)
    gdf = gdf.dropna(subset=["lat", "lon"])
    gdf = gdf[np.isfinite(gdf["lat"]) & np.isfinite(gdf["lon"])]
    return gdf


with _Timer("load_data (dashboard_data.parquet)"):
    gdf = load_data(DATA_PATH)


@st.cache_resource
def build_spatial_tree(lats: np.ndarray, lons: np.ndarray):
    coords_rad = np.radians(np.column_stack([lats, lons]))
    return BallTree(coords_rad, metric="haversine")


with _Timer("build_spatial_tree"):
    tree = build_spatial_tree(gdf["lat"].values, gdf["lon"].values)


@st.cache_data
def load_fema_layer(path):
    """
    Loads the lightweight, merged FEMA flood-zone layer (built once,
    offline, via extract_dashboard_data.py) — dissolved down to just 2
    shapes (A-type zones, V-type zones) with BFE_min/BFE_max as a summary
    range per zone type. Vector rendering is cheap at this scale (2
    polygons, not thousands), so no raster/image-overlay trick is needed
    here. Makes NO live calls to FEMA's NFHL REST service.
    """
    if path is None:
        return None
    try:
        fema_gdf = gpd.read_parquet(path)
        if fema_gdf.crs != "EPSG:4326":
            fema_gdf = fema_gdf.to_crs(epsg=4326)
        return json.loads(fema_gdf.to_json())
    except Exception as e:
        st.warning(f"Could not load FEMA zones parquet: {e}")
        return None


with _Timer("load_fema_layer"):
    fema_geojson = load_fema_layer(FEMA_PATH)

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("Coastwide — AEP Comparison Dashboard")

col_map, col_plot = st.columns([3, 2])

# -----------------------------------------------------------------------
# Handle a click from the PREVIOUS run before building the map, so the
# selected marker reflects the latest click on this render.
# -----------------------------------------------------------------------
if "aep_interactive_map" in st.session_state and st.session_state["aep_interactive_map"]:
    last_click = st.session_state["aep_interactive_map"].get("last_clicked")
    if last_click:
        lat_click, lon_click = last_click["lat"], last_click["lng"]
        dist, idx = tree.query([[np.radians(lat_click), np.radians(lon_click)]], k=1)
        nearest_idx = idx[0][0]
        distance_m = dist[0][0] * 6371000
        if distance_m < 2500 and st.session_state.selected_idx != nearest_idx:
            st.session_state.selected_idx = nearest_idx
            clicked_row = gdf.iloc[nearest_idx]
            st.session_state.map_center = [float(clicked_row["lat"]), float(clicked_row["lon"])]
            st.session_state.map_zoom = 14

# -----------------------------------------------------------------------
# Map (left column)
#
# IMPORTANT: the map is built FRESH every rerun, NOT cached. st_folium
# implements feature_group_to_add by calling .add_to(m) on the passed-in
# map object, i.e. it MUTATES it. Earlier this map was wrapped in
# @st.cache_resource to avoid rebuilding it on every click — but since
# cache_resource returns the exact same singleton object every time (and
# is shared server-wide, not per-session), every click permanently welded
# another copy of the selection marker onto that one shared map, which is
# what caused the map to break/disappear after a point was selected.
#
# The actual expensive part was never the Python-side construction here
# (that was always well under 1s even with markers/FEMA polygons included)
# — it was st_folium's browser-side rendering, which is already reduced by
# capping the point count and simplifying/dissolving the FEMA polygons at
# export time. So rebuilding fresh each run is both correct AND fast.
# -----------------------------------------------------------------------
with col_map:
    with _Timer("build map"):
        m = folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_MAP_ZOOM, tiles="OpenStreetMap")

        if fema_geojson:
            def style_fema(feature):
                zone_type = str(feature["properties"].get("zone_type", ""))
                color = "#8E24AA" if zone_type == "V" else "#0288D1"
                return {"fillColor": color, "color": color, "weight": 1, "fillOpacity": 0.4}

            folium.GeoJson(
                fema_geojson,
                name="FEMA Flood Zones",
                style_function=style_fema,
                tooltip=folium.GeoJsonTooltip(
                    fields=["zone_type", "BFE_min", "BFE_max"],
                    aliases=["Zone type:", "BFE min (ft):", "BFE max (ft):"],
                    localize=True,
                ),
                show=False,  # off by default — toggle on via the layer control (top-right)
            ).add_to(m)

        data_points = [[float(lat), float(lon)] for lat, lon in zip(gdf["lat"], gdf["lon"])]
        callback_js = """
        function (row) {
            var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
                radius: 3, fillColor: '#2171b5', color: '#2171b5', weight: 1, fillOpacity: 0.6
            });
            return marker;
        };
        """
        # disableClusteringAtZoom=1 keeps FastMarkerCluster's fast JS-side
        # marker creation (still much lighter than 25k individual Python
        # folium.Marker objects) but tells the underlying clustering library
        # to never actually group points into numbered bubbles — every
        # point renders individually at any zoom level instead of hiding
        # behind a cluster count.
        FastMarkerCluster(
            data=data_points,
            callback=callback_js,
            name="Sampled RAS cells",
            options={
                'disableClusteringAtZoom': 1,
                'spiderfyOnMaxZoom': False,
                'zoomToBoundsOnClick': False,
            },
        ).add_to(m)
        folium.LayerControl().add_to(m)

        fg_selected = folium.FeatureGroup(name="Selected Cell Marker")
        selected_row = gdf.iloc[st.session_state.selected_idx]
        folium.Marker(
            location=[float(selected_row["lat"]), float(selected_row["lon"])],
            popup=f"Cell: {selected_row.get('cell_id', st.session_state.selected_idx)}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(fg_selected)

    with _Timer("st_folium render (browser round-trip)"):
        map_data = st_folium(
            m,
            center=st.session_state.map_center,
            zoom=st.session_state.map_zoom,
            feature_group_to_add=fg_selected,
            width='stretch',
            height=650,
            returned_objects=["last_clicked"],
            key="aep_interactive_map",
        )

# -----------------------------------------------------------------------
# Handle a click from THIS run
# -----------------------------------------------------------------------
if map_data and map_data.get("last_clicked"):
    lat_click, lon_click = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    dist, idx = tree.query([[np.radians(lat_click), np.radians(lon_click)]], k=1)
    nearest_idx = idx[0][0]
    distance_m = dist[0][0] * 6371000
    if distance_m < 2500 and st.session_state.selected_idx != nearest_idx:
        st.session_state.selected_idx = nearest_idx
        clicked_row = gdf.iloc[nearest_idx]
        st.session_state.map_center = [float(clicked_row["lat"]), float(clicked_row["lon"])]
        st.session_state.map_zoom = 14

# -----------------------------------------------------------------------
# AEP comparison plot (right column)
# -----------------------------------------------------------------------
selected_row = gdf.iloc[st.session_state.selected_idx]
aep_raw = selected_row["aep"]
aep_data = json.loads(aep_raw) if isinstance(aep_raw, str) else aep_raw

COLOR_MAP = {
    "CPRA_Base": dict(color="#1E88E5", dash="solid", width=3),
    "LWI_TC_Base": dict(color="#F57C00", dash="solid", width=3),
}

LABEL_MAP = {
    "CPRA_Base": "TC Surge AEP (CPRA)",
    "LWI_TC_Base": "TC Compound AEP",
}

with col_plot:
    _t0_plot = time.perf_counter()
    raw_bfe = selected_row.get("fema_bfe")
    try:
        fema_bfe = float(raw_bfe) if raw_bfe is not None and not pd.isna(raw_bfe) else None
    except (ValueError, TypeError):
        fema_bfe = None
    bfe_str = f"{fema_bfe:.2f} ft" if fema_bfe is not None else "N/A"

    cell_id_val = selected_row.get("cell_id", st.session_state.selected_idx)
    st.markdown(f"**Cell:** {cell_id_val} | **FEMA BFE (100yr):** {bfe_str}")

    fig = go.Figure()
    all_y_values = []
    for label, data in aep_data.items():
        if not data:
            continue
        parsed = sorted(
            [(float(k), float(v)) for k, v in data.items() if v is not None and not pd.isna(v)],
            key=lambda item: item[0],
        )
        if not parsed:
            continue
        x, y = [p[0] for p in parsed], [p[1] for p in parsed]
        all_y_values.extend(y)
        style = COLOR_MAP.get(label, dict(color="gray", dash="solid", width=2))
        display_label = LABEL_MAP.get(label, label)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=display_label,
            line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
            marker=dict(size=4),
        ))

    if fema_bfe is not None:
        fig.add_hline(
            y=fema_bfe, line_dash="dashdot", line_color="#9C27B0", line_width=2.5,
            annotation_text=f"FEMA 100-Yr BFE ({fema_bfe:.2f} ft)",
            annotation_position="top left", annotation_font=dict(size=11, color="#9C27B0"),
        )

    # y-axis upper limit: 2 ft of headroom above whichever is higher — the
    # tallest curve value, or the FEMA BFE line if it's above the curves —
    # so nothing (including the FEMA line or its annotation) sits flush
    # against the top of the plot.
    y_candidates = list(all_y_values)
    if fema_bfe is not None:
        y_candidates.append(fema_bfe)
    if y_candidates:
        y_upper = max(y_candidates) + 2
        y_lower = min(y_candidates)
    else:
        y_upper, y_lower = None, None

    for rp in [10, 100, 500, 1000]:
        fig.add_vline(x=rp, line_dash="dash", line_color="gray", opacity=0.4)

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            type="log", title="Return Period (years)",
            range=[np.log10(1), np.log10(10000)],
            tickvals=[2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
            ticktext=["2", "5", "10", "25", "50", "100", "250", "500", "1000", "2000", "5000", "10000"],
        ),
        yaxis=dict(
            title="WSE (ft)",
            range=[y_lower, y_upper] if y_upper is not None else None,
        ),
        height=600,
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(title="Scenario", orientation="h", y=1.02, x=0.0, xanchor="left"),
    )
    st.plotly_chart(fig, width='stretch')
    print(f"[TIMER] AEP plot build (JSON parse + plotly figure): "
          f"{time.perf_counter() - _t0_plot:.3f}s")
