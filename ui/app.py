"""
Interstellar — Streamlit Frontend

Main application with 4 tabs:
1. Search (text + image semantic search)
2. Change Detection
3. Review Queue (analyst workflow)
4. Admin / Ingestion
"""

import streamlit as st
import requests
import sys
from pathlib import Path

# Add project root to path so we can import app modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Page Config ---
st.set_page_config(
    page_title="Interstellar — Satellite Imagery Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Base URL ---
API_BASE = "http://localhost:8000"


def check_api_health():
    """Check if the backend API is reachable."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# --- Custom CSS ---
st.markdown("""
<style>
    /* Global dark theme enhancements */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
    }

    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .metric-card h3 {
        color: #e2e8f0;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card .value {
        color: #60a5fa;
        font-size: 2rem;
        font-weight: 700;
    }

    /* Result card */
    .result-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }

    .result-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-pending { background: rgba(234, 179, 8, 0.2); color: #fbbf24; }
    .badge-accepted { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
    .badge-rejected { background: rgba(239, 68, 68, 0.2); color: #f87171; }

    /* Confidence bar */
    .confidence-bar {
        width: 100%;
        height: 8px;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 4px;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
        border-right: 1px solid rgba(99, 102, 241, 0.1);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.1);
        color: #94a3b8;
        padding: 0.5rem 1.25rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        color: #e2e8f0 !important;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Offline indicator */
    .offline-badge {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ade80;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown('<h1 class="main-title">🛰️ Interstellar</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Semantic & Change-Aware Satellite Imagery Search System</p>',
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🔧 System Status")

    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline — Start with `uvicorn app.main:app`")

    st.markdown(
        '<span class="offline-badge">🔒 Offline / On-Premises</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")

    # AOI selection
    aoi_options = ["All AOIs", "default", "new_delhi", "mumbai"]
    if api_healthy:
        try:
            r = requests.get(f"{API_BASE}/")
            if r.status_code == 200:
                pass  # Could fetch dynamic AOI list
        except Exception:
            pass

    selected_aoi = st.selectbox("Area of Interest", aoi_options, key="global_aoi")
    if selected_aoi == "All AOIs":
        selected_aoi = None

    # Date range
    import datetime

    col1, col2 = st.columns(2)
    with col1:
        date_start = st.date_input("From", value=datetime.date(2023, 1, 1), key="global_date_start")
    with col2:
        date_end = st.date_input("To", value=datetime.date.today(), key="global_date_end")

    # Sensor filter
    sensor_options = ["All Sensors", "sentinel-2", "sentinel-1", "landsat", "bhuvan"]
    selected_sensor = st.selectbox("Sensor", sensor_options, key="global_sensor")
    if selected_sensor == "All Sensors":
        selected_sensor = None

    st.markdown("---")
    st.markdown(
        "**Interstellar v1.0** — Built for fully offline satellite imagery intelligence."
    )


# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Search",
    "🔄 Change Detection",
    "📋 Review Queue",
    "⚙️ Admin & Ingestion",
])


# --- Tab 1: Search ---
with tab1:
    from ui.pages.search import render_search_tab
    render_search_tab(API_BASE, selected_aoi, date_start, date_end, selected_sensor)


# --- Tab 2: Change Detection ---
with tab2:
    from ui.pages.change_detection import render_change_detection_tab
    render_change_detection_tab(API_BASE, selected_aoi, date_start, date_end)


# --- Tab 3: Review Queue ---
with tab3:
    from ui.pages.review_queue import render_review_queue_tab
    render_review_queue_tab(API_BASE)


# --- Tab 4: Admin & Ingestion ---
with tab4:
    from ui.pages.admin import render_admin_tab
    render_admin_tab(API_BASE)
