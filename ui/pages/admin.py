"""
Admin and ingestion UI.
"""
import streamlit as st
import requests

def render_admin_tab(api_base: str):
    st.markdown("## ⚙️ System Administration")
    
    col1, col2, col3 = st.columns(3)
    
    # Fetch stats
    index_size = 0
    try:
        r = requests.get(f"{api_base}/")
        if r.status_code == 200:
            index_size = r.json().get("index_size", 0)
    except:
        pass
        
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Index Size</h3>
            <div class="value">{index_size}</div>
            <p style="color:#94a3b8; font-size:0.8rem; margin:0">vectors in FAISS</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Mode</h3>
            <div class="value">Offline</div>
            <p style="color:#94a3b8; font-size:0.8rem; margin:0">Airgapped execution</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📥 Incremental Ingestion")
    st.info("Upload a multi-band GeoTIFF to add to the index without rebuilding.")
    
    with st.form("ingest_form"):
        upload_file = st.file_uploader("Imagery file (.tif)", type=["tif", "tiff", "npy"])
        
        col_a, col_b, col_c = st.columns(3)
        aoi = col_a.text_input("AOI ID", value="default")
        sensor = col_b.selectbox("Sensor", ["sentinel-2", "sentinel-1", "landsat", "bhuvan"])
        date = col_c.text_input("Date (YYYY-MM-DD)", value="2024-01-01")
        
        submit = st.form_submit_button("Process & Index", type="primary")
        
        if submit and upload_file:
            with st.spinner("Processing image, extracting tiles, generating embeddings, and updating FAISS..."):
                try:
                    files = {"file": (upload_file.name, upload_file.getvalue())}
                    data = {"aoi_id": aoi, "sensor": sensor, "date": date}
                    
                    r = requests.post(f"{api_base}/ingest/add", files=files, data=data)
                    if r.status_code == 200:
                        res = r.json()
                        st.success(f"Success! {res['message']}")
                        st.metric("New Index Size", res['index_size_after'], res['tiles_added'])
                    else:
                        st.error(f"Ingestion failed: {r.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
