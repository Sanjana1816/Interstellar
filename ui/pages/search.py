"""
Search tab for Streamlit UI.
"""
import streamlit as st
import requests

def render_search_tab(api_base: str, selected_aoi: str, date_start, date_end, selected_sensor: str):
    st.markdown("## 🔍 Semantic Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search imagery by description", placeholder="e.g. 'construction site near a river'")
    with col2:
        image_upload = st.file_uploader("Or search by image", type=["png", "jpg", "jpeg"])
        
    if st.button("Search", type="primary", use_container_width=True):
        with st.spinner("Searching..."):
            filters = {}
            if selected_aoi: filters["aoi_id"] = selected_aoi
            if date_start: filters["date_start"] = str(date_start)
            if date_end: filters["date_end"] = str(date_end)
            if selected_sensor: filters["sensor"] = selected_sensor
            
            try:
                if image_upload is not None:
                    files = {"file": image_upload.getvalue()}
                    data = {"k": 20}
                    data.update(filters)
                    r = requests.post(f"{api_base}/search/image", files=files, data=data)
                else:
                    payload = {"query": search_query, "k": 20, "filters": filters}
                    r = requests.post(f"{api_base}/search/text", json=payload)
                    
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    st.success(f"Found {len(results)} results")
                    
                    if not results:
                        st.info("No matching tiles found.")
                    
                    # Display as grid
                    cols = st.columns(4)
                    for i, res in enumerate(results):
                        with cols[i % 4]:
                            st.markdown(f"""
                            <div class="result-card">
                                <b>{res['date']}</b> | {res['sensor']}<br>
                                <small>Score: {res['score']:.4f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            if res.get("thumbnail_url"):
                                st.image(f"{api_base}{res['thumbnail_url']}", use_container_width=True)
                            
                            if st.button("Find Similar", key=f"sim_{res['tile_id']}"):
                                st.session_state["find_similar_target"] = res["tile_id"]
                                st.rerun()
                else:
                    st.error(f"Search failed: {r.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
                
    if "find_similar_target" in st.session_state:
        target_id = st.session_state["find_similar_target"]
        st.markdown(f"### 🔗 Similar to {target_id}")
        if st.button("Clear Similar"):
            del st.session_state["find_similar_target"]
            st.rerun()
            
        with st.spinner("Finding similar tiles..."):
            try:
                r = requests.post(f"{api_base}/cluster/similar", json={"tile_id": target_id, "k": 12})
                if r.status_code == 200:
                    results = r.json().get("similar_tiles", [])
                    cols = st.columns(4)
                    for i, res in enumerate(results):
                        with cols[i % 4]:
                            st.markdown(f"""
                            <div class="result-card">
                                <b>{res['date']}</b> | {res['sensor']}<br>
                                <small>Score: {res['score']:.4f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            if res.get("thumbnail_url"):
                                st.image(f"{api_base}{res['thumbnail_url']}", use_container_width=True)
            except Exception as e:
                st.error(f"Connection error: {e}")
