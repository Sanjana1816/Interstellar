"""
Change detection UI tab.
"""
import streamlit as st
import requests
from ui.components.image_compare import render_comparison

def render_change_detection_tab(api_base: str, selected_aoi: str, date_start, date_end):
    st.markdown("## 🔄 Change Detection")
    
    if not selected_aoi:
        st.warning("Please select a specific Area of Interest (AOI) in the sidebar.")
        return
        
    st.info(f"Analyzing changes for **{selected_aoi}** between **{date_start}** and **{date_end}**.")
    
    change_types = st.multiselect(
        "Filter by change type", 
        ["construction", "clearance", "water_change", "new_road"],
        default=["construction", "clearance", "water_change", "new_road"]
    )
    
    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing temporal pairs..."):
            payload = {
                "aoi_id": selected_aoi,
                "date_start": str(date_start),
                "date_end": str(date_end),
                "change_types": change_types
            }
            try:
                r = requests.post(f"{api_base}/change/analyze", json=payload)
                if r.status_code == 200:
                    data = r.json()
                    candidates = data.get("candidates", [])
                    
                    st.success(f"Found {len(candidates)} change candidates.")
                    
                    for cand in candidates:
                        with st.expander(f"[{cand['change_type'].upper()}] Confidence: {cand['confidence']:.2f} - {cand['earliest_observed_date']}"):
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                url_before = f"{api_base}{cand['tile_before']['thumbnail_url']}" if cand['tile_before'].get('thumbnail_url') else None
                                url_after = f"{api_base}{cand['tile_after']['thumbnail_url']}" if cand['tile_after'].get('thumbnail_url') else None
                                
                                if url_before and url_after:
                                    render_comparison(url_before, url_after, cand['tile_before']['date'], cand['tile_after']['date'])
                                else:
                                    st.warning("Thumbnails missing.")
                            
                            with col2:
                                st.markdown(f"**Type:** {cand['change_type']}")
                                st.markdown(f"**Confidence:** {cand['confidence']:.2f}")
                                st.markdown(f"**Date:** {cand['earliest_observed_date']}")
                                
                                st.progress(cand['confidence'])
                                
                                if st.button("Send to Review Queue", key=f"q_{cand['candidate_id']}"):
                                    st.success("Candidate queued for analyst review.")
                                    # Actually in this system, analyzing creates the pending record automatically.
                                    # This button is just a confirmation UX.
                else:
                    st.error(f"Analysis failed: {r.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
