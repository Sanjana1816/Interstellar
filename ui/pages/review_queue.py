"""
Review queue and audit trail UI.
"""
import streamlit as st
import requests

def render_review_queue_tab(api_base: str):
    st.markdown("## 📋 Analyst Review Queue")
    
    try:
        r = requests.get(f"{api_base}/review/queue?sort_by=confidence&limit=20")
        if r.status_code == 200:
            queue = r.json().get("items", [])
            
            if not queue:
                st.info("Queue is empty. Great job!")
                
            for item in queue:
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: space-between;">
                            <h4>{item['change_type'].upper()} ({item['earliest_observed_date']})</h4>
                            <span class="badge badge-pending">PENDING</span>
                        </div>
                        <p>Confidence: {item['confidence']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        img_col1, img_col2 = st.columns(2)
                        url_before = f"{api_base}{item['tile_before']['thumbnail_url']}" if item['tile_before'].get('thumbnail_url') else None
                        url_after = f"{api_base}{item['tile_after']['thumbnail_url']}" if item['tile_after'].get('thumbnail_url') else None
                        
                        if url_before: img_col1.image(url_before, caption=f"Before ({item['tile_before']['date']})", use_container_width=True)
                        if url_after: img_col2.image(url_after, caption=f"After ({item['tile_after']['date']})", use_container_width=True)
                        
                    with col2:
                        note = st.text_area("Analyst Note", key=f"note_{item['candidate_id']}")
                        
                        btn_col1, btn_col2 = st.columns(2)
                        if btn_col1.button("Accept", key=f"acc_{item['candidate_id']}", type="primary"):
                            _submit_decision(api_base, item['candidate_id'], "accept", note)
                        if btn_col2.button("Reject", key=f"rej_{item['candidate_id']}"):
                            _submit_decision(api_base, item['candidate_id'], "reject", note)
                            
                    st.markdown("---")
    except Exception as e:
        st.error(f"Error loading queue: {e}")
        
    st.markdown("## 📜 Audit History")
    if st.button("Refresh History"):
        st.rerun()
        
    try:
        r = requests.get(f"{api_base}/review/history?limit=10")
        if r.status_code == 200:
            history = r.json().get("decisions", [])
            for h in history:
                color = "green" if h["decision"] == "accept" else "red"
                st.markdown(f"- **{h['timestamp']}** | {h['reviewer']} | <span style='color:{color}'>{h['decision'].upper()}</span> | {h['candidate_id']} | Note: {h['note']}", unsafe_allow_html=True)
    except Exception as e:
        pass

def _submit_decision(api_base, candidate_id, decision, note):
    try:
        payload = {
            "candidate_id": candidate_id,
            "reviewer": "analyst_1",
            "decision": decision,
            "note": note
        }
        r = requests.post(f"{api_base}/review/decision", json=payload)
        if r.status_code == 200:
            st.success("Decision recorded.")
            st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")
