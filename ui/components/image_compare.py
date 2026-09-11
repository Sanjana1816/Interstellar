"""
Image comparison component.
"""
import streamlit as st

def render_comparison(img1_url: str, img2_url: str, label1: str, label2: str):
    """Render a side-by-side image comparison (using simple columns for now to avoid custom component dependency issues)."""
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1_url, caption=f"Before: {label1}", use_container_width=True)
    with col2:
        st.image(img2_url, caption=f"After: {label2}", use_container_width=True)
