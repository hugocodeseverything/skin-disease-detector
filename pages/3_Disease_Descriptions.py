import streamlit as st
from disease_descriptions import DISEASE_DESCRIPTIONS

st.set_page_config(page_title="Disease Descriptions")

st.title("Skin Disease Descriptions")

for disease, desc in DISEASE_DESCRIPTIONS.items():
    st.subheader(disease)
    st.write(desc)
    st.divider()
