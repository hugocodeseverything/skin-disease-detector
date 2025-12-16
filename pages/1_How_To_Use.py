import streamlit as st

st.set_page_config(page_title="How to Use")

st.title("How to Use the Prediction System")

st.write("""
1. Open the **Skin Disease Detection** page.
2. Upload a clear image of the affected skin area.
3. Ensure proper lighting and focus.
4. Click **Predict Disease**.
5. The system will display:
   - Predicted disease
   - Confidence score
   - Top 3 predictions
   - Disease description
""")

st.info(
    "This application is designed for academic and demonstration purposes."
)
