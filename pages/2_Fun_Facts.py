import streamlit as st

st.set_page_config(page_title="Fun Facts")

st.title("Fun Facts About Skin Diseases")

facts = [
    "Skin is the largest organ in the human body.",
    "Many skin diseases are influenced by genetics and environment.",
    "Early detection improves treatment outcomes.",
    "Not all skin diseases are contagious.",
    "Machine learning can assist clinicians as decision support tools."
]

for fact in facts:
    st.write("•", fact)
