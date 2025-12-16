import streamlit as st

st.set_page_config(page_title="Fun Facts")

st.title("Fun Facts About Skin Diseases")

facts = [
    """
    The skin is the largest organ in the human body and plays a critical role in protecting
    us from external threats such as bacteria, viruses, and harmful radiation. Because it
    is constantly exposed to the environment, the skin is also one of the most vulnerable
    organs to disease and damage.
    """,

    """
    Many skin diseases are influenced by a combination of genetic factors and environmental
    triggers. Conditions such as eczema, psoriasis, and acne may run in families, but their
    severity can be affected by lifestyle, stress, climate, and exposure to allergens.
    """,

    """
    Early detection of skin diseases significantly improves treatment outcomes. In particular,
    serious conditions like melanoma have much higher survival rates when identified at an
    early stage. This is why routine skin checks and early screening tools are so important.
    """,

    """
    Not all skin diseases are contagious. While some conditions are caused by infections,
    many others—such as autoimmune or inflammatory skin disorders—cannot be transmitted
    from person to person. Misunderstanding this often leads to unnecessary stigma.
    """,

    """
    Machine learning models can assist clinicians as decision-support tools by analyzing
    visual patterns that may not be obvious to the human eye. These systems are designed
    to support medical professionals, not replace them, by providing additional insights
    and prioritizing high-risk cases.
    """,

    """
    The appearance of a skin condition can vary greatly depending on lighting, camera quality,
    and skin tone. This variability makes automated analysis challenging and highlights the
    importance of image preprocessing techniques such as normalization and contrast enhancement.
    """,

    """
    Skin disease datasets must be diverse to ensure fair and accurate predictions. Models
    trained on limited skin tones or demographics may perform poorly on underrepresented
    groups, making dataset diversity a critical issue in medical artificial intelligence.
    """
]

for fact in facts:
    st.markdown(fact)
