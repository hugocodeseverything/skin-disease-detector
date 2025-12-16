import streamlit as st

st.set_page_config(page_title="Fun Facts")

st.title("Fun Facts About Skin Diseases")

facts = [
    "The human skin is the largest organ in the body and serves as the first line of defense against environmental threats. Because of this, skin conditions often manifest early and visibly, making dermatological diseases some of the most commonly diagnosed medical conditions worldwide. However, many skin diseases share very similar visual characteristics, such as redness, scaling, or inflammation, which makes accurate diagnosis challenging even for trained professionals.

One of the biggest challenges in skin disease diagnosis is visual overlap between different conditions. For example, fungal infections, eczema, and psoriasis can all present with similar redness and texture patterns. This visual similarity is one reason why machine learning models do not rely solely on color information. Instead, they focus heavily on texture, shape, and structural features, such as lesion borders, surface roughness, and distribution patterns across the skin.

Lighting conditions play a significant role in both human and machine diagnosis. Variations in lighting, shadows, camera quality, and skin tone can drastically alter the appearance of a lesion. In medical datasets, the same disease can look very different depending on how and where the photo was taken. This is why preprocessing techniques—such as contrast enhancement and normalization—are critical in machine learning pipelines to reduce visual bias.

Machine learning models trained on skin images often perform feature extraction to identify meaningful patterns. These features may include texture descriptors, color histograms, and shape-related metrics. Interestingly, models can detect subtle patterns that are not immediately obvious to the human eye, such as microscopic texture irregularities or statistical variations in color distribution. This makes AI a valuable decision-support tool, especially in early screening scenarios.

Despite advances in artificial intelligence, skin disease classification models are not a replacement for medical professionals. AI systems do not have clinical context, patient history, or physical examination capabilities. Instead, they are designed to assist by narrowing down possible conditions and highlighting high-risk cases that may require urgent attention. Final diagnosis and treatment decisions should always be made by qualified healthcare providers.

Another important consideration is data diversity. Skin disease datasets must include a wide range of skin tones, ages, and demographic backgrounds to avoid biased predictions. Historically, many medical image datasets have been skewed toward lighter skin tones, which can reduce model performance on underrepresented groups. Addressing this issue is an active area of research in medical AI.

Finally, early detection of skin diseases—especially malignant conditions such as melanoma—can dramatically improve treatment outcomes. Studies show that early-stage detection significantly increases survival rates and reduces treatment complexity. AI-powered screening tools have the potential to improve accessibility, especially in regions with limited access to dermatologists, by providing an initial assessment and encouraging timely medical consultation. "
    "Skin is the largest organ in the human body.",
    "Many skin diseases are influenced by genetics and environment.",
    "Early detection improves treatment outcomes.",
    "Not all skin diseases are contagious.",
    "Machine learning can assist clinicians as decision support tools."
]

for fact in facts:
    st.write("•", fact)
