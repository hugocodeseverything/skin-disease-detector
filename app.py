# ================= IMPORTS =================
import streamlit as st
import numpy as np
import joblib
import xgboost as xgb
from PIL import Image
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage import measure
from scipy.stats import skew

from disease_descriptions import DISEASE_DESCRIPTIONS


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Skin Disease Detection",
    layout="centered"
)

st.markdown("""
<style>
    body { background-color: #0E1117; color: #FAFAFA; }
    .stApp { background-color: #0E1117; }
</style>
""", unsafe_allow_html=True)


# ================= LOAD MODEL =================
@st.cache_resource
def load_models():
    model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_models()


# ================= SIDEBAR =================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Prediction", "Fun Facts", "How To Use"]
)


# ================= PREPROCESS =================
def preprocess_image(img):
    img = np.array(img)

    img = resize(img, (256, 256), anti_aliasing=True)

    hsv = rgb2hsv(img)
    hsv[..., 2] = equalize_adapthist(hsv[..., 2], clip_limit=0.03)
    img = hsv2rgb(hsv)

    for i in range(3):
        p2, p98 = np.percentile(img[..., i], (2, 98))
        img[..., i] = rescale_intensity(img[..., i], in_range=(p2, p98))

    img = np.clip(img, 0, 1)
    img = gaussian(img, sigma=0.5, channel_axis=-1)

    return img


# ================= FEATURE EXTRACTION =================
def extract_features_from_image(img):
    gray = rgb2gray(img)
    gray_uint8 = (gray * 255).astype(np.uint8)

    lbp = local_binary_pattern(gray_uint8, P=16, R=2, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=18, range=(0, 18), density=True)

    gray_q = (gray_uint8 // 8).astype(np.uint8)
    glcm = graycomatrix(
        gray_q,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32,
        symmetric=True,
        normed=True
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    glcm_features = np.hstack([graycoprops(glcm, p).ravel() for p in props])

    small_gray = resize(gray, (64, 64), anti_aliasing=True)
    hog_feat = hog(
        small_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    hsv = rgb2hsv(img)
    hsv_hist = np.concatenate([
        np.histogram(hsv[..., 0], bins=32, range=(0, 1), density=True)[0],
        np.histogram(hsv[..., 1], bins=16, range=(0, 1), density=True)[0],
        np.histogram(hsv[..., 2], bins=16, range=(0, 1), density=True)[0],
    ])

    color_moments = []
    for c in range(3):
        ch = img[..., c].ravel()
        color_moments.extend([np.mean(ch), np.std(ch), skew(ch, bias=False)])
    color_moments = np.array(color_moments)

    try:
        thresh = threshold_otsu(gray)
        mask = (gray > thresh).astype(np.uint8)
        labeled = measure.label(mask)
        regions = measure.regionprops(labeled)

        if regions:
            largest = max(regions, key=lambda x: x.area)
            area = largest.area / (img.shape[0] * img.shape[1])
            perimeter = largest.perimeter
            eccentricity = largest.eccentricity
            solidity = largest.solidity
            compactness = (4 * np.pi * largest.area) / (perimeter ** 2) if perimeter > 0 else 0
        else:
            area = perimeter = eccentricity = solidity = compactness = 0.0
    except:
        area = perimeter = eccentricity = solidity = compactness = 0.0

    return np.concatenate([
        lbp_hist,
        glcm_features,
        hog_feat,
        hsv_hist,
        color_moments,
        np.array([area, eccentricity, solidity, compactness, perimeter])
    ])


# ================= PAGE: PREDICTION =================
if page == "Prediction":
    st.title("Skin Disease Detection")
    st.caption("Machine Learning–based Skin Disease Classification")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a skin image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Disease"):
            img_prep = preprocess_image(image)
            features = extract_features_from_image(img_prep)
            features = scaler.transform([features])

            dmatrix = xgb.DMatrix(features)
            probs = model.predict(dmatrix)[0]

            classes = label_encoder.classes_
            top_idx = np.argsort(probs)[::-1][:3]

            main_class = classes[top_idx[0]]
            main_conf = probs[top_idx[0]] * 100

            st.success(f"Prediction: **{main_class}** ({main_conf:.2f}%)")

            st.subheader("Disease Description")
            st.write(
                DISEASE_DESCRIPTIONS.get(
                    main_class,
                    "Description not available for this condition."
                )
            )

            st.subheader("Prediction Confidence")

            labels = [classes[i] for i in top_idx]
            values = [probs[i] * 100 for i in top_idx]

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")

            colors = ["#4CAF50", "#2196F3", "#FFC107"]
            bars = ax.barh(labels, values, color=colors)

            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            ax.set_xlabel("Confidence (%)", color="white")

            ax.tick_params(colors="white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("white")
            ax.spines["bottom"].set_color("white")

            for bar, value in zip(bars, values):
                ax.text(
                    value - 2,
                    bar.get_y() + bar.get_height()/2,
                    f"{value:.2f}%",
                    va="center",
                    ha="right",
                    color="black",
                    fontweight="bold"
                )

            plt.subplots_adjust(left=0.45)
            st.pyplot(fig)

            st.warning(
                "This application is for educational purposes only and not a medical diagnosis."
            )


# ================= PAGE: FUN FACTS =================
elif page == "Fun Facts":
    st.title("Fun Facts About Skin Diseases")
    st.markdown("""
- Skin is the **largest organ** in the human body  
- Some skin diseases **mimic each other visually**, making diagnosis difficult  
- AI models often rely on **texture patterns**, not color alone  
- Lighting conditions can significantly affect predictions  
- Early detection dramatically improves treatment outcomes
""")


# ================= PAGE: HOW TO USE =================
elif page == "How To Use":
    st.title("How To Use This App")
    st.markdown("""
1. Upload a clear photo of the affected skin area  
2. Ensure good lighting and focus  
3. Click **Predict Disease**  
4. Review the top predictions and confidence scores  
5. Consult a medical professional for diagnosis
""")
