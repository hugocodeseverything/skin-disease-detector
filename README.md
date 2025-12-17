# DermaVision
Skin disease classification web app using Manual Feature Extraction, XGBoost, trained on DermNet dataset and deployed with Streamlit.
Our group member consist of:

-Sefanya Putri Novira

Contribution:

-Making Demo Video

- Keep track of the progress

- Making PPT

-Hugo Sachio Wijaya

- With Sefanya, making PPT

- With Renaldo, Designing the entire pipeline, training model, suggest improvement, accuracy 

-Deploying the model to streamlit

-Writing this Readme.md, writing introduction, and methodology

-Deploying all of the file to github

-Renaldo

-With Hugo, training the model, suggest improvement, improving the model, pipeline and accuracy

-Running the code and suggest where to deploy, writing literature review 

-Testing streamlit model to see if there were any bugs

# DermaVision — Intelligent Skin Disease Classification System

DermaVision is a computer vision–based skin disease classification system designed to assist early screening and improve diagnostic efficiency through automated image analysis. This project combines advanced image preprocessing, handcrafted feature extraction, and an optimized XGBoost classifier to deliver fast, consistent, and reliable predictions.

---

## Project Overview

Skin diseases affect millions of people worldwide and often require expert visual examination. However, access to dermatology specialists can be limited, and manual diagnosis may be subjective.  
DermaVision aims to provide a **lightweight, efficient, and deployable** machine learning solution that supports early detection and automated screening of skin diseases.

**Key objectives:**
- Classify multiple types of skin diseases from images
- Reduce subjectivity in visual diagnosis
- Provide fast and reproducible predictions
- Enable deployment in web-based applications

---

## Dataset

This project uses the **DermNet Dataset**, a widely used dermatology image dataset.

- Total images: **15,447**
- Number of classes: **23 skin disease categories**
- Source: DermNet via Kaggle  
  https://www.kaggle.com/datasets/shubhamgoel27/dermnet/data

The dataset contains diverse clinical images representing various skin conditions, making it suitable for multi-class classification tasks.

---

##  Methodology & Pipeline

### 1. Image Preprocessing
Each image undergoes the following preprocessing steps:
- Resizing to **256 × 256**
- RGB → HSV color space conversion
- CLAHE for local contrast enhancement
- Contrast stretching (2nd–98th percentile)
- Gaussian denoising
- Pixel normalization (0–1)

### 2. Feature Extraction
Handcrafted features are extracted to represent texture, color, and shape:
- Local Binary Pattern (LBP)
- GLCM texture features
- Histogram of Oriented Gradients (HOG)
- HSV color histograms
- Color moments (mean, std, skewness)
- Shape features (area ratio, eccentricity, solidity, perimeter)

### 3. Feature Scaling & Balancing
- **StandardScaler** is used to standardize all features
- **SMOTE** is applied to handle class imbalance and improve generalization

### 4. Model Training
- Algorithm: **XGBoost**
- Tree method: `hist`
- Grow policy: `lossguide`
- Training up to **1500 boosting rounds**
- Evaluation metric: **multi-class log loss (mlogloss)**
- Early stopping applied to prevent overfitting

---

## Evaluation & Results

- Overall accuracy: **~84%**
- Strong class-wise performance across most skin disease categories
- ROC-AUC (One-vs-Rest): **≈ 0.98++** for many classes
- Stable generalization with balanced precision and recall

### Model Comparison (Same Dataset)
- **XGBoost (handcrafted features)** achieved competitive accuracy with significantly lower computational cost compared to:
  - ResNet-152
  - Vision Transformer (ViT)

We have also compare the model to SVM, SVM only achieved 7.9% accuracy with random forest achieving 9.3% accuracy. This is likely due to incompatibility with complex tabular data with the model.
This demonstrates the effectiveness of combining traditional feature engineering with gradient-boosting models.

##  Application Demo

The trained model is deployed using **Streamlit**, allowing users to:
- Upload skin images
- Receive predicted disease classes
- View brief disease descriptions
Link to Application Demo: https://skin-disease-detector-a4qmfqtqo9baf9u2mpev2p.streamlit.app/
A demo video of the Streamlit application is included in the presentation.


Project Structure:
├── app.py # Main Streamlit application
├── pages/ # Additional Streamlit pages
├── disease_descriptions.py # Disease information mapping
├── xgboost_model.pkl # Trained XGBoost model
├── scaler.pkl # Feature scaler
├── label_encoder.pkl # Label encoder
├── requirements.txt # Dependencies
├── .streamlit/ # Streamlit configuration
└── README.md

## How to Use

1. Open the **Skin Disease Detection** page in the Streamlit application.  
2. Upload a **clear image** of the affected skin area.
   - Ensure the image is well-lit.
   - Avoid blur and excessive shadows for better prediction accuracy.
3. Click the **Predict Disease** button.
4. The system will analyze the image and display:
   - **Predicted disease**
   - **Confidence score**
   - **Top 3 prediction results**
   - **Brief disease description**

This interface is designed to provide fast and interpretable results while maintaining a simple and user-friendly workflow.

