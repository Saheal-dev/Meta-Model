import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ✅ Import your working meta-feature extraction function directly
from meta_feature_extraction import extract_meta_features

# Page title
st.title("📊 Meta-ML Model Recommender")

# Upload dataset
uploaded_file = st.file_uploader("📁 Upload your dataset (CSV only)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📌 Preview of Uploaded Dataset:")
    st.dataframe(df.head())

    # Extract meta-features
    st.markdown("🔍 **Extracting meta-features...**")
    meta_features = extract_meta_features(df)

    if not meta_features or len(meta_features) == 0:
        st.error("❌ Failed to extract meta-features. Check the dataset format.")
    else:
        st.subheader("📈 Extracted Meta-Features:")
        st.json(meta_features)

        # ✅ Load the trained model and label encoder
        try:
            model = joblib.load("meta_model.pkl")
            label_encoder = joblib.load("meta_model_label_encoder.pkl")

            # ✅ Convert to DataFrame and align feature columns
            meta_df = pd.DataFrame([meta_features])

            # Ensure columns are in the same order as training
            expected_features = model.feature_names_in_  # Only available in scikit-learn 1.0+
            missing_features = [col for col in expected_features if col not in meta_df.columns]
            for col in missing_features:
                meta_df[col] = 0  # or np.nan or mean value if you know it

            # Align column order
            meta_df = meta_df[expected_features]

            # Predict
            pred = model.predict(meta_df)
            recommended_model = label_encoder.inverse_transform(pred)[0]

            st.success(f"✅ **Recommended Model:** `{recommended_model}`")

        except Exception as e:
            st.error(f"❌ Error loading model or making prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Tech Sahil")
