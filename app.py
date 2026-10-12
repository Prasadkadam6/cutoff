///     /g`import streamlit as st
import pandas as pd
import joblib
import os

# Page Config
st.set_page_config(
    page_title="University Cutoff Predictor",
    page_icon="🎓",
    layout="centered"
)

# Header
st.title("🎓 University Cutoff Predictor App")
st.write("This interactive app predicts the cutoff scores/ranks for university courses using a machine learning pipeline trained on past datasets.")

# Cache model loading
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.warning("⚠️ 'model.pkl' not found in the directory!")
    uploaded_file = st.file_uploader("Upload your trained model.pkl file", type=["pkl"])
    if uploaded_file is not None:
        with open("model.pkl", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Model uploaded and saved successfully! Refreshing...")
        st.rerun()
else:
    st.success("✅ Model loaded successfully!")

    st.markdown("### Enter Input Parameters")

    # Create form fields for the 10 features expected by the preprocessor
    col1, col2 = st.columns(2)

    with col1:
        university_id = st.text_input("University ID", value="banaras-hindu-university")
        programme_id = st.text_input("Programme ID", value="banaras-hindu-university-b-a-honours")
        college = st.text_input("College", value="main_campus")
        variant = st.text_input("Variant", value="standard_variant")
        year = st.number_input("Year", min_value=2020, max_value=2030, value=2025, step=1)

    with col2:
        category = st.selectbox("Category", ["UR", "OBC", "SC", "ST", "EWS"])
        seat_type = st.selectbox("Seat Type", ["general", "Paid Fee Course", "All-India open merit", "J&K open merit"])
        basis = st.selectbox("Basis", ["official", "merit-list"])
        round_val = st.selectbox("Round", ["1", "2", "3", "4"])

    # Perform the Advanced Feature Engineering combining college, university, and programme
    uni_prog_variant = f"{university_id}_{programme_id}_{variant}"

    # Prepare the feature dictionary matching training features order:
    input_data = {
        'programme_id': [programme_id],
        'university_id': [university_id],
        'college': [college],
        'variant': [variant],
        'uni_prog_variant': [uni_prog_variant],
        'category': [category],
        'seat_type': [seat_type],
        'basis': [basis],
        'round': [str(round_val)],
        'year': [year]
    }

    input_df = pd.DataFrame(input_data)

    if st.button("Predict Cutoff Value"):
        try:
            prediction = model.predict(input_df)
            st.markdown("---")
            st.metric(label="Predicted Cutoff Figure (Capped limit)", value=f"{prediction[0]:.4f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")