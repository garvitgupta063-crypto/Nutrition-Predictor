"""Streamlit interface for the supplied nutrition prediction model."""

from __future__ import annotations

import streamlit as st

from predictor import FEATURES, NutritionPredictor


st.set_page_config(page_title="Nutrition prediction", page_icon="🥗")
st.title("Nutrition prediction")
st.caption("Enter the nutritional values below to generate a prediction.")


@st.cache_resource
def load_predictor() -> NutritionPredictor:
    return NutritionPredictor()


try:
    predictor = load_predictor()
except Exception as error:
    st.error(f"Could not load the supplied model files: {error}")
    st.stop()

with st.form("prediction_form"):
    values = {}
    for feature in FEATURES:
        label = feature.replace("_", " ").capitalize()
        values[feature] = st.number_input(label, min_value=0.0, value=0.0, step=0.1)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        prediction = predictor.predict_one(values)
        st.success(f"Prediction: {prediction:,.2f}")
    except ValueError as error:
        st.error(str(error))
