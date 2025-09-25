import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Decision Tree Classifier", page_icon="🌳", layout="centered")
st.title("Decision Tree")
st.write("Target kolom 'A' (UMUR_TAHUN).")

@st.cache_resource
def load_assets():
    clf = joblib.load("model.pkl")
    # Tentukan fitur sama seperti training
    df = pd.read_excel("BlaBla.xlsx")
    feature_cols = [c for c in df.columns if c != "A"]
    if "UMUR_TAHUN" in feature_cols:
        df["UMUR_TAHUN"] = pd.to_numeric(df["UMUR_TAHUN"], errors="coerce")
    feature_cols = [c for c in feature_cols if df[c].nunique() > 1]
    return clf, feature_cols

clf, feature_cols = load_assets()

st.subheader("Input Fitur")
inputs = {}
for col in feature_cols:
    if col == "UMUR_TAHUN":
        inputs[col] = st.number_input(col, min_value=0, max_value=120, value=25, step=1)
    else:
        # asumsikan fitur biner/int (0/1)
        inputs[col] = st.number_input(col, min_value=0, max_value=1, value=0, step=1)

if st.button("Prediksi"):
    X = pd.DataFrame([inputs], columns=feature_cols)
    # Imputasi sederhana seperti di training
    X = X.fillna(X.median(numeric_only=True))
    y_pred = clf.predict(X)[0]
    st.success(f"Prediksi kelas: **{y_pred}**")

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        proba_df = pd.DataFrame({"Probabilitas": proba}, index=[str(c) for c in clf.classes_])
        st.bar_chart(proba_df)

st.caption("Model: DecisionTreeClassifier (scikit-learn)")
