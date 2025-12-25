import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import base64
from pathlib import Path

# ============================================================================
# CONFIGURATION ET DESIGN
# ============================================================================
st.set_page_config(
    page_title="CardioAI",
    layout="wide",
    page_icon="🫀"
)

def apply_custom_design():
    # 1. Gestion de l'image d'arrière-plan locale (image.jpg)
    image_path = Path("image.jpg")
    if image_path.exists():
        with open(image_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
        
        st.markdown(
            f"""
            <style>
            .bg-image {{
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                filter: blur(12px);
                -webkit-filter: blur(12px);
                z-index: -1;
                opacity: 0.25;
            }}
            .stApp {{ background: transparent; }}
            </style>
            <div class="bg-image"></div>
            """,
            unsafe_allow_html=True
        )

    # 2. Chargement du CSS externe
    css_file = Path("style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

apply_custom_design()

# ============================================================================
# BARRE LATÉRALE (SIDEBAR) - AJOUT DU NOM ET TEXTE
# ============================================================================
st.sidebar.markdown("# 🫀 CardioAI")
st.sidebar.markdown("""
**Assistant Intelligent** Analyse des risques cardiovasculaires basée sur l'IA pour un suivi préventif et rapide.
""")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox("Navigation", ["Exploration", "Prédiction et Conseils"])

# ============================================================================
# LOGIQUE DE CHARGEMENT (ORIGINALE)
# ============================================================================
@st.cache_data
def load_data():
    file_path = Path("cardio_train.csv")
    if not file_path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, sep=";")
    except:
        df = pd.read_csv(file_path)
    
    df["imc"] = df["weight"] / ((df["height"] / 100) ** 2)
    if "age" in df.columns and df["age"].mean() > 200:
        df["age"] = (df["age"] / 365).round(0).astype(int)
    
    df = df[(df["imc"] > 10) & (df["imc"] < 60) & (df["ap_hi"] > 50) & (df["ap_hi"] < 250)]
    return df

df = load_data()

# ============================================================================
# PAGE EXPLORATION
# ============================================================================
if page == "Exploration":
    st.title("📊 Exploration des Données")
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patients", f"{df.shape[0]:,}")
        col2.metric("Variables", df.shape[1])
        col3.metric("IMC Moyen", f"{df['imc'].mean():.1f}")
        col4.metric("Risque Moyen", f"{(df['cardio'].mean()*100):.1f}%")

        st.markdown("---")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        c1, c2 = st.columns([1, 3])
        with c1:
            selected_col = st.selectbox("Variable", numeric_cols)
            chart = st.radio("Graphique", ["Histogramme", "Boxplot"])
        with c2:
            colors = {0: "#1e293b", 1: "#28a745"}
            if chart == "Histogramme":
                fig = px.histogram(df, x=selected_col, color="cardio", color_discrete_map=colors)
            else:
                fig = px.box(df, y=selected_col, x="cardio", color="cardio", color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE PRÉDICTION
# ============================================================================
else:
    st.title("🧪 Prédiction du Risque")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 👤 Personnel")
            age = st.slider("Âge (années)", 18, 100, 50)
            gender = st.selectbox("Sexe", ["Homme", "Femme"])
            height = st.slider("Taille (cm)", 140, 210, 170)
            weight = st.slider("Poids (kg)", 40, 150, 70)
        with c2:
            st.markdown("### 🩺 Médical")
            systolic = st.slider("Pression systolique", 80, 250, 120)
            diastolic = st.slider("Pression diastolique", 50, 150, 80)
            cholesterol = st.select_slider("Cholestérol", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Élevé", 3:"Très élevé"}[x])
            gluc = st.select_slider("Glucose", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Élevé", 3:"Très élevé"}[x])
        
        st.markdown("### 🚬 Mode de vie")
        l1, l2, l3 = st.columns(3)
        smoke = l1.checkbox("Fumeur")
        alco = l2.checkbox("Alcool")
        active = l3.checkbox("Actif")
        
        submitted = st.form_submit_button("🚀 Lancer l'analyse", use_container_width=True)

    if submitted:
        payload = {
            "age": age * 365, "gender": 1 if gender == "Femme" else 2,
            "height": height, "weight": weight, "ap_hi": systolic, "ap_lo": diastolic,
            "cholesterol": cholesterol, "gluc": gluc, "smoke": int(smoke), "alco": int(alco), "active": int(active)
        }
        try:
            res = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=5)
            if res.status_code == 200:
                data = res.json()
                st.markdown("---")
                if data["prediction"] == 1:
                    st.error(f"⚠️ Risque élevé ({data['probability']*100:.1f}%)")
                else:
                    st.success(f"✅ Risque faible ({data['probability']*100:.1f}%)")
            else: st.error("Erreur API")

        except: st.error("L'API ne répond pas. Lancez 'uvicorn api:app'.")

