import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import joblib
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
    # Utilisation de votre image 'cardi.jpeg'
    image_path = Path("cardi.jpeg")
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
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                filter: blur(12px);
                -webkit-filter: blur(12px);
                z-index: -1;
                opacity: 0.3;
            }}
            .stApp {{ background: transparent; }}
            </style>
            <div class="bg-image"></div>
            """,
            unsafe_allow_html=True
        )

    css_file = Path("style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

apply_custom_design()

# ============================================================================
# CHARGEMENT DES RESSOURCES
# ============================================================================
@st.cache_resource
def load_prediction_model():
    model_path = Path("random_forest_model.pkl")
    return joblib.load(model_path) if model_path.exists() else None

model = load_prediction_model()

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
    return df

df = load_data()

# ============================================================================
# BARRE LATÉRALE - PRÉDICTION PAR DÉFAUT
# ============================================================================
st.sidebar.markdown("# 🫀 CardioAI")
st.sidebar.markdown("**Assistant Santé Intelligent**")
st.sidebar.markdown("---")

# index=0 place "Prédiction et Conseils" en premier (par défaut)
page = st.sidebar.selectbox(
    "Navigation", 
    ["Prédiction et Conseils", "Exploration et Corrélations"],
    index=0 
)

# ============================================================================
# PAGE PRÉDICTION (PAR DÉFAUT)
# ============================================================================
if page == "Prédiction et Conseils":
    st.title("🧪 Analyse du Risque Cardiovasculaire")
    
    if model is None:
        st.error("❌ Modèle 'random_forest_model.pkl' introuvable.")
        st.stop()

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 👤 Profil")
            age = st.slider("Âge (années)", 18, 100, 50)
            gender = st.selectbox("Sexe", ["Homme", "Femme"])
            height = st.slider("Taille (cm)", 140, 210, 170)
            weight = st.slider("Poids (kg)", 40, 150, 70)
        with c2:
            st.markdown("### 🩺 Constantes")
            systolic = st.slider("Pression systolique", 80, 250, 120)
            diastolic = st.slider("Pression diastolique", 50, 150, 80)
            cholesterol = st.select_slider("Cholestérol", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Élevé", 3:"Très élevé"}[x])
            gluc = st.select_slider("Glucose", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Élevé", 3:"Très élevé"}[x])
        
        st.markdown("### 🚬 Mode de vie")
        l1, l2, l3 = st.columns(3)
        smoke = l1.checkbox("Fumeur")
        alco = l2.checkbox("Alcool")
        active = l3.checkbox("Activité Physique")
        
        submitted = st.form_submit_button("🚀 Lancer le diagnostic", use_container_width=True)

    if submitted:
        features = np.array([[age*365.25, 1 if gender=="Femme" else 2, height, weight, systolic, diastolic, cholesterol, gluc, int(smoke), int(alco), int(active)]])
        prob = model.predict_proba(features)[0][1]
        
        st.markdown("---")
        if prob > 0.5:
            st.error(f"⚠️ **Risque élevé** : {prob*100:.1f}%")
        else:
            st.success(f"✅ **Risque faible** : {prob*100:.1f}%")

# ============================================================================
# PAGE EXPLORATION ET CORRÉLATIONS
# ============================================================================
else:
    st.title("📊 Analyses et Corrélations")
    
    if not df.empty:
        st.subheader("🔗 Corrélation avec la variable cible (Cardio)")
        
        # Calcul de la corrélation
        corr_matrix = df.corr()[['cardio']].sort_values(by='cardio', ascending=False)
        
        fig_corr = px.bar(
            corr_matrix, 
            x=corr_matrix.index, 
            y='cardio',
            title="Impact des variables sur le risque cardiaque",
            labels={'index': 'Variables', 'cardio': 'Coefficient de Corrélation'},
            color='cardio',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.info("💡 Plus la barre est haute, plus la variable a une influence directe sur le risque cardiovasculaire.")
        
        st.markdown("---")
        st.subheader("📈 Distributions interactives")
        col_sel, col_graph = st.columns([1, 3])
        with col_sel:
            var = st.selectbox("Choisir une variable", df.columns)
        with col_graph:
            fig_hist = px.histogram(df, x=var, color="cardio", barmode="overlay", color_discrete_map={0:"#1e293b", 1:"#28a745"})
            st.plotly_chart(fig_hist, use_container_width=True)
