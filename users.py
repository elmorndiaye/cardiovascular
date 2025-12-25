import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import joblib
from pathlib import Path

# ============================================================================
# 1. CONFIGURATION ET DESIGN (CSS & BACKGROUND)
# ============================================================================
st.set_page_config(
    page_title="CardioAI",
    layout="wide",
    page_icon="🫀"
)

def apply_custom_design():
    # Gestion de l'image d'arrière-plan 'cardi.jpeg'
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
                filter: blur(15px);
                -webkit-filter: blur(15px);
                z-index: -1;
                opacity: 0.3;
            }}
            .stApp {{ background: transparent; }}
            </style>
            <div class="bg-image"></div>
            """,
            unsafe_allow_html=True
        )

    # Chargement du fichier style.css pour les cartes blanches
    css_file = Path("style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

apply_custom_design()

# ============================================================================
# 2. CHARGEMENT DU MODÈLE ET DES DONNÉES
# ============================================================================
@st.cache_resource
def load_prediction_model():
    model_path = Path("random_forest_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None

model = load_prediction_model()

@st.cache_data
def load_data():
    file_path = Path("cardio_train.csv")
    if not file_path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, sep=";")
    except:
        df = pd.read_csv(file_path)
    
    # Nettoyage et création de variables
    df["imc"] = df["weight"] / ((df["height"] / 100) ** 2)
    if "age" in df.columns and df["age"].mean() > 200:
        df["age"] = (df["age"] / 365.25).round(0).astype(int)
    
    return df

df = load_data()

# ============================================================================
# 3. BARRE LATÉRALE (SIDEBAR)
# ============================================================================
st.sidebar.markdown("# 🫀 CardioAI")
st.sidebar.markdown("""
**Assistant Santé Intelligent**
Analyse prédictive des risques cardiaques.
""")
st.sidebar.markdown("---")

# Navigation : La prédiction est la page par défaut (index 0)
page = st.sidebar.selectbox(
    "Menu de l'application", 
    ["Prédiction et Diagnostic", "Exploration des données"],
    index=0
)

# ============================================================================
# 4. PAGE : PRÉDICTION ET DIAGNOSTIC (PAR DÉFAUT)
# ============================================================================
if page == "Prédiction et Diagnostic":
    st.title("🧪 Analyse du Risque Personnalisée")
    
    if model is None:
        st.error("⚠️ Erreur : Le fichier 'random_forest_model.pkl' est introuvable sur le serveur.")
        st.stop()

    with st.form("main_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 👤 Profil Physique")
            age_ans = st.slider("Âge (années)", 18, 100, 50)
            gender = st.selectbox("Sexe", ["Homme", "Femme"])
            h = st.slider("Taille (cm)", 140, 210, 170)
            w = st.slider("Poids (kg)", 40, 150, 75)
        
        with col2:
            st.markdown("### 🩺 Paramètres Médicaux")
            sys = st.slider("Pression Systolique (Max)", 80, 220, 120)
            dia = st.slider("Pression Diastolique (Min)", 40, 120, 80)
            chol = st.select_slider("Cholestérol", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Élevé", 3:"Très élevé"}[x])
            gluc = st.select_slider("Glucose", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Élevé", 3:"Très élevé"}[x])
        
        st.markdown("### 🏃 Mode de Vie")
        c3a, c3b, c3c = st.columns(3)
        smk = c3a.checkbox("Fumeur")
        alc = c3b.checkbox("Consommation d'alcool")
        act = c3c.checkbox("Activité physique régulière")
        
        btn = st.form_submit_button("Lancer l'Analyse IA", use_container_width=True)

    if btn:
        # Préparation des données pour le modèle
        # Format attendu : age(jours), gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
        input_data = np.array([[
            age_ans * 365.25, 
            1 if gender == "Femme" else 2, 
            h, w, sys, dia, chol, gluc, 
            int(smk), int(alc), int(act)
        ]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        st.subheader("📊 Résultats de l'analyse")

        # Affichage des visuels après prédiction
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            # Graphique de Jauge (Donut)
            fig_gauge = px.pie(
                values=[probability, 1-probability], 
                names=["Risque", "Sain"],
                hole=0.7,
                color_discrete_sequence=["#e74c3c" if probability > 0.5 else "#27ae60", "#f8f9fa"]
            )
            fig_gauge.update_traces(textinfo='none')
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if prediction == 1:
                st.error(f"**Risque élevé : {probability*100:.1f}%**")
            else:
                st.success(f"**Risque faible : {probability*100:.1f}%**")

        with res_col2:
            st.markdown("#### Comparaison de vos constantes")
            # Petit graphique de comparaison
            comp_df = pd.DataFrame({
                "Paramètres": ["Pression Systolique", "Pression Diastolique"],
                "Vos valeurs": [sys, dia],
                "Cible idéale": [120, 80]
            }).set_index("Paramètres")
            st.bar_chart(comp_df)
            
            if probability < 0.5:
                st.balloons()

# ============================================================================
# 5. PAGE : EXPLORATION ET CORRÉLATIONS
# ============================================================================
else:
    st.title("📊 Analyse Globale des Données")
    
    if not df.empty:
        st.subheader("🔗 Corrélation avec le risque cardiaque")
        # Calcul des corrélations numériques uniquement
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()['cardio'].sort_values(ascending=False).drop('cardio')
        
        fig_corr = px.bar(
            x=correlations.index, 
            y=correlations.values,
            labels={'x': 'Variables', 'y': 'Force du lien'},
            title="Quels facteurs influencent le plus le risque ?",
            color=correlations.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        

        st.markdown("---")
        st.subheader("🕵️ Exploration individuelle")
        sel_var = st.selectbox("Sélectionner une variable pour voir sa distribution :", correlations.index)
        fig_dist = px.histogram(df, x=sel_var, color="cardio", barmode="overlay", color_discrete_map={0:"#1e293b", 1:"#27ae60"})
        st.plotly_chart(fig_dist, use_container_width=True)
