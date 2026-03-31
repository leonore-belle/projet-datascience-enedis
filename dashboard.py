import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Dashboard Enedis", page_icon="⚡", layout="wide")

# CSS minimaliste pour réduire les marges en haut de la page
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ DASHBOARD ANALYTIQUE - PROJET DATA SCIENCES")
st.markdown("---")

# --- 2. PREMIÈRE LIGNE (3 COLONNES) ---
col1, col2, col3 = st.columns(3)

# ESPACE 1 : KPI
with col1:
    with st.container(border=True): # Crée une belle boîte encadrée
        st.markdown("#### 📊 KPI METRICS")
        
        # Sous-colonnes pour mettre les chiffres côte à côte
        c1, c2 = st.columns(2)
        c1.metric("Clients Suivis", "487", "12 ce mois")
        c2.metric("Conso Totale", "5.2 TWh", "-0.1 TWh", delta_color="inverse")
        
        # Petit graphique
        df_abo = pd.DataFrame({'Abo': ['Vert', 'Standard'], 'Val': [102, 385]})
        fig1 = px.pie(df_abo, values='Val', names='Abo', hole=.6, color_discrete_sequence=['#4db6ac', '#80cbc4'])
        # On fixe la hauteur et les marges pour aligner parfaitement les boîtes
        fig1.update_layout(height=230, margin=dict(l=0, r=0, t=30, b=0), title_text="Répartition Abonnements")
        st.plotly_chart(fig1, use_container_width=True)

# ESPACE 2 : CLASSIFICATION
with col2:
    with st.container(border=True):
        st.markdown("#### 🏷️ CLASSIFICATION")
        st.metric("Profil Majoritaire", "Résidentiel (82%)", "Modèle Précision: 94%")
        
        df_fun = pd.DataFrame({'Type': ['Résidentiel', 'Commercial', 'Indus.'], 'Val': [82, 12, 6]})
        fig2 = px.funnel(df_fun, x='Val', y='Type', color_discrete_sequence=['#26a69a'])
        fig2.update_layout(height=230, margin=dict(l=0, r=0, t=30, b=0), title_text="Volume par type")
        st.plotly_chart(fig2, use_container_width=True)

# ESPACE 3 : GRAPHIC
with col3:
    with st.container(border=True):
        st.markdown("#### 📈 GRAPHIC (TEMP.)")
        st.metric("Tendance Actuelle", "À la hausse", "Hiver approchant", delta_color="off")
        
        dates = pd.date_range(start="2024-01-01", periods=6, freq='ME')
        df_line = pd.DataFrame({'Mois': dates, 'Conso': [450, 420, 390, 380, 510, 600]})
        fig3 = px.line(df_line, x='Mois', y='Conso', markers=True, color_discrete_sequence=['#00897b'])
        fig3.update_layout(height=230, margin=dict(l=0, r=0, t=30, b=0), title_text="Évolution Mensuelle (MWh)")
        st.plotly_chart(fig3, use_container_width=True)


# --- 3. DEUXIÈME LIGNE (3 COLONNES) ---
col4, col5, col6 = st.columns(3)

# ESPACE 4 : REGRESSION
with col4:
    with st.container(border=True):
        st.markdown("#### 🔮 REGRESSION")
        st.metric("Prévision J+7", "461 MWh", "-2% vs semaine passée")
        
        df_bar = pd.DataFrame({'Région': ['Nord', 'Sud', 'Est', 'Ouest'], 'Prévue': [120, 95, 140, 106]})
        fig4 = px.bar(df_bar, x='Région', y='Prévue', color='Région', color_discrete_sequence=['#4db6ac', '#26a69a', '#00897b', '#00695c'])
        fig4.update_layout(height=230, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title_text="Prévision par Région")
        st.plotly_chart(fig4, use_container_width=True)

# ESPACE 5 : GENERATOR
with col5:
    with st.container(border=True):
        st.markdown("#### ⚙️ GENERATOR")
        st.metric("Données Simulées", "1.2M Lignes", "Génération terminée")
        
        df_gen = pd.DataFrame({'Profil': ['Type A', 'Type B', 'Type C'], 'Points': [500, 300, 200]})
        fig5 = px.bar(df_gen, x='Points', y='Profil', orientation='h', color_discrete_sequence=['#4db6ac'])
        fig5.update_layout(height=230, margin=dict(l=0, r=0, t=30, b=0), title_text="Distribution Synthétique")
        st.plotly_chart(fig5, use_container_width=True)

# ESPACE 6 : INTERPRETATION
with col6:
    with st.container(border=True):
        st.markdown("#### 🧠 INTERPRETATION")
        st.metric("Variable Critique", "Température", "Impact direct fort")
        
        df_shap = pd.DataFrame({'Feature': ['Température', 'Heure', 'Jour', 'Vent'], 'Impact': [45, 25, 15, 5]})
        fig6 = px.bar(df_shap, x='Impact', y='Feature', orientation='h', color_discrete_sequence=['#00897b'])
        fig6.update_layout(height=230, margin=dict(l=0, r=0, t=30, b=0), title_text="Importance Variables (SHAP)")
        fig6.update_yaxes(autorange="reversed") # Met la variable la plus importante en haut
        st.plotly_chart(fig6, use_container_width=True)