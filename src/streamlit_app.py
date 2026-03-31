import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Dashboard Enedis - Analyse Consommation", page_icon="⚡", layout="wide")

# CSS pour réduire les marges
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ DASHBOARD ANALYTIQUE - PROJET DATA SCIENCES (ENEDIS)")
st.markdown("---")

# --- 2. CHARGEMENT DES DONNÉES (simulées ou réelles) ---
# Pour l'exemple, on va simuler des données basées sur le notebook
# Dans un vrai projet, charger les données depuis un fichier CSV
@st.cache_data
def load_data():
    # Simulation de données basées sur le notebook
    np.random.seed(42)
    n_clients = 500
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="30T")
    data = []
    for client_id in range(n_clients):
        for date in dates:
            # Simuler une consommation réaliste (plus élevée le matin/soir, plus basse la nuit)
            hour = date.hour + date.minute / 60
            base_conso = 0.5 + np.random.normal(0, 0.2)
            if 6 <= hour < 10:  # Pic matin
                base_conso += 1.5
            elif 17 <= hour < 22:  # Pic soir
                base_conso += 2.0
            elif 0 <= hour < 6:  # Nuit
                base_conso *= 0.3
            data.append({
                "pdl_id": f"PDL_{client_id:03d}",
                "datetime": date,
                "p_kw": max(0, base_conso),
                "is_weekend": date.dayofweek >= 5,
            })
    df = pd.DataFrame(data)
    df["date"] = df["datetime"].dt.date
    df["dow"] = df["datetime"].dt.dayofweek
    df["hh_index"] = ((df["datetime"].dt.hour * 60) + df["datetime"].dt.minute) // 30
    return df

df = load_data()

# Simulation des labels RP/RS (basé sur le notebook)
# On suppose que 80% sont des RP et 20% des RS
np.random.seed(42)
df["label"] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

# --- 3. PREMIÈRE LIGNE (3 COLONNES) ---
col1, col2, col3 = st.columns(3)

# ESPACE 1 : KPI
with col1:
    with st.container(border=True):
        st.markdown("#### 📊 KPI GLOBAUX")
        n_clients = df["pdl_id"].nunique()
        total_conso = df["p_kw"].sum() / 1000  # en MWh
        c1, c2 = st.columns(2)
        c1.metric("Clients Analysés", n_clients)
        c2.metric("Consommation Totale", f"{total_conso:.1f} MWh")

        # Répartition RP/RS
        label_counts = df.groupby("label")["pdl_id"].nunique()
        df_abo = pd.DataFrame({
            "Type": ["Résidence Principale", "Résidence Secondaire"],
            "Nombre": label_counts.values
        })
        fig1 = px.pie(df_abo, values="Nombre", names="Type", hole=0.6, color_discrete_sequence=["#4db6ac", "#ff9800"])
        fig1.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), title_text="Répartition RP/RS")
        st.plotly_chart(fig1, use_container_width=True)

# ESPACE 2 : CLASSIFICATION
with col2:
    with st.container(border=True):
        st.markdown("#### 🏷️ CLASSIFICATION RP/RS")
        # Simulation des métriques de classification (basé sur le notebook)
        st.metric("Précision Modèle", "92%", "F1-Score: 0.89")

        # Répartition des clusters (exemple)
        cluster_counts = pd.DataFrame({
            "Cluster": [f"Cluster {i}" for i in range(5)],
            "Nombre": np.random.randint(50, 150, 5)
        })
        fig2 = px.bar(cluster_counts, x="Cluster", y="Nombre", color="Cluster", color_discrete_sequence=px.colors.qualitative.Set3)
        fig2.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), title_text="Clients par Cluster")
        st.plotly_chart(fig2, use_container_width=True)

# ESPACE 3 : TRENDS
with col3:
    with st.container(border=True):
        st.markdown("#### 📈 TENDANCES SAISONNIÈRES")
        st.metric("Tendance Hivernale", "+35%", "Chauffage électrique")

        # Simulation de données mensuelles
        months = pd.date_range(start="2023-01-01", periods=12, freq="MS").month_name()
        conso_mensuelle = [300 + 100 * np.sin(i * np.pi / 6) for i in range(12)]  # Variation saisonnière
        df_line = pd.DataFrame({"Mois": months, "Conso (MWh)": conso_mensuelle})
        fig3 = px.line(df_line, x="Mois", y="Conso (MWh)", markers=True, color_discrete_sequence=["#00897b"])
        fig3.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), title_text="Consommation Mensuelle Moyenne")
        st.plotly_chart(fig3, use_container_width=True)

# --- 4. DEUXIÈME LIGNE : FORECASTING & GÉNÉRATION ---
col4, col5, col6 = st.columns(3)

# ESPACE 4 : FORECASTING (Random Forest)
with col4:
    with st.container(border=True):
        st.markdown("#### 🔮 FORECASTING (RANDOM FOREST)")
        # Exemple de prédiction pour un client aléatoire
        client_id = np.random.choice(df["pdl_id"].unique())
        st.metric("Client", client_id, "Prévision J+7")

        # Simulation de données de forecasting
        dates = pd.date_range(start="2023-12-01", periods=7)
        reelle = [10 + np.random.normal(0, 1) for _ in range(7)]
        predite = [9.5 + np.random.normal(0, 0.5) for _ in range(7)]
        df_forecast = pd.DataFrame({"Date": dates, "Réelle": reelle, "Prédite": predite})

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Réelle"], name="Réelle", line=dict(color="#00897b")))
        fig4.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Prédite"], name="Prédite", line=dict(color="#ff9800", dash="dot")))
        fig4.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), title_text="Consommation Prédite vs Réelle")
        st.plotly_chart(fig4, use_container_width=True)

# ESPACE 5 : GÉNÉRATEUR DE COURBES
with col5:
    with st.container(border=True):
        st.markdown("#### ⚙️ GÉNÉRATEUR DE PROFILS")
        st.metric("Profils Générés", "500", "RP: 400, RS: 100")

        # Exemple de courbe générée pour une RP et une RS
        hours = list(range(48))  # 48 demi-heures
        rp_curve = [0.5 + 1.5 * np.sin(h * np.pi / 24) for h in hours]  # Courbe lissée pour RP
        rs_curve = [0.1 + 0.3 * np.sin(h * np.pi / 24) for h in hours]  # Courbe plus plate pour RS

        df_gen = pd.DataFrame({
            "Heure": [f"{h//2:02d}:{30*(h%2):02d}" for h in hours],
            "RP (kW)": rp_curve,
            "RS (kW)": rs_curve
        })

        fig5 = px.line(df_gen, x="Heure", y=["RP (kW)", "RS (kW)"], color_discrete_sequence=["#26a69a", "#ff9800"])
        fig5.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), title_text="Profil Type RP vs RS")
        st.plotly_chart(fig5, use_container_width=True)

# ESPACE 6 : INTERPRÉTATION DES CLUSTERS
with col6:
    with st.container(border=True):
        st.markdown("#### 🧠 INTERPRÉTATION CLUSTERS")
        st.metric("Cluster Critique", "Cluster 3", "RS Estivales")

        # Exemple de features discriminantes
        df_features = pd.DataFrame({
            "Feature": ["active_day_rate", "max_gap_len", "r_summer", "winter_minus_summer"],
            "Importance": [0.45, 0.30, 0.15, 0.10]
        })
        fig6 = px.bar(df_features, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#00897b"])
        fig6.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), title_text="Features Discriminantes (SHAP)")
        st.plotly_chart(fig6, use_container_width=True)

# --- 5. TROISIÈME LIGNE : DÉTAILS CLIENT ---
st.markdown("---")
st.header("🔍 DÉTAILS PAR CLIENT")

# Sélection d'un client
client_list = df["pdl_id"].unique()
selected_client = st.selectbox("Sélectionnez un client", client_list, index=0)

# Affichage des données du client
client_data = df[df["pdl_id"] == selected_client]
st.subheader(f"Profil de Consommation : {selected_client}")
col7, col8 = st.columns(2)

with col7:
    st.markdown("#### 📅 Consommation Journalière (Dernière Semaine)")
    last_week = client_data.tail(336)  # 7 jours * 48 demi-heures
    fig7 = px.line(last_week, x="datetime", y="p_kw", title="Consommation par Demi-Heure")
    fig7.update_layout(height=300)
    st.plotly_chart(fig7, use_container_width=True)

with col8:
    st.markdown("#### 📊 Statistiques")
    daily_conso = client_data.groupby("date")["p_kw"].sum().reset_index()
    daily_conso["date"] = pd.to_datetime(daily_conso["date"])
    fig8 = px.bar(daily_conso, x="date", y="p_kw", title="Consommation Quotidienne (kWh)")
    fig8.update_layout(height=300)
    st.plotly_chart(fig8, use_container_width=True)

    # Métriques clés
    mean_conso = daily_conso["p_kw"].mean()
    max_conso = daily_conso["p_kw"].max()
    st.metric("Moyenne Quotidienne", f"{mean_conso:.2f} kWh")
    st.metric("Pic de Consommation", f"{max_conso:.2f} kWh", f"le {daily_conso[daily_conso['p_kw'] == max_conso]['date'].values[0]}")

# --- 6. FOOTER ---
st.markdown("---")
st.markdown("""
    **Dashboard réalisé avec Streamlit** | Données simulées basées sur le notebook `correction_td1.ipynb`.
    Pour plus d'informations, contactez l'équipe Data Science.
""")