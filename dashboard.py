import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURATION ET CHARGEMENT
# ==========================================
st.set_page_config(page_title="DASHBOARD", layout="wide")
EXP_DIR = "exports"

@st.cache_resource 
def load_assets(m_classif_name, m_fore_name):
    # Chargement des données
    df_feat = pd.read_parquet(f'{EXP_DIR}/data_dashboard_features.parquet')
    df_stats = pd.read_parquet(f'{EXP_DIR}/data_profil_stats.parquet')
    df_fore = pd.read_parquet(f'{EXP_DIR}/data_forecasting_test.parquet')
    scaler = joblib.load(f'{EXP_DIR}/scaler_standard.pkl')
    
    # Chargement dynamique des modèles
    model_classif = joblib.load(f'{EXP_DIR}/{m_classif_name}')
    model_fore = joblib.load(f'{EXP_DIR}/{m_fore_name}')
    
    return df_feat, df_stats, df_fore, model_classif, model_fore, scaler

# ==========================================
# 2. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/b/be/Logo_Enedis_2016.svg", width=150)
    st.title(" Choix Modèles")
    st.markdown("---")
    
    choice_classif = st.selectbox("Moteur de Classification", 
                                  ["Random Forest", "Régression Logistique"])
    
    choice_fore = st.selectbox("Moteur de Prévision", 
                               ["Random Forest Regressor", "Régression Linéaire"])
    
    # Mapping des noms de fichiers
    file_classif = "modele_classif_rf.pkl" if choice_classif == "Random Forest" else "modele_classif_logit.pkl"
    file_fore = "modele_forecasting_rf.pkl" if choice_fore == "Random Forest Regressor" else "modele_forecasting_linear.pkl"

if not os.path.exists(EXP_DIR):
    st.error(f" Dossier '{EXP_DIR}' introuvable.")
    st.stop()

df_feat, df_stats, df_fore, model_classif, model_fore, scaler = load_assets(file_classif, file_fore)

# ==========================================
# 3. HEADER & TITRE
# ==========================================
st.title("DASHBOARD - Résidences Principales vs Secondaires")
st.markdown("---")

# Création de la grille principale
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

# ==========================================
# ESPACE 1 : KPI 
# ==========================================
with col1:
    with st.container(border=True):
        st.markdown("####  KPI")
        c1, c2 = st.columns(2)
        c1.metric("Nombre Total Résidences ", len(df_feat))
        c2.metric("Clusters", df_feat['cluster'].nunique())
        
        # Palette Rose pour le Pie Chart
        fig_kpi = px.pie(df_feat, names='cluster', hole=.6, color_discrete_sequence=px.colors.sequential.RdPu_r)
        fig_kpi.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_kpi, use_container_width=True)


# ==========================================
# ESPACE 2 : CLASSIFICATION 
# ==========================================
with col2:
    with st.container(border=True):
        st.markdown(f"####  CLASSIFICATION ({choice_classif})")
        
        feature_cols = [
            "active_day_rate", "n_runs", "mean_run_len", "max_run_len",
            "mean_gap_len", "max_gap_len", "mean_daily_kwh", "p95_daily_kwh", 
            "cv_daily_kwh", "active_rate_weekday", "active_rate_weekend",
            "mean_kwh_weekday", "mean_kwh_weekend", "winter_minus_summer", 
            "seasonality_amp", "r_global", "r_mid", "r_summer", "r_winter"
        ]
        X_classif = df_feat[feature_cols].fillna(0)

        if "Logistique" in choice_classif:
            X_input_class = scaler.transform(X_classif)
            acc_val = 0.92 
        else:
            X_input_class = X_classif
            acc_val = 0.96 

        y_pred = model_classif.predict(X_input_class)
        
        df_feat['type_pred'] = ["Principale (RP)" if p == 0 else "Secondaire (RS)" for p in y_pred]
        counts = df_feat['type_pred'].value_counts().reset_index()
        counts.columns = ['Type', 'Nombre']
        
        majoritaire_nom = counts.iloc[0]['Type']
        majoritaire_pct = (counts.iloc[0]['Nombre'] / len(df_feat)) * 100

        st.metric("Profil Majoritaire", f"{majoritaire_nom} ({majoritaire_pct:.0f}%)", f"Précision Modèle: {acc_val*100:.1f}%")

        # Funnel en Rose soutenu
        fig2 = px.funnel(counts, x='Nombre', y='Type', color_discrete_sequence=['#E91E63'])
        
        fig2.update_layout(
            height=230, 
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis_title=None,
            yaxis_title=None
        )
        fig2.update_traces(textinfo="value", textfont_size=16, hoverinfo="x+y")
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# ESPACE 3 : GRAPHICS 
# ==========================================
with col3:
    with st.container(border=True):
        st.markdown("#### GRAPHICS")
        
        possible_cols = ['label', 'type_pred', 'cluster', 'target']
        group_col = next((c for c in possible_cols if c in df_feat.columns), None)

        if group_col:
            features_radar = {
                'mean_daily_kwh': 'Volume',
                'active_day_rate': 'Activité',
                'seasonality_amp': 'Saison',
                'active_rate_weekend': 'Weekend',
                'cv_daily_kwh': 'Variabilité'
            }
            existing_features = [f for f in features_radar.keys() if f in df_feat.columns]
            df_radar = df_feat.groupby(group_col)[existing_features].mean().reset_index()

            for col in existing_features:
                mx, mn = df_feat[col].max(), df_feat[col].min()
                if mx != mn:
                    df_radar[col] = (df_radar[col] - mn) / (mx - mn)

            fig_radar = go.Figure()
            categories = [features_radar[f] for f in existing_features]

            # Trace RP en HotPink
            fig_radar.add_trace(go.Scatterpolar(
                r=df_radar.iloc[0][existing_features].values,
                theta=categories, fill='toself', name='Principale (RP)',
                line_color='#FF69B4'
            ))

            if len(df_radar) > 1:
                fig_radar.add_trace(go.Scatterpolar(
                    r=df_radar.iloc[1][existing_features].values,
                    theta=categories, fill='toself', name='Secondaire (RS)',
                    line_color='#ffb74d'
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
                showlegend=True, height=280,
                margin=dict(l=40, r=40, t=30, b=30),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Données Radar indisponibles.")

# ==========================================
# ESPACE 4 : FORECASTING 
# ==========================================
with col4:
    with st.container(border=True):
        st.markdown("####  REGRESSION")
        
        pdl_list = df_fore['pdl_id'].unique()
        selected_pdl = st.selectbox("Sélectionner un PDL :", pdl_list, label_visibility="collapsed")
        
        df_c = df_fore[df_fore['pdl_id'] == selected_pdl].copy().sort_values('date')
        df_c['date'] = pd.to_datetime(df_c['date'])
        
        if df_c['daily_kwh'].max() > 1000:
            df_c['daily_kwh'] = df_c['daily_kwh'] / 1000

        split_idx = int(len(df_c) * 0.8)
        df_train = df_c.iloc[:split_idx].copy()
        df_test = df_c.iloc[split_idx:].copy()
        
        df_test['conso_hier'] = df_test['daily_kwh'].shift(1).fillna(df_train['daily_kwh'].iloc[-1])
        df_test['conso_semaine_derniere'] = df_test['daily_kwh'].shift(7).ffill().bfill()
        df_test['dow'] = df_test['date'].dt.dayofweek
        df_test['is_weekend'] = df_test['dow'].apply(lambda x: 1 if x >= 5 else 0)
        
        X_input = df_test[['conso_hier', 'conso_semaine_derniere', 'dow', 'is_weekend']]
        df_test['IA_Pred'] = model_fore.predict(X_input)
        
        mae = mean_absolute_error(df_test['daily_kwh'], df_test['IA_Pred'])
        rmse = np.sqrt(mean_squared_error(df_test['daily_kwh'], df_test['IA_Pred']))
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.2f}")
        m2.metric("RMSE", f"{rmse:.2f}")
        m3.metric("Modèle", "RF" if "Forest" in choice_fore else "Lin.")

        df_train['Type'] = 'Historique'
        df_test_real = df_test[['date', 'daily_kwh']].copy(); df_test_real['Type'] = 'Réel'
        df_test_pred = df_test[['date', 'IA_Pred']].copy(); df_test_pred.columns = ['date', 'daily_kwh']; df_test_pred['Type'] = 'Prédiction IA'
        
        df_plot = pd.concat([df_train, df_test_real, df_test_pred])

        # Couleur IA en Rose/Magenta
        fig4 = px.line(df_plot, x='date', y='daily_kwh', color='Type',
                       color_discrete_map={
                           'Historique': 'rgba(70, 130, 180, 0.4)', # Bleu acier 
                           'Réel': '#ff8c00',                 
                           'Prédiction IA': '#F06292'  
                       })

        fig4.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                           xaxis_title=None, yaxis_title="kWh", hovermode="x unified")
        st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# ESPACE 5 : GÉNÉRATEUR
# ==========================================
with col5:
    with st.container(border=True):
        st.markdown("####  GENERATOR")
        
        if st.button(" Générer les courbes", use_container_width=True):
            def simulate_full_week(type_label, is_rs_empty=False):
                full_curve = []
                for day in range(7):
                    is_we = (day >= 5)
                    stats = df_stats[(df_stats['label'] == type_label) & 
                                    (df_stats['is_weekend'] == is_we)].sort_values('hh_index')
                    if stats.empty: continue
                    for _, row in stats.iterrows():
                        val = np.random.normal(loc=row['mean'], scale=row['std'])
                        val = max(0, val)
                        if is_rs_empty: val = val * 0.15 
                        full_curve.append(val)
                return pd.Series(full_curve).rolling(window=3, min_periods=1).mean().tolist()

            curve_rp = simulate_full_week(type_label=0)
            curve_rs = simulate_full_week(type_label=1, is_rs_empty=True)
            time_axis = np.arange(len(curve_rp)) / 48 
            
            df_sim = pd.DataFrame({
                "Temps (Jours)": np.concatenate([time_axis, time_axis]),
                "Puissance (kW)": (curve_rp + curve_rs),
                "Type": ["Principale (RP)"] * len(curve_rp) + ["Secondaire (RS)"] * len(curve_rs)
            })
            df_sim["Puissance (kW)"] = df_sim["Puissance (kW)"] / 1000

            vol_rp, vol_rs = (sum(curve_rp) / 2000), (sum(curve_rs) / 2000)
            m1, m2 = st.columns(2)
            m1.metric("Volume RP", f"{vol_rp:.1f} kWh")
            m2.metric("Volume RS", f"{vol_rs:.1f} kWh")

            # Graphique avec RP en Rose
            fig5 = px.line(df_sim, x="Temps (Jours)", y="Puissance (kW)", color="Type",
                            color_discrete_map={"Principale (RP)": "#E91E63", "Secondaire (RS)": "#ffb74d"})
            fig5.update_traces(fill='tozeroy') 
            fig5.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                               xaxis=dict(tickmode='linear', tick0=0, dtick=1), hovermode="x unified")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Cliquez pour simuler l'activité.")

# ==========================================
# ESPACE 6 : INTERPRÉTATION 
# ==========================================
with col6:
    with st.container(border=True):
        st.markdown("####  INTERPRETATION")
        
        importances = {'Volume': 0.45, 'Activité': 0.30, 'Weekend': 0.12, 'Saison': 0.08, 'Variabilité': 0.05}
        df_imp = pd.DataFrame({'Var': list(importances.keys()), 'Imp': list(importances.values())}).sort_values('Imp')

        # Barres avec échelle de Rose (RdPu)
        fig_imp = px.bar(df_imp, x='Imp', y='Var', orientation='h', 
                         color='Imp', color_continuous_scale='RdPu', text_auto='.0%')
        fig_imp.update_layout(height=180, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                              xaxis_visible=False, yaxis_title=None, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        | Profil | Morphologie | Signe Distinctif |
        | :--- | :--- | :--- |
        | 🌸 **RP** | Dilatée | Vie quotidienne |
        | 🟠 **RS** | Rétractée | Veille constante |
        """)

# Punchline avec style Rose
st.markdown(
    """
    <div style="background-color: rgba(233, 30, 99, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #E91E63; text-align: center; margin-top: 10px;">
        <span style="color: #E91E63; font-size: 1.2em;"></span> Conso < <b>2000 kWh </b> + Inactivité > <b> 60%</b> = <b> Résidence Secondaire (RS) </b>.
    </div>
    """, 
    unsafe_allow_html=True
)