"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   RETAIL SALES FORECASTING — Professional BI + AI Dashboard                 ║
║   Gradient Boosting Regressor  ·  R² = 0.9991  ·  MAPE = 2.69%             ║
║   PFE Data Science — 2025/2026                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils.helpers import (
    load_data, load_uploaded_file, load_model,
    predict_sales, compute_metrics,
    generate_forecast_for_year, aggregate_forecast,
    generate_2026_forecast, generate_ai_insights,
    MONTH_NAMES, FEATURES
)

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailIQ",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#060d1a;}
.main{background:#060d1a;}
.block-container{padding:1.5rem 2rem 3rem;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:#0d1b2e;}
::-webkit-scrollbar-thumb{background:#1e4d8c;border-radius:3px;}

[data-testid="stSidebar"]{background:linear-gradient(180deg,#020810 0%,#071428 40%,#0a1e3d 100%);border-right:1px solid #1a3a6c;}
[data-testid="stSidebar"] *{color:#c9d8f0!important;}
[data-testid="stSidebar"] hr{border-color:#1a3a6c!important;opacity:.5;}
[data-testid="stSidebar"] .stSelectbox>div,[data-testid="stSidebar"] .stMultiSelect>div{background:#0d1b2e!important;border:1px solid #1e4d8c!important;border-radius:8px!important;}

.main-header{background:linear-gradient(135deg,#020c1b 0%,#0d2a5e 40%,#1565c0 100%);padding:2.2rem 2.5rem 2rem;border-radius:18px;margin-bottom:1.8rem;border:1px solid #1e4d8c;position:relative;overflow:hidden;box-shadow:0 8px 40px rgba(21,101,192,.3),inset 0 1px 0 rgba(255,255,255,.05);}
.main-header::before{content:'';position:absolute;top:-60px;right:-60px;width:280px;height:280px;background:radial-gradient(circle,rgba(100,181,246,.08) 0%,transparent 70%);border-radius:50%;}
.main-header h1{color:#fff;font-size:2rem;font-weight:800;margin:0;letter-spacing:-.03em;text-shadow:0 0 40px rgba(100,181,246,.4);}
.main-header .subtitle{color:#64b5f6;font-size:.92rem;margin:.5rem 0 .8rem;}
.badge{display:inline-block;background:rgba(100,181,246,.12);color:#90caf9;font-size:.72rem;font-weight:600;padding:4px 12px;border-radius:20px;border:1px solid rgba(100,181,246,.25);margin-right:8px;margin-top:4px;letter-spacing:.04em;}
.badge.green{background:rgba(102,187,106,.12);color:#a5d6a7;border-color:rgba(102,187,106,.25);}
.badge.orange{background:rgba(255,167,38,.12);color:#ffcc80;border-color:rgba(255,167,38,.25);}

.kpi-card{background:linear-gradient(135deg,#0d1b2e 0%,#0f2240 100%);border-radius:16px;padding:1.4rem 1.5rem;border:1px solid #1a3a6c;position:relative;overflow:hidden;transition:all .25s ease;box-shadow:0 4px 20px rgba(0,0,0,.4);}
.kpi-card:hover{transform:translateY(-4px);border-color:#2979ff;box-shadow:0 8px 30px rgba(41,121,255,.2);}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--accent-color,#1565c0);border-radius:16px 16px 0 0;}
.kpi-glow{position:absolute;top:-20px;right:-20px;width:80px;height:80px;background:radial-gradient(circle,var(--glow-color,rgba(21,101,192,.15)) 0%,transparent 70%);border-radius:50%;}
.kpi-icon-big{font-size:2rem;opacity:.8;margin-bottom:.5rem;display:block;}
.kpi-label{font-size:.7rem;font-weight:600;color:#546e7a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;}
.kpi-value{font-size:1.9rem;font-weight:800;color:#e3f2fd;line-height:1;margin-bottom:.3rem;}
.kpi-sub{font-size:.75rem;color:#546e7a;}

.section-title{font-size:1.1rem;font-weight:700;color:#000000 !important;padding:.6rem 0 .6rem 1rem;border-left:4px solid #42a5f5;margin:2rem 0 1.2rem;background:linear-gradient(90deg,rgba(66,165,245,.15) 0%,transparent 70%);border-radius:0 8px 8px 0;text-shadow:0 0 20px rgba(100,181,246,.5);}

.insight-card{border-radius:14px;padding:1.3rem 1.5rem;margin-bottom:.9rem;border-left:4px solid #1565c0;position:relative;transition:transform .2s;}
.insight-card:hover{transform:translateX(4px);}
.insight-card.danger{background:rgba(183,28,28,.12);border-left-color:#ef5350;}
.insight-card.warning{background:rgba(230,81,0,.12);border-left-color:#ff7043;}
.insight-card.success{background:rgba(27,94,32,.12);border-left-color:#66bb6a;}
.insight-card.info{background:rgba(13,71,161,.15);border-left-color:#42a5f5;}
.insight-title{font-size:.88rem;font-weight:700;color:#5D89A1;margin-bottom:.4rem;}
.insight-text{font-size:.82rem;color:#90a4ae;line-height:1.6;}

.rec-card{background:linear-gradient(135deg,#0a1628 0%,#0d1e38 100%);border-radius:14px;padding:1.4rem;border:1px solid #1a3a6c;margin-bottom:1rem;transition:all .2s;}
.rec-card:hover{border-color:#2979ff;box-shadow:0 4px 20px rgba(41,121,255,.15);}
.rec-priority{font-size:.65rem;font-weight:700;padding:3px 10px;border-radius:10px;text-transform:uppercase;letter-spacing:.06em;}
.priority-haute{background:rgba(239,83,80,.2);color:#ef9a9a;}
.priority-moyenne{background:rgba(255,167,38,.2);color:#ffcc80;}
.priority-basse{background:rgba(102,187,106,.2);color:#a5d6a7;}
.rec-title{font-size:.9rem;font-weight:700;color:#e3f2fd;margin:.5rem 0 .4rem;}
.rec-desc{font-size:.8rem;color:#78909c;line-height:1.6;}

.upload-zone{background:linear-gradient(135deg,#0a1628 0%,#0d1e38 100%);border:2px dashed #1e4d8c;border-radius:16px;padding:2.5rem;text-align:center;margin-bottom:1.5rem;}
.data-stat{background:linear-gradient(135deg,#0a1628,#0d1e38);border-radius:12px;padding:1rem 1.2rem;border:1px solid #1a3a6c;text-align:center;}
.data-stat .val{font-size:1.5rem;font-weight:700;color:#64b5f6;}
.data-stat .lbl{font-size:.72rem;color:#546e7a;text-transform:uppercase;letter-spacing:.06em;margin-top:3px;}

.stTabs [data-baseweb="tab-list"]{background:#0a1628!important;border-radius:12px 12px 0 0!important;gap:4px!important;padding:4px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#546e7a!important;border-radius:8px!important;font-size:.83rem!important;font-weight:600!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1565c0,#1e88e5)!important;color:#fff!important;}
.stTabs [data-baseweb="tab-panel"]{background:#0a1628!important;border:1px solid #1a3a6c!important;border-radius:0 12px 12px 12px!important;padding:1.2rem!important;}

.stDownloadButton>button,.stButton>button{background:linear-gradient(135deg,#1565c0,#1e88e5)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;padding:.55rem 1.6rem!important;transition:all .2s!important;box-shadow:0 4px 15px rgba(21,101,192,.3)!important;}
.stDownloadButton>button:hover,.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(21,101,192,.4)!important;}

.stMultiSelect [data-baseweb="tag"]{
    background-color:#1565c0 !important;
    color:white !important;
    border-radius:8px !important;
    border:none !important;
}

footer,#MainMenu,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY DARK THEME ────────────────────────────────────────────────────────
PTH = dict(
    paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
    font=dict(family="Inter,sans-serif", size=12, color="#90a4ae"),
    margin=dict(l=40,r=20,t=45,b=40),
    xaxis=dict(showgrid=True, gridcolor="#0d2040", zeroline=False,
               tickfont=dict(color="#546e7a"), linecolor="#1a3a6c"),
    yaxis=dict(showgrid=True, gridcolor="#0d2040", zeroline=False,
               tickfont=dict(color="#546e7a"), linecolor="#1a3a6c"),
)
LEG = dict(bgcolor="rgba(10,22,40,.8)", bordercolor="#1a3a6c",
           borderwidth=1, font=dict(color="#90a4ae"),
           orientation="h", y=1.05)

C = dict(blue="#1e88e5", lblue="#42a5f5", green="#66bb6a",
         orange="#ff7043", red="#ef5350", yellow="#ffd54f",
         purple="#ce93d8", cyan="#00bcd4")

# ── SESSION STATE & CACHE ─────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model("model_gbr_sales.pkl","encoders.pkl")

@st.cache_data
def get_default_data():
    return load_data("data/cleaned_data.csv")

if "df" not in st.session_state:
    st.session_state.df = get_default_data()
if "data_source" not in st.session_state:
    st.session_state.data_source = "default"

model, encoders = get_model()
df = st.session_state.df

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 .8rem;">
        <div style="font-size:2.6rem;filter:drop-shadow(0 0 12px rgba(100,181,246,.5));">📊</div>
        <div style="font-size:1.1rem;font-weight:800;color:#e3f2fd;letter-spacing:-.02em;margin-top:4px;">RetailIQ</div>
        <div style="font-size:.72rem;color:#546e7a;margin-top:2px;letter-spacing:.04em;">SALES FORECASTING PLATFORM</div>
    </div><hr/>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:.7rem;color:#546e7a;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", [
        "Accueil & Import",
        "Dashboard Prévisions",
        "Prévisions 2026",
        "Prédiction Manuelle",
        "Insights IA",
        "Évaluation Modèle",
        "Tableau Détaillé",
    ], label_visibility="collapsed")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.7rem;color:#546e7a;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>Filtres</div>", unsafe_allow_html=True)

    sel_regions  = st.multiselect("📍 Région",    sorted(df["Region"].unique()),   default=sorted(df["Region"].unique()))
    sel_segments = st.multiselect("🏢 Segment",   sorted(df["Segment"].unique()),  default=sorted(df["Segment"].unique()))
    sel_cats     = st.multiselect("📦 Catégorie", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.7rem;color:#546e7a;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>Période</div>", unsafe_allow_html=True)
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    date_range = st.date_input("", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.7rem;color:#546e7a;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>Horizon Prévision</div>", unsafe_allow_html=True)
    horizon = st.slider("Mois", 1, 24, 12)

    st.markdown("<hr/>", unsafe_allow_html=True)
    src = "📂 Fichier importé" if st.session_state.data_source == "uploaded" else "Données par défaut"
    st.markdown(f"""<div style="font-size:.72rem;color:#546e7a;text-align:center;line-height:1.8;">
        {src}<br/>{len(df):,} transactions<br/>
        <span style="color:#1e4d8c;">PFE ID · 2025/2026</span></div>""",
        unsafe_allow_html=True)

# ── FILTER ────────────────────────────────────────────────────────────────────
d0 = pd.to_datetime(date_range[0])
d1 = pd.to_datetime(date_range[1]) if len(date_range) > 1 else pd.to_datetime(date_range[0])

df_f = df[
    df["Region"].isin(sel_regions) &
    df["Segment"].isin(sel_segments) &
    df["Category"].isin(sel_cats) &
    (df["Date"] >= d0) & (df["Date"] <= d1)
].copy()

if df_f.empty:
    st.warning("⚠️ Aucune donnée pour les filtres sélectionnés.")
    st.stop()

df_f["_pred"] = predict_sales(df_f, model, encoders)
m = compute_metrics(df_f["Sales"].values, df_f["_pred"].values)

# ── HEADER ────────────────────────────────────────────────────────────────────
TITLES = {
    "Accueil & Import":     ("Accueil & Import",           "Chargez vos données et explorez le dataset"),
    "Dashboard Prévisions": ("Dashboard des Prévisions",    f"Horizon : +{horizon} mois · {len(df_f):,} transactions"),
    "Prévisions 2026":      ("Prévisions 2026",             "Prévisions annuelles complètes — exercice 2026"),
    "Prédiction Manuelle":  ("Prédiction Manuelle",         "Saisissez vos paramètres — le modèle prédit instantanément"),
    "Insights IA":          ("AI Business Insights",        "Analyse automatique & recommandations intelligentes"),
    "Évaluation Modèle":    ("Évaluation du Modèle",        "Performance du Gradient Boosting Regressor"),
    "Tableau Détaillé":     ("Tableau des Prévisions",      "Données détaillées avec export CSV"),
}
title, subtitle = TITLES[page]
st.markdown(f"""
<div class='main-header'>
    <h1>{title}</h1>
    <p class='subtitle'>{subtitle}</p>
    <span class='badge'>🤖 Gradient Boosting</span>
    <span class='badge green'>R² = 0.9991</span>
    <span class='badge orange'>MAPE = {m['MAPE']:.2f}%</span>
    <span class='badge'>{d0.strftime('%b %Y')} → {d1.strftime('%b %Y')}</span>
    <span class='badge'>{len(df_f):,} transactions</span>
</div>""", unsafe_allow_html=True)

# ── GLOBAL KPIs ───────────────────────────────────────────────────────────────
def kpi(ac, gc, icon, lbl, val, sub):
    return f"""<div class='kpi-card' style='--accent-color:{ac};--glow-color:{gc};'>
        <div class='kpi-glow'></div><span class='kpi-icon-big'>{icon}</span>
        <div class='kpi-label'>{lbl}</div><div class='kpi-value'>{val}</div>
        <div class='kpi-sub'>{sub}</div></div>"""

monthly_s = df_f.set_index("Date")["Sales"].resample("MS").sum()
gr = ((monthly_s.iloc[-1]-monthly_s.iloc[-2])/monthly_s.iloc[-2]*100) if len(monthly_s)>=2 else 0
profit_total = df_f["Profit"].sum()
real_total   = df_f["Sales"].sum()
pred_total   = df_f["_pred"].sum()

cols = st.columns(5)
for col, args in zip(cols, [
    ("#1565c0","rgba(21,101,192,.15)","💰","Ventes Réelles",  f"€{real_total:,.0f}", "Total période sélectionnée"),
    ("#1e88e5","rgba(30,136,229,.15)","🎯","Ventes Prédites", f"€{pred_total:,.0f}", f"Précision : {m['Accuracy']:.1f}%"),
    ("#26a69a","rgba(38,166,154,.15)","💎","Profit Total",    f"€{profit_total:,.0f}", f"Marge : {profit_total/real_total*100:.1f}%"),
    ("#66bb6a","rgba(102,187,106,.15)","📉","RMSE / MAE",     f"€{m['RMSE']:.0f}", f"MAE : €{m['MAE']:.0f}"),
    ("#ff7043" if gr<0 else "#66bb6a","rgba(102,187,106,.12)","📈","Croissance MoM", f"{gr:+.1f}%","vs mois précédent"),
]):
    with col:
        st.markdown(kpi(*args), unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL & IMPORT
# ══════════════════════════════════════════════════════════════════════════════
if page == "Accueil & Import":
    t1, t2 = st.tabs(["📂 Import de fichier", "🔎 Aperçu des données"])

    with t1:
        st.markdown("<div class='section-title'>📂 Charger un nouveau fichier</div>", unsafe_allow_html=True)
        st.markdown("""<div class='upload-zone'>
            <div style='font-size:2.5rem;margin-bottom:.8rem;'>📁</div>
            <div style='color:#64b5f6;font-weight:600;font-size:.95rem;'>Glissez-déposez votre fichier ici</div>
            <div style='color:#546e7a;font-size:.8rem;margin-top:4px;'>Formats supportés : CSV · XLSX · XLS</div>
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("", type=["csv","xlsx","xls"], label_visibility="collapsed")

        if uploaded:
            with st.spinner("Chargement et validation..."):
                df_new, msg = load_uploaded_file(uploaded)
            if df_new is None:
                st.error(msg)
            else:
                st.success(msg)
                st.session_state.df = df_new
                st.session_state.data_source = "uploaded"
                c1,c2,c3,c4 = st.columns(4)
                for col,(lbl,val) in zip([c1,c2,c3,c4],[
                    ("Lignes",    f"{len(df_new):,}"),
                    ("Colonnes",  str(df_new.shape[1])),
                    ("Manquants", str(df_new.isnull().sum().sum())),
                    ("Période",   f"{df_new['Date'].min().strftime('%b %Y')} → {df_new['Date'].max().strftime('%b %Y')}"),
                ]):
                    with col:
                        st.markdown(f"<div class='data-stat'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)
                st.markdown("<br/>", unsafe_allow_html=True)
                st.dataframe(df_new.head(20), use_container_width=True, height=350)
                st.info("✅ Les dashboards ont été mis à jour automatiquement.")
        else:
            st.markdown("<div class='section-title'>📋 Colonnes attendues</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Colonne":     ["Order_ID","Region","Segment","Category","Sub_Category","Year","Month","Quantity","Unit_Cost","Unit_Price","Discount","Sales","Profit","Date"],
                "Type":        ["string","string","string","string","string","int","int","int","float","float","float","float","float","date"],
                "Description": ["ID commande","Zone géographique","Segment client","Catégorie produit","Sous-catégorie","Année","Mois 1-12","Quantité","Coût unitaire","Prix unitaire","Remise 0-1","Chiffre d'affaires","Bénéfice","Date YYYY-MM-DD"],
            }), use_container_width=True, hide_index=True)

    with t2:
        st.markdown("<div class='section-title'>🔎 Dataset actuel</div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col,(lbl,val) in zip([c1,c2,c3,c4],[
            ("Lignes",   f"{len(df):,}"),("Colonnes", str(df.shape[1])),
            ("Manquants",str(df.isnull().sum().sum())),
            ("Source","Importé" if st.session_state.data_source=="uploaded" else "Défaut"),
        ]):
            with col:
                st.markdown(f"<div class='data-stat'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=400)
        st.markdown("<div class='section-title'>📊 Statistiques descriptives</div>", unsafe_allow_html=True)
        st.dataframe(df[["Sales","Profit","Quantity","Discount","Unit_Price","Unit_Cost"]].describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD PRÉVISIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard Prévisions":
    monthly = df_f.groupby("Date").agg(Real=("Sales","sum"), Predicted=("_pred","sum")).reset_index().sort_values("Date")
    rmse_g = m["RMSE"]
    ci_lo = monthly["Predicted"] - 1.96*rmse_g*np.sqrt(len(monthly))
    ci_hi = monthly["Predicted"] + 1.96*rmse_g*np.sqrt(len(monthly))

    t1,t2,t3,t4 = st.tabs(["📈 Réel vs Prédit","🔮 Forecast","📉 Tendance & Saisonnalité","🗓️ Heatmap"])

    with t1:
        fig = go.Figure([
            go.Scatter(x=list(monthly["Date"])+list(monthly["Date"][::-1]),
                       y=list(ci_hi)+list(ci_lo[::-1]),
                       fill="toself",fillcolor="rgba(30,136,229,.07)",
                       line=dict(color="rgba(0,0,0,0)"),name="IC 95%",hoverinfo="skip"),
            go.Scatter(x=monthly["Date"],y=monthly["Real"],mode="lines+markers",name="Réel",
                       line=dict(color="#e3f2fd",width=2.5),marker=dict(size=5,color="#e3f2fd")),
            go.Scatter(x=monthly["Date"],y=monthly["Predicted"],mode="lines+markers",name="Prédit",
                       line=dict(color=C["blue"],width=2.5,dash="dot"),marker=dict(size=5,color=C["blue"])),
        ])
        fig.update_layout(**PTH,title="Ventes Mensuelles — Réel vs Prédit",hovermode="x unified",height=420,
                          legend=LEG)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        last_row  = df_f.sort_values("Date").iloc[-1]
        last_date = df_f["Date"].max()
        future_dates = pd.date_range(start=last_date+pd.DateOffset(months=1),periods=horizon,freq="MS")
        rows = []
        for d in future_dates:
            r = last_row.copy(); r["Date"]=d; r["Year"]=d.year; r["Month"]=d.month
            rows.append(r)
        df_fut   = pd.DataFrame(rows)
        fut_pred = predict_sales(df_fut, model, encoders)
        fut_lo   = fut_pred - 1.96*rmse_g
        fut_hi   = fut_pred + 1.96*rmse_g

        fig2 = go.Figure([
            go.Scatter(x=monthly["Date"],y=monthly["Real"],mode="lines",name="Historique",line=dict(color="#546e7a",width=2)),
            go.Scatter(x=list(future_dates)+list(future_dates[::-1]),y=list(fut_hi)+list(fut_lo[::-1]),
                       fill="toself",fillcolor="rgba(30,136,229,.12)",
                       line=dict(color="rgba(0,0,0,0)"),name="IC 95%",hoverinfo="skip"),
            go.Scatter(x=future_dates,y=fut_pred,mode="lines+markers",name=f"+{horizon}m",
                       line=dict(color=C["blue"],width=3),
                       marker=dict(size=8,color=C["blue"],line=dict(color="white",width=2))),
        ])
        fig2.update_layout(**PTH,title=f"Prévision — {horizon} mois à venir",hovermode="x unified",height=420,
                           legend=LEG)
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        c1,c2 = st.columns(2)
        with c1:
            monthly["Trend"] = monthly["Real"].rolling(3,center=True).mean()
            fig3 = go.Figure([
                go.Bar(x=monthly["Date"],y=monthly["Real"],name="Ventes",
                       marker_color="rgba(30,136,229,.4)",marker_line=dict(color=C["blue"],width=1)),
                go.Scatter(x=monthly["Date"],y=monthly["Trend"],name="Tendance 3M",
                           line=dict(color=C["yellow"],width=3)),
            ])
            fig3.update_layout(**PTH,title="Ventes + Tendance 3M",height=350,
                               legend=LEG)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            monthly["MN"] = monthly["Date"].dt.month
            seas = monthly.groupby("MN")["Real"].mean().reset_index()
            seas["Nm"] = seas["MN"].map(lambda x: MONTH_NAMES[x][:3])
            fig4 = go.Figure(go.Bar(
                x=seas["Nm"],y=seas["Real"],
                marker=dict(color=seas["Real"],colorscale=[[0,"#0d2040"],[1,C["blue"]]],showscale=False),
                text=seas["Real"].round(0).astype(int),textposition="outside",
                textfont=dict(size=9,color="#90a4ae")))
            fig4.update_layout(**PTH,title="Saisonnalité Moyenne",height=350,showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    with t4:
        df_f["YN"] = df_f["Date"].dt.year
        df_f["MN"] = df_f["Date"].dt.month
        pivot = df_f.groupby(["YN","MN"])["Sales"].sum().unstack(fill_value=0)
        pivot.columns = [MONTH_NAMES[c][:3] for c in pivot.columns]
        fig5 = go.Figure(go.Heatmap(
            z=pivot.values,x=pivot.columns.tolist(),y=[str(y) for y in pivot.index],
            colorscale=[[0,"#020c1b"],[.3,"#0d2a5e"],[.6,"#1565c0"],[1,"#64b5f6"]],
            showscale=True,colorbar=dict(tickfont=dict(color="#546e7a"),thickness=15),
            text=[[f"€{v:,.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",textfont=dict(size=10,color="#e3f2fd")))
        fig5.update_layout(**PTH,title="Heatmap Ventes — Année × Mois",height=320)
        st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRÉVISIONS 2026
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prévisions 2026":

    # ── Contrôles utilisateur ──────────────────────────────────────────────
    st.markdown("<div class=\'section-title\'>⚙️ Paramètres de Prévision</div>", unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])

    with ctrl1:
        last_year    = int(df_f["Date"].dt.year.max())
        target_year  = st.selectbox(
            "📅 Année à prédire",
            options=list(range(last_year + 1, last_year + 6)),  # +1 à +5 ans
            index=0,
            help="Choisissez l\'année pour laquelle vous voulez générer les prévisions."
        )

    with ctrl2:
        growth_pct = st.slider(
            "📈 Taux de croissance annuel (%)",
            min_value=-10, max_value=30, value=5, step=1,
            help="Taux appliqué à Quantity et Unit_Price pour simuler l\'évolution future."
        )
        growth_rate = growth_pct / 100.0

    with ctrl3:
        st.markdown(f"""
        <div class=\'insight-card info\' style=\'margin-top:0.5rem;\'>
            <div class=\'insight-title\'>🧠 Comment ça marche ?</div>
            <div class=\'insight-text\'>
                Un <b>Dummy DataFrame</b> est construit pour chaque combinaison
                Mois × Région × Segment × Catégorie avec les features projetées
                (Quantity, Unit_Price × {1+growth_rate:.2f}ˣ).
                Ces données sont ensuite passées au modèle <b>GBR</b> pour prédire les ventes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Génération des prévisions ──────────────────────────────────────────
    with st.spinner(f"Construction du Dummy DataFrame et prévisions {target_year}..."):
        df_forecast  = generate_forecast_for_year(df_f, model, encoders,
                                                   target_year=target_year,
                                                   growth_rate=growth_rate)
        m2026        = aggregate_forecast(df_forecast, rmse=m["RMSE"])

    total_s   = m2026["Sales"].sum()
    total_p   = m2026["Profit"].sum()
    best_m    = m2026.loc[m2026["Sales"].idxmax()]
    weak_m    = m2026.loc[m2026["Sales"].idxmin()]
    prev_year = df_f[df_f["Date"].dt.year == (target_year - 1)]["Sales"].sum()
    if prev_year == 0:
        prev_year = df_f["Sales"].sum() / df_f["Date"].dt.year.nunique()
    yoy = ((total_s - prev_year) / prev_year * 100) if prev_year > 0 else 0

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,args in zip([c1,c2,c3,c4,c5],[
        ("#1565c0","rgba(21,101,192,.15)","💰",f"Ventes {target_year}",  f"€{total_s:,.0f}","Total annuel prévu"),
        ("#26a69a","rgba(38,166,154,.15)","💎",f"Profit {target_year}",  f"€{total_p:,.0f}",f"Marge {total_p/total_s*100:.1f}%"),
        ("#66bb6a","rgba(102,187,106,.15)","📈","Croissance YoY",         f"{yoy:+.1f}%",   f"vs {target_year-1}"),
        ("#ff7043","rgba(255,112,67,.15)","🏆","Meilleur Mois",           best_m["Mois"],   f"€{best_m['Sales']:,.0f}"),
        ("#ef5350","rgba(239,83,80,.12)","📉","Mois le plus faible",      weak_m["Mois"],   f"€{weak_m['Sales']:,.0f}"),
    ]):
        with col:
            st.markdown(kpi(*args), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    t1,t2,t3,t4 = st.tabs(["📊 Ventes & Profits","📈 Comparaison Historique","🔍 Dummy DataFrame","📋 Tableau Mensuel"])

    with t1:
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=m2026["Mois"],y=m2026["Sales"],name="Ventes",
                             marker_color="rgba(30,136,229,.5)",
                             marker_line=dict(color=C["blue"],width=1.5)),secondary_y=False)
        fig.add_trace(go.Scatter(x=m2026["Mois"],y=m2026["Profit"],name="Profit",
                                 mode="lines+markers",line=dict(color=C["green"],width=3),
                                 marker=dict(size=8,color=C["green"],line=dict(color="white",width=2))),secondary_y=True)
        fig.update_layout(**PTH,title=f"Prévisions Mensuelles {target_year} — Ventes & Profits",height=420,legend=LEG)
        fig.update_yaxes(title_text="Ventes (€)",secondary_y=False,gridcolor="#0d2040",tickfont=dict(color="#546e7a"))
        fig.update_yaxes(title_text="Profit (€)",secondary_y=True,gridcolor="rgba(0,0,0,0)",tickfont=dict(color="#546e7a"))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure([
            go.Scatter(x=list(m2026["Mois"])+list(m2026["Mois"][::-1]),
                       y=list(m2026["IC_Haute"])+list(m2026["IC_Basse"][::-1]),
                       fill="toself",fillcolor="rgba(30,136,229,.1)",
                       line=dict(color="rgba(0,0,0,0)"),name="IC 95%",hoverinfo="skip"),
            go.Scatter(x=m2026["Mois"],y=m2026["Sales"],mode="lines+markers+text",
                       text=[f"€{v/1000:.0f}k" for v in m2026["Sales"]],
                       textposition="top center",textfont=dict(size=9,color="#90caf9"),
                       line=dict(color=C["blue"],width=3),
                       marker=dict(size=9,color=C["blue"],line=dict(color="white",width=2)),
                       name="Prévision"),
        ])
        fig2.update_layout(**PTH,title=f"Courbe {target_year} avec Intervalles de Confiance",height=380,legend=LEG)
        st.plotly_chart(fig2, use_container_width=True)

    with t2:
        hist = df_f.groupby(pd.Grouper(key="Date",freq="MS")).agg(Sales=("Sales","sum")).reset_index()
        hist["Year"] = hist["Date"].dt.year
        hist["MN"]   = hist["Date"].dt.month
        fig3 = go.Figure()
        clrs = {2022:"#37474f",2023:"#546e7a",2024:"#78909c",2025:"#64b5f6"}
        for yr in sorted(hist["Year"].unique()):
            s = hist[hist["Year"]==yr].sort_values("MN")
            fig3.add_trace(go.Scatter(x=s["MN"].map(lambda x:MONTH_NAMES[x][:3]),y=s["Sales"],
                mode="lines+markers",name=str(yr),
                line=dict(color=clrs.get(yr,"#546e7a"),width=2),marker=dict(size=5)))
        fig3.add_trace(go.Scatter(x=m2026["Mois"].str[:3],y=m2026["Sales"],
            mode="lines+markers",name=f"{target_year} (prédit)",
            line=dict(color=C["blue"],width=3,dash="dot"),
            marker=dict(size=9,color=C["blue"],line=dict(color="white",width=2))))
        fig3.update_layout(**PTH,title=f"Comparaison Historique vs Prévisions {target_year}",height=420,
                           hovermode="x unified",legend=LEG)
        st.plotly_chart(fig3, use_container_width=True)

    with t3:
        st.markdown(f"""
        <div class=\'insight-card info\'>
            <div class=\'insight-title\'>🧠 Dummy DataFrame — Explication</div>
            <div class=\'insight-text\'>
                Ce tableau montre les <b>features projetées</b> qui ont été passées au modèle GBR
                pour générer les prévisions de <b>{target_year}</b>.<br/><br/>
                • <b>Taux de croissance appliqué :</b> {growth_pct}% par an<br/>
                • <b>Années projetées :</b> {target_year - int(df_f["Date"].dt.year.max())} an(s) depuis {int(df_f["Date"].dt.year.max())}<br/>
                • <b>Facteur multiplicateur :</b> ×{(1+growth_rate)**(target_year - int(df_f["Date"].dt.year.max())):.3f}<br/>
                • <b>Combinaisons générées :</b> {len(df_forecast):,} lignes (12 mois × régions × segments × catégories)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Afficher un échantillon du Dummy DataFrame
        dummy_display = df_forecast[[
            "Date","Region","Segment","Category","Sub_Category",
            "Year","Month","Quantity","Unit_Cost","Unit_Price","Discount","Predicted_Sales"
        ]].head(50).copy()
        dummy_display.columns = [
            "Date","Région","Segment","Catégorie","Sous-Cat.",
            "Année","Mois","Quantité","Coût Unit.","Prix Unit.","Remise","Ventes Prédites"
        ]
        st.markdown(f"<div style=\'color:#546e7a;font-size:.8rem;margin-bottom:.5rem;\'>Aperçu des 50 premières lignes sur {len(df_forecast):,}</div>",unsafe_allow_html=True)
        st.dataframe(dummy_display.reset_index(drop=True), use_container_width=True, height=400)

    with t4:
        tbl = m2026[["Mois","Sales","Profit","Marge","IC_Basse","IC_Haute"]].copy()
        tbl.columns=["Mois","Ventes Prévues (€)","Profit Prévu (€)","Marge (%)","IC Basse (€)","IC Haute (€)"]
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=460)
        st.download_button(
            f"⬇️ Exporter Prévisions {target_year} (CSV)",
            tbl.to_csv(index=False),
            f"previsions_{target_year}.csv","text/csv"
        )
elif page == "Prédiction Manuelle":

    # ── Mapping catégorie → sous-catégories ──────────────────────────────
    CAT_SUB = {
        "Office Supplies": ["Binders", "Paper", "Storage"],
        "Furniture":       ["Bookcases", "Chairs", "Tables"],
        "Technology":      ["Accessories", "Copiers", "Phones"],
    }
    # Valeurs moyennes par catégorie pour les suggestions
    CAT_DEFAULTS = {
        "Office Supplies": dict(qty=5, cost=271.0, price=325.0, discount=0.10),
        "Furniture":       dict(qty=5, cost=250.0, price=300.0, discount=0.10),
        "Technology":      dict(qty=5, cost=255.0, price=306.0, discount=0.10),
    }
    PROFIT_RATIO = {"Office Supplies": 0.163, "Furniture": 0.167, "Technology": 0.163}

    # ── Formulaire de saisie ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>📝 Paramètres de la commande</div>",
                unsafe_allow_html=True)

    with st.form("prediction_form"):
        # ── Ligne 1 : Date ──
        col_yr, col_mo, col_sp1 = st.columns([1, 1, 2])
        with col_yr:
            input_year = st.selectbox("📅 Année", list(range(2022, 2035)), index=4)
        with col_mo:
            input_month = st.selectbox("🗓️ Mois",
                list(range(1, 13)),
                format_func=lambda x: MONTH_NAMES[x], index=0)

        st.markdown("<hr style='border-color:#1a3a6c;opacity:.4;'>", unsafe_allow_html=True)

        # ── Ligne 2 : Catégorielles ──
        col_r, col_s, col_c, col_sc = st.columns(4)
        with col_r:
            input_region = st.selectbox("📍 Région",
                ["Centre", "Est", "Nord", "Sud"])
        with col_s:
            input_segment = st.selectbox("🏢 Segment",
                ["Consommateur", "Corporate", "Entreprise"])
        with col_c:
            input_category = st.selectbox("📦 Catégorie",
                ["Office Supplies", "Furniture", "Technology"])
        with col_sc:
            input_subcat = st.selectbox("🗂️ Sous-Catégorie",
                CAT_SUB[input_category])

        st.markdown("<hr style='border-color:#1a3a6c;opacity:.4;'>", unsafe_allow_html=True)

        # ── Ligne 3 : Numériques ──
        defaults = CAT_DEFAULTS[input_category]
        col_q, col_uc, col_up, col_d = st.columns(4)
        with col_q:
            input_qty = st.number_input("📦 Quantité", min_value=1, max_value=20,
                value=defaults["qty"], step=1,
                help="Nombre d'unités commandées")
        with col_uc:
            input_cost = st.number_input("💵 Coût Unitaire (€)", min_value=10.0, max_value=600.0,
                value=defaults["cost"], step=5.0,
                help="Coût d'achat par unité")
        with col_up:
            input_price = st.number_input("💰 Prix Unitaire (€)", min_value=10.0, max_value=700.0,
                value=defaults["price"], step=5.0,
                help="Prix de vente par unité")
        with col_d:
            input_discount = st.slider("🏷️ Remise (%)", min_value=0, max_value=30,
                value=int(defaults["discount"]*100), step=1,
                help="Taux de remise accordé au client") / 100.0

        st.markdown("<br/>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀  Lancer la Prédiction",
                                          use_container_width=True)

    # ── Résultats ────────────────────────────────────────────────────────
    if submitted:
        # Construire la ligne d'input (1 seule ligne = 1 commande)
        input_row = pd.DataFrame([{
            "Region":       input_region,
            "Segment":      input_segment,
            "Category":     input_category,
            "Sub_Category": input_subcat,
            "Year":         input_year,
            "Month":        input_month,
            "Quantity":     input_qty,
            "Unit_Cost":    input_cost,
            "Unit_Price":   input_price,
            "Discount":     input_discount,
        }])

        # Prédiction via le modèle GBR
        predicted_sales  = float(predict_sales(input_row, model, encoders)[0])
        profit_ratio     = PROFIT_RATIO.get(input_category, 0.165)
        predicted_profit = predicted_sales * profit_ratio
        predicted_margin = (predicted_profit / predicted_sales * 100) if predicted_sales > 0 else 0
        revenue_line     = input_qty * input_price * (1 - input_discount)
        cost_line        = input_qty * input_cost

        st.markdown("<div class='section-title'>📊 Résultats de la Prédiction</div>",
                    unsafe_allow_html=True)

        # ── KPI Cards résultat ──
        r1, r2, r3, r4, r5 = st.columns(5)
        for col, args in zip([r1, r2, r3, r4, r5], [
            ("#1565c0","rgba(21,101,192,.2)","💰","Ventes Prédites",
             f"€{predicted_sales:,.2f}", f"Modèle GBR · R²=0.9997"),
            ("#26a69a","rgba(38,166,154,.2)","💎","Profit Estimé",
             f"€{predicted_profit:,.2f}", f"Ratio : {profit_ratio*100:.1f}%"),
            ("#66bb6a" if predicted_margin>15 else "#ff7043",
             "rgba(102,187,106,.2)","📈","Marge",
             f"{predicted_margin:.1f}%", "Bonne" if predicted_margin>15 else "Faible"),
            ("#42a5f5","rgba(66,165,245,.15)","🧾","CA Brut Ligne",
             f"€{revenue_line:,.2f}", f"{input_qty} unité(s) × €{input_price*(1-input_discount):.2f}"),
            ("#ff7043","rgba(255,112,67,.15)","🏭","Coût Total",
             f"€{cost_line:,.2f}", f"{input_qty} × €{input_cost:.2f}"),
        ]):
            with col:
                st.markdown(kpi(*args), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        col_left, col_right = st.columns([3, 2])

        with col_left:
            # ── Récapitulatif des inputs ──
            st.markdown("<div class='section-title'>📋 Récapitulatif des Inputs</div>",
                        unsafe_allow_html=True)
            recap = pd.DataFrame({
                "Paramètre": ["Année","Mois","Région","Segment","Catégorie",
                              "Sous-Catégorie","Quantité","Prix Unitaire",
                              "Coût Unitaire","Remise"],
                "Valeur":    [input_year, MONTH_NAMES[input_month], input_region,
                              input_segment, input_category, input_subcat,
                              f"{input_qty} unité(s)", f"€{input_price:.2f}",
                              f"€{input_cost:.2f}", f"{input_discount*100:.0f}%"],
            })
            st.dataframe(recap, use_container_width=True, hide_index=True, height=380)

        with col_right:
            # ── Jauge de profitabilité ──
            st.markdown("<div class='section-title'>📈 Analyse de Profitabilité</div>",
                        unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_margin,
                delta={"reference": 15, "valueformat": ".1f",
                       "increasing": {"color": "#66bb6a"},
                       "decreasing": {"color": "#ef5350"}},
                title={"text": "Marge (%)", "font": {"color": "#90a4ae", "size": 14}},
                number={"suffix": "%", "font": {"color": "#e3f2fd", "size": 32}},
                gauge={
                    "axis": {"range": [0, 40], "tickcolor": "#546e7a",
                             "tickfont": {"color": "#546e7a"}},
                    "bar":  {"color": "#1e88e5", "thickness": 0.3},
                    "bgcolor": "#0a1628",
                    "bordercolor": "#1a3a6c",
                    "steps": [
                        {"range": [0,  10], "color": "rgba(239,83,80,.2)"},
                        {"range": [10, 20], "color": "rgba(255,167,38,.2)"},
                        {"range": [20, 40], "color": "rgba(102,187,106,.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffd54f", "width": 3},
                        "thickness": 0.8, "value": 15
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0a1628", font=dict(family="Inter"),
                margin=dict(l=20, r=20, t=40, b=20), height=280,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Verdict ──
            if predicted_margin >= 20:
                verdict_cls, verdict_icon, verdict_txt = "success", "✅", "Excellente rentabilité"
                verdict_msg = "Cette commande génère une marge supérieure à 20%. Validez et priorisez."
            elif predicted_margin >= 10:
                verdict_cls, verdict_icon, verdict_txt = "info",    "📊", "Rentabilité correcte"
                verdict_msg = "Marge acceptable. Réduire la remise pourrait améliorer le profit."
            else:
                verdict_cls, verdict_icon, verdict_txt = "warning", "⚠️", "Marge trop faible"
                verdict_msg = "Cette commande risque d'être peu rentable. Augmentez le prix ou réduisez la remise."

            st.markdown(f"""
            <div class='insight-card {verdict_cls}'>
                <div class='insight-title'>{verdict_icon} {verdict_txt}</div>
                <div class='insight-text'>{verdict_msg}</div>
            </div>""", unsafe_allow_html=True)

        # ── Simulation : impact de la remise ──
        st.markdown("<div class='section-title'>🔄 Simulation — Impact de la Remise sur les Ventes</div>",
                    unsafe_allow_html=True)

        discount_range = [i/100 for i in range(0, 31, 5)]
        sim_rows = []
        for d in discount_range:
            row = input_row.copy()
            row["Discount"] = d
            s = float(predict_sales(row, model, encoders)[0])
            p = s * profit_ratio
            sim_rows.append({"Remise (%)": f"{d*100:.0f}%", "Ventes": round(s, 2),
                              "Profit": round(p, 2), "Marge (%)": round(p/s*100, 1)})
        df_sim = pd.DataFrame(sim_rows)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=df_sim["Remise (%)"], y=df_sim["Ventes"],
            mode="lines+markers+text",
            text=[f"€{v:,.0f}" for v in df_sim["Ventes"]],
            textposition="top center", textfont=dict(size=9, color="#90caf9"),
            line=dict(color=C["blue"], width=3),
            marker=dict(size=10, color=[
                "#66bb6a" if v == df_sim["Ventes"].max() else
                "#ef5350" if v == df_sim["Ventes"].min() else C["blue"]
                for v in df_sim["Ventes"]
            ], line=dict(color="white", width=2)),
            name="Ventes Prédites",
        ))
        # Marquer la remise actuelle
        current_disc_label = f"{input_discount*100:.0f}%"
        if current_disc_label in df_sim["Remise (%)"].values:
            cur_val = df_sim[df_sim["Remise (%)"]==current_disc_label]["Ventes"].values[0]
            fig_sim.add_trace(go.Scatter(
                x=[current_disc_label], y=[cur_val],
                mode="markers", name="Remise actuelle",
                marker=dict(size=16, color="#ffd54f",
                            symbol="star", line=dict(color="white", width=2)),
            ))
        fig_sim.update_layout(**PTH,
            title="Impact de la Remise sur les Ventes Prédites",
            xaxis_title="Taux de remise", yaxis_title="Ventes prédites (€)",
            height=350, legend=LEG)
        st.plotly_chart(fig_sim, use_container_width=True)

        # Tableau simulation
        st.dataframe(df_sim, use_container_width=True, hide_index=True)

        # Export
        export_data = {
            "Année": [input_year], "Mois": [MONTH_NAMES[input_month]],
            "Région": [input_region], "Segment": [input_segment],
            "Catégorie": [input_category], "Sous-Catégorie": [input_subcat],
            "Quantité": [input_qty], "Prix Unitaire": [input_price],
            "Coût Unitaire": [input_cost], "Remise (%)": [input_discount*100],
            "Ventes Prédites (€)": [round(predicted_sales,2)],
            "Profit Estimé (€)": [round(predicted_profit,2)],
            "Marge (%)": [round(predicted_margin,1)],
        }
        st.download_button(
            "⬇️ Exporter ce résultat (CSV)",
            pd.DataFrame(export_data).to_csv(index=False),
            f"prediction_{input_year}_{input_month:02d}.csv", "text/csv"
        )

    else:
        # Message d'accueil avant soumission
        st.markdown("""
        <div class='insight-card info' style='text-align:center; padding:2.5rem;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>🎯</div>
            <div class='insight-title' style='font-size:1rem;'>
                Remplissez le formulaire ci-dessus et cliquez sur "Lancer la Prédiction"
            </div>
            <div class='insight-text' style='margin-top:.5rem;'>
                Le modèle <b>Gradient Boosting</b> (R² = 0.9997) analysera vos paramètres
                et vous donnera instantanément les ventes prédites, le profit estimé,
                la marge et une simulation de l'impact des remises.
            </div>
        </div>
        """, unsafe_allow_html=True)


elif page == "Insights IA":
    with st.spinner("Analyse IA en cours..."):
        ins = generate_ai_insights(df_f)

    st.markdown("<div class='section-title'>🚨 Alertes Intelligentes</div>", unsafe_allow_html=True)
    for al in ins["alerts"]:
        st.markdown(f"""<div class='insight-card {al["type"]}'>
            <div class='insight-title'>{al["icon"]} {al["title"]}</div>
            <div class='insight-text'>{al["msg"]}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>💡 Recommandations Stratégiques</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    for i,rec in enumerate(ins["recommendations"]):
        with (c1 if i%2==0 else c2):
            pc = f"priority-{rec['priority'].lower()}"
            st.markdown(f"""<div class='rec-card'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='font-size:1.5rem;'>{rec['icon']}</span>
                    <span class='rec-priority {pc}'>{rec['priority']}</span>
                </div>
                <div style='font-size:.7rem;color:#546e7a;margin-top:4px;'>{rec['category']}</div>
                <div class='rec-title'>{rec['title']}</div>
                <div class='rec-desc'>{rec['desc']}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📊 Analyses Détaillées</div>", unsafe_allow_html=True)
    t1,t2,t3,t4 = st.tabs(["📅 Mensuel","📦 Catégories","🌍 Régions & Segments","💳 Impact Remises"])

    with t1:
        mn = ins["monthly"]
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                            subplot_titles=("Ventes Mensuelles","Marge (%)"),vertical_spacing=.12)
        fig.add_trace(go.Bar(x=mn["Month_Name"],y=mn["Sales"],name="Ventes",
            marker_color=[C["blue"] if v>=mn["Sales"].mean() else "#1a3a6c" for v in mn["Sales"]]),row=1,col=1)
        fig.add_trace(go.Scatter(x=mn["Month_Name"],y=mn["Sales"].rolling(3,center=True).mean(),
            line=dict(color=C["yellow"],width=2),name="Tendance"),row=1,col=1)
        fig.add_trace(go.Bar(x=mn["Month_Name"],y=mn["Profit_Margin"],name="Marge %",
            marker_color=[C["green"] if v>15 else C["orange"] for v in mn["Profit_Margin"]]),row=2,col=1)
        fig.update_layout(**PTH,height=480,legend=LEG)
        fig.update_xaxes(tickfont=dict(color="#546e7a"))
        fig.update_yaxes(gridcolor="#0d2040",tickfont=dict(color="#546e7a"))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        cat = ins["category_data"]
        c1,c2 = st.columns(2)
        with c1:
            fig_c = go.Figure(go.Pie(labels=cat["Category"],values=cat["Sales"],hole=.5,
                marker=dict(colors=["#1565c0","#1e88e5","#42a5f5"]),
                textinfo="label+percent",textfont=dict(color="#e3f2fd",size=12),
                hovertemplate="<b>%{label}</b><br>€%{value:,.0f}<br>%{percent}<extra></extra>"))
            fig_c.update_layout(**PTH,title="Part des Ventes par Catégorie",height=320,showlegend=False)
            st.plotly_chart(fig_c, use_container_width=True)
        with c2:
            fig_m = go.Figure(go.Bar(x=cat["Category"],y=cat["Profit_Margin"],
                marker=dict(color=cat["Profit_Margin"],colorscale=[[0,"#1a3a6c"],[.5,"#1565c0"],[1,"#66bb6a"]],showscale=False),
                text=[f"{v:.1f}%" for v in cat["Profit_Margin"]],textposition="outside",textfont=dict(size=11,color="#90a4ae")))
            fig_m.update_layout(**PTH,title="Marge par Catégorie (%)",height=320,showlegend=False)
            st.plotly_chart(fig_m, use_container_width=True)

    with t3:
        reg,seg = ins["region_data"],ins["segment_data"]
        c1,c2 = st.columns(2)
        with c1:
            fig_r = go.Figure(go.Bar(x=reg["Sales"],y=reg["Region"],orientation="h",
                marker=dict(color=reg["Sales"],colorscale=[[0,"#0d2a5e"],[1,C["blue"]]],showscale=False),
                text=[f"€{v:,.0f}" for v in reg["Sales"]],textposition="outside",textfont=dict(size=10,color="#90a4ae")))
            fig_r.update_layout(**PTH,title="Ventes par Région",height=300,showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)
        with c2:
            fig_s = go.Figure(go.Bar(x=seg["Segment"],y=seg["Profit_Margin"],
                marker=dict(color=["#1565c0","#1e88e5","#42a5f5"]),
                text=[f"{v:.1f}%" for v in seg["Profit_Margin"]],textposition="outside",textfont=dict(size=11,color="#90a4ae")))
            fig_s.update_layout(**PTH,title="Marge par Segment (%)",height=300,showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)

    with t4:
        disc = ins["discount_data"]
        c1,c2 = st.columns(2)
        with c1:
            fig_d1 = go.Figure(go.Bar(x=disc["Discount_Band"].astype(str),y=disc["Sales"],
                marker_color=[C["blue"],C["lblue"],C["orange"],C["red"]],
                text=[f"€{v:,.0f}" for v in disc["Sales"]],textposition="outside",textfont=dict(size=10,color="#90a4ae")))
            fig_d1.update_layout(**PTH,title="Ventes Moy. par Tranche de Remise",height=320,showlegend=False)
            st.plotly_chart(fig_d1, use_container_width=True)
        with c2:
            fig_d2 = go.Figure(go.Bar(x=disc["Discount_Band"].astype(str),y=disc["Profit"],
                marker_color=[C["green"],"#a5d6a7",C["orange"],C["red"]],
                text=[f"€{v:,.0f}" for v in disc["Profit"]],textposition="outside",textfont=dict(size=10,color="#90a4ae")))
            fig_d2.update_layout(**PTH,title="Profit Moy. par Tranche de Remise",height=320,showlegend=False)
            st.plotly_chart(fig_d2, use_container_width=True)
        st.markdown("""<div class='insight-card info'>
            <div class='insight-title'>📊 Analyse Impact Remises</div>
            <div class='insight-text'>Corrélation négative entre taux de remise et profit.
            Les remises >20% causent une chute significative de la marge.<br/><br/>
            <b>Recommandation :</b> Plafonner à 15% pour préserver la rentabilité.</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ÉVALUATION MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Évaluation Modèle":
    y_true    = df_f["Sales"].values
    y_pred_v  = df_f["_pred"].values
    residuals = y_true - y_pred_v

    c1,c2,c3,c4 = st.columns(4)
    for col,args in zip([c1,c2,c3,c4],[
        ("#66bb6a","rgba(102,187,106,.15)","🏆","R² Score",  f"{m['R2']:.4f}","Coefficient de détermination"),
        ("#1e88e5","rgba(30,136,229,.15)","🎯","Précision",  f"{m['Accuracy']:.1f}%",f"MAPE = {m['MAPE']:.2f}%"),
        ("#ff7043","rgba(255,112,67,.15)","📉","RMSE",        f"€{m['RMSE']:.2f}","Erreur quadratique moyenne"),
        ("#ce93d8","rgba(206,147,216,.12)","📏","MAE",         f"€{m['MAE']:.2f}","Erreur absolue moyenne"),
    ]):
        with col:
            st.markdown(kpi(*args), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    t1,t2,t3 = st.tabs(["🎯 Réel vs Prédit","📊 Résidus","🔑 Importance Variables"])

    with t1:
        c1,c2 = st.columns([2,1])
        with c1:
            lim=[min(y_true.min(),y_pred_v.min()),max(y_true.max(),y_pred_v.max())]
            fig = go.Figure([
                go.Scatter(x=y_true,y=y_pred_v,mode="markers",
                    marker=dict(color=abs(residuals),colorscale="Blues",size=6,opacity=.5,showscale=True,
                                colorbar=dict(title="Erreur €",tickfont=dict(color="#546e7a"))),
                    name="Prédictions",
                    hovertemplate="Réel: €%{x:,.0f}<br>Prédit: €%{y:,.0f}<extra></extra>"),
                go.Scatter(x=lim,y=lim,mode="lines",line=dict(color=C["red"],width=2,dash="dash"),name="Parfait"),
            ])
            fig.update_layout(**PTH,title="Réel vs Prédit",xaxis_title="Réel (€)",yaxis_title="Prédit (€)",height=400)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("""
            <div class='insight-card success'>
                <div class='insight-title'>✅ Modèle Excellent</div>
                <div class='insight-text'>R² = 0.9991 → le modèle explique <b>99.91%</b> de la variance.<br/><br/>
                Points très proches de la droite parfaite.</div>
            </div>
            <div class='insight-card info'>
                <div class='insight-title'>🤖 Algorithme</div>
                <div class='insight-text'><b>Gradient Boosting</b><br/>
                • 100 estimateurs<br/>• LR : 0.1<br/>• Depth : 5<br/>• Split : 80/20</div>
            </div>""", unsafe_allow_html=True)

    with t2:
        c1,c2 = st.columns(2)
        with c1:
            fig_r = go.Figure(go.Scatter(x=y_pred_v,y=residuals,mode="markers",
                marker=dict(color=residuals,colorscale="RdBu_r",size=5,opacity=.5,showscale=True,
                            colorbar=dict(tickfont=dict(color="#546e7a"))),name="Résidus"))
            fig_r.add_hline(y=0,line=dict(color=C["red"],width=2,dash="dash"))
            fig_r.update_layout(**PTH,title="Résidus vs Prédites",xaxis_title="Prédit",yaxis_title="Résidu",height=350)
            st.plotly_chart(fig_r, use_container_width=True)
        with c2:
            fig_h = go.Figure(go.Histogram(x=residuals,nbinsx=35,
                marker=dict(color=C["blue"],opacity=.7,line=dict(color="rgba(30,136,229,.3)",width=.5))))
            fig_h.add_vline(x=0,line=dict(color=C["red"],width=2,dash="dash"))
            fig_h.update_layout(**PTH,title="Distribution des Résidus",xaxis_title="Erreur (€)",height=350,showlegend=False)
            st.plotly_chart(fig_h, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        for col,(lbl,val) in zip([c1,c2,c3,c4],[
            ("Moyenne",f"€{residuals.mean():.2f}"),("Std",f"€{residuals.std():.2f}"),
            ("Min",f"€{residuals.min():.2f}"),("Max",f"€{residuals.max():.2f}"),
        ]):
            with col:
                st.markdown(f"<div class='data-stat'><div class='val' style='font-size:1.2rem;'>{val}</div><div class='lbl'>{lbl}</div></div>",unsafe_allow_html=True)

    with t3:
        imp = pd.Series(model.feature_importances_,index=FEATURES).sort_values()
        fig_fi = go.Figure(go.Bar(x=imp.values,y=imp.index,orientation="h",
            marker=dict(color=[C["blue"] if v>=imp.median() else "#1a3a6c" for v in imp.values],
                        line=dict(color="rgba(30,136,229,.3)",width=.5)),
            text=[f"{v:.3f}" for v in imp.values],textposition="outside",textfont=dict(size=10,color="#90a4ae")))
        fig_fi.update_layout(**PTH,title="Importance des Variables — Gradient Boosting",xaxis_title="Importance",height=420,showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — TABLEAU DÉTAILLÉ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Tableau Détaillé":
    st.markdown("<div class='section-title'>📋 Tableau Complet des Prévisions</div>", unsafe_allow_html=True)

    c1,c2 = st.columns([3,1])
    with c1:
        search = st.text_input("🔍 Rechercher (Région, Catégorie...)", "", placeholder="ex: Nord, Technology...")
    with c2:
        sort_by = st.selectbox("Trier par",["Date","Sales","_pred","Profit","Discount"])

    df_tbl = df_f.copy().sort_values(sort_by,ascending=(sort_by=="Date"))
    df_tbl["Error_€"] = (df_tbl["Sales"]-df_tbl["_pred"]).round(2)
    df_tbl["Error_%"] = (df_tbl["Error_€"]/df_tbl["Sales"]*100).round(2)
    df_tbl["CI_Low"]  = (df_tbl["_pred"]-1.96*m["RMSE"]).round(2)
    df_tbl["CI_High"] = (df_tbl["_pred"]+1.96*m["RMSE"]).round(2)

    disp = df_tbl[["Date","Region","Segment","Category","Sub_Category","Sales","_pred","Error_€","Error_%","Profit","Discount","CI_Low","CI_High"]].copy()
    disp.columns=["Date","Région","Segment","Catégorie","Sous-Cat.","Ventes Réelles","Ventes Prédites","Erreur €","Erreur %","Profit","Remise","IC Basse","IC Haute"]
    disp["Ventes Réelles"]=disp["Ventes Réelles"].round(2)
    disp["Ventes Prédites"]=disp["Ventes Prédites"].round(2)

    if search:
        mask = disp.apply(lambda r: search.lower() in str(r).lower(),axis=1)
        disp = disp[mask]

    st.markdown(f"<div style='color:#546e7a;font-size:.8rem;margin-bottom:.5rem;'>{len(disp):,} lignes</div>",unsafe_allow_html=True)
    st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=500)

    c1,c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Export CSV Complet", disp.to_csv(index=False), "previsions_completes.csv","text/csv")
    with c2:
        summary = df_f.groupby(pd.Grouper(key="Date",freq="MS")).agg(
            Ventes=("Sales","sum"),Predites=("_pred","sum"),Profit=("Profit","sum"),Commandes=("Sales","count")).reset_index()
        summary["Marge_%"]=(summary["Profit"]/summary["Ventes"]*100).round(1)
        st.download_button("⬇️ Export Résumé Mensuel", summary.to_csv(index=False),"resume_mensuel.csv","text/csv")

    st.markdown("<div class='section-title'>📊 Résumé Mensuel</div>", unsafe_allow_html=True)
    fig = go.Figure([
        go.Bar(x=summary["Date"],y=summary["Ventes"],name="Réel",
               marker_color="rgba(30,136,229,.5)",marker_line=dict(color=C["blue"],width=1)),
        go.Bar(x=summary["Date"],y=summary["Predites"],name="Prédit",
               marker_color="rgba(102,187,106,.4)",marker_line=dict(color=C["green"],width=1)),
    ])
    fig.update_layout(**PTH,title="Ventes Mensuelles — Réel vs Prédit",barmode="group",height=380,
                      hovermode="x unified",legend=LEG)
    st.plotly_chart(fig, use_container_width=True)
