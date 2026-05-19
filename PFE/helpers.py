"""
utils/helpers.py
────────────────
Fonctions utilitaires : chargement, prétraitement, prédiction, insights IA.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Colonnes obligatoires ────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Order_ID", "Region", "Segment", "Category", "Sub_Category",
    "Year", "Month", "Quantity", "Unit_Cost", "Unit_Price",
    "Discount", "Sales", "Profit", "Date"
]

FEATURES = [
    "Region", "Segment", "Category", "Sub_Category",
    "Year", "Month", "Quantity", "Unit_Cost", "Unit_Price", "Discount"
]

MONTH_NAMES = {
    1:"Janvier", 2:"Février", 3:"Mars", 4:"Avril",
    5:"Mai", 6:"Juin", 7:"Juillet", 8:"Août",
    9:"Septembre", 10:"Octobre", 11:"Novembre", 12:"Décembre"
}

# ── Chargement des données ───────────────────────────────────────────────────
def load_data(path: str = "data/cleaned_data.csv") -> pd.DataFrame:
    """Charge et prépare le dataset principal."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_uploaded_file(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """
    Charge un fichier CSV ou XLSX uploadé via Streamlit.
    Retourne (dataframe, message_erreur).
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "❌ Format non supporté. Utilisez CSV ou XLSX."

        # Vérification colonnes
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return None, f"❌ Colonnes manquantes : {', '.join(missing)}"

        df["Date"] = pd.to_datetime(df["Date"])
        return df, "✅ Fichier chargé avec succès !"

    except Exception as e:
        return None, f"❌ Erreur lors du chargement : {str(e)}"


# ── Chargement du modèle ─────────────────────────────────────────────────────
def load_model(model_path="models/model_gbr_sales.pkl",
               enc_path="models/encoders.pkl"):
    """Charge le modèle GBR et les encodeurs."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


# ── Prétraitement ────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Encode les variables catégorielles pour la prédiction."""
    df_enc = df.copy()
    df_enc["Region"]       = encoders["region"].transform(df["Region"])
    df_enc["Segment"]      = encoders["segment"].transform(df["Segment"])
    df_enc["Category"]     = encoders["category"].transform(df["Category"])
    df_enc["Sub_Category"] = encoders["subcat"].transform(df["Sub_Category"])
    return df_enc


# ── Prédiction ───────────────────────────────────────────────────────────────
def predict_sales(df_raw: pd.DataFrame, model, encoders) -> np.ndarray:
    """Retourne les prédictions de ventes."""
    df_enc = preprocess_data(df_raw, encoders)
    return model.predict(df_enc[FEATURES])


# ── Métriques ────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    """Calcule MAE, RMSE, R², MAPE."""
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {
        "MAE":      round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE":     round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R2":       round(r2_score(y_true, y_pred), 4),
        "MAPE":     round(mape, 2),
        "Accuracy": round(100 - mape, 2),
    }


# ── Prévisions 2026 ──────────────────────────────────────────────────────────
def generate_2026_forecast(df: pd.DataFrame, model, encoders) -> pd.DataFrame:
    """
    Génère les prévisions mensuelles Sales + Profit pour 2026
    en se basant sur les moyennes par mois/région/segment/catégorie.
    """
    rows = []
    regions    = df["Region"].unique()
    segments   = df["Segment"].unique()
    categories = df["Category"].unique()

    for month in range(1, 13):
        for region in regions:
            for segment in segments:
                for category in categories:
                    sub_cats = df[df["Category"] == category]["Sub_Category"].unique()
                    sub_cat = sub_cats[0]

                    # Moyennes historiques pour ce groupe
                    mask = (
                        (df["Month"] == month) &
                        (df["Region"] == region) &
                        (df["Segment"] == segment) &
                        (df["Category"] == category)
                    )
                    subset = df[mask]
                    if subset.empty:
                        subset = df[(df["Month"] == month) & (df["Category"] == category)]
                    if subset.empty:
                        subset = df[df["Month"] == month]

                    row = {
                        "Date":         pd.Timestamp(f"2026-{month:02d}-01"),
                        "Year":         2026,
                        "Month":        month,
                        "Region":       region,
                        "Segment":      segment,
                        "Category":     category,
                        "Sub_Category": sub_cat,
                        "Quantity":     subset["Quantity"].mean(),
                        "Unit_Cost":    subset["Unit_Cost"].mean(),
                        "Unit_Price":   subset["Unit_Price"].mean(),
                        "Discount":     subset["Discount"].mean(),
                        "Profit_Ratio": (subset["Profit"] / subset["Sales"]).mean()
                                        if not subset.empty else 0.15,
                    }
                    rows.append(row)

    df_2026 = pd.DataFrame(rows)
    df_2026["Predicted_Sales"]  = predict_sales(df_2026, model, encoders)
    df_2026["Predicted_Profit"] = df_2026["Predicted_Sales"] * df_2026["Profit_Ratio"]

    return df_2026


# ── Insights IA ──────────────────────────────────────────────────────────────
def generate_ai_insights(df: pd.DataFrame) -> dict:
    """
    Analyse automatique des données et génère des recommandations intelligentes.
    """
    insights = {}

    # ── Analyse mensuelle ──
    monthly = df.groupby("Month").agg(
        Sales=("Sales","sum"), Profit=("Profit","sum"),
        Discount=("Discount","mean"), Quantity=("Quantity","sum")
    ).reset_index()
    monthly["Profit_Margin"] = monthly["Profit"] / monthly["Sales"] * 100
    monthly["Month_Name"] = monthly["Month"].map(MONTH_NAMES)

    peak_month = monthly.loc[monthly["Sales"].idxmax()]
    weak_month = monthly.loc[monthly["Sales"].idxmin()]
    avg_sales  = monthly["Sales"].mean()

    insights["peak_month"]  = peak_month
    insights["weak_month"]  = weak_month
    insights["monthly"]     = monthly

    # ── Analyse par catégorie ──
    cat = df.groupby("Category").agg(
        Sales=("Sales","sum"), Profit=("Profit","sum"),
        Discount=("Discount","mean"), Quantity=("Quantity","sum")
    ).reset_index()
    cat["Profit_Margin"] = cat["Profit"] / cat["Sales"] * 100
    cat["Perf_Score"]    = (cat["Sales"] - cat["Sales"].mean()) / cat["Sales"].std()

    insights["best_category"] = cat.loc[cat["Sales"].idxmax(), "Category"]
    insights["weak_category"] = cat.loc[cat["Sales"].idxmin(), "Category"]
    insights["category_data"] = cat

    # ── Analyse par région ──
    reg = df.groupby("Region").agg(
        Sales=("Sales","sum"), Profit=("Profit","sum"),
        Discount=("Discount","mean")
    ).reset_index()
    reg["Profit_Margin"] = reg["Profit"] / reg["Sales"] * 100

    insights["best_region"] = reg.loc[reg["Sales"].idxmax(), "Region"]
    insights["weak_region"] = reg.loc[reg["Sales"].idxmin(), "Region"]
    insights["region_data"] = reg

    # ── Analyse par segment ──
    seg = df.groupby("Segment").agg(
        Sales=("Sales","sum"), Profit=("Profit","sum"),
        Discount=("Discount","mean")
    ).reset_index()
    seg["Profit_Margin"] = seg["Profit"] / seg["Sales"] * 100

    insights["best_segment"] = seg.loc[seg["Profit_Margin"].idxmax(), "Segment"]
    insights["segment_data"] = seg

    # ── Impact des remises ──
    df["Discount_Band"] = pd.cut(df["Discount"],
        bins=[-0.01, 0.05, 0.15, 0.25, 1.0],
        labels=["0-5%","5-15%","15-25%",">25%"])
    disc = df.groupby("Discount_Band", observed=True).agg(
        Sales=("Sales","mean"), Profit=("Profit","mean"),
        Count=("Sales","count")
    ).reset_index()
    insights["discount_data"] = disc

    # ── Alertes ──
    alerts = []
    high_disc = df[df["Discount"] > 0.25]
    if len(high_disc) > 0:
        loss = high_disc[high_disc["Profit"] < 0]
        if len(loss) > 0:
            alerts.append({
                "type": "danger",
                "icon": "🔴",
                "title": "Remises excessives détectées",
                "msg": f"{len(loss)} transactions avec remise >25% génèrent des pertes. "
                       f"Perte totale : €{abs(loss['Profit'].sum()):,.0f}"
            })

    low_margin_cat = cat[cat["Profit_Margin"] < 10]
    for _, row in low_margin_cat.iterrows():
        alerts.append({
            "type": "warning",
            "icon": "🟡",
            "title": f"Marge faible — {row['Category']}",
            "msg": f"Marge bénéficiaire de {row['Profit_Margin']:.1f}% pour {row['Category']}. "
                   f"Réviser la stratégie tarifaire."
        })

    best_margin_seg = seg.loc[seg["Profit_Margin"].idxmax()]
    alerts.append({
        "type": "success",
        "icon": "🟢",
        "title": f"Segment le plus rentable : {best_margin_seg['Segment']}",
        "msg": f"Marge de {best_margin_seg['Profit_Margin']:.1f}%. "
               f"Augmenter les efforts marketing sur ce segment."
    })

    # Saisonnalité
    q1 = df[df["Month"].isin([1,2,3])]["Sales"].sum()
    q4 = df[df["Month"].isin([10,11,12])]["Sales"].sum()
    if q4 > q1 * 1.2:
        alerts.append({
            "type": "info",
            "icon": "🔵",
            "title": "Pic saisonnier T4 détecté",
            "msg": f"Le T4 génère {((q4/q1)-1)*100:.0f}% de plus que le T1. "
                   f"Augmenter les stocks en Octobre-Novembre."
        })

    insights["alerts"] = alerts

    # ── Recommandations ──
    recommendations = [
        {
            "icon": "📉",
            "category": "Remises",
            "priority": "Haute",
            "color": "#ef5350",
            "title": "Optimiser la politique de remises",
            "desc": f"Les remises >20% réduisent significativement la marge. "
                    f"Remise moyenne actuelle : {df['Discount'].mean()*100:.1f}%. "
                    f"Recommandation : plafonner à 15% sauf exceptions justifiées."
        },
        {
            "icon": "📦",
            "category": "Stock",
            "priority": "Haute",
            "color": "#ff7043",
            "title": f"Renforcer les stocks avant {peak_month['Month_Name']}",
            "desc": f"{peak_month['Month_Name']} est le mois le plus fort avec "
                    f"€{peak_month['Sales']:,.0f} de ventes. Anticiper les commandes "
                    f"fournisseurs 6-8 semaines à l'avance."
        },
        {
            "icon": "🎯",
            "category": "Marketing",
            "priority": "Moyenne",
            "color": "#42a5f5",
            "title": f"Campagne promotionnelle en {weak_month['Month_Name']}",
            "desc": f"{weak_month['Month_Name']} affiche les ventes les plus faibles "
                    f"(€{weak_month['Sales']:,.0f}). Lancer des offres spéciales "
                    f"pour stimuler la demande pendant cette période creuse."
        },
        {
            "icon": "🌍",
            "category": "Régions",
            "priority": "Moyenne",
            "color": "#26a69a",
            "title": f"Développer la région {insights['weak_region']}",
            "desc": f"La région {insights['weak_region']} sous-performe. "
                    f"Analyser les barrières à l'entrée et envisager des partenariats "
                    f"locaux ou une force commerciale dédiée."
        },
        {
            "icon": "💎",
            "category": "Segments",
            "priority": "Haute",
            "color": "#7e57c2",
            "title": f"Fidéliser le segment {insights['best_segment']}",
            "desc": f"{insights['best_segment']} est le segment le plus rentable. "
                    f"Investir dans des programmes de fidélité et des offres premium "
                    f"pour augmenter la valeur vie client (CLV)."
        },
        {
            "icon": "🚀",
            "category": "Produits",
            "priority": "Basse",
            "color": "#66bb6a",
            "title": f"Développer la catégorie {insights['weak_category']}",
            "desc": f"{insights['weak_category']} présente le plus faible volume. "
                    f"Étudier l'élargissement de la gamme et une meilleure visibilité "
                    f"en ligne et en point de vente."
        },
    ]
    insights["recommendations"] = recommendations

    # ── Tendance YoY ──
    yearly = df.groupby("Year")["Sales"].sum()
    if len(yearly) >= 2:
        last_yr  = yearly.iloc[-1]
        prev_yr  = yearly.iloc[-2]
        yoy = (last_yr - prev_yr) / prev_yr * 100
        insights["yoy_growth"] = round(yoy, 1)
        insights["yoy_last"]   = last_yr
        insights["yoy_prev"]   = prev_yr
    else:
        insights["yoy_growth"] = 0

    return insights