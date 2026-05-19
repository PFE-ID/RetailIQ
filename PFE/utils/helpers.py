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


def load_uploaded_file(uploaded_file):
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


# ════════════════════════════════════════════════════════════════════════════
#  DUMMY DATAFRAME — Génération des prévisions pour n'importe quelle année
# ════════════════════════════════════════════════════════════════════════════

def build_dummy_dataframe(df_hist: pd.DataFrame, target_year: int,
                          growth_rate: float = 0.05) -> pd.DataFrame:
    """
    Construit un Dummy DataFrame pour une année cible (ex: 2026, 2027...).

    Logique :
    ─────────
    Pour chaque combinaison (Mois × Région × Segment × Catégorie) :
      1. Cherche les données historiques du même groupe + même mois
      2. Calcule les moyennes des features numériques (Quantity, Unit_Cost,
         Unit_Price, Discount)
      3. Applique un taux de croissance annuel sur Quantity et Unit_Price
         pour simuler l'évolution future
      4. Conserve la même distribution catégorielle (Region, Segment,
         Category, Sub_Category)

    Paramètres :
    ────────────
    df_hist     : DataFrame historique (données réelles)
    target_year : Année à prédire (2026, 2027, 2028...)
    growth_rate : Taux de croissance annuel appliqué (défaut 5%)

    Retourne :
    ──────────
    DataFrame avec toutes les colonnes nécessaires au modèle + colonnes
    de résultat (Predicted_Sales, Predicted_Profit, Profit_Ratio).
    """

    # Nombre d'années écoulées depuis la dernière année historique
    last_hist_year = int(df_hist["Year"].max())
    years_ahead    = target_year - last_hist_year
    growth_factor  = (1 + growth_rate) ** years_ahead  # ex: 1.05^2 pour 2027

    rows = []
    regions    = df_hist["Region"].unique()
    segments   = df_hist["Segment"].unique()
    categories = df_hist["Category"].unique()

    for month in range(1, 13):
        for region in regions:
            for segment in segments:
                for category in categories:

                    # ── Sous-catégorie la plus fréquente pour cette catégorie ──
                    sub_cat = (
                        df_hist[df_hist["Category"] == category]["Sub_Category"]
                        .mode()
                        .iloc[0]
                    )

                    # ── Recherche des données historiques (du plus précis au plus large) ──
                    # Niveau 1 : même mois + région + segment + catégorie
                    mask_l1 = (
                        (df_hist["Month"]    == month)   &
                        (df_hist["Region"]   == region)  &
                        (df_hist["Segment"]  == segment) &
                        (df_hist["Category"] == category)
                    )
                    subset = df_hist[mask_l1]

                    # Niveau 2 : même mois + catégorie (si level 1 vide)
                    if subset.empty:
                        mask_l2 = (
                            (df_hist["Month"]    == month) &
                            (df_hist["Category"] == category)
                        )
                        subset = df_hist[mask_l2]

                    # Niveau 3 : même catégorie seulement (fallback global)
                    if subset.empty:
                        subset = df_hist[df_hist["Category"] == category]

                    # Niveau 4 : tout le dataset (dernier recours)
                    if subset.empty:
                        subset = df_hist

                    # ── Calcul des moyennes historiques ──
                    avg_quantity   = subset["Quantity"].mean()
                    avg_unit_cost  = subset["Unit_Cost"].mean()
                    avg_unit_price = subset["Unit_Price"].mean()
                    avg_discount   = subset["Discount"].mean()
                    avg_profit_ratio = (
                        (subset["Profit"] / subset["Sales"])
                        .replace([np.inf, -np.inf], np.nan)
                        .mean()
                    )
                    if pd.isna(avg_profit_ratio):
                        avg_profit_ratio = 0.15  # valeur par défaut

                    # ── Application du taux de croissance ──
                    # Quantity et Unit_Price évoluent avec la croissance
                    # Unit_Cost suit Unit_Price (marge constante)
                    # Discount reste stable (décision commerciale)
                    projected_quantity   = avg_quantity   * growth_factor
                    projected_unit_price = avg_unit_price * growth_factor
                    projected_unit_cost  = avg_unit_cost  * growth_factor
                    projected_discount   = avg_discount   # inchangé

                    # ── Construction de la ligne du Dummy DataFrame ──
                    row = {
                        # Identifiants temporels
                        "Date":         pd.Timestamp(f"{target_year}-{month:02d}-01"),
                        "Year":         target_year,
                        "Month":        month,

                        # Variables catégorielles (inchangées)
                        "Region":       region,
                        "Segment":      segment,
                        "Category":     category,
                        "Sub_Category": sub_cat,

                        # Features numériques projetées
                        "Quantity":     round(projected_quantity, 2),
                        "Unit_Cost":    round(projected_unit_cost, 2),
                        "Unit_Price":   round(projected_unit_price, 2),
                        "Discount":     round(projected_discount, 4),

                        # Métadonnées pour calcul du profit
                        "Profit_Ratio": avg_profit_ratio,

                        # Traçabilité
                        "_hist_year":   last_hist_year,
                        "_growth_rate": growth_rate,
                        "_years_ahead": years_ahead,
                    }
                    rows.append(row)

    df_dummy = pd.DataFrame(rows)

    # ── Prédiction via le modèle GBR ──
    # Le modèle reçoit les features projetées et génère Sales prédit
    # (la logique de growth est déjà intégrée dans les features)
    return df_dummy


def generate_forecast_for_year(df_hist: pd.DataFrame, model, encoders,
                                target_year: int,
                                growth_rate: float = 0.05) -> pd.DataFrame:
    """
    Fonction principale : génère les prévisions complètes pour une année.

    Étapes :
    ────────
    1. build_dummy_dataframe()  → construit le Dummy DataFrame
    2. predict_sales()          → passe le Dummy au modèle GBR
    3. Calcule Predicted_Profit = Predicted_Sales × Profit_Ratio

    Retourne un DataFrame agrégé par mois avec :
    - Date, Mois, Predicted_Sales, Predicted_Profit, Marge, IC Basse, IC Haute
    """
    # Étape 1 : construire le Dummy DataFrame
    df_dummy = build_dummy_dataframe(df_hist, target_year, growth_rate)

    # Étape 2 : prédire les ventes via le modèle
    df_dummy["Predicted_Sales"] = predict_sales(df_dummy, model, encoders)

    # Étape 3 : calculer le profit prédit
    df_dummy["Predicted_Profit"] = (
        df_dummy["Predicted_Sales"] * df_dummy["Profit_Ratio"]
    )

    return df_dummy


def aggregate_forecast(df_forecast: pd.DataFrame, rmse: float = 0.0) -> pd.DataFrame:
    """
    Agrège les prévisions par mois et ajoute les intervalles de confiance.

    Paramètres :
    ────────────
    df_forecast : résultat de generate_forecast_for_year()
    rmse        : RMSE du modèle (pour calculer IC à 95%)

    Retourne un DataFrame mensuel avec IC ±8% si rmse=0, sinon IC basé sur RMSE.
    """
    agg = df_forecast.groupby("Date").agg(
        Sales=("Predicted_Sales",  "sum"),
        Profit=("Predicted_Profit", "sum"),
    ).reset_index()

    agg["Month"]  = agg["Date"].dt.month
    agg["Mois"]   = agg["Month"].map(MONTH_NAMES)
    agg["Marge"]  = (agg["Profit"] / agg["Sales"] * 100).round(1)

    # Intervalles de confiance
    if rmse > 0:
        # IC basé sur RMSE du modèle (statistique)
        n_groups = df_forecast.groupby("Date").size().mean()
        margin   = 1.96 * rmse * np.sqrt(n_groups)
        agg["IC_Basse"] = (agg["Sales"] - margin).round(2)
        agg["IC_Haute"] = (agg["Sales"] + margin).round(2)
    else:
        # IC ±8% (conservateur)
        agg["IC_Basse"] = (agg["Sales"] * 0.92).round(2)
        agg["IC_Haute"] = (agg["Sales"] * 1.08).round(2)

    agg["Sales"]  = agg["Sales"].round(2)
    agg["Profit"] = agg["Profit"].round(2)

    return agg


# Alias pour compatibilité avec le code existant
def generate_2026_forecast(df, model, encoders):
    """Alias vers generate_forecast_for_year(2026) — compatibilité."""
    return generate_forecast_for_year(df, model, encoders, target_year=2026, growth_rate=0.05)


# ── Insights IA ──────────────────────────────────────────────────────────────
def generate_ai_insights(df: pd.DataFrame) -> dict:
    """Analyse automatique des données et génère des recommandations intelligentes."""
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

    insights["peak_month"] = peak_month
    insights["weak_month"] = weak_month
    insights["monthly"]    = monthly

    # ── Analyse par catégorie ──
    cat = df.groupby("Category").agg(
        Sales=("Sales","sum"), Profit=("Profit","sum"),
        Discount=("Discount","mean"), Quantity=("Quantity","sum")
    ).reset_index()
    cat["Profit_Margin"] = cat["Profit"] / cat["Sales"] * 100

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
    df_tmp = df.copy()
    df_tmp["Discount_Band"] = pd.cut(df_tmp["Discount"],
        bins=[-0.01, 0.05, 0.15, 0.25, 1.0],
        labels=["0-5%","5-15%","15-25%",">25%"])
    disc = df_tmp.groupby("Discount_Band", observed=True).agg(
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
                "type": "danger", "icon": "🔴",
                "title": "Remises excessives détectées",
                "msg": f"{len(loss)} transactions avec remise >25% génèrent des pertes. "
                       f"Perte totale : €{abs(loss['Profit'].sum()):,.0f}"
            })

    low_margin_cat = cat[cat["Profit_Margin"] < 10]
    for _, row in low_margin_cat.iterrows():
        alerts.append({
            "type": "warning", "icon": "🟡",
            "title": f"Marge faible — {row['Category']}",
            "msg": f"Marge bénéficiaire de {row['Profit_Margin']:.1f}% pour {row['Category']}. "
                   f"Réviser la stratégie tarifaire."
        })

    best_margin_seg = seg.loc[seg["Profit_Margin"].idxmax()]
    alerts.append({
        "type": "success", "icon": "🟢",
        "title": f"Segment le plus rentable : {best_margin_seg['Segment']}",
        "msg": f"Marge de {best_margin_seg['Profit_Margin']:.1f}%. "
               f"Augmenter les efforts marketing sur ce segment."
    })

    q1 = df[df["Month"].isin([1,2,3])]["Sales"].sum()
    q4 = df[df["Month"].isin([10,11,12])]["Sales"].sum()
    if q4 > q1 * 1.2:
        alerts.append({
            "type": "info", "icon": "🔵",
            "title": "Pic saisonnier T4 détecté",
            "msg": f"Le T4 génère {((q4/q1)-1)*100:.0f}% de plus que le T1. "
                   f"Augmenter les stocks en Octobre-Novembre."
        })

    insights["alerts"] = alerts

    # ── Recommandations ──
    insights["recommendations"] = [
        {
            "icon": "📉", "category": "Remises", "priority": "Haute",
            "title": "Optimiser la politique de remises",
            "desc": f"Les remises >20% réduisent significativement la marge. "
                    f"Remise moyenne : {df['Discount'].mean()*100:.1f}%. Plafonner à 15%."
        },
        {
            "icon": "📦", "category": "Stock", "priority": "Haute",
            "title": f"Renforcer les stocks avant {peak_month['Month_Name']}",
            "desc": f"{peak_month['Month_Name']} est le mois le plus fort "
                    f"(€{peak_month['Sales']:,.0f}). Anticiper 6-8 semaines à l'avance."
        },
        {
            "icon": "🎯", "category": "Marketing", "priority": "Moyenne",
            "title": f"Campagne promotionnelle en {weak_month['Month_Name']}",
            "desc": f"{weak_month['Month_Name']} affiche les ventes les plus faibles "
                    f"(€{weak_month['Sales']:,.0f}). Lancer des offres spéciales."
        },
        {
            "icon": "🌍", "category": "Régions", "priority": "Moyenne",
            "title": f"Développer la région {insights['weak_region']}",
            "desc": f"La région {insights['weak_region']} sous-performe. "
                    f"Envisager des partenariats locaux ou une force commerciale dédiée."
        },
        {
            "icon": "💎", "category": "Segments", "priority": "Haute",
            "title": f"Fidéliser le segment {insights['best_segment']}",
            "desc": f"{insights['best_segment']} est le segment le plus rentable. "
                    f"Investir dans des programmes de fidélité et des offres premium."
        },
        {
            "icon": "🚀", "category": "Produits", "priority": "Basse",
            "title": f"Développer la catégorie {insights['weak_category']}",
            "desc": f"{insights['weak_category']} présente le plus faible volume. "
                    f"Étudier l'élargissement de la gamme et la visibilité en ligne."
        },
    ]

    # ── Tendance YoY ──
    yearly = df.groupby("Year")["Sales"].sum()
    if len(yearly) >= 2:
        insights["yoy_growth"] = round((yearly.iloc[-1]-yearly.iloc[-2])/yearly.iloc[-2]*100, 1)
    else:
        insights["yoy_growth"] = 0

    return insights