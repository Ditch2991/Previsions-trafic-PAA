# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.pipeline import Pipeline  # juste pour le type, ton modèle joblib est un Pipeline


# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(page_title="Prévisions annuelles - Ridge (PAA)", layout="wide")
st.title("📈 Prévisions annuelles (12 mois) — Modèle Ridge (Pipeline sauvegardé)")
st.caption("Excel (données brutes) → agrégation mensuelle → features (lag_1, lag_12, roll_mean_3) → prédictions.")


# ============================================================
# HELPERS
# ============================================================
def _normalize_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def load_excel_and_build_monthly_series(file):
    """
    Charge Excel (Feuil1), renomme les colonnes, gère mois en texte (Janvier...),
    construit une série mensuelle du tonnage (somme).
    Retourne: (df_brut_renomme, df_mensuel)
    """
    df = pd.read_excel(file, sheet_name="Feuil1")

    # Renommage des colonnes (comme ton notebook)
    df = df.rename(columns={
        "Sens trafic 2": "sens_trafic",
        "Transbordement": "transbordement",
        "Année": "annee",
        "Mois": "mois",
        "Nom Navire": "nom_navire",
        "Type Navire": "type_navire",
        "Produits des Tab Statistiques": "produit",
        "Somme de Tonne": "tonnage"
    })

    required_cols = ["annee", "mois", "tonnage"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans Excel: {missing}")

    df = df.dropna(subset=["annee", "mois", "tonnage"]).copy()

    # Année robuste
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
    df = df.dropna(subset=["annee"])
    df["annee"] = df["annee"].astype(int)

    # Mois robuste (numérique ou texte FR)
    mois_raw = _normalize_str(df["mois"])

    mois_map = {
        "janvier": 1, "janv": 1, "jan": 1,
        "février": 2, "fevrier": 2, "févr": 2, "fevr": 2, "fév": 2, "fev": 2,
        "mars": 3,
        "avril": 4, "avr": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7, "juil": 7,
        "août": 8, "aout": 8,
        "septembre": 9, "sept": 9, "sep": 9,
        "octobre": 10, "oct": 10,
        "novembre": 11, "nov": 11,
        "décembre": 12, "decembre": 12, "déc": 12, "dec": 12
    }

    mois_num = pd.to_numeric(mois_raw, errors="coerce")
    mois_mapped = mois_raw.map(mois_map)
    df["mois"] = mois_num.fillna(mois_mapped)

    if df["mois"].isna().any():
        bad_examples = df.loc[df["mois"].isna(), "mois"].astype(str).head(10).tolist()
        raise ValueError(f"Mois non reconnus (exemples): {bad_examples}. Corrige l'Excel ou complète le mapping.")

    df["mois"] = df["mois"].astype(int)

    # Tonnage robuste
    df["tonnage"] = pd.to_numeric(df["tonnage"], errors="coerce")
    df = df.dropna(subset=["tonnage"])

    # Date mensuelle
    df["date_mois"] = pd.to_datetime(
        df["annee"].astype(str) + "-" + df["mois"].astype(str) + "-01",
        errors="coerce"
    )
    df = df.dropna(subset=["date_mois"])

    # Agrégation mensuelle (somme)
    df_mensuel = (
        df.groupby("date_mois")["tonnage"]
          .sum()
          .to_frame()
          .sort_index()
    )
    df_mensuel.index = df_mensuel.index.to_period("M").to_timestamp()

    return df, df_mensuel


def make_ml_frame(df_mensuel: pd.DataFrame) -> pd.DataFrame:
    df_ml = df_mensuel.copy()
    df_ml["lag_1"] = df_ml["tonnage"].shift(1)
    df_ml["lag_12"] = df_ml["tonnage"].shift(12)
    df_ml["roll_mean_3"] = df_ml["tonnage"].rolling(3).mean()
    return df_ml


@st.cache_resource
def load_model_and_meta():
    """
    Charge le modèle Ridge (Pipeline scaler+ridge) et meta.json depuis le dossier models/
    Compatible Streamlit Cloud (chemins relatifs au fichier app.py).
    """
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "ridge_best.joblib"
    meta_path = base_dir / "models" / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta introuvable: {meta_path}")

    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta


def forecast_year_recursive(model: Pipeline, df_mensuel: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Prévision récursive avec features: lag_1, lag_12, roll_mean_3
    IMPORTANT: les features doivent être les mêmes que lors de l'entraînement du modèle sauvegardé.
    """
    history = df_mensuel.copy()

    start = pd.Timestamp(f"{target_year}-01-01").to_period("M").to_timestamp()
    end = pd.Timestamp(f"{target_year}-12-01").to_period("M").to_timestamp()
    future_index = pd.date_range(start=start, end=end, freq="MS")

    preds = []
    for d in future_index:
        prev1 = (d.to_period("M") - 1).to_timestamp()
        prev2 = (d.to_period("M") - 2).to_timestamp()
        prev3 = (d.to_period("M") - 3).to_timestamp()
        prev12 = (d.to_period("M") - 12).to_timestamp()

        lag_1 = float(history.loc[prev1, "tonnage"]) if prev1 in history.index else float(history["tonnage"].iloc[-1])
        lag_12 = float(history.loc[prev12, "tonnage"]) if prev12 in history.index else float(history["tonnage"].tail(12).mean())

        last3 = []
        for pm in [prev1, prev2, prev3]:
            if pm in history.index:
                last3.append(float(history.loc[pm, "tonnage"]))
        roll_mean_3 = float(np.mean(last3)) if len(last3) == 3 else float(history["tonnage"].tail(3).mean())

        X_future = pd.DataFrame({
            "lag_1": [lag_1],
            "lag_12": [lag_12],
            "roll_mean_3": [roll_mean_3]
        }, index=[d])

        yhat = float(model.predict(X_future)[0])
        preds.append(yhat)
        history.loc[d, "tonnage"] = yhat

    out = pd.DataFrame({
        "date_mois": future_index,
        "prediction_tonnage": preds
    })
    out["date_mois_str"] = out["date_mois"].dt.strftime("%Y-%m")
    return out


# ============================================================
# LOAD MODEL (cached)
# ============================================================
try:
    model, meta = load_model_and_meta()
    FEATURES = meta.get("features", ["lag_1", "lag_12", "roll_mean_3"])
    BEST_ALPHA = meta.get("best_alpha", None)
except Exception as e:
    st.error(f"❌ Impossible de charger le modèle/meta depuis le dossier models/. Détail: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Paramètres")
    uploaded = st.file_uploader("Charge le fichier Excel (.xlsx)", type=["xlsx"])
    target_year = st.number_input("Année à prédire", min_value=1900, max_value=2100, value=2027, step=1)

    if BEST_ALPHA is not None:
        st.info(f"✅ Modèle chargé. Best alpha (RandomizedSearch) = {BEST_ALPHA:.6f}")
    else:
        st.info("✅ Modèle chargé. (best_alpha non trouvé dans meta.json)")

    st.divider()
    st.caption("Feuille attendue: 'Feuil1'. Colonnes attendues: Année, Mois, Somme de Tonne (au minimum).")


# ============================================================
# APP FLOW
# ============================================================
if uploaded is None:
    st.info("⬅️ Charge ton fichier Excel dans la barre latérale pour démarrer.")
    st.stop()

try:
    df_brut, df_mensuel = load_excel_and_build_monthly_series(uploaded)
except Exception as e:
    st.error(f"Erreur lors du chargement/formatage du fichier Excel: {e}")
    st.stop()

st.subheader("🧾 Série mensuelle (tonnage total) construite depuis l'Excel")
hist_show = df_mensuel.copy()
hist_show.index = hist_show.index.strftime("%Y-%m")
st.dataframe(hist_show.tail(36), use_container_width=True)

if df_mensuel.shape[0] < 15:
    st.warning("⚠️ Série courte : idéalement ≥ 15 mois pour lag_12 et roll_mean_3. Résultats potentiellement instables.")

# On construit les features uniquement pour vérifier qu'on a assez d'historique (dropna)
df_ml = make_ml_frame(df_mensuel)
df_ml_clean = df_ml.dropna(subset=FEATURES + ["tonnage"]).copy()

if df_ml_clean.empty:
    st.error("Pas assez de données après création des lags/rolling (dropna). Ajoute plus d'historique.")
    st.stop()

# ICI: on n'entraîne PLUS. On utilise le modèle sauvegardé.
pred_df = forecast_year_recursive(model, df_mensuel, int(target_year))

c1, c2 = st.columns([1.2, 1])

with c1:
    st.subheader("📋 Prédictions des 12 mois")
    st.dataframe(pred_df[["date_mois_str", "prediction_tonnage"]], use_container_width=True)

    csv_bytes = pred_df[["date_mois_str", "prediction_tonnage"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger les prédictions (CSV)",
        data=csv_bytes,
        file_name=f"predictions_ridge_{target_year}.csv",
        mime="text/csv"
    )

with c2:
    st.subheader("📉 Courbe des prédictions")
    st.line_chart(pred_df.set_index("date_mois")[["prediction_tonnage"]], height=320)

st.divider()
st.subheader("🔎 Aperçu des données brutes (après renommage)")
st.dataframe(df_brut.head(25), use_container_width=True)

with st.expander("🛠️ Infos modèle"):
    st.write("Features attendues par le modèle:", FEATURES)
    if BEST_ALPHA is not None:
        st.write("Best alpha:", BEST_ALPHA)
    st.write("Meta:", meta)
