# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(page_title="Prévisions annuelles - Ridge", layout="wide")
st.title("📈 Prévisions annuelles (12 mois) — Modèle Ridge sauvegardé (joblib)")
st.caption("Excel → agrégation mensuelle → features (lag_1, lag_12, roll_mean_3) → prédictions récursives.")


# ============================================================
# HELPERS
# ============================================================
def _normalize_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def load_excel_and_build_monthly_series(file):
    df = pd.read_excel(file, sheet_name="Feuil1")

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

    # année robuste
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
    df = df.dropna(subset=["annee"])
    df["annee"] = df["annee"].astype(int)

    # mois robuste (numérique ou FR)
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
        bad = df.loc[df["mois"].isna(), "mois"].astype(str).head(10).tolist()
        raise ValueError(f"Mois non reconnus (exemples): {bad}")

    df["mois"] = df["mois"].astype(int)

    # tonnage robuste
    df["tonnage"] = pd.to_numeric(df["tonnage"], errors="coerce")
    df = df.dropna(subset=["tonnage"])

    # date mensuelle
    df["date_mois"] = pd.to_datetime(
        df["annee"].astype(str) + "-" + df["mois"].astype(str) + "-01",
        errors="coerce"
    )
    df = df.dropna(subset=["date_mois"])

    # agrégation mensuelle
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


def forecast_until(model, df_mensuel: pd.DataFrame, months_ahead: int, features: list[str]) -> pd.DataFrame:
    """
    Prévisions récursives 'months_ahead' à partir du dernier mois observé.
    C'est cette logique qui permet de prédire 2027 puis 2040 différemment.
    """
    history = df_mensuel.copy()
    last_date = history.index.max()
    future_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=months_ahead, freq="MS")

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

    out = pd.DataFrame({"date_mois": future_index, "prediction_tonnage": preds})
    out["date_mois_str"] = out["date_mois"].dt.strftime("%Y-%m")
    return out


# ============================================================
# Charger modèle + meta (avant sidebar)
# ============================================================
try:
    model, meta = load_model_and_meta()
    FEATURES = meta.get("features", ["lag_1", "lag_12", "roll_mean_3"])
    BEST_ALPHA = meta.get("best_alpha", None)
except Exception as e:
    st.error(f"❌ Impossible de charger le modèle sauvegardé: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Paramètres")
    uploaded = st.file_uploader("Charge le fichier Excel (.xlsx)", type=["xlsx"])

    st.caption("Ton modèle est déjà entraîné et sauvegardé (joblib).")
    st.write(f"**Best alpha (notebook)** : {BEST_ALPHA}" if BEST_ALPHA is not None else "**Best alpha** : (non trouvé)")

    target_year = st.number_input("Année à prédire", min_value=2025, max_value=2100, value=2027, step=1)

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

# Calcul horizon = du mois suivant le dernier observé jusqu'à décembre de target_year
last_obs = df_mensuel.index.max()
end_target = pd.Timestamp(f"{int(target_year)}-12-01").to_period("M").to_timestamp()

if end_target <= last_obs:
    st.warning(f"⚠️ Ton historique va déjà jusqu'à {last_obs:%Y-%m}. Choisis une année > {last_obs.year}.")
    st.stop()

months_ahead = (end_target.to_period("M") - last_obs.to_period("M")).n

st.info(f"Horizon calculé: **{months_ahead} mois** (de {last_obs:%Y-%m} → {end_target:%Y-%m})")

# Construire df_ml juste pour contrôle data suffisante
df_ml = make_ml_frame(df_mensuel)
df_ml_clean = df_ml.dropna(subset=FEATURES + ["tonnage"]).copy()

if df_ml_clean.empty:
    st.error("Pas assez de données après création des lags/rolling. Ajoute plus d'historique.")
    st.stop()

# Prévision jusqu'à l'année cible
pred_all = forecast_until(model, df_mensuel, months_ahead=months_ahead, features=FEATURES)

# Extraire uniquement les 12 mois de l'année cible
pred_year = pred_all[pred_all["date_mois"].dt.year == int(target_year)].copy()

c1, c2 = st.columns([1.2, 1])

with c1:
    st.subheader(f"📋 Prédictions des 12 mois — {int(target_year)}")
    st.dataframe(pred_year[["date_mois_str", "prediction_tonnage"]], use_container_width=True)

    csv_bytes = pred_year[["date_mois_str", "prediction_tonnage"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger les prédictions (CSV)",
        data=csv_bytes,
        file_name=f"predictions_ridge_{int(target_year)}.csv",
        mime="text/csv"
    )

with c2:
    st.subheader("📉 Courbe des prédictions")
    st.line_chart(pred_year.set_index("date_mois")[["prediction_tonnage"]], height=320)

st.divider()
st.subheader("🔎 Aperçu des données brutes (après renommage)")
st.dataframe(df_brut.head(25), use_container_width=True)
