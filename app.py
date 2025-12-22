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
st.set_page_config(page_title="Prévisions annuelles - PAA", layout="wide")
st.title("📈 Prévisions annuelles (12 mois) — PAA")
st.caption("Excel → agrégation mensuelle → prévision (Ridge OU Hybride Holt-Winters + Ridge).")

# ============================================================
# HELPERS DATA
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

    df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
    df = df.dropna(subset=["annee"])
    df["annee"] = df["annee"].astype(int)

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

    df["tonnage"] = pd.to_numeric(df["tonnage"], errors="coerce")
    df = df.dropna(subset=["tonnage"])

    df["date_mois"] = pd.to_datetime(
        df["annee"].astype(str) + "-" + df["mois"].astype(str) + "-01",
        errors="coerce"
    )
    df = df.dropna(subset=["date_mois"])

    df_mensuel = (
        df.groupby("date_mois")["tonnage"]
          .sum()
          .to_frame()
          .sort_index()
    )
    df_mensuel.index = df_mensuel.index.to_period("M").to_timestamp()  # MS

    return df, df_mensuel

def months_between_exclusive(start_month: pd.Timestamp, end_month: pd.Timestamp) -> int:
    sp = start_month.to_period("M")
    ep = end_month.to_period("M")
    return (ep - sp).n  # exclu start, inclu end quand on génère depuis start+1

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_artifacts(model_choice: str):
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"

    if model_choice == "Ridge (baseline)":
        mdir = models_dir / "ridge"
        model_path = mdir / "model.joblib"
        meta_path = mdir / "meta.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Artefacts Ridge manquants dans: {mdir}")
        model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"type": "ridge", "model": model, "meta": meta}

    if model_choice == "Hybride Holt-Winters + Ridge":
        mdir = models_dir / "hw_ridge"
        hw_path = mdir / "hw_fit.joblib"
        ridge_path = mdir / "ridge_resid.joblib"
        meta_path = mdir / "meta.json"
        if not hw_path.exists() or not ridge_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Artefacts Hybride manquants dans: {mdir}")
        hw_fit = joblib.load(hw_path)
        ridge_resid = joblib.load(ridge_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"type": "hw_ridge", "hw_fit": hw_fit, "ridge_resid": ridge_resid, "meta": meta}

    raise ValueError("Choix de modèle invalide")

# ============================================================
# FORECAST: RIDGE baseline (on tonnage)
# ============================================================
def forecast_ridge_tonnage(model, df_mensuel: pd.DataFrame, months_ahead: int) -> pd.DataFrame:
    history = df_mensuel.copy()
    last_date = history.index.max().to_period("M").to_timestamp()
    start_future = (last_date.to_period("M") + 1).to_timestamp()
    future_index = pd.date_range(start=start_future, periods=months_ahead, freq="MS")

    preds = []
    for d in future_index:
        prev1 = (d.to_period("M") - 1).to_timestamp()
        prev2 = (d.to_period("M") - 2).to_timestamp()
        prev3 = (d.to_period("M") - 3).to_timestamp()
        prev12 = (d.to_period("M") - 12).to_timestamp()

        lag_1 = float(history.loc[prev1, "tonnage"]) if prev1 in history.index else float(history["tonnage"].iloc[-1])
        lag_12 = float(history.loc[prev12, "tonnage"]) if prev12 in history.index else float(history["tonnage"].tail(12).mean())

        vals = []
        for pm in [prev1, prev2, prev3]:
            if pm in history.index:
                vals.append(float(history.loc[pm, "tonnage"]))
        roll_mean_3 = float(np.mean(vals)) if len(vals) == 3 else float(history["tonnage"].tail(3).mean())

        X_future = pd.DataFrame({"lag_1": [lag_1], "lag_12": [lag_12], "roll_mean_3": [roll_mean_3]}, index=[d])
        yhat = float(model.predict(X_future)[0])

        preds.append(yhat)
        history.loc[d, "tonnage"] = yhat

    out = pd.DataFrame({"date_mois": future_index, "prediction_tonnage": preds})
    out["date_mois_str"] = out["date_mois"].dt.strftime("%Y-%m")
    return out

# ============================================================
# FORECAST: HYBRID HW + Ridge (ridge predicts residuals)
# ============================================================
def _compute_hw_residual_series(df_mensuel: pd.DataFrame, hw_fit) -> pd.Series:
    # fittedvalues index peut différer; on aligne sur l'index de df_mensuel
    fitted = pd.Series(hw_fit.fittedvalues)
    fitted.index = pd.to_datetime(fitted.index).to_period("M").to_timestamp()
    fitted = fitted.reindex(df_mensuel.index)
    resid = (df_mensuel["tonnage"] - fitted).dropna()
    resid.name = "resid_hw"
    return resid

def forecast_hybrid_hw_ridge(hw_fit, ridge_resid, df_mensuel: pd.DataFrame, months_ahead: int) -> pd.DataFrame:
    """
    Prévision = base Holt-Winters (forecast) + résidus prédits par Ridge (récursif sur résidus).
    """
    last_date = df_mensuel.index.max().to_period("M").to_timestamp()
    start_future = (last_date.to_period("M") + 1).to_timestamp()
    future_index = pd.date_range(start=start_future, periods=months_ahead, freq="MS")

    # 1) Base Holt-Winters
    base_forecast = pd.Series(hw_fit.forecast(months_ahead))
    base_forecast.index = future_index
    base_forecast.name = "base_hw"

    # 2) Historique des résidus HW (sur passé)
    resid_hist = _compute_hw_residual_series(df_mensuel, hw_fit).to_frame()

    # 3) Prévoir résidus récursivement
    resid_preds = []
    for d in future_index:
        prev1 = (d.to_period("M") - 1).to_timestamp()
        prev2 = (d.to_period("M") - 2).to_timestamp()
        prev3 = (d.to_period("M") - 3).to_timestamp()
        prev12 = (d.to_period("M") - 12).to_timestamp()

        # lags sur résidus
        lag_1 = float(resid_hist.loc[prev1, "resid_hw"]) if prev1 in resid_hist.index else float(resid_hist["resid_hw"].iloc[-1])
        lag_12 = float(resid_hist.loc[prev12, "resid_hw"]) if prev12 in resid_hist.index else float(resid_hist["resid_hw"].tail(12).mean())

        vals = []
        for pm in [prev1, prev2, prev3]:
            if pm in resid_hist.index:
                vals.append(float(resid_hist.loc[pm, "resid_hw"]))
        roll_mean_3 = float(np.mean(vals)) if len(vals) == 3 else float(resid_hist["resid_hw"].tail(3).mean())

        Xr = pd.DataFrame({"lag_1": [lag_1], "lag_12": [lag_12], "roll_mean_3": [roll_mean_3]}, index=[d])
        rhat = float(ridge_resid.predict(Xr)[0])

        resid_preds.append(rhat)
        resid_hist.loc[d, "resid_hw"] = rhat

    resid_forecast = pd.Series(resid_preds, index=future_index, name="resid_pred")
    hybrid = base_forecast + resid_forecast

    out = pd.DataFrame({
        "date_mois": future_index,
        "base_hw": base_forecast.values,
        "resid_pred": resid_forecast.values,
        "prediction_tonnage": hybrid.values
    })
    out["date_mois_str"] = out["date_mois"].dt.strftime("%Y-%m")
    return out

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Paramètres")
    model_choice = st.selectbox(
        "Choisir le modèle",
        ["Ridge (baseline)", "Hybride Holt-Winters + Ridge"]
    )
    uploaded = st.file_uploader("Charge le fichier Excel (.xlsx)", type=["xlsx"])
    target_year = st.number_input("Année à prédire", min_value=2025, max_value=2100, value=2027, step=1)

    st.divider()
    st.caption("Feuille: 'Feuil1'. Colonnes minimales: Année, Mois, Somme de Tonne.")

# ============================================================
# APP FLOW
# ============================================================
if uploaded is None:
    st.info("⬅️ Charge ton fichier Excel pour démarrer.")
    st.stop()

try:
    df_brut, df_mensuel = load_excel_and_build_monthly_series(uploaded)
except Exception as e:
    st.error(f"Erreur Excel: {e}")
    st.stop()

# Historique
st.subheader("🧾 Série mensuelle (tonnage total) construite depuis l'Excel")
hist_show = df_mensuel.copy()
hist_show.index = hist_show.index.strftime("%Y-%m")
st.dataframe(hist_show.tail(36), use_container_width=True)

# Horizon jusqu'à décembre target_year
last_obs = df_mensuel.index.max().to_period("M").to_timestamp()
end_target = pd.Timestamp(f"{int(target_year)}-12-01").to_period("M").to_timestamp()

if end_target <= last_obs:
    st.warning(f"⚠️ Historique déjà jusqu'à {last_obs:%Y-%m}. Choisis une année > {last_obs.year}.")
    st.stop()

months_ahead = months_between_exclusive(last_obs, end_target)
st.info(f"Horizon: **{months_ahead} mois** (de {last_obs:%Y-%m} → {end_target:%Y-%m})")

# Charger artefacts
try:
    art = load_artifacts(model_choice)
except Exception as e:
    st.error(f"❌ Impossible de charger les artefacts du modèle: {e}")
    st.stop()

# Prédire
if art["type"] == "ridge":
    pred_all = forecast_ridge_tonnage(art["model"], df_mensuel, months_ahead=months_ahead)
else:
    pred_all = forecast_hybrid_hw_ridge(art["hw_fit"], art["ridge_resid"], df_mensuel, months_ahead=months_ahead)

# Garder uniquement l'année cible (12 mois)
target_months = pd.date_range(start=f"{int(target_year)}-01-01", end=f"{int(target_year)}-12-01", freq="MS")
pred_year = pred_all.set_index("date_mois").reindex(target_months).reset_index().rename(columns={"index": "date_mois"})
pred_year["date_mois_str"] = pred_year["date_mois"].dt.strftime("%Y-%m")

missing = pred_year["prediction_tonnage"].isna().sum()
if missing > 0:
    st.warning(
        f"⚠️ Il manque {missing} mois sur {target_year}. "
        "Vérifie que l'horizon couvre bien jusqu'à décembre."
    )

# Affichage
c1, c2 = st.columns([1.2, 1])

with c1:
    st.subheader(f"📋 Prédictions — {int(target_year)} ({model_choice})")
    cols = ["date_mois_str", "prediction_tonnage"]
    if "base_hw" in pred_year.columns:
        cols = ["date_mois_str", "base_hw", "resid_pred", "prediction_tonnage"]
    st.dataframe(pred_year[cols], use_container_width=True)

    csv_bytes = pred_year[cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger (CSV)",
        data=csv_bytes,
        file_name=f"predictions_{model_choice.replace(' ', '_')}_{int(target_year)}.csv",
        mime="text/csv"
    )

with c2:
    st.subheader("📉 Courbe des prédictions")
    st.line_chart(pred_year.set_index("date_mois")[["prediction_tonnage"]], height=320)

st.divider()
st.subheader("🔎 Aperçu des données brutes")
st.dataframe(df_brut.head(25), use_container_width=True)
