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
st.caption("Choisir un modèle (Ridge ou Hybride Holt-Winters + Ridge) → charger Excel → prédire jusqu'à l'année choisie.")


# ============================================================
# HELPERS (DATA)
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

    # date mensuelle (début de mois)
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
    df_mensuel.index = df_mensuel.index.to_period("M").to_timestamp()  # MS

    return df, df_mensuel


def months_between_exclusive(start_month: pd.Timestamp, end_month: pd.Timestamp) -> int:
    sp = start_month.to_period("M")
    ep = end_month.to_period("M")
    return (ep - sp).n


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_ridge_artifacts():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "ridge" / "ridge_best.joblib"
    meta_path = base_dir / "models" / "ridge" / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Ridge modèle introuvable: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Ridge meta introuvable: {meta_path}")

    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta


@st.cache_resource
def load_hw_ridge_artifacts():
    base_dir = Path(__file__).resolve().parent
    folder = base_dir / "models" / "hw_ridge"

    hw_model_path = folder / "hw_model.joblib"
    ridge_resid_path = folder / "hw_ridge_resid.joblib"
    meta_path = folder / "hw_hybrid_meta.json"

    missing = [p for p in [hw_model_path, ridge_resid_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Artefacts Hybride manquants: {', '.join([str(p) for p in missing])}")

    hw_model = joblib.load(hw_model_path)
    ridge_resid = joblib.load(ridge_resid_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return hw_model, ridge_resid, meta


# ============================================================
# FORECASTING
# ============================================================
def ridge_forecast_next_months(model, df_mensuel: pd.DataFrame, months_ahead: int) -> pd.DataFrame:
    """Ridge récursif basé sur lags uniquement."""
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


def hw_ridge_forecast_next_months(hw_model, ridge_resid, df_mensuel: pd.DataFrame, months_ahead: int, meta: dict) -> pd.DataFrame:
    """
    Hybride = Forecast Holt-Winters + Ridge(residus).
    Hypothèses:
      - hw_model a une méthode .forecast(steps=months_ahead) OU .predict(...)
      - ridge_resid attend les mêmes features de résidus que pendant l'entraînement
      - meta contient "features_resid" et éventuellement "seasonal_periods" etc.
    """
    history = df_mensuel.copy()
    last_date = history.index.max().to_period("M").to_timestamp()
    start_future = (last_date.to_period("M") + 1).to_timestamp()
    future_index = pd.date_range(start=start_future, periods=months_ahead, freq="MS")

    # 1) Prévision HW (base)
    # Compatible statsmodels Holt-Winters fitted model: forecast(steps)
    try:
        base_hw = hw_model.forecast(steps=months_ahead)
    except Exception:
        # fallback
        base_hw = hw_model.predict(start=len(history), end=len(history) + months_ahead - 1)

    base_hw = pd.Series(base_hw, index=future_index, name="base_hw").astype(float)

    # 2) Résidus via Ridge (récursif sur history, si features dépend du tonnage passé)
    features_resid = meta.get("features_resid", ["lag_1", "lag_12", "roll_mean_3"])
    preds = []

    for d in future_index:
        prev1 = (d.to_period("M") - 1).to_timestamp()
        prev2 = (d.to_period("M") - 2).to_timestamp()
        prev3 = (d.to_period("M") - 3).to_timestamp()
        prev12 = (d.to_period("M") - 12).to_timestamp()

        # mêmes features lags (adaptable si tu as d'autres features de résidus)
        lag_1 = float(history.loc[prev1, "tonnage"]) if prev1 in history.index else float(history["tonnage"].iloc[-1])
        lag_12 = float(history.loc[prev12, "tonnage"]) if prev12 in history.index else float(history["tonnage"].tail(12).mean())

        vals = []
        for pm in [prev1, prev2, prev3]:
            if pm in history.index:
                vals.append(float(history.loc[pm, "tonnage"]))
        roll_mean_3 = float(np.mean(vals)) if len(vals) == 3 else float(history["tonnage"].tail(3).mean())

        X_resid = pd.DataFrame({"lag_1": [lag_1], "lag_12": [lag_12], "roll_mean_3": [roll_mean_3]}, index=[d])
        resid_hat = float(ridge_resid.predict(X_resid)[0])

        yhat = float(base_hw.loc[d] + resid_hat)
        preds.append(yhat)

        # injection pour que les lags évoluent avec la trajectoire hybride
        history.loc[d, "tonnage"] = yhat

    out = pd.DataFrame({"date_mois": future_index, "prediction_tonnage": preds})
    out["date_mois_str"] = out["date_mois"].dt.strftime("%Y-%m")
    return out


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Paramètres")
    model_choice = st.selectbox("Modèle", ["Ridge (joblib)", "Hybride Holt-Winters + Ridge (joblib)"])
    uploaded = st.file_uploader("Charge le fichier Excel (.xlsx)", type=["xlsx"])
    target_year = st.number_input("Année à prédire", min_value=2025, max_value=2100, value=2027, step=1)

    st.divider()
    st.caption("Feuille attendue: 'Feuil1'. Colonnes attendues: Année, Mois, Somme de Tonne (au minimum).")


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

st.subheader("🧾 Série mensuelle (tonnage total)")
hist_show = df_mensuel.copy()
hist_show.index = hist_show.index.strftime("%Y-%m")
st.dataframe(hist_show.tail(36), use_container_width=True)

last_obs = df_mensuel.index.max().to_period("M").to_timestamp()
end_target = pd.Timestamp(f"{int(target_year)}-12-01").to_period("M").to_timestamp()

if end_target <= last_obs:
    st.warning(f"⚠️ Ton historique va déjà jusqu'à {last_obs:%Y-%m}. Choisis une année > {last_obs.year}.")
    st.stop()

months_ahead = months_between_exclusive(last_obs, end_target)
st.info(f"Horizon: **{months_ahead} mois** (de {last_obs:%Y-%m} → {end_target:%Y-%m})")

# 1) Prévisions jusqu'à end_target
try:
    if model_choice.startswith("Ridge"):
        ridge_model, ridge_meta = load_ridge_artifacts()
        best_alpha = ridge_meta.get("best_alpha", None)
        if best_alpha is not None:
            st.caption(f"Ridge best_alpha (notebook): {best_alpha:.6f}")
        pred_all = ridge_forecast_next_months(ridge_model, df_mensuel, months_ahead)

    else:
        hw_model, ridge_resid, hw_meta = load_hw_ridge_artifacts()
        st.caption("Hybride: Holt-Winters + Ridge(residus)")
        pred_all = hw_ridge_forecast_next_months(hw_model, ridge_resid, df_mensuel, months_ahead, hw_meta)

except Exception as e:
    st.error(f"❌ Impossible de charger / utiliser le modèle: {e}")
    st.stop()

# 2) Extraire uniquement Jan..Dec de l'année cible
target_months = pd.date_range(start=f"{int(target_year)}-01-01", end=f"{int(target_year)}-12-01", freq="MS")
pred_year = pred_all.set_index("date_mois").reindex(target_months).reset_index().rename(columns={"index": "date_mois"})
pred_year["date_mois_str"] = pred_year["date_mois"].dt.strftime("%Y-%m")

missing = pred_year["prediction_tonnage"].isna().sum()
if missing > 0:
    st.warning(f"⚠️ Il manque {missing} mois sur {target_year} (horizon insuffisant).")

c1, c2 = st.columns([1.2, 1])

with c1:
    st.subheader(f"📋 Prédictions — {int(target_year)}")
    st.dataframe(pred_year[["date_mois_str", "prediction_tonnage"]], use_container_width=True)

    csv_bytes = pred_year[["date_mois_str", "prediction_tonnage"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger (CSV)", data=csv_bytes, file_name=f"predictions_{int(target_year)}.csv", mime="text/csv")

with c2:
    st.subheader("📉 Courbe des prédictions")
    st.line_chart(pred_year.set_index("date_mois")[["prediction_tonnage"]], height=320)

st.divider()
st.subheader("🔎 Données brutes (aperçu)")
st.dataframe(df_brut.head(25), use_container_width=True)
