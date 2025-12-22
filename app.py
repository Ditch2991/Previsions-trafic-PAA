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


def months_ahead_to_reach(last_obs_ms: pd.Timestamp, end_target_ms: pd.Timestamp) -> int:
    """
    Nombre de mois à prédire à partir du mois suivant last_obs_ms
    pour atteindre end_target_ms inclus.
    Ex: last_obs=2025-07-01, end_target=2026-12-01 -> 17 mois (Aug..Dec).
    """
    lp = last_obs_ms.to_period("M")
    ep = end_target_ms.to_period("M")
    # mois futurs = (ep - lp) car on démarre le mois suivant
    return (ep - lp).n


def forecast_next_months(model, df_mensuel: pd.DataFrame, months_ahead: int) -> pd.DataFrame:
    """
    Prévisions récursives sur months_ahead mois, à partir du dernier mois observé.
    """
    history = df_mensuel.copy()
    last_date = history.index.max().to_period("M").to_timestamp()  # MS

    # On prédit à partir du mois suivant last_date, sur months_ahead périodes
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

        X_future = pd.DataFrame({
            "lag_1": [lag_1],
            "lag_12": [lag_12],
            "roll_mean_3": [roll_mean_3],
        }, index=[d])

        yhat = float(model.predict(X_future)[0])
        preds.append(yhat)
        history.loc[d, "tonnage"] = yhat

    out = pd.DataFrame({"date_mois": future_index, "prediction_tonnage": preds})
    out["date_mois_str"] = out["date_mois"].dt.strftime("%Y-%m")
    return out


# ============================================================
# Charger modèle + meta
# ============================================================
try:
    model, meta = load_model_and_meta()
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
    if BEST_ALPHA is not None:
        st.write(f"**Best alpha (notebook)** : {float(BEST_ALPHA):.6f}")
    else:
        st.write("**Best alpha** : (non trouvé)")

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

# Dates de référence (toujours MS)
last_obs = df_mensuel.index.max().to_period("M").to_timestamp()
end_target = pd.Timestamp(f"{int(target_year)}-12-01").to_period("M").to_timestamp()

if end_target <= last_obs:
    st.warning(f"⚠️ Ton historique va déjà jusqu'à {last_obs:%Y-%m}. Choisis une année > {last_obs.year}.")
    st.stop()

months_ahead = months_ahead_to_reach(last_obs, end_target)
st.info(f"Horizon calculé: **{months_ahead} mois** (de {last_obs:%Y-%m} → {end_target:%Y-%m})")

# 1) Prévision jusqu'à Décembre de l'année cible (incluse)
pred_all = forecast_next_months(model, df_mensuel, months_ahead=months_ahead)

# 2) Construire table complète = historique + futures
pred_all = pred_all.set_index("date_mois")
full_series = pd.concat([df_mensuel, pred_all.rename(columns={"prediction_tonnage": "tonnage"})[["tonnage"]]], axis=0)

# 3) Extraire exactement Jan..Dec de l'année cible (12 mois garantis)
target_months = pd.date_range(start=f"{int(target_year)}-01-01", end=f"{int(target_year)}-12-01", freq="MS")
pred_year = full_series.reindex(target_months).rename(columns={"tonnage": "prediction_tonnage"}).reset_index()
pred_year = pred_year.rename(columns={"index": "date_mois"})
pred_year["date_mois_str"] = pred_year["date_mois"].dt.strftime("%Y-%m")

# Si jamais certains mois sont encore NaN, c'est que l'horizon n'a pas atteint ces mois (ne devrait plus arriver)
missing = int(pred_year["prediction_tonnage"].isna().sum())
if missing > 0:
    st.error(
        f"❌ Il manque encore {missing} mois sur {target_year}. "
        "Vérifie que le dernier mois observé (dans Excel) est bien interprété correctement."
    )
    st.stop()

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
