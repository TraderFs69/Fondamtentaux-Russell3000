# app_funda_russell3000_yahoo.py
import os, io, time, math, requests
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Yahoo fundamentals via yahooquery (plus riche que yfinance)
try:
    from yahooquery import Ticker
except ImportError:
    Ticker = None

# ------------------------
# SETUP
# ------------------------
st.set_page_config(page_title="Russell 3000 — Score Fondamental", layout="wide")
st.title("📘 Russell 3000 — Score Fondamental")

load_dotenv()  # pas de clé requise pour Yahoo, mais on garde la compatibilité

if Ticker is None:
    st.error("Le paquet 'yahooquery' n'est pas installé.\n\n"
             "Installe :  \n"
             "`pip install yahooquery streamlit pandas numpy python-dotenv requests lxml yahooquery`")
    st.stop()

# ------------------------
# UNIVERSE : RUSSELL 3000 DEPUIS EXCEL
# ------------------------

# 👉 Ajuste ce chemin si ton script n'est pas au même endroit
# Ici on suppose un sous-dossier `sp500_builder` à côté du script.
RUSSELL_PATH = os.path.join("sp500_builder", "russell3000_constituents.xlsx")

@st.cache_data(ttl=3600)
def fetch_russell3000_universe(path: str = RUSSELL_PATH) -> pd.DataFrame:
    """
    Charge l'univers Russell 3000 depuis un fichier Excel.
    Le fichier peut avoir diverses colonnes (Symbol, Ticker, Name, Sector, Industry, etc.).
    On essaie de mapper proprement vers : ticker, name, sector, industry.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier Russell 3000 introuvable : {path}\n"
            f"Vérifie le chemin (RUSSELL_PATH)."
        )

    df = pd.read_excel(path)

    # Mapping automatique des colonnes
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if "symbol" in lc or "ticker" in lc:
            colmap[c] = "ticker"
        elif "security" in lc or "name" in lc or "company" in lc:
            colmap[c] = "name"
        elif "sector" in lc:
            colmap[c] = "sector"
        elif "sub" in lc and "industry" in lc:
            # Sub-Industry
            colmap[c] = "industry"
        elif "industry" in lc and "sector" not in lc:
            # fallback si une colonne "Industry" simple
            colmap[c] = "industry"

    df = df.rename(columns=colmap)

    if "ticker" not in df.columns:
        raise ValueError(
            "Impossible de trouver une colonne ticker/symbol dans le fichier "
            "(colonnes attendues : Symbol, Ticker, etc.)."
        )

    df["ticker"] = (
        df["ticker"].astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^\w\.\-]", "", regex=True)
    )

    keep_cols = [c for c in ["ticker", "name", "sector", "industry"] if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df

# ------------------------
# HELPERS
# ------------------------
def pctile_or_neutral(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s2.dropna().empty:
        return pd.Series(50.0, index=s.index)  # neutre si colonne vide
    return 100 * s2.fillna(s2.median()).rank(pct=True)

# ------------------------
# RÉCUP FUNDAMENTAUX YAHOO (yahooquery)
# ------------------------
YQ_CHUNK = 40  # batch pour éviter rate-limit

# Map Yahoo -> nos colonnes
YQ_FIELDS = {
    # Qualité
    "grossMargins": ("quality", "gross_margin"),            # marge brute %
    "profitMargins": ("quality", "net_margin"),             # marge nette %
    "returnOnEquity": ("quality", "roe"),                   # ROE %
    # Croissance
    "revenueGrowth": ("growth", "revenue_growth"),          # YoY
    "earningsQuarterlyGrowth": ("growth", "eps_growth"),    # proxy EPS growth
    # Solidité
    "debtToEquity": ("safety", "debt_to_equity"),           # %
    "currentRatio": ("safety", "current_ratio"),            # ratio
    "quickRatio": ("safety", "quick_ratio"),
    # Bonus info
    "trailingPE": ("info", "pe"),
    "priceToBook": ("info", "pb"),
    "beta": ("info", "beta"),
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Récupère par batch via yahooquery:
      - financial_data
      - summary_detail
      - key_stats  (en renfort si dispo)
    Renvoie 1 ligne par ticker avec nos colonnes cibles.
    """
    all_rows = []
    for i in range(0, len(tickers), YQ_CHUNK):
        batch = tickers[i:i+YQ_CHUNK]
        t = Ticker(batch, asynchronous=True)

        fd = t.financial_data or {}
        sd = t.summary_detail or {}
        ks = t.key_stats or {}

        for sym in batch:
            row = {"ticker": sym}

            def g(key):
                v = None
                if isinstance(fd.get(sym), dict):
                    v = fd[sym].get(key, None)
                if v is None and isinstance(sd.get(sym), dict):
                    v = sd[sym].get(key, None)
                if v is None and isinstance(ks.get(sym), dict):
                    v = ks[sym].get(key, None)
                return v

            for yq_key, (_, our_name) in YQ_FIELDS.items():
                row[our_name] = g(yq_key)

            all_rows.append(row)

        time.sleep(0.4)  # pause douce pour éviter le rate-limit

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return df

# ------------------------
# SCORING (score en 1ère colonne)
# ------------------------
def compute_score_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Sous-scores : listes de colonnes
    q_cols = ["gross_margin", "net_margin", "roe"]
    g_cols = ["revenue_growth", "eps_growth"]

    # D/E inversé (plus bas = mieux)
    inv_de = -pd.to_numeric(d["debt_to_equity"], errors="coerce")
    d["_inv_de"] = inv_de

    # Percentiles avec neutralisation si colonne vide
    q = [pctile_or_neutral(pd.to_numeric(d[c], errors="coerce")) for c in q_cols]
    quality = sum(q) / len(q)

    g = [pctile_or_neutral(pd.to_numeric(d[c], errors="coerce")) for c in g_cols]
    growth = sum(g) / len(g)

    s = [
        pctile_or_neutral(d["_inv_de"]),
        pctile_or_neutral(pd.to_numeric(d["current_ratio"], errors="coerce"))
    ]
    safety = sum(s) / len(s)

    score = 0.40 * quality + 0.35 * growth + 0.25 * safety

    out = pd.DataFrame({
        "score": score.round(2),  # ✅ score en premier
        "ticker": d["ticker"],
        "quality": quality.round(2),
        "growth": growth.round(2),
        "safety": safety.round(2),
        # Champs bruts utiles
        "gross_margin": d["gross_margin"],
        "net_margin": d["net_margin"],
        "roe": d["roe"],
        "revenue_growth": d["revenue_growth"],
        "eps_growth": d["eps_growth"],
        "debt_to_equity": d["debt_to_equity"],
        "current_ratio": d["current_ratio"],
        "quick_ratio": d.get("quick_ratio", np.nan),
        "pe": d.get("pe", np.nan),
        "pb": d.get("pb", np.nan),
        "beta": d.get("beta", np.nan),
    })
    return out

# ------------------------
# UI
# ------------------------
col1, col2, col3 = st.columns(3)
with col1:
    min_score = st.slider("Score minimal", 0, 100, 50, 1)
with col2:
    limit_names = st.number_input(
        "Limiter l'univers (0 = tous)", 
        min_value=0, 
        value=300,     # 👈 par défaut 300 au lieu de 100, car Russell 3000
        step=100
    )
with col3:
    show_raw = st.checkbox("Afficher un échantillon brut", value=False)

run_btn = st.button("🔎 Lancer l’analyse fondamentale (Yahoo)")

# ------------------------
# MAIN
# ------------------------
if run_btn:
    with st.spinner("Chargement de l'univers Russell 3000..."):
        universe = fetch_russell3000_universe()

    if limit_names and limit_names > 0:
        universe = universe.head(int(limit_names))
    st.write(f"Univers chargé : **{len(universe)}** tickers")

    tickers = universe["ticker"].tolist()

    with st.spinner("Récupération des fondamentaux Yahoo (batchs)..."):
        funda = fetch_yahoo_fundamentals(tickers)

    st.write(f"📦 Fondamentaux récupérés via Yahoo : **{len(funda)}** / {len(universe)}")

    if len(funda) == 0:
        st.warning("Aucune donnée récupérée via Yahoo. Réduis le nombre de tickers ou réessaie plus tard.")
        st.stop()

    if show_raw:
        st.subheader("Échantillon brut (Yahoo)")
        st.dataframe(funda.head(10))

    scored = compute_score_yahoo(funda)

    # Merge pour récupérer secteur/industrie et calculer les moyennes par secteur
    merged = universe.merge(scored, on="ticker", how="inner")
    out = (
        merged[merged["score"] >= min_score]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    # ------------------------
    # 🧮 Moyennes par secteur
    # ------------------------
    sector_cols = [
        "score", "quality", "growth", "safety",
        "gross_margin", "net_margin", "roe",
        "revenue_growth", "eps_growth",
        "debt_to_equity", "current_ratio"
    ]
    sector_summary = (
        merged[["sector"] + sector_cols]
        .groupby("sector", dropna=False)
        .mean(numeric_only=True)
        .round(2)
        .sort_values("score", ascending=False)
        .reset_index()
    )

    st.subheader("📋 Résultats (Yahoo fondamentaux) — Russell 3000")
    st.dataframe(out)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Exporter CSV — Résultats",
            data=out.to_csv(index=False),
            file_name="russell3000_fundamentals_yahoo_results.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "📥 Exporter CSV — Moyennes par secteur",
            data=sector_summary.to_csv(index=False),
            file_name="russell3000_fundamentals_yahoo_sector_summary.csv",
            mime="text/csv"
        )

    st.subheader("🏷️ Moyennes par secteur (score, sous-scores, ratios)")
    st.dataframe(sector_summary)
