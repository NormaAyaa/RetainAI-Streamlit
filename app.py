"""
RetainAI — Streamlit App
Jalankan: streamlit run app.py
"""

import io
import warnings
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ─── Paths ────────────────────────────────────────────────────
BASE   = Path(__file__).parent
DATA   = BASE / "data" / "raw"
MODELS = BASE / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RetainAI — Employee Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Header */
.retain-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.retain-logo {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #00e5a0;
    letter-spacing: -0.5px;
}
.retain-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #6e7681;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Stat cards */
.stat-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.stat-green::before  { background: #00e5a0; }
.stat-blue::before   { background: #3d6eff; }
.stat-orange::before { background: #ff6b35; }
.stat-amber::before  { background: #f0b429; }

.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.val-green  { color: #00e5a0; }
.val-blue   { color: #3d6eff; }
.val-orange { color: #ff6b35; }
.val-amber  { color: #f0b429; }
.stat-sub { font-size: 11px; color: #6e7681; }

/* Risk badge */
.badge-high   { background: rgba(255,107,53,.12); color: #ff6b35; border: 1px solid rgba(255,107,53,.25); padding: 2px 10px; border-radius: 20px; font-size: 11px; font-family:'DM Mono',monospace; font-weight:600; }
.badge-medium { background: rgba(240,180,41,.12); color: #f0b429; border: 1px solid rgba(240,180,41,.25); padding: 2px 10px; border-radius: 20px; font-size: 11px; font-family:'DM Mono',monospace; font-weight:600; }
.badge-low    { background: rgba(0,229,160,.12);  color: #00e5a0; border: 1px solid rgba(0,229,160,.25);  padding: 2px 10px; border-radius: 20px; font-size: 11px; font-family:'DM Mono',monospace; font-weight:600; }

/* Section title */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Sidebar style */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #6e7681;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

/* Recommendation card */
.rec-card {
    background: rgba(0,229,160,.05);
    border: 1px solid rgba(0,229,160,.15);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    color: #c9d1d9;
    font-size: 13px;
    line-height: 1.5;
}

/* Risk factor bar */
.factor-bar {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 6px;
}
.factor-label { font-size: 12px; color: #8b949e; margin-bottom: 5px; }
.factor-name  { font-size: 13px; color: #e6edf3; font-weight: 600; margin-bottom: 5px; }
.factor-bar-bg { background: #21262d; border-radius: 3px; height: 6px; overflow: hidden; }
.factor-bar-fill { height: 100%; border-radius: 3px; }

/* Upload zone */
.upload-hint {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #6e7681;
    padding: 4px 0;
}

/* Info box */
.info-box {
    background: rgba(61,110,255,.08);
    border: 1px solid rgba(61,110,255,.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #a5b4fc;
    margin-bottom: 12px;
}

div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "model": None,
        "features": [],
        "trained_at": None,
        "metrics": {},
        "df_master": None,
        "df_candidates": None,
        "shap_explainer": None,
        "upload_success": None,
        "_nav": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (sama seperti backend)
# ═══════════════════════════════════════════════════════════════

FEATURE_LABELS = {
    "market_ratio":                "Kesenjangan Gaji vs Pasar",
    "avg_overtime":                "Rata-rata Overtime/Bulan",
    "satisfaction_score":          "Skor Kepuasan Kerja",
    "career_growth":               "Peluang Karir",
    "manager_relation":            "Hubungan dengan Manajer",
    "months_since_last_promotion": "Bulan Sejak Promosi Terakhir",
    "high_performer_no_promo":     "High Performer Tanpa Promosi",
    "work_life_balance":           "Work-Life Balance",
    "sat_trend":                   "Tren Kepuasan",
    "perf_score":                  "Skor Performa",
    "tenure_months":               "Masa Kerja (Bulan)",
    "num_prev_companies":          "Jumlah Perusahaan Sebelumnya",
    "avg_absent":                  "Rata-rata Absen/Bulan",
    "is_overloaded":               "Status Burnout",
    "distance_km":                 "Jarak Rumah ke Kantor",
}

FEATURE_COLS = [
    "age","is_female","edu_num","is_married","distance_km","num_prev_companies",
    "tenure_months","tenure_years",
    "market_ratio","avg_overtime","avg_absent","avg_late","is_overloaded",
    "satisfaction_score","work_life_balance","manager_relation","career_growth","sat_trend",
    "perf_score","months_since_last_promotion","has_promotion_plan","high_performer_no_promo",
]


def build_features(emp, salary=None, attendance=None, survey=None, performance=None):
    df = emp.copy()

    if "hire_date" in df.columns:
        df["hire_date"] = pd.to_datetime(df["hire_date"], errors="coerce")
        today = pd.Timestamp.today()
        df["tenure_months"] = ((today - df["hire_date"]).dt.days / 30).round(1)
        df["tenure_years"]  = (df["tenure_months"] / 12).round(2)
    else:
        df["tenure_months"] = 24.0
        df["tenure_years"]  = 2.0

    if salary is not None and not salary.empty:
        sal = (salary.sort_values("effective_date")
                     .groupby("employee_id").last().reset_index()
               [["employee_id","amount","market_ratio"]]
               .rename(columns={"amount":"current_salary"}))
        df = df.merge(sal, on="employee_id", how="left")
    else:
        df["current_salary"] = np.nan
        df["market_ratio"]   = 1.0
    df["market_ratio"] = df["market_ratio"].fillna(1.0)

    if attendance is not None and not attendance.empty:
        att = (attendance.sort_values("year_month")
                         .groupby("employee_id").tail(3)
                         .groupby("employee_id")
                         .agg(avg_overtime=("overtime_hours","mean"),
                              avg_absent=("absent_days","mean"),
                              avg_late=("late_count","mean"))
                         .reset_index())
        df = df.merge(att, on="employee_id", how="left")
    else:
        df["avg_overtime"] = 20.0
        df["avg_absent"]   = 0.0
        df["avg_late"]     = 0.0

    df["avg_overtime"] = df["avg_overtime"].fillna(20.0)
    df["avg_absent"]   = df["avg_absent"].fillna(0.0)
    df["avg_late"]     = df["avg_late"].fillna(0.0)
    df["is_overloaded"] = (df["avg_overtime"] > 60).astype(int)

    if survey is not None and not survey.empty:
        surv_last = (survey.sort_values("survey_date")
                           .groupby("employee_id").last().reset_index()
                    [["employee_id","satisfaction_score","work_life_balance",
                      "manager_relation","career_growth"]])

        def _trend(g):
            v = g.sort_values("survey_date")["satisfaction_score"].values
            return float(v[-1] - v[-2]) if len(v) >= 2 else 0.0

        trend = (survey.groupby("employee_id")
                       .apply(_trend).reset_index()
                       .rename(columns={0:"sat_trend"}))
        df = df.merge(surv_last, on="employee_id", how="left")
        df = df.merge(trend, on="employee_id", how="left")
    else:
        df["satisfaction_score"] = 3.5
        df["work_life_balance"]  = 3.5
        df["manager_relation"]   = 3.5
        df["career_growth"]      = 3.5
        df["sat_trend"]          = 0.0

    for c in ["satisfaction_score","work_life_balance","manager_relation","career_growth","sat_trend"]:
        df[c] = df[c].fillna(3.5 if c != "sat_trend" else 0.0)

    if performance is not None and not performance.empty:
        perf = (performance.sort_values("period")
                           .groupby("employee_id").last().reset_index()
               [["employee_id","score","months_since_last_promotion","has_promotion_plan"]]
               .rename(columns={"score":"perf_score"}))
        perf["has_promotion_plan"] = (perf["has_promotion_plan"]
                                      .astype(str).str.lower()
                                      .isin(["true","1","yes"]).astype(int))
        df = df.merge(perf, on="employee_id", how="left")
    else:
        df["perf_score"]               = 3.5
        df["months_since_last_promotion"] = 12.0
        df["has_promotion_plan"]       = 0

    df["perf_score"]                  = df["perf_score"].fillna(3.5)
    df["months_since_last_promotion"] = df["months_since_last_promotion"].fillna(12.0)
    df["has_promotion_plan"]          = df["has_promotion_plan"].fillna(0)
    df["high_performer_no_promo"] = (
        (df["perf_score"] >= 4.0) &
        (df["months_since_last_promotion"] >= 18)
    ).astype(int)

    df["is_female"]  = (df.get("gender","") == "Female").astype(int)
    df["is_married"] = (df.get("marital_status","") == "Married").astype(int)
    edu_map = {"SMA":1,"D3":2,"S1":3,"S2":4,"S3":5}
    df["edu_num"]   = df.get("education_level", pd.Series("S1", index=df.index)).map(edu_map).fillna(3)
    df["age"]       = pd.to_numeric(df.get("age", 30), errors="coerce").fillna(30)
    df["distance_km"] = pd.to_numeric(df.get("distance_from_home_km",10), errors="coerce").fillna(10)
    df["num_prev_companies"] = pd.to_numeric(df.get("num_companies_worked",2), errors="coerce").fillna(2)

    return df


def make_label(df):
    return (
        (df["market_ratio"] < 0.82) |
        (df["avg_overtime"] > 62) |
        (df["satisfaction_score"] < 2.6) |
        (df["high_performer_no_promo"] == 1)
    ).astype(int)


def load_default_data():
    def _load(name):
        p = DATA / name
        return pd.read_csv(p) if p.exists() else None

    emp  = _load("employee.csv")
    sal  = _load("salary_history.csv")
    att  = _load("attendance.csv")
    surv = _load("engagement_survey.csv")
    perf = _load("performance.csv")
    cand = _load("candidates.csv")

    if emp is not None:
        df = build_features(emp, sal, att, surv, perf)
        st.session_state["df_master"] = df

    if cand is not None:
        st.session_state["df_candidates"] = cand

    return emp is not None


def train_model(df):
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].fillna(0)
    y = make_label(df)

    if y.nunique() < 2:
        y.iloc[:max(1, len(y)//4)] = 1

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if y.sum() >= 2 else None
    )

    estimators = []
    if HAS_XGB:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        )))
    estimators.append(("rf", RandomForestClassifier(
        n_estimators=150, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1
    )))
    estimators.append(("gb", GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )))
    estimators.append(("lr", Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])))

    weights = []
    for name, _ in estimators:
        if name in ("xgb","rf"): weights.append(3)
        elif name == "gb":       weights.append(2)
        else:                    weights.append(1)

    model = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    model.fit(X_tr, y_tr)

    metrics = {}
    if len(X_te) > 0 and y_te.nunique() > 1:
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)
        metrics["roc_auc"]   = round(roc_auc_score(y_te, y_prob), 4)
        rpt = classification_report(y_te, y_pred, output_dict=True)
        metrics["precision"] = round(rpt.get("1",{}).get("precision",0), 3)
        metrics["recall"]    = round(rpt.get("1",{}).get("recall",0), 3)
        metrics["f1"]        = round(rpt.get("1",{}).get("f1-score",0), 3)

    explainer = None
    if HAS_SHAP:
        try:
            for name, est in model.named_estimators_.items():
                if name in ("xgb","rf","gb"):
                    explainer = shap.TreeExplainer(est)
                    break
        except Exception:
            pass

    path = MODELS / "resign_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model":model,"features":feats,"metrics":metrics,
                     "trained_at":datetime.now().isoformat()}, f)

    st.session_state["model"]          = model
    st.session_state["features"]       = feats
    st.session_state["metrics"]        = metrics
    st.session_state["trained_at"]     = datetime.now().isoformat()
    st.session_state["shap_explainer"] = explainer

    return metrics, feats


def predict_all(df):
    model = st.session_state["model"]
    feats = st.session_state["features"]
    if model is None:
        return []

    X     = df[[c for c in feats if c in df.columns]].fillna(0)
    proba = model.predict_proba(X)[:, 1]

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        risk  = round(float(proba[i]) * 100, 1)
        level = "Tinggi" if risk >= 70 else "Sedang" if risk >= 40 else "Rendah"
        results.append({
            "employee_id":        str(row.get("employee_id", f"EMP{i}")),
            "full_name":          str(row.get("full_name", "—")),
            "department":         str(row.get("department", "—")),
            "position":           str(row.get("position", "—")),
            "tenure_years":       round(float(row.get("tenure_years", 0)), 1),
            "resign_probability": risk,
            "risk_level":         level,
            "market_ratio":       round(float(row.get("market_ratio", 1)), 2),
            "avg_overtime":       round(float(row.get("avg_overtime", 0)), 1),
            "satisfaction":       round(float(row.get("satisfaction_score", 3.5)), 2),
        })

    results.sort(key=lambda x: x["resign_probability"], reverse=True)
    return results


def explain_employee(emp_id):
    df        = st.session_state["df_master"]
    model     = st.session_state["model"]
    feats     = st.session_state["features"]
    explainer = st.session_state["shap_explainer"]

    if df is None or model is None:
        return None

    row = df[df["employee_id"] == emp_id]
    if row.empty:
        return None

    X    = row[[c for c in feats if c in row.columns]].fillna(0)
    prob = float(model.predict_proba(X)[0, 1])

    risk_factors = []
    safe_factors = []
    method = "Heuristic"

    if explainer is not None and HAS_SHAP:
        try:
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            vals = sv[0]
            factors = []
            for feat, v, actual in zip(feats, vals, X.values[0]):
                factors.append({
                    "feature": feat,
                    "label":   FEATURE_LABELS.get(feat, feat),
                    "shap":    round(float(v), 4),
                    "value":   round(float(actual), 3),
                    "impact":  round(abs(float(v)) / (sum(abs(float(x)) for x in vals) + 1e-9) * 100, 1),
                })
            factors.sort(key=lambda x: abs(x["shap"]), reverse=True)
            risk_factors = [f for f in factors if f["shap"] > 0][:5]
            safe_factors = [f for f in factors if f["shap"] < 0][:3]
            method = "SHAP (TreeExplainer)"
        except Exception:
            pass

    if not risk_factors:
        row0 = row.iloc[0]
        checks = [
            ("market_ratio",                lambda v: (1-v)*40 if v < 1 else 0),
            ("avg_overtime",                lambda v: (v-20)/100*30 if v > 20 else 0),
            ("satisfaction_score",          lambda v: (3.5-v)/3.5*25 if v < 3.5 else 0),
            ("career_growth",               lambda v: (3.5-v)/3.5*20 if v < 3.5 else 0),
            ("manager_relation",            lambda v: (3.5-v)/3.5*15 if v < 3.5 else 0),
            ("months_since_last_promotion", lambda v: min(v/60*15, 15)),
        ]
        for feat, fn in checks:
            if feat in feats:
                val    = float(row0.get(feat, 0))
                impact = fn(val)
                if impact > 0:
                    risk_factors.append({
                        "feature": feat,
                        "label":   FEATURE_LABELS.get(feat, feat),
                        "shap":    round(impact/100, 3),
                        "value":   round(val, 2),
                        "impact":  round(impact, 1),
                    })
        risk_factors.sort(key=lambda x: x["impact"], reverse=True)
        risk_factors = risk_factors[:5]

    # Rekomendasi
    recs = []
    feats_hit = {f["feature"] for f in risk_factors}
    if "market_ratio" in feats_hit:
        mr  = float(row.iloc[0].get("market_ratio", 1))
        gap = round((1 - mr) * 100, 1)
        recs.append(f"💰 Lakukan salary review — gaji saat ini {gap}% di bawah pasar. Sesuaikan ke market rate.")
    if "avg_overtime" in feats_hit:
        ot = float(row.iloc[0].get("avg_overtime", 0))
        recs.append(f"⏱️ Kurangi overtime dari rata-rata {ot:.0f} jam/bln — distribusi ulang workload atau tambah resource.")
    if "satisfaction_score" in feats_hit or "work_life_balance" in feats_hit:
        recs.append("💬 Lakukan 1-on-1 meeting dengan karyawan untuk menggali kebutuhan dan keluhan.")
    if "career_growth" in feats_hit or "months_since_last_promotion" in feats_hit:
        recs.append("📈 Buat Individual Development Plan (IDP) dengan target promosi jelas dalam 6 bulan.")
    if "manager_relation" in feats_hit:
        recs.append("🤝 Fasilitasi mediasi atau coaching antara karyawan dan manajer langsung.")
    if not recs:
        recs.append("👁️ Pantau terus kondisi karyawan ini melalui survey engagement rutin.")

    return {
        "employee_id":        emp_id,
        "full_name":          str(row.iloc[0].get("full_name","—")),
        "resign_probability": round(prob * 100, 1),
        "risk_level":         "Tinggi" if prob >= 0.7 else "Sedang" if prob >= 0.4 else "Rendah",
        "risk_factors":       risk_factors,
        "protective_factors": safe_factors,
        "recommendations":    recs,
        "explanation_method": method,
        "data":               row.iloc[0],
    }


def match_candidates(candidates, vacancy):
    required_skills = [s.lower().strip() for s in vacancy.get("required_skills", [])]
    jd_text  = vacancy.get("job_description", " ".join(required_skills))
    exp_min  = int(vacancy.get("exp_min_years", 2))
    exp_max  = int(vacancy.get("exp_max_years", 8))
    budget_min = float(vacancy.get("budget_min", 0))
    budget_max = float(vacancy.get("budget_max", 999_000_000))

    skill_texts = candidates.get("skills", pd.Series([""]*len(candidates))).fillna("").tolist()
    corpus = [jd_text] + skill_texts
    tfidf  = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
    mat    = tfidf.fit_transform(corpus)
    nlp_scores = cosine_similarity(mat[1:], mat[0]).flatten()

    results = []
    for i, (_, cand) in enumerate(candidates.iterrows()):
        exp  = float(cand.get("years_experience", 0))
        sal  = float(cand.get("salary_expectation", 0))
        cand_skills = set(s.lower().strip() for s in str(cand.get("skills","")).split(","))

        skill_sc = len(set(required_skills) & cand_skills) / len(required_skills) if required_skills else 0.5
        if exp_min <= exp <= exp_max:   exp_sc = 1.0
        elif exp < exp_min:             exp_sc = max(0, 1 - (exp_min - exp) * 0.15)
        else:                           exp_sc = max(0.5, 1 - (exp - exp_max) * 0.1)

        if budget_min <= sal <= budget_max: sal_sc = 1.0
        elif sal < budget_min:              sal_sc = 0.9
        else:                               sal_sc = max(0, 1 - ((sal - budget_max)/budget_max) * 2)

        struct_sc = skill_sc*0.5 + exp_sc*0.3 + sal_sc*0.2

        tenure = float(cand.get("last_tenure_years", 2))
        n_comp = float(cand.get("num_companies_worked", 2))
        resign_risk = 0.25
        if tenure < 1: resign_risk += 0.30
        elif tenure < 2: resign_risk += 0.15
        if n_comp > 4: resign_risk += 0.20
        elif n_comp > 3: resign_risk += 0.10
        resign_risk = min(resign_risk, 0.95)

        model = st.session_state["model"]
        feats = st.session_state["features"]
        if model is not None:
            try:
                row_dict = {
                    "market_ratio": sal / budget_max if budget_max else 1,
                    "avg_overtime": 25,
                    "satisfaction_score": 3.5,
                    "tenure_months": tenure * 12,
                    "tenure_years": tenure,
                    "num_prev_companies": n_comp,
                }
                X_c = pd.DataFrame([{f: row_dict.get(f, 0) for f in feats}]).fillna(0)
                resign_risk = float(model.predict_proba(X_c)[0, 1])
            except Exception:
                pass

        retention_fit = 1 - resign_risk
        final_score = nlp_scores[i]*0.35 + struct_sc*0.40 + retention_fit*0.25

        results.append({
            "rank":              0,
            "candidate_id":      str(cand.get("candidate_id", f"CND{i+1:03d}")),
            "full_name":         str(cand.get("full_name","—")),
            "position_applied":  str(cand.get("position_applied","—")),
            "years_experience":  exp,
            "skills":            str(cand.get("skills","")),
            "salary_expectation": sal,
            "match_score":       round(final_score * 100, 1),
            "nlp_similarity":    round(nlp_scores[i] * 100, 1),
            "structured_score":  round(struct_sc * 100, 1),
            "resign_risk_pct":   round(resign_risk * 100, 1),
            "resign_risk_level": "Tinggi" if resign_risk >= 0.5 else "Sedang" if resign_risk >= 0.25 else "Rendah",
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
        <div style='padding:16px 0 20px; border-bottom:1px solid #21262d;'>
            <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#00e5a0;letter-spacing:-0.5px;'>RetainAI</div>
            <div style='font-family:DM Mono,monospace;font-size:10px;color:#6e7681;letter-spacing:1.5px;text-transform:uppercase;margin-top:3px;'>Employee Intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    # Handle redirect dari tombol navigasi post-upload
    nav_override = st.session_state.get("_nav")
    if nav_override:
        st.session_state["_nav"] = None

    page = st.radio(
        "Navigasi",
        ["🏠 Dashboard", "⚠️ Risiko Karyawan", "🔍 Analisis XAI", "🎯 Kandidat Matching", "📂 Upload Data", "⚙️ Model"],
        label_visibility="collapsed",
        index=["🏠 Dashboard", "⚠️ Risiko Karyawan", "🔍 Analisis XAI", "🎯 Kandidat Matching", "📂 Upload Data", "⚙️ Model"].index(nav_override) if nav_override else 0,
    )

    # Model status
    if st.session_state["model"] is not None:
        auc = st.session_state["metrics"].get("roc_auc","—")
        st.markdown(f"""
            <div style='margin-top:20px;background:rgba(0,229,160,.07);border:1px solid rgba(0,229,160,.15);
                        border-radius:8px;padding:10px 12px;font-family:DM Mono,monospace;font-size:11px;color:#00e5a0;'>
                <span style='display:inline-block;width:6px;height:6px;background:#00e5a0;border-radius:50%;margin-right:6px;'></span>
                Model Aktif · AUC {auc}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='margin-top:20px;background:rgba(255,107,53,.07);border:1px solid rgba(255,107,53,.15);
                        border-radius:8px;padding:10px 12px;font-family:DM Mono,monospace;font-size:11px;color:#ff6b35;'>
                ⚠️ Model belum ditraining
            </div>
        """, unsafe_allow_html=True)


# ─── Auto load data on first run ────────────────────────────────
if st.session_state["df_master"] is None:
    loaded = load_default_data()
    if loaded and st.session_state["model"] is None:
        # Try loading saved model
        mp = MODELS / "resign_model.pkl"
        if mp.exists():
            with open(mp,"rb") as f:
                saved = pickle.load(f)
            st.session_state["model"]      = saved["model"]
            st.session_state["features"]   = saved["features"]
            st.session_state["metrics"]    = saved.get("metrics",{})
            st.session_state["trained_at"] = saved.get("trained_at","")
        else:
            df = st.session_state["df_master"]
            if df is not None:
                with st.spinner("🔄 Training model..."):
                    train_model(df)


# ═══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("""
        <div class='retain-header'>
            <div>
                <div class='retain-logo'>🎯 RetainAI</div>
                <div class='retain-tagline'>Employee Intelligence Platform · v2.0</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    df   = st.session_state["df_master"]
    cand = st.session_state["df_candidates"]

    if df is None:
        st.info("📂 Belum ada data. Silakan upload data melalui menu **Upload Data** atau letakkan CSV di folder `data/raw/`.")
        st.stop()

    results = predict_all(df) if st.session_state["model"] else []
    high    = sum(1 for r in results if r["risk_level"] == "Tinggi")
    medium  = sum(1 for r in results if r["risk_level"] == "Sedang")
    low     = len(results) - high - medium

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class='stat-card stat-green'>
                <div class='stat-label'>Total Karyawan</div>
                <div class='stat-value val-green'>{len(df)}</div>
                <div class='stat-sub'>Active headcount</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='stat-card stat-orange'>
                <div class='stat-label'>Risiko Tinggi</div>
                <div class='stat-value val-orange'>{high}</div>
                <div class='stat-sub'>Perlu perhatian segera</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class='stat-card stat-amber'>
                <div class='stat-label'>Risiko Sedang</div>
                <div class='stat-value val-amber'>{medium}</div>
                <div class='stat-sub'>Perlu monitoring</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        cand_count = len(cand) if cand is not None else 0
        st.markdown(f"""
            <div class='stat-card stat-blue'>
                <div class='stat-label'>Kandidat</div>
                <div class='stat-value val-blue'>{cand_count}</div>
                <div class='stat-sub'>Siap dimatching</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='section-title'>📊 Distribusi Risiko per Departemen</div>", unsafe_allow_html=True)
        if results:
            df_res = pd.DataFrame(results)
            dept_risk = df_res.groupby(["department","risk_level"]).size().unstack(fill_value=0)
            for lvl in ["Tinggi","Sedang","Rendah"]:
                if lvl not in dept_risk.columns:
                    dept_risk[lvl] = 0
            dept_risk = dept_risk[["Tinggi","Sedang","Rendah"]].reset_index()

            fig = go.Figure()
            colors = {"Tinggi":"#ff6b35","Sedang":"#f0b429","Rendah":"#00e5a0"}
            for lvl in ["Tinggi","Sedang","Rendah"]:
                fig.add_trace(go.Bar(
                    name=lvl, x=dept_risk["department"], y=dept_risk[lvl],
                    marker_color=colors[lvl],
                ))

            fig.update_layout(
                barmode="stack",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e", family="DM Sans"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0,r=0,t=10,b=0),
                height=280,
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>🎯 Komposisi Risiko</div>", unsafe_allow_html=True)
        if results:
            fig2 = go.Figure(go.Pie(
                labels=["Risiko Tinggi","Risiko Sedang","Risiko Rendah"],
                values=[high, medium, low],
                hole=0.65,
                marker_colors=["#ff6b35","#f0b429","#00e5a0"],
                textinfo="percent",
                hovertemplate="%{label}: %{value} karyawan<extra></extra>",
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e", family="DM Sans"),
                legend=dict(bgcolor="rgba(0,0,0,0)", x=0, y=0),
                margin=dict(l=0,r=0,t=10,b=0),
                height=280,
                annotations=[dict(text=f"<b>{len(results)}</b><br>Karyawan", x=0.5, y=0.5,
                                  font=dict(size=14, color="#e6edf3"), showarrow=False)]
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Top 5 high risk
    st.markdown("<div class='section-title'>🚨 Top 5 Karyawan Berisiko Tertinggi</div>", unsafe_allow_html=True)
    if results:
        top5 = [r for r in results if r["risk_level"] == "Tinggi"][:5]
        if not top5:
            top5 = results[:5]

        df_top = pd.DataFrame(top5)[["full_name","department","position","resign_probability","risk_level","satisfaction","avg_overtime"]]
        df_top.columns = ["Nama","Departemen","Posisi","Risiko (%)","Level","Kepuasan","Overtime"]
        st.dataframe(df_top, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: RISIKO KARYAWAN
# ═══════════════════════════════════════════════════════════════

elif page == "⚠️ Risiko Karyawan":
    st.markdown("<h2 style='font-family:Syne,sans-serif;color:#e6edf3;margin-bottom:20px;'>⚠️ Analisis Risiko Karyawan</h2>", unsafe_allow_html=True)

    df = st.session_state["df_master"]
    if df is None:
        st.warning("Belum ada data karyawan. Upload terlebih dahulu.")
        st.stop()
    if st.session_state["model"] is None:
        st.warning("Model belum ditraining.")
        st.stop()

    results = predict_all(df)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_level = st.selectbox("Filter Level Risiko", ["Semua","Tinggi","Sedang","Rendah"])
    with col2:
        depts = ["Semua"] + sorted(df["department"].dropna().unique().tolist()) if "department" in df.columns else ["Semua"]
        filter_dept = st.selectbox("Filter Departemen", depts)
    with col3:
        min_prob = st.slider("Probabilitas Minimum (%)", 0, 100, 0)

    filtered = results
    if filter_level != "Semua":
        filtered = [r for r in filtered if r["risk_level"] == filter_level]
    if filter_dept != "Semua":
        filtered = [r for r in filtered if r["department"] == filter_dept]
    filtered = [r for r in filtered if r["resign_probability"] >= min_prob]

    st.markdown(f"<div style='color:#6e7681;font-size:12px;margin:8px 0 16px;font-family:DM Mono,monospace;'>Menampilkan {len(filtered)} dari {len(results)} karyawan</div>", unsafe_allow_html=True)

    if filtered:
        df_show = pd.DataFrame(filtered)
        df_show["Risk Level"] = df_show["risk_level"]
        display_cols = ["employee_id","full_name","department","position","tenure_years",
                        "resign_probability","risk_level","market_ratio","avg_overtime","satisfaction"]
        display_cols = [c for c in display_cols if c in df_show.columns]

        df_display = df_show[display_cols].copy()
        df_display.columns = ["ID","Nama","Departemen","Posisi","Masa Kerja (th)",
                               "Risiko (%)","Level Risiko","Market Ratio","Overtime","Kepuasan"]

        def color_risk(val):
            if val == "Tinggi":   return "background-color: rgba(255,107,53,.12); color: #ff6b35"
            elif val == "Sedang": return "background-color: rgba(240,180,41,.12); color: #f0b429"
            else:                 return "background-color: rgba(0,229,160,.12); color: #00e5a0"

        styled = df_display.style.applymap(color_risk, subset=["Level Risiko"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

        # Chart distribusi probabilitas
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 Distribusi Probabilitas Resign</div>", unsafe_allow_html=True)
        fig = px.histogram(
            df_show, x="resign_probability", nbins=20,
            color_discrete_sequence=["#3d6eff"],
            labels={"resign_probability":"Probabilitas Resign (%)"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", family="DM Sans"),
            margin=dict(l=0,r=0,t=10,b=0), height=240,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: ANALISIS XAI
# ═══════════════════════════════════════════════════════════════

elif page == "🔍 Analisis XAI":
    st.markdown("<h2 style='font-family:Syne,sans-serif;color:#e6edf3;margin-bottom:20px;'>🔍 Explainable AI — Analisis Individual</h2>", unsafe_allow_html=True)

    df = st.session_state["df_master"]
    if df is None or st.session_state["model"] is None:
        st.warning("Data/model belum tersedia.")
        st.stop()

    # Employee selector
    results  = predict_all(df)
    emp_opts = {f"{r['full_name']} ({r['employee_id']}) — {r['risk_level']} {r['resign_probability']}%": r["employee_id"]
                for r in results}

    selected_label = st.selectbox(
        "Pilih Karyawan untuk Dianalisis",
        list(emp_opts.keys()),
        help="Karyawan diurutkan berdasarkan tingkat risiko"
    )
    emp_id = emp_opts[selected_label]

    if st.button("🔍 Analisis", type="primary"):
        with st.spinner("Menganalisis..."):
            result = explain_employee(emp_id)

        if result is None:
            st.error("Karyawan tidak ditemukan.")
            st.stop()

        prob  = result["resign_probability"]
        level = result["risk_level"]
        color = "#ff6b35" if level == "Tinggi" else "#f0b429" if level == "Sedang" else "#00e5a0"

        # Header card
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nama", result["full_name"])
        with col2:
            st.metric("Probabilitas Resign", f"{prob}%")
        with col3:
            st.metric("Level Risiko", level)
        with col4:
            st.metric("Metode", result["explanation_method"])

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("### 🚨 Faktor Risiko Utama")
            if result["risk_factors"]:
                for f in result["risk_factors"]:
                    impact = min(f["impact"], 100)
                    bar_color = "#ff6b35" if impact > 25 else "#f0b429"
                    st.markdown(f"""
                        <div class='factor-bar'>
                            <div class='factor-name'>{f['label']}</div>
                            <div class='factor-label'>Nilai: {f['value']} · Impact: {f['impact']}%</div>
                            <div class='factor-bar-bg'>
                                <div class='factor-bar-fill' style='width:{impact}%;background:{bar_color};'></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # Chart
                labels  = [f["label"] for f in result["risk_factors"]]
                impacts = [f["impact"] for f in result["risk_factors"]]
                fig = go.Figure(go.Bar(
                    x=impacts, y=labels, orientation='h',
                    marker_color=["#ff6b35","#f06b35","#f0b429","#f0c429","#f0d429"][:len(labels)],
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b949e", family="DM Sans"),
                    margin=dict(l=0,r=0,t=10,b=0), height=220,
                    xaxis=dict(gridcolor="#21262d", title="Impact (%)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada faktor risiko signifikan terdeteksi.")

        with col_r:
            st.markdown("### ✅ Rekomendasi Aksi")
            for rec in result["recommendations"]:
                st.markdown(f"<div class='rec-card'>{rec}</div>", unsafe_allow_html=True)

            if result["protective_factors"]:
                st.markdown("### 🛡️ Faktor Pelindung")
                for f in result["protective_factors"]:
                    st.markdown(f"""
                        <div class='factor-bar'>
                            <div class='factor-name'>{f['label']}</div>
                            <div class='factor-label'>Nilai: {f['value']}</div>
                        </div>
                    """, unsafe_allow_html=True)

        # Gauge
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Risk Gauge")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix":"%","font":{"color":color,"size":32,"family":"Syne"}},
            gauge={
                "axis": {"range":[0,100],"tickcolor":"#6e7681","tickfont":{"color":"#6e7681"}},
                "bar":  {"color":color,"thickness":0.25},
                "bgcolor":"#161b22",
                "bordercolor":"#21262d",
                "steps":[
                    {"range":[0,40],   "color":"rgba(0,229,160,.1)"},
                    {"range":[40,70],  "color":"rgba(240,180,41,.1)"},
                    {"range":[70,100], "color":"rgba(255,107,53,.1)"},
                ],
                "threshold":{
                    "line":{"color":color,"width":3},
                    "thickness":0.75,
                    "value":prob,
                },
            },
            title={"text":"Probabilitas Resign","font":{"color":"#8b949e","size":14}},
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", family="DM Sans"),
            margin=dict(l=20,r=20,t=30,b=20),
            height=250,
        )
        st.plotly_chart(fig_g, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: KANDIDAT MATCHING
# ═══════════════════════════════════════════════════════════════

elif page == "🎯 Kandidat Matching":
    st.markdown("<h2 style='font-family:Syne,sans-serif;color:#e6edf3;margin-bottom:20px;'>🎯 Smart Candidate Matching</h2>", unsafe_allow_html=True)

    cand = st.session_state["df_candidates"]
    if cand is None:
        st.warning("Belum ada data kandidat. Upload `candidates.csv` terlebih dahulu.")
        st.stop()

    st.markdown(f"<div class='info-box'>📋 {len(cand)} kandidat tersedia dalam database</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pos_title = st.text_input("Judul Posisi", placeholder="mis. Senior Backend Developer")
        jd_text   = st.text_area("Deskripsi Pekerjaan", height=100,
                                  placeholder="Deskripsikan tanggung jawab dan requirement posisi...")
        skills_raw = st.text_input("Required Skills (pisahkan koma)", placeholder="Python, FastAPI, PostgreSQL, Docker")
    with col2:
        exp_min, exp_max = st.slider("Pengalaman (tahun)", 0, 15, (2, 8))
        budget_min = st.number_input("Budget Gaji Minimum (Rp)", value=5_000_000, step=500_000)
        budget_max = st.number_input("Budget Gaji Maksimum (Rp)", value=20_000_000, step=500_000)
        top_n      = st.slider("Tampilkan Top-N Kandidat", 1, 20, 10)

    if st.button("🔍 Cari Kandidat Terbaik", type="primary"):
        required_skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []

        vacancy = {
            "position_title":  pos_title,
            "job_description": jd_text or " ".join(required_skills),
            "required_skills": required_skills,
            "exp_min_years":   exp_min,
            "exp_max_years":   exp_max,
            "budget_min":      budget_min,
            "budget_max":      budget_max,
        }

        with st.spinner("Memproses matching..."):
            ranked = match_candidates(cand, vacancy)

        st.markdown(f"<br><div class='section-title'>🏆 Top {top_n} Kandidat untuk: {pos_title or 'Posisi'}</div>", unsafe_allow_html=True)

        top_cands = ranked[:top_n]
        df_cands  = pd.DataFrame(top_cands)

        col1, col2, col3 = st.columns(3)
        with col1:
            if top_cands:
                best = top_cands[0]
                st.markdown(f"""
                    <div class='stat-card stat-green'>
                        <div class='stat-label'>🥇 Kandidat Terbaik</div>
                        <div style='font-family:Syne,sans-serif;font-size:16px;font-weight:700;color:#e6edf3;margin:6px 0;'>{best['full_name']}</div>
                        <div style='font-family:DM Mono,monospace;font-size:11px;color:#00e5a0;'>Match Score: {best['match_score']}%</div>
                    </div>
                """, unsafe_allow_html=True)
        with col2:
            avg_match = round(sum(c["match_score"] for c in top_cands) / len(top_cands), 1) if top_cands else 0
            st.markdown(f"""
                <div class='stat-card stat-blue'>
                    <div class='stat-label'>Avg Match Score</div>
                    <div class='stat-value val-blue'>{avg_match}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            low_risk = sum(1 for c in top_cands if c["resign_risk_level"] == "Rendah")
            st.markdown(f"""
                <div class='stat-card stat-green'>
                    <div class='stat-label'>Retensi Tinggi</div>
                    <div class='stat-value val-green'>{low_risk}</div>
                    <div class='stat-sub'>dari {len(top_cands)} kandidat</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Table
        display_cols = ["rank","full_name","years_experience","skills","salary_expectation",
                        "match_score","nlp_similarity","structured_score","resign_risk_pct","resign_risk_level"]
        df_show = df_cands[display_cols].copy()
        df_show.columns = ["#","Nama","Pengalaman","Skills","Ekspektasi Gaji",
                           "Match (%)","NLP (%)","Struktural (%)","Resign Risk (%)","Risk Level"]

        def color_match(val):
            if isinstance(val, (int,float)):
                if val >= 70:   return "color: #00e5a0"
                elif val >= 50: return "color: #f0b429"
                else:           return "color: #ff6b35"
            return ""

        def color_risk_cand(val):
            if val == "Tinggi":   return "color: #ff6b35"
            elif val == "Sedang": return "color: #f0b429"
            else:                 return "color: #00e5a0"

        styled = df_show.style \
            .applymap(color_match, subset=["Match (%)"]) \
            .applymap(color_risk_cand, subset=["Risk Level"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Score comparison chart
        st.markdown("<br><div class='section-title'>📊 Perbandingan Score Kandidat</div>", unsafe_allow_html=True)
        fig = go.Figure()
        names = df_cands["full_name"].tolist()[:top_n]
        fig.add_trace(go.Bar(name="Match Score",  x=names, y=df_cands["match_score"].tolist()[:top_n], marker_color="#3d6eff"))
        fig.add_trace(go.Bar(name="NLP Similarity", x=names, y=df_cands["nlp_similarity"].tolist()[:top_n], marker_color="#00e5a0"))
        fig.add_trace(go.Bar(name="Resign Risk",  x=names, y=df_cands["resign_risk_pct"].tolist()[:top_n], marker_color="#ff6b35"))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=10,b=0), height=300,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATA
# ═══════════════════════════════════════════════════════════════

elif page == "📂 Upload Data":

    DATA_INFO = {
        "employee":    ("employee.csv",          "👤 Karyawan",    "employee_id, full_name, department, position, hire_date, age, gender, education_level, marital_status, distance_from_home_km, num_companies_worked"),
        "salary":      ("salary_history.csv",    "💰 Gaji",        "employee_id, amount, market_ratio, effective_date"),
        "attendance":  ("attendance.csv",        "🕐 Kehadiran",   "employee_id, year_month, overtime_hours, absent_days, late_count"),
        "survey":      ("engagement_survey.csv", "📋 Survey",      "employee_id, survey_date, satisfaction_score, work_life_balance, manager_relation, career_growth"),
        "performance": ("performance.csv",       "📈 Performa",    "employee_id, period, score, months_since_last_promotion, has_promotion_plan"),
        "candidates":  ("candidates.csv",        "🎯 Kandidat",    "candidate_id, full_name, position_applied, years_experience, skills, salary_expectation, last_tenure_years, num_companies_worked"),
    }

    # ── Tampilkan hasil upload jika baru saja selesai ───────────
    if st.session_state.get("upload_success"):
        result = st.session_state["upload_success"]

        st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(0,229,160,.1),rgba(61,110,255,.08));
                        border:1px solid rgba(0,229,160,.3);border-radius:14px;padding:24px 28px;margin-bottom:24px;'>
                <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#00e5a0;margin-bottom:4px;'>
                    ✅ Upload & Training Selesai!
                </div>
                <div style='font-size:13px;color:#8b949e;'>
                    {result.get("n_files", 1)} file berhasil disimpan · {result.get("n_sources", 1)} sumber data digabungkan · Model siap digunakan.
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── Stat ringkasan hasil ────────────────────────────────
        r = result
        c1, c2, c3, c4, c5 = st.columns(5)
        cards = [
            (c1, "Total Karyawan",  str(r.get("total_emp","—")),  "val-green",  "stat-green"),
            (c2, "Risiko Tinggi",   str(r.get("high","—")),        "val-orange", "stat-orange"),
            (c3, "Risiko Sedang",   str(r.get("medium","—")),      "val-amber",  "stat-amber"),
            (c4, "Risiko Rendah",   str(r.get("low","—")),         "val-green",  "stat-green"),
            (c5, "ROC-AUC Model",   str(r.get("auc","—")),         "val-blue",   "stat-blue"),
        ]
        for col, label, val, vcls, ccls in cards:
            with col:
                st.markdown(f"""
                    <div class='stat-card {ccls}'>
                        <div class='stat-label'>{label}</div>
                        <div class='stat-value {vcls}' style='font-size:26px;'>{val}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metrik model ────────────────────────────────────────
        col_m, col_chart = st.columns([1, 2])
        with col_m:
            st.markdown("#### 📊 Metrik Model")
            metrics_display = [
                ("ROC-AUC",   r.get("auc","—"),       "#3d6eff"),
                ("Precision", r.get("precision","—"), "#00e5a0"),
                ("Recall",    r.get("recall","—"),    "#f0b429"),
                ("F1-Score",  r.get("f1","—"),        "#00c97a"),
            ]
            for mname, mval, mcolor in metrics_display:
                bar_w = int(float(mval)*100) if isinstance(mval, float) else 0
                st.markdown(f"""
                    <div style='background:#161b22;border:1px solid #21262d;border-radius:8px;
                                padding:10px 14px;margin-bottom:6px;'>
                        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;'>
                            <span style='font-size:12px;color:#8b949e;font-family:DM Mono,monospace;'>{mname}</span>
                            <span style='font-size:14px;font-weight:700;color:{mcolor};font-family:Syne,sans-serif;'>{mval}</span>
                        </div>
                        <div style='background:#21262d;border-radius:3px;height:5px;'>
                            <div style='width:{bar_w}%;background:{mcolor};height:5px;border-radius:3px;'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with col_chart:
            st.markdown("#### 📈 Distribusi Risiko Karyawan")
            df_res = st.session_state.get("df_master")
            if df_res is not None and st.session_state["model"] is not None:
                preds = predict_all(df_res)
                if preds:
                    df_preds = pd.DataFrame(preds)

                    col_pie, col_bar = st.columns(2)
                    with col_pie:
                        high_n   = sum(1 for p in preds if p["risk_level"]=="Tinggi")
                        medium_n = sum(1 for p in preds if p["risk_level"]=="Sedang")
                        low_n    = len(preds) - high_n - medium_n
                        fig_pie = go.Figure(go.Pie(
                            labels=["Tinggi","Sedang","Rendah"],
                            values=[high_n, medium_n, low_n],
                            hole=0.6,
                            marker_colors=["#ff6b35","#f0b429","#00e5a0"],
                            textinfo="percent+value",
                        ))
                        fig_pie.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#8b949e", family="DM Sans"),
                            showlegend=False,
                            margin=dict(l=0,r=0,t=10,b=0), height=200,
                            annotations=[dict(text=f"<b>{len(preds)}</b>", x=0.5, y=0.5,
                                              font=dict(size=18,color="#e6edf3"), showarrow=False)]
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_bar:
                        fig_hist = px.histogram(
                            df_preds, x="resign_probability", nbins=15,
                            color_discrete_sequence=["#3d6eff"],
                            labels={"resign_probability": "Risiko (%)"},
                        )
                        fig_hist.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#8b949e", family="DM Sans"),
                            margin=dict(l=0,r=0,t=10,b=0), height=200,
                            xaxis=dict(gridcolor="#21262d"),
                            yaxis=dict(gridcolor="#21262d", title="Jumlah"),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Top 5 high risk langsung ────────────────────────────
        st.markdown("#### 🚨 Karyawan Risiko Tertinggi")
        preds = predict_all(st.session_state["df_master"]) if st.session_state["model"] else []
        top5  = [p for p in preds if p["risk_level"] == "Tinggi"][:5] or preds[:5]
        if top5:
            df_top = pd.DataFrame(top5)[["full_name","department","position","resign_probability","risk_level","satisfaction","avg_overtime"]]
            df_top.columns = ["Nama","Departemen","Posisi","Risiko (%)","Level","Kepuasan","Overtime"]
            st.dataframe(df_top, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tombol navigasi ─────────────────────────────────────
        st.markdown("#### 🧭 Langkah Selanjutnya")
        nav1, nav2, nav3, nav4 = st.columns(4)
        with nav1:
            if st.button("🏠 Ke Dashboard", use_container_width=True, type="primary"):
                st.session_state["upload_success"] = None
                st.session_state["_nav"] = "🏠 Dashboard"
                st.rerun()
        with nav2:
            if st.button("⚠️ Analisis Risiko", use_container_width=True):
                st.session_state["upload_success"] = None
                st.session_state["_nav"] = "⚠️ Risiko Karyawan"
                st.rerun()
        with nav3:
            if st.button("🔍 Analisis XAI", use_container_width=True):
                st.session_state["upload_success"] = None
                st.session_state["_nav"] = "🔍 Analisis XAI"
                st.rerun()
        with nav4:
            if st.button("🔄 Upload File Lain", use_container_width=True):
                st.session_state["upload_success"] = None
                st.rerun()

        st.stop()

    # ── Halaman upload normal ───────────────────────────────────
    st.markdown("<h2 style='font-family:Syne,sans-serif;color:#e6edf3;margin-bottom:8px;'>📂 Upload Semua Dataset</h2>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>💡 Upload semua dataset sekaligus, lalu klik <b>Proses & Training</b> — model akan ditraining dari gabungan seluruh data yang tersedia. Minimal <b>employee.csv</b> diperlukan.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status 6 file yang sudah ada ───────────────────────────
    st.markdown("<div class='section-title'>📋 Status Dataset Saat Ini</div>", unsafe_allow_html=True)
    status_cols = st.columns(6)
    for i, (dtype, (fname, label, _)) in enumerate(DATA_INFO.items()):
        p = DATA / fname
        with status_cols[i]:
            if p.exists():
                n = len(pd.read_csv(p))
                st.markdown(f"""
                    <div style='background:rgba(0,229,160,.07);border:1px solid rgba(0,229,160,.2);
                                border-radius:8px;padding:10px;text-align:center;'>
                        <div style='font-size:18px;'>{label.split()[0]}</div>
                        <div style='font-family:DM Mono,monospace;font-size:10px;color:#00e5a0;margin-top:4px;'>✅ {n} baris</div>
                        <div style='font-family:DM Mono,monospace;font-size:9px;color:#6e7681;margin-top:2px;'>{fname.split(".")[0]}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background:rgba(255,107,53,.05);border:1px solid rgba(255,107,53,.15);
                                border-radius:8px;padding:10px;text-align:center;'>
                        <div style='font-size:18px;'>{label.split()[0]}</div>
                        <div style='font-family:DM Mono,monospace;font-size:10px;color:#ff6b35;margin-top:4px;'>❌ Belum ada</div>
                        <div style='font-family:DM Mono,monospace;font-size:9px;color:#6e7681;margin-top:2px;'>{fname.split(".")[0]}</div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Form upload semua sekaligus ─────────────────────────────
    st.markdown("<div class='section-title'>⬆️ Upload Dataset Baru</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6e7681;margin-bottom:16px;'>Upload satu atau lebih file CSV di bawah. File yang tidak diupload akan tetap menggunakan data yang sudah ada.</div>", unsafe_allow_html=True)

    uploaded_files = {}
    previews = {}

    rows = [
        [("employee", DATA_INFO["employee"]), ("salary", DATA_INFO["salary"]), ("attendance", DATA_INFO["attendance"])],
        [("performance", DATA_INFO["performance"]), ("survey", DATA_INFO["survey"]), ("candidates", DATA_INFO["candidates"])],
    ]

    for row in rows:
        cols = st.columns(3)
        for col, (dtype, (fname, label, columns_hint)) in zip(cols, row):
            with col:
                icon = label.split()[0]
                name = " ".join(label.split()[1:])
                st.markdown(f"""
                    <div style='background:#161b22;border:1px solid #21262d;border-radius:10px;
                                padding:12px 14px 8px;margin-bottom:4px;'>
                        <div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;color:#e6edf3;'>
                            {icon} {name}
                        </div>
                        <div style='font-family:DM Mono,monospace;font-size:10px;color:#6e7681;margin-top:3px;'>{fname}</div>
                    </div>
                """, unsafe_allow_html=True)
                with st.expander("📌 Kolom yang dibutuhkan", expanded=False):
                    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:10px;color:#a5b4fc;line-height:1.7;'>{columns_hint}</div>", unsafe_allow_html=True)

                uf = st.file_uploader(
                    label=f"Upload {fname}",
                    type=["csv"],
                    key=f"bulk_uploader_{dtype}",
                    label_visibility="collapsed",
                )
                uploaded_files[dtype] = uf

                if uf is not None:
                    try:
                        df_prev = pd.read_csv(uf)
                        previews[dtype] = df_prev
                        st.markdown(f"""
                            <div style='background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.2);
                                        border-radius:6px;padding:8px 10px;font-size:11px;color:#00e5a0;
                                        font-family:DM Mono,monospace;margin-top:4px;'>
                                ✅ {len(df_prev)} baris · {len(df_prev.columns)} kolom
                            </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gagal membaca: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Preview semua yang diupload ─────────────────────────────
    if previews:
        st.markdown("<div class='section-title'>👁️ Preview File yang Diupload</div>", unsafe_allow_html=True)
        for dtype, df_prev in previews.items():
            fname = DATA_INFO[dtype][0]
            label = DATA_INFO[dtype][1]
            with st.expander(f"{label} — {len(df_prev)} baris"):
                st.dataframe(df_prev.head(5), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Tombol proses semua sekaligus ───────────────────────────
    n_new      = len(previews)
    has_emp    = "employee" in previews or (DATA / "employee.csv").exists()
    retrain_cb = st.checkbox("🧠 Training ulang model dari semua dataset", value=True, key="bulk_retrain_cb")

    if n_new == 0:
        st.markdown("""
            <div style='border:2px dashed #21262d;border-radius:10px;padding:28px 20px;
                        text-align:center;color:#6e7681;font-size:12px;'>
                📂 Belum ada file yang diupload di atas<br>
                <span style='font-size:11px;font-family:DM Mono,monospace;'>Upload minimal satu file CSV untuk melanjutkan</span>
            </div>
        """, unsafe_allow_html=True)
    elif not has_emp:
        st.warning("⚠️ **employee.csv** wajib ada (upload baru atau sudah tersedia). Tanpanya model tidak dapat ditraining.")
    else:
        btn_label = f"🚀 Simpan {n_new} File & {'Training Model' if retrain_cb else 'Update Data'}"
        if st.button(btn_label, type="primary", use_container_width=True):

            prog = st.progress(0, text="💾 Menyimpan semua file...")

            # ── Step 1: Simpan semua file baru ──────────────────
            saved_types = []
            has_cand_new = False
            step = 0
            total_steps = n_new
            for dtype, df_new in previews.items():
                fname = DATA_INFO[dtype][0]
                df_new.to_csv(DATA / fname, index=False)
                saved_types.append(dtype)
                if dtype == "candidates":
                    has_cand_new = True
                    st.session_state["df_candidates"] = df_new
                step += 1
                prog.progress(int(step / total_steps * 30), text=f"💾 Menyimpan {fname}...")

            prog.progress(30, text=f"✅ {n_new} file tersimpan")

            # ── Step 2: Rebuild master dataset ──────────────────
            prog.progress(38, text="🔄 Membangun dataset gabungan...")
            emp_path = DATA / "employee.csv"
            emp  = pd.read_csv(emp_path)
            sal  = pd.read_csv(DATA/"salary_history.csv")   if (DATA/"salary_history.csv").exists()   else None
            att  = pd.read_csv(DATA/"attendance.csv")        if (DATA/"attendance.csv").exists()        else None
            surv = pd.read_csv(DATA/"engagement_survey.csv") if (DATA/"engagement_survey.csv").exists() else None
            perf = pd.read_csv(DATA/"performance.csv")       if (DATA/"performance.csv").exists()       else None
            cand_df = pd.read_csv(DATA/"candidates.csv")     if (DATA/"candidates.csv").exists()        else None

            df_master = build_features(emp, sal, att, surv, perf)
            st.session_state["df_master"] = df_master
            if cand_df is not None:
                st.session_state["df_candidates"] = cand_df

            n_sources = sum([
                sal is not None, att is not None,
                surv is not None, perf is not None,
            ])
            prog.progress(50, text=f"✅ Dataset gabungan: {len(df_master)} karyawan · {n_sources+1} sumber data")

            summary = {
                "total_emp": len(df_master),
                "type":      "bulk",
                "n_files":   n_new,
                "n_sources": n_sources + 1,
            }

            if retrain_cb:
                # ── Step 3: Training ────────────────────────────
                prog.progress(55, text="🤖 Menyiapkan fitur training...")
                import time; time.sleep(0.2)
                prog.progress(65, text="🧠 Training Random Forest + Gradient Boosting...")
                import time; time.sleep(0.2)
                prog.progress(75, text="⚡ Training XGBoost + Logistic Regression...")
                metrics_new, feats_new = train_model(df_master)
                prog.progress(90, text="📊 Menghitung prediksi semua karyawan...")

                preds    = predict_all(df_master)
                high_n   = sum(1 for p in preds if p["risk_level"]=="Tinggi")
                medium_n = sum(1 for p in preds if p["risk_level"]=="Sedang")
                low_n    = len(preds) - high_n - medium_n

                prog.progress(100, text="🎉 Selesai! Model aktif.")

                summary.update({
                    "auc":        metrics_new.get("roc_auc","—"),
                    "precision":  metrics_new.get("precision","—"),
                    "recall":     metrics_new.get("recall","—"),
                    "f1":         metrics_new.get("f1","—"),
                    "n_features": len(feats_new),
                    "high":       high_n,
                    "medium":     medium_n,
                    "low":        low_n,
                })
            else:
                prog.progress(100, text="✅ Data diperbarui.")

            st.session_state["upload_success"] = summary
            import time; time.sleep(0.5)
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL
# ═══════════════════════════════════════════════════════════════

elif page == "⚙️ Model":
    st.markdown("<h2 style='font-family:Syne,sans-serif;color:#e6edf3;margin-bottom:20px;'>⚙️ Model Management</h2>", unsafe_allow_html=True)

    model   = st.session_state["model"]
    metrics = st.session_state["metrics"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Informasi Model")
        if model is not None:
            info = {
                "Status": "✅ Aktif",
                "Trained At": st.session_state["trained_at"] or "—",
                "Jumlah Fitur": len(st.session_state["features"]),
                "XGBoost": "✅ Ya" if HAS_XGB else "❌ Tidak",
                "SHAP": "✅ Ya" if HAS_SHAP else "❌ Tidak",
            }
            for k, v in info.items():
                st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #21262d;font-size:13px;'>
                        <span style='color:#6e7681;font-family:DM Mono,monospace;font-size:11px;'>{k}</span>
                        <span style='color:#e6edf3;'>{v}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Model belum ditraining.")

    with col2:
        st.markdown("### 📈 Metrik Evaluasi")
        if metrics:
            metric_items = [
                ("ROC-AUC",    metrics.get("roc_auc","—"),  "Semakin mendekati 1.0 semakin baik"),
                ("Precision",  metrics.get("precision","—"), "Akurasi prediksi positif"),
                ("Recall",     metrics.get("recall","—"),    "Kemampuan mendeteksi kasus resign"),
                ("F1-Score",   metrics.get("f1","—"),        "Harmonic mean precision & recall"),
            ]
            for name, val, desc in metric_items:
                color = "#00e5a0" if isinstance(val,float) and val >= 0.8 else "#f0b429" if isinstance(val,float) and val >= 0.6 else "#ff6b35"
                st.markdown(f"""
                    <div style='background:#161b22;border:1px solid #21262d;border-radius:8px;padding:12px 16px;margin-bottom:8px;'>
                        <div style='display:flex;justify-content:space-between;align-items:center;'>
                            <div>
                                <div style='font-family:Syne,sans-serif;font-size:13px;color:#e6edf3;font-weight:600;'>{name}</div>
                                <div style='font-size:11px;color:#6e7681;margin-top:2px;font-family:DM Mono,monospace;'>{desc}</div>
                            </div>
                            <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:{color};'>{val}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Belum ada metrik. Training model terlebih dahulu.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔁 Manual Training")
    df = st.session_state["df_master"]

    col_a, col_b = st.columns([2,1])
    with col_a:
        if df is not None:
            st.markdown(f"<div class='info-box'>Dataset siap: <b>{len(df)} karyawan</b> · {len([c for c in FEATURE_COLS if c in df.columns])} fitur tersedia</div>", unsafe_allow_html=True)
        else:
            st.warning("Belum ada data. Upload employee.csv terlebih dahulu.")

    with col_b:
        if df is not None and st.button("🚀 Train Ulang Model", type="primary"):
            with st.spinner("Training model ensemble..."):
                metrics_new, feats = train_model(df)
            st.success(f"✅ Training selesai! AUC = {metrics_new.get('roc_auc','—')}")
            st.rerun()

    # Feature importance
    if model is not None and st.session_state["features"]:
        st.markdown("<br><div class='section-title'>🔑 Feature Importance</div>", unsafe_allow_html=True)
        try:
            feats = st.session_state["features"]
            importances = None
            for name, est in model.named_estimators_.items():
                if name in ("rf","gb","xgb") and hasattr(est, "feature_importances_"):
                    importances = est.feature_importances_
                    break

            if importances is not None:
                fi_df = pd.DataFrame({
                    "feature": feats,
                    "label":   [FEATURE_LABELS.get(f,f) for f in feats],
                    "importance": importances
                }).sort_values("importance", ascending=True).tail(12)

                fig = go.Figure(go.Bar(
                    x=fi_df["importance"], y=fi_df["label"],
                    orientation="h", marker_color="#3d6eff",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b949e", family="DM Sans"),
                    margin=dict(l=0,r=0,t=10,b=0), height=360,
                    xaxis=dict(gridcolor="#21262d", title="Importance"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Feature importance tidak tersedia: {e}")
