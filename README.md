# RetainAI — Streamlit Version

Employee Intelligence Platform untuk analisis risiko resign dan candidate matching.

## Cara Menjalankan

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi:
```bash
streamlit run app.py
```

3. Buka browser di `http://localhost:8501`

## Struktur File

```
retainai_streamlit/
├── app.py                    # Main Streamlit app
├── requirements.txt
├── data/
│   └── raw/
│       ├── employee.csv
│       ├── salary_history.csv
│       ├── attendance.csv
│       ├── engagement_survey.csv
│       ├── performance.csv
│       └── candidates.csv
└── models/
    └── resign_model.pkl      # Saved model (auto-generated)
```

## Fitur

- 🏠 **Dashboard** — Overview statistik dan distribusi risiko
- ⚠️ **Risiko Karyawan** — Tabel lengkap dengan filter & probabilitas resign
- 🔍 **Analisis XAI** — Explainable AI per karyawan dengan SHAP/Heuristic
- 🎯 **Kandidat Matching** — Smart matching dengan NLP + retention scoring
- 📂 **Upload Data** — Upload CSV langsung dari UI
- ⚙️ **Model** — Metrik evaluasi, feature importance, manual retrain
