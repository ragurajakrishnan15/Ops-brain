# OpsBrain: AI-Powered Supply Chain Analytics & RAG Assistant

An operations intelligence platform combining demand forecasting, statistical analysis, interactive dashboards, and an LLM-powered assistant for supply chain analytics. Built end to end: SQL data pipeline, forecasting with honest out-of-sample evaluation and significance tests, a sandboxed text-to-SQL agent plus RAG over operations documents, and a Streamlit data product.

**Live demo:** _link added after deployment_ · **EDA report:** [ragurajakrishnan15.github.io/Ops-brain/eda.html](https://ragurajakrishnan15.github.io/Ops-brain/eda.html)

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Streamlit Dashboard                        │
│ Overview │ Forecasts │ Stat Tests │ Risk │ Data Quality │ Chat 💬 │
├──────────────────────────────────────────────────────────────────┤
│                        Analytics Engine                          │
│  ┌──────────────────┐  ┌──────────┐  ┌────────────────────────┐  │
│  │ XGBoost vs.      │  │  SHAP    │  │ Router → SQL agent     │  │
│  │ baselines,       │  │ drivers  │  │        → RAG (Chroma)  │  │
│  │ champion/series  │  │          │  │ Google Gemini          │  │
│  └────────┬─────────┘  └────┬─────┘  └───────────┬────────────┘  │
│           │   MLflow (runs, models, intervals)   │               │
├───────────┴─────────────────┴────────────────────┴───────────────┤
│                           Data Layer                             │
│  DuckDB warehouse (orders → daily demand → features, risk view)  │
│  Data quality checks     │  Document embeddings (MiniLM, ONNX)   │
└──────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Tools | Why |
|-------|-------|-----|
| **Language** | Python 3.11 | Industry standard for data science |
| **SQL Engine** | DuckDB | Analytical SQL with Snowflake-style window functions, runs locally |
| **ML** | XGBoost, scikit-learn, Optuna | Forecasting + hyperparameter search |
| **Statistics** | statsmodels, SciPy | Diebold–Mariano tests (HAC), Benjamini–Hochberg FDR, logistic regression, chi-square / Cramér's V |
| **Explainability** | SHAP | Per-prediction drivers of each forecast |
| **Experiment Tracking** | MLflow | Every series' model, metrics and intervals are logged |
| **Gen AI / RAG** | LangChain, ChromaDB (ONNX MiniLM embeddings), Google Gemini API | Text-to-SQL and document Q&A |
| **Dashboard** | Streamlit, Plotly | Interactive data product |
| **Deployment** | Posit Connect Cloud, Docker, docker compose | Hosted demo; one-command local deployment |
| **CI/CD** | GitHub Actions | Lint, format check, tests with coverage gate, image build |
| **Documentation** | Quarto | Reproducible EDA report ([`docs/eda.html`](docs/eda.html)) |

## Dataset

- **Supply chain orders**: [DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (Constante, Silva & Pereira, [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6), CC BY 4.0): 180,519 order lines, 2015-01-01 to 2018-01-31, 50 product categories.
- **Operations docs**: 7 SOPs and policies in `data/docs/` (inventory, shipping, data quality, demand planning, KPI glossary, order status, supplier management) for the RAG assistant.

**Structural break:** after 2017-10-02 the catalog changes. Most long-running categories stop and monthly volume falls 81%. Forecasting models are trained and evaluated on data up to that date (`MODEL_END_DATE`). The rest stays in the warehouse for analysis and the assistant.

## Quick Start

The assistant needs a free [Gemini API key](https://aistudio.google.com/apikey). Everything else works without one.

```bash
git clone https://github.com/ragurajakrishnan15/Ops-brain.git
cd Ops-brain
cp .env.example .env          # then set GOOGLE_API_KEY
```

**Fastest: run the dashboard on the prebuilt data.** Without a local warehouse, the app downloads the
[data bundle](https://github.com/ragurajakrishnan15/Ops-brain/releases/tag/data-v1) (18 MB: warehouse,
document store, trained models) on first start:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To rebuild everything from the raw data, put `DataCoSupplyChainDataset.csv` in `data/raw/` and use one of the options below.

**Option 1: Docker (recommended)**

```bash
docker compose up --build                     # first start builds the warehouse and trains models (~5 min)
docker compose --profile mlflow up            # optional: MLflow UI on http://localhost:5000
```

Open http://localhost:8501. `TRAIN_TRIALS` (default 10) sets the Optuna trials used on first start.

**Option 2: Local**

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python scripts/setup_data.py                          # warehouse + document embeddings (~20 s)
python -m src.forecasting.train                       # all eligible series (~5 min; --trials, --level, --series)
python -m src.utils.llm                               # check the Gemini key and model names
streamlit run streamlit_app.py
```

## Configuration (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_API_KEY` | (none) | Gemini key for the assistant (`GEMINI_API_KEY` also works) |
| `GEMINI_MODEL` / `GEMINI_ROUTER_MODEL` | `gemini-3.5-flash` / `gemini-3.5-flash-lite` | Answer/SQL model and question router |
| `RAW_CSV_PATH` | `data/raw/DataCoSupplyChainDataset.csv` | Input dataset |
| `DUCKDB_PATH`, `CHROMA_PERSIST_DIR`, `MLFLOW_TRACKING_URI` | under `data/processed/`, `mlruns/` | Storage locations (relative to the project root) |
| `MODEL_END_DATE` | `2017-10-02` | Last date used for training/evaluation |
| `MIN_HISTORY_DAYS`, `FORECAST_HORIZON_DAYS` | `365`, `28` | Series eligibility and forecast horizon |
| `DATA_BUNDLE_URL`, `DATA_BUNDLE_SHA256` | `data-v1` release asset | Prebuilt data downloaded when the warehouse is missing |
| `ASSISTANT_SESSION_LIMIT`, `ASSISTANT_DAILY_LIMIT` | `10`, `200` | Question limits for the public demo |

## Deployment (Posit Connect Cloud)

The hosted demo runs on [Posit Connect Cloud](https://connect.posit.cloud) from this public repo:

1. **Publish → Streamlit**, choose this repository and branch `main`, primary file `streamlit_app.py`, Python 3.11.
2. Under **Advanced settings → Secret variables**, add `GOOGLE_API_KEY`.
3. Publish. On first start the app downloads the data bundle (no raw data or training in the cloud).
   Pushes to `main` republish automatically.

To refresh the bundle after retraining: `python scripts/build_bundle.py` (also precomputes the slow
dashboard tables), then `gh release upload data-v1 dist/opsbrain-data.zip --clobber`.

## Project Structure

```
opsbrain/
├── src/
│   ├── data_pipeline/
│   │   ├── ingest.py          # CSV → DuckDB: date parsing, PII removal, canceled/fraud flags
│   │   ├── transforms.py      # Dense daily demand (category + department), features, risk view
│   │   ├── feature_store.py   # Parameterised loaders for both series levels
│   │   └── quality.py         # Checks from the data quality standards
│   ├── forecasting/
│   │   ├── features.py        # Python mirror of the SQL features; recursive forecasting; baselines
│   │   ├── train.py           # Optuna + rolling backtests → champion, intervals → MLflow
│   │   ├── predict.py         # History / holdout backtest / 28-day future with 95% intervals
│   │   ├── explain.py         # SHAP global importance and per-day drivers
│   │   ├── significance.py    # Diebold–Mariano test (Newey–West HAC), Benjamini–Hochberg
│   │   └── alerts.py          # Forecast health flags (WAPE target, coverage, interval width)
│   ├── analytics/
│   │   └── late_delivery.py   # Chi-square + Cramér's V, logistic regression, LR test
│   ├── rag_assistant/
│   │   ├── ingest_docs.py     # Header-aware chunking → Chroma (idempotent)
│   │   ├── retriever.py       # Similarity search with file + section citations
│   │   ├── sql_agent.py       # Text-to-SQL with a read-only, no-file-access sandbox and retry
│   │   └── chat.py            # Routes questions to SQL or RAG
│   ├── dashboard/
│   │   ├── views/             # overview, forecasts, statistics, inventory_risk, data_quality, assistant
│   │   ├── data.py            # Cached data access
│   │   └── components.py      # Plotly charts
│   └── utils/                 # config, DuckDB helper, Gemini client, data bundle, precomputed tables
├── data/
│   ├── raw/                   # Kaggle CSV (gitignored)
│   ├── processed/             # DuckDB warehouse + Chroma store (gitignored)
│   └── docs/                  # Operations SOPs for RAG
├── notebooks/eda.qmd          # Quarto EDA → docs/eda.html
├── streamlit_app.py           # Dashboard entry point (st.navigation); fetches the data bundle if needed
├── scripts/
│   ├── setup_data.py          # Build warehouse + embeddings
│   ├── build_bundle.py        # Deployment data bundle + precomputed tables
│   └── bootstrap.py           # Docker entrypoint: setup/train only if missing
├── tests/                     # pytest: synthetic-data pipeline/forecasting tests, sandbox, retrieval, dashboard
├── .github/workflows/ci.yml   # Lint + format + tests (coverage ≥ 70%) + Docker build
├── Dockerfile, docker-compose.yml
├── requirements.txt, requirements-dev.txt
└── BUILD_PLAN.md              # How the project was completed, phase by phase
```

## Modules

### 1. Data Pipeline (`src/data_pipeline/`)
Loads the CSV into DuckDB and builds everything else in SQL:
- **Cleaning:** parses DataCo's `M/D/YYYY H:MM` timestamps, drops PII columns, trims stray whitespace in names, and flags canceled and suspected-fraud orders so they're excluded from demand.
- **Daily demand:** densified to one row per category per day, with zero-filled gaps, so a 7-row lag really is 7 days.
- **Features and risk:** lag and rolling features use only past days (windows end at `1 PRECEDING`). The inventory risk view blends late-delivery rate with demand volatility on a 0–100 scale.

### 2. Forecasting Engine (`src/forecasting/`)
Each series is a category or a department with at least 365 days of history: 24 categories and 6 departments.
1. **Test window:** hold out the last 28 days.
2. **Tuning:** Optuna tunes an XGBoost model that predicts the *deviation from the trailing 28-day average*.
3. **Champion selection:** six rolling 28-day backtests compare XGBoost against a 28-day moving average and a same-weekday-last-week forecast. The lowest backtest WAPE becomes the series' champion, and the backtest errors at each horizon set its 95% interval widths.
4. **Holdout scoring:** every method is scored once on the 28 held-out days.
5. **Significance:** a Diebold–Mariano test compares XGBoost's and the moving average's daily absolute errors over the 196 consecutive tested days (Newey–West variance, lag 27, because errors from the same forecast origin are correlated).
6. **Final model:** XGBoost is refit on all the data and logged to MLflow with its backtests, holdout and test result.

Multi-day forecasts are recursive, using a Python copy of the SQL features (a test checks every row matches).

### 3. Operations Assistant (`src/rag_assistant/`)
A router sends each question to one of two modes:
- **Data questions:** a text-to-SQL agent.
  - The generated SQL must be a single `SELECT`. It runs on a read-only DuckDB connection with external access disabled and the configuration locked, and results are capped at 200 rows.
  - A failed query is retried once with the error message, and results are summarized in plain language.
- **Document questions:** RAG over the SOPs, with file and section citations. Embeddings run locally (MiniLM); Gemini writes the answers.

### 4. Statistical Analysis (`src/forecasting/significance.py`, `src/analytics/`)
- **Forecast significance:** Diebold–Mariano tests for all 30 series, with Benjamini–Hochberg adjustment so 30 tests don't produce false discoveries.
- **Late-delivery drivers:** chi-square tests of independence with Cramér's V effect sizes; a logistic regression (odds ratios with 95% CIs) of late delivery on shipping mode, market, segment, department, quantity and discount; and a likelihood-ratio test of whether anything beyond shipping mode matters. First Class is perfectly separated (every line is late), so it's reported as a rule and excluded from the fit.

### 5. Dashboard (`streamlit_app.py`, `src/dashboard/`)
Six pages:
- **Overview:** what the project is, key findings (computed live), KPIs, the demand trend with the catalog break, and forecast alerts.
- **Demand Forecasts:** history, holdout and future, a method comparison, SHAP drivers, and a warning for discontinued series.
- **Statistical Tests:** the significance tests and late-delivery models, as forest plots and tables.
- **Inventory Risk:** late-delivery rate vs. volatility.
- **Data Quality:** checks from the data quality standards.
- **Operations Assistant:** chat that keeps its SQL, result tables and sources; per-visit and daily question limits for the public demo.

## Results

Holdout = the last 28 days before `MODEL_END_DATE`. These days were never used for training or method selection.

| Series volume | Series | Median holdout WAPE | XGBoost chosen | 95% interval coverage |
|---|---|---|---|---|
| 20+ units/day | 10 | 0.20 | 5 | 0.93 |
| 5–20 units/day | 6 | 0.36 | 4 | 0.93 |
| < 5 units/day | 14 | 0.96 | 13 | 0.98 |

- XGBoost and the moving average are within ~0.01 WAPE of each other on most series. With leakage removed, DataCo daily demand is close to noise around a stable level:
  - weekdays differ by only 2.2%;
  - the within-category lag-7 correlation is 0.01;
  - discounts don't move daily volume.

  See the [EDA report](https://ragurajakrishnan15.github.io/Ops-brain/eda.html). Choosing a champion per series, rather than assuming the ML model wins, is what keeps the published forecast honest.
- **Is the difference real?** Diebold–Mariano tests with Benjamini–Hochberg adjustment (5% FDR): XGBoost is significantly more accurate on **5 of 30** series, the moving average on **0**, and there is **no significant difference on 25**. Four of the five are sparse categories (~1–1.6 units/day) whose level was falling, which a trailing average follows slowly; the fifth is Electronics (errors 1.4% of demand lower). Without the adjustment, two more series would have looked significant.
- Low-volume categories are intermittent. Their daily WAPE is high whatever method is used, and the dashboard flags them against the planning targets.
- **Late deliveries are a scheduling problem.** Among valid orders, **100%** of First Class lines are late (promised 1 day, shipped in 2) and Second Class lines have **6.0×** the odds of Standard Class (95% CI 5.8–6.2). Shipping mode has Cramér's V = 0.48; every other driver is ≤ 0.02, and a likelihood-ratio test of market, segment, department, quantity and discount together gives p = 0.71 (pseudo-R² 0.0850 → 0.0850).

## Testing

```bash
pytest                       # everything that can run locally (Gemini tests skip without a key)
pytest -m llm -v             # 10-question live assistant evaluation (needs GOOGLE_API_KEY + setup_data)
pytest --cov=src             # coverage
```

The pipeline and forecasting tests build synthetic DataCo-format data, so CI doesn't need the Kaggle dataset. They cover:
- date parsing and zero-filling of missing days;
- no-leakage checks;
- parity between the SQL and Python feature definitions;
- end-to-end training, logging, forecasting and SHAP.

Other suites cover the SQL sandbox (DROP, COPY, ATTACH, SET and `read_csv` are all blocked), retrieval quality on the real docs, and headless rendering of every dashboard page.

## Interview Talking Points

- **"Why DuckDB?"** It runs the same analytical SQL patterns as Snowflake (window functions, CTEs) locally, at zero cloud cost. The whole feature pipeline is SQL.
- **"Your model barely beats a moving average. Isn't that bad?"** The first version scored far better, but it was leaking: it used same-day order counts and rolling windows that included the target day. After removing the leakage, the EDA shows there's almost no within-series signal, so the right answer is to choose the champion per series on backtests and report the honest holdout. A Diebold–Mariano test then shows the difference is statistically significant on only 5 of 30 series after a false-discovery-rate correction.
- **"What statistics did you use?"** Diebold–Mariano tests with Newey–West standard errors (multi-step errors are autocorrelated), Benjamini–Hochberg for multiple testing, chi-square tests with Cramér's V (with 173k rows, effect size matters more than p-values), and a logistic regression with a likelihood-ratio test. The regression also hit perfect separation (First Class is always late), which I reported as a rule rather than letting the fit diverge.
- **"How do you know the intervals are right?"** They're calibrated on out-of-sample backtest errors at each horizon, not on training residuals. Holdout coverage averages 95% for a 95% interval.
- **"How does the RAG work?"**
  - Markdown docs are split by section, then into 512-character chunks with 50 characters of overlap.
  - The chunks are embedded locally with MiniLM and stored in Chroma (cosine similarity). Re-ingesting replaces the collection.
  - The top 5 chunks are passed to Gemini with their file and section, and the prompt requires citations.
- **"Isn't text-to-SQL dangerous?"** The LLM never gets a writable connection. Queries are parsed by DuckDB and must be a single SELECT, and they run with external access disabled and the configuration locked. Tests try DROP, COPY, ATTACH, SET and `read_csv`.
- **"What does SHAP show?"** For a chosen day, how each feature moved the forecast away from the 28-day average, in units. This is how a planner decides whether to trust a forecast.
- **"Why Streamlit over a notebook?"** A notebook is analysis; a Streamlit app is a product someone can use without writing code. The Quarto report covers the analysis side.

## Next Steps

1. Add an intermittent-demand model (Croston/TSB via `statsforecast`) as a fourth candidate for low-volume categories.
2. Forecast at weekly granularity for sparse series, where daily WAPE is dominated by noise.
3. Add authentication with `streamlit-authenticator` before sharing the dashboard.
4. Connect to Snowflake (free trial) instead of DuckDB; the SQL is portable.

## License
MIT
