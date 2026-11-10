# OpsBrain — Build Plan to Completion

Status as of 2026-11-10 (start): all modules were scaffolded (~1,800 lines), but the pipeline did not
produce usable data end to end. **Phases 0–6 are now complete**; open items are listed at the end of Phase 6. This plan orders the work so each phase leaves the project
runnable and verifiable before the next one starts.

**Definition of done:** a fresh clone → `docker compose up` (or the local steps) → dashboard shows
real forecasts, a correct risk view and a working assistant; CI is green; README claims match the code.

---

## Inventory: what exists vs. what's missing

| Area | State | Missing / broken |
|---|---|---|
| Raw data | ✅ `DataCoSupplyChainDataset.csv` (180,519 rows), field dictionary, access logs | Filename the code expects (`supply_chain.csv`) |
| Ops docs for RAG | ✅ 3 markdown SOPs (~860 words) | More docs for useful retrieval (optional) |
| Ingest / clean | ⚠️ Written | Date parsing gives NULL for all rows; no order-status filtering |
| Feature engineering | ⚠️ Written | Uses same-day values as inputs (leakage); lags count rows, not days; Electronics has 2 departments |
| Training | ⚠️ Written | No `category` param logged; 24/50 categories too short |
| Prediction | ❌ Broken | MLflow filter uses `OR` (unsupported); in-sample "forecast"; no future horizon |
| SHAP | ⚠️ Functions exist | Not wired into the dashboard |
| RAG | ⚠️ Written | Duplicate chunks on re-ingest; models reloaded per query; deprecated imports |
| SQL agent | ⚠️ Written | `tabulate` missing; no file-access sandboxing |
| Dashboard | ⚠️ Single file | README-listed `pages/*.py` don't exist; no caching; SHAP & data-quality views missing |
| Tests | ❌ 4 trivial tests | Nothing exercises the real SQL or models |
| CI | ❌ Fails | 28 ruff errors; installs ~2 GB of deps (torch) on every run |
| Docker | ⚠️ | `curl` missing for healthcheck; no data bootstrap; obsolete `version:` key |
| EDA notebook | ⚠️ Python only | README says "R + Python"; Quarto not installed locally |
| Git | ❌ Not a repo | `git init`, first commit, push to GitHub |

---

## Phase 0 — Environment (½ day)

- [x] Create `.venv`, install `requirements.txt` (+ `tabulate`, now pinned in `requirements.txt`)
- [x] Pre-download the embedding model `sentence-transformers/all-MiniLM-L6-v2` (in the Hugging Face cache)
- [x] Install Quarto 1.11.9 — `pip install quarto-cli` fails on Windows (260-char path limit while building the wheel),
      so the official zip is unpacked at `D:\opsbrain-project\tools\quarto\bin\quarto.exe` (outside the repo)
- [ ] `git init`, commit the current state as a baseline
- [x] LLM switched from Anthropic Claude to **Google Gemini** (`langchain-google-genai==2.1.12`, `src/utils/llm.py`)
- [ ] Copy `.env.example` → `.env`, add `GOOGLE_API_KEY` (free at aistudio.google.com/apikey), then run `python -m src.utils.llm`
- [ ] Split dependencies: `requirements.txt` (runtime) and `requirements-dev.txt` (pytest, ruff)

**Check:** `python -c "import duckdb, xgboost, chromadb, langchain_google_genai"` succeeds. ✅

## Phase 1 — Make the data pipeline correct (1 day) ✅ done 2026-11-10

1. [x] `ingest.py`: default path → `DataCoSupplyChainDataset.csv`, configurable via `RAW_CSV_PATH`.
   (Kept pandas for loading: DuckDB's `read_csv(encoding='latin-1')` rejects this file.)
2. [x] Parse dates with `try_strptime(..., '%m/%d/%Y %H:%M')`; setup now fails loudly if any date doesn't parse.
3. [x] Canceled / suspected-fraud lines kept in `orders` with `is_valid_demand = FALSE`, excluded from demand (7,754 lines).
4. [x] PII columns dropped at ingest.
5. [x] `daily_demand` keyed by category; department = most common one (`MODE`), which fixes Electronics.
6. [x] Calendar densified per category (8,259 zero-demand days filled, flagged `is_filled`).
7. [x] Leakage removed: rolling windows end at `1 PRECEDING`; same-day counts replaced by `lag_1d_order_count`,
   `lag_1d_unique_customers`; `FEATURE_COLS` is now 16 columns.
8. [x] `inventory_risk`: late % = late lines / order lines; volatility CV capped at 100 → score in 0–100.
9. [x] `setup_data.py`, `.env.example` and the SQL-agent schema text updated.
10. [x] (found during the run) Trailing/double spaces trimmed from text keys (`'Cameras '`, `'South of  USA '`).
11. [x] `tests/test_pipeline_sql.py`: 8 tests run the real SQL on a synthetic DataCo-format CSV.

**Result on the real data:** 180,519 orders → 25,222 daily rows → 23,828 feature rows, 0 NULL features, 50 categories;
late-delivery % ranges 49.3–68.9, risk score 43.3–81.3. Setup takes ~16 s.

> **Decision (2026-11-10): both** — model categories with ≥ 365 days *and* departments.

## Phase 2 — Forecasting that holds up ✅ done 2026-11-10

1. [x] Parameterised feature-store queries; series at two levels (`category`, `department`; new
   `daily_demand_department` / `department_features` tables built from the same SQL template).
2. [x] **Structural break handled:** after 2017-10-02 volume drops ~80% and most long-running categories end, so models
   train/evaluate on data up to `MODEL_END_DATE=2017-10-02`. Eligible: 24 categories + 6 departments.
3. [x] Experiments showed the 4 revenue/order lags didn't help (better on 2 of 6 series, worse on 4) → 12 units-only features, which
   makes recursive multi-day forecasting exact. `features.py` mirrors the SQL; a parity test checks every row.
4. [x] XGBoost predicts the deviation from the 28-day average (`reg:absoluteerror`); Optuna tunes on pre-holdout data only.
5. [x] **Champion per series:** XGBoost vs. 28-day moving average vs. seasonal naive, chosen on 6 rolling 28-day recursive
   backtests; the last 28 days are held out and only reported. 95% intervals per horizon from backtest errors.
6. [x] MLflow: experiment `opsbrain-forecast`, tags `series_level/series_name/champion`, `forecast_meta.json` + `holdout.csv`
   artifacts; `predict.py` filters by name in pandas (apostrophes safe, no `OR`). Windows paths → `file://` URIs.
7. [x] WAPE as primary metric; MAPE only on non-zero days. SHAP explains deviation from the 28-day average, in units.
8. [ ] (Stretch) A statistical model (`statsforecast` AutoETS / Croston for intermittent series) as a 4th candidate.

**Result (holdout, 30 series):** champion = XGBoost on 22, moving average on 8 — margins are usually < 0.01 WAPE.
High-volume categories reach 0.17–0.33 WAPE; sparse categories (< 5 units/day) ~0.6–1.3. Mean 95% interval
coverage ≈ 0.95. Outdoors (dept) holdout WAPE 1.38 reflects a mid-September level drop no method could anticipate.
**Honest takeaway:** once leakage is removed, DataCo daily demand is mostly noise around a level; the tuned model only
matches a moving average. That's the expected interview answer, not a bug.

## Phase 3 — RAG + SQL assistant hardening ✅ done 2026-11-10 (live LLM eval pending a Gemini key)

1. [x] Maintained packages: `langchain-chroma==0.2.3`, `langchain-huggingface==0.3.1`, `langchain_core`, `langchain_text_splitters`.
2. [x] Model ids in `config.py` (`GEMINI_MODEL`, `GEMINI_ROUTER_MODEL`); one cached client per role (`get_llm`).
3. [x] Ingest recreates the collection (verified: 56 chunks after two runs); `source` = file name; header-aware splitting
   adds a `section` field used in citations; cosine similarity.
4. [x] Embedder and vector store cached per process. Chroma's broken telemetry silenced.
5. [x] SQL sandbox: `duckdb.extract_statements` must yield one SELECT; read-only connection with
   `enable_external_access=false` + `lock_configuration`; 200-row cap; one retry with the DuckDB error fed back.
6. [x] Results summarised by Gemini (1–3 sentences); table + SQL in expanders.
7. [x] Corpus 3 → 7 docs: demand planning process, KPI glossary, order status & cancellation policy, supplier management.
   Inventory SOP aligned to WAPE targets so documents don't contradict each other.

**Check:** offline — retrieval finds the expected doc in the top 3 for 10/10 questions; sandbox blocks DROP/DELETE/COPY/ATTACH/
SET/INSTALL/multi-statement/`read_csv`. Live — `pytest -m llm` (5 doc + 5 data questions vs. DuckDB ground truth): **not yet run**.

## Phase 4 — Dashboard completion ✅ done 2026-11-10

1. [x] `st.navigation` with 5 views in `src/dashboard/views/` (renamed from `pages/` — Streamlit warns when both are used);
   `app.py` puts the project root on `sys.path`, which `streamlit run` didn't do before.
2. [x] All queries/forecasts behind `st.cache_data`; models behind `st.cache_resource` (`src/dashboard/data.py`).
3. [x] Forecasts: level + series pickers, holdout metrics vs. 28-day average, history/holdout/future chart with intervals,
   next-28-days table, method comparison, SHAP importance + waterfall for any of the last 120 days.
4. [x] Data Quality view (`src/data_pipeline/quality.py`): 16 rules from the standards doc — 13/13 completeness+validity pass;
   459 demand-day outliers and 18.7% negative-profit lines flagged for review; table freshness/coverage; null rates.
5. [x] Alerts (`src/forecasting/alerts.py`): WAPE above target (25% / 60% by volume), coverage < 85%, and — for high-volume
   series only — intervals > 1.5× the forecast. Shown on the Overview.
6. [x] Chat history keeps mode, SQL, result table and sources; assistant shows setup steps when no key is set.
7. [x] Inventory Risk: late-delivery vs. volatility scatter (the category × department heatmap was mostly empty) with a
   minimum-volume filter.

**Check:** `tests/test_dashboard.py` renders every view with real data via `AppTest` (no exceptions/errors, expected headers,
series switching); `streamlit run` boots and `/_stcore/health` returns ok. Note: `AppTest` in Streamlit 1.42 doesn't execute
`st.navigation` pages, so views are tested as standalone scripts.

## Phase 5 — Tests & CI ✅ done 2026-11-10

1. [x] Synthetic DataCo-format fixtures instead of a sample of the Kaggle file (no licensed data in the repo):
   40-day pipeline fixture (`test_pipeline_sql.py`) and a 450-day forecasting fixture (`test_forecasting.py`).
2. [x] Tests: date parsing, densification, canceled/fraud exclusion, no-leakage, SQL↔Python feature parity, apostrophe names,
   risk % bounds, department sums, quality rules (outlier count cross-checked in pandas), end-to-end train → MLflow → forecast
   → SHAP (contributions add up) → alerts, recursive day 1 = one-step, SQL sandbox, retry loop, ingest idempotency,
   retrieval top-3 on the real docs, headless dashboard rendering. 58 pass locally; 10 live-Gemini tests skip without a key.
3. [x] `ruff check` + `ruff format --check` clean on `src/ tests/ scripts/` (was 28 lint errors).
4. [x] CI: pip + Hugging Face caches, CPU-only torch, coverage gate (`--cov-fail-under=70`), import checks, coverage artifact,
   Docker build job (Buildx with GHA cache). `requirements-dev.txt` split from runtime `requirements.txt`.

**Check:** coverage of `data_pipeline` + `forecasting` + `rag_assistant` + `utils` is **79%** with the real data hidden
(CI conditions). CI itself hasn't run yet — it runs on the first push.

## Phase 6 — Packaging, docs, release ✅ done 2026-11-10 (push to GitHub pending)

1. [x] Dockerfile: CPU-only torch, `xgboost-cpu` (drops ~470 MB of NCCL/CUDA), non-root `opsbrain` user, embedding model
   baked in, Python healthcheck (no curl needed), `.dockerignore`. Image 4.1 GB (from 5.5 GB).
2. [x] `scripts/bootstrap.py` as the container entrypoint: builds the warehouse / trains models only when missing.
3. [x] docker compose: no obsolete `version:`, optional `.env`, named `mlruns` volume (MLflow stores absolute artifact paths,
   so host-trained runs can't load in the container), `TRAIN_TRIALS`, optional `mlflow` profile on :5000.
   **Verified:** `docker compose up` → trained 30 series in-container → healthy in ~4.5 min → forecasts load inside the container.
4. [x] `notebooks/eda.qmd` rewritten: all 4 original "key findings" were contradicted by the data (no weekly seasonality; late
   deliveries driven by shipping mode, not category; lag-7 correlation 0.89 pooled but 0.01 within category; discounts
   don't move volume). Findings are now computed in code. Rendered with Quarto 1.11.9 → `docs/eda.html`. Python only.
5. [x] README rewritten: real structure, Gemini setup, config table, results table, corrected talking points.
6. [x] Config resolves relative paths against the project root; `.gitattributes` keeps LF endings.
7. [x] `git init` + initial commit on `main`.
8. [ ] Create the GitHub repo and push; tag `v1.0.0` once CI is green.
9. [ ] Screenshots for the README (needs a browser session).
10. [ ] Run `pytest -m llm` once a Gemini key is in `.env`.

## Suggested order & effort

| Phase | Effort | Unblocks |
|---|---|---|
| 0 Environment | 0.5 d | everything |
| 1 Data pipeline | 1 d | 2, 4 |
| 2 Forecasting | 1–1.5 d | 4 (forecast page) |
| 3 Assistant | 1 d | 4 (assistant page) |
| 4 Dashboard | 1–1.5 d | 6 |
| 5 Tests & CI | 1 d (do alongside 1–4) | release |
| 6 Packaging & docs | 0.5–1 d | release |
| **Total** | **≈ 6–7.5 days** | |

## External resources

| Resource | Source | Status |
|---|---|---|
| DataCo dataset | Kaggle `shashwatwork/dataco-smart-supply-chain-for-big-data-analysis` | ✅ in `data/raw/` |
| Python deps | PyPI (`requirements.txt`) | installed into `.venv` |
| `tabulate` | PyPI | installed (needed by `DataFrame.to_markdown`) |
| Embedding model `all-MiniLM-L6-v2` | Hugging Face Hub | ✅ in HF cache (and baked into the Docker image) |
| Quarto CLI | github.com/quarto-dev/quarto-cli (zip; pip wheel fails on Windows) | ✅ `D:\opsbrain-project	ools\quarto` |
| Gemini API key | aistudio.google.com/apikey (free tier) | **user must provide** |
| R + reticulate | CRAN | not needed — notebook and README are Python-only |
