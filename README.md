# Car Value Assistant

An interactive Streamlit app that helps users find used cars that meet their requirements and budget. It parses natural-language queries (e.g., "Toyota automatic under 15000 after 2016 <80k miles"), filters a CSV dataset, predicts a fair price with a machine learning model, and ranks results by overall value and closeness to the user's request.

## What this project is

- **Goal**: Recommend cars that best match a user's constraints and preferences while surfacing good-value listings.
- **Data source**: `car_price_prediction_ (1).csv` loaded at runtime.
- **Interface**: Fully chat-based via Streamlit (`app/app.py`).
- **Parsing**: `app/query_parser.py` extracts budget, year range, mileage cap, fuel, transmission, condition, and brand/model hints. It is fuzzy-tolerant to misspellings.
- **Recommendation logic**: `app/model.py` applies filters and ranks candidates using a trained RandomForestRegressor to estimate fair price, combining it with brand preference and nearness-to-constraints into a composite score. If no exact matches exist, it falls back to closely related picks and, when brands are specified, focuses on those brands first.

## How recommendations are computed (algorithm overview)

The app treats recommendation as a two-stage process:

1. **Filtering** (`_apply_filters()`):
   - Numeric filters: `budget` (with optional tolerance), `year_min/year_max`, `mileage_max`.
   - Categorical filters with fuzzy tolerance: `Brand`, `Model`, `Fuel Type`, `Transmission`, `Condition` using RapidFuzz token-set ratio.

2. **Ranking** (`recommend()`):
   - Train-time: A `RandomForestRegressor` is fitted on numeric features (`Engine Size`, `Mileage`, `Age`, `Mileage_per_year`) and one-hot encoded categoricals (`Brand`, `Fuel Type`, `Transmission`, `Condition`, `Model`).
   - Inference: For filtered cars, predict `pred_price` (fair price) and compute `value_score = pred_price - Price`.
   - Compute soft signals:
     - Brand priority for requested brands (ensures asked-for brands surface first when available).
     - Preference score: +1 for `prefer_brands`, −1 for `avoid_brands`.
     - Nearness to numeric constraints (graded closeness to budget, mileage, year bounds).
   - Combine into a composite score and sort: requested-brand-first, then composite descending, then price ascending.
   - Fallback when no exact matches: restrict to requested brands (if present anywhere) and use value + similarity to the request to rank related best picks.

## The RandomForestRegressor explained

- **What it is**: An ensemble of many decision trees trained on bootstrapped samples (bagging) with feature randomness. Each tree learns a mapping from car attributes to price; the forest predicts the average of all trees.
- **Why it works well here**:
  - Handles mixed numeric/categorical features (after one-hot encoding) without assuming linear relationships.
  - Robust to outliers and noisy data via ensembling.
  - Captures non-linear interactions (e.g., certain models in specific years holding value differently).
- **Pipeline used**:
  - `ColumnTransformer`: `StandardScaler` for numeric features; `OneHotEncoder(handle_unknown="ignore", min_frequency=5)` for categoricals.
  - `RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)` as the final step.

## How ML helps in AI, and how this project shows it

- **From rules to learning**: Instead of hardcoding prices or simple heuristics, the model learns price patterns directly from historical data. This allows the assistant to estimate a fair price for any car in the dataset.
- **Decision support**: By comparing the predicted fair price to the listed price, the assistant surfaces high-value options—something difficult to do reliably with static rules.
- **Natural-language robustness**: The system uses fuzzy NLP parsing to interpret imperfect inputs (misspellings, shorthand), integrating them with ML predictions. This blend of language understanding and learned pricing demonstrates how ML components make AI assistants more useful and adaptive.
- **How this app proves this point**:
  - Predicted `pred_price` vs. listed `Price` drives `value_score` and ranking.
  - Recommendations adapt to new constraints without any code changes—only data and model behavior drive results.
  - Fuzzy matching + model-based ranking yields sensible suggestions even for incomplete queries.

## Project structure

- `app/app.py` — Streamlit chat UI and session handling.
- `app/model.py` — Data loading, feature engineering, ML pipeline, filtering, and recommendation logic.
- `app/query_parser.py` — Natural-language parsing and fuzzy keyword extraction.
- `car_price_prediction_ (1).csv` — Dataset of car listings.

## Setup & Run

Prerequisites: Python 3.12+ recommended.

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If requirements.txt is minimal, install directly:
# pip install streamlit scikit-learn pandas numpy rapidfuzz
```

2. Launch the app:

```bash
streamlit run app/app.py --server.port 8501 --server.address 127.0.0.1
```

3. Open the browser at:

- http://127.0.0.1:8501

If you are using an IDE that proxies the server, use the provided preview URL instead.

## Usage tips

- Try natural prompts like:
  - "Toyota automatic under 15000 after 2016 <80k miles"
  - "electric near 25k before 2020"
  - "prefer Honda avoid BMW manual max 12000"
- If your query is too vague, the app will ask for more detail.
- The system is fuzzy-tolerant: misspellings like "automatc" or "toyyta" are often understood.

## System constraints and behavior on specific queries

- **Dataset-bound**: Results come only from `car_price_prediction_ (1).csv`. If a brand/model/year combo isn’t in the CSV, it cannot be recommended.
- **Cold-start training**: The model trains in-memory at startup; extremely large CSVs increase load time.
- **Fuzzy matching limits**: Fuzzy tolerance helps with misspellings but won’t infer entirely different categories (e.g., "gear" won’t map to transmission reliably).
- **Numeric nearness, not absolutes**: When an item slightly exceeds a bound (e.g., budget), it can still rank if close, via a graded "nearness" component.
- **Brand-first ranking**: If you specify brands, matches in those brands are prioritized. If none survive constraints, the app shows best picks from your brands across the dataset with a clear note.
- **Vague queries**: If the message contains no usable signal (no budget/year/mileage nor recognized categories/brands), the app asks for clarification instead of returning a generic list.

### How it acts on specific queries

- **Incomplete text (e.g., "some cars")**
  - Behavior: Asks for more detail (budget, year range, mileage, brand, etc.).

- **Misspellings (e.g., "toyyta automatc gas ~20k 2018-2021")**
  - Behavior: Fuzzy parsing maps tokens to `Toyota`, `automatic`, `petrol/gas` where possible; filters and ranks accordingly.

- **Brand specified but budget too low (e.g., "Toyota under 5000 after 2020")**
  - Behavior: If no Toyota fits numeric bounds, shows a Toyota-focused fallback: "No requested brand within constraints; showing best from requested brands" and lists closest Toyota picks.

- **Conflicting constraints (e.g., "electric diesel manual")**
  - Behavior: Only values present in the dataset after fuzzy matching will survive. If conflicts zero out matches, fallback engages (see above).

- **Only budget given (e.g., "under 15000")**
  - Behavior: Filters by budget, then ranks by value (predicted vs. listed price) and nearness to any other provided numeric constraints.

- **Prefer/avoid phrases (e.g., "prefer Toyota avoid BMW")**
  - Behavior: Adds soft preference signals to ranking and prioritizes requested brands; avoid brands receive negative preference.

- **Model name with brand hint (e.g., "Corolla automatic <80k miles")**
  - Behavior: Tokens are used as brand/model hints; fuzzy filters reduce to likely matches, then ranking applies as usual.

## Notes & limitations

- The model is trained in-memory at startup on the provided CSV; training time and quality depend on data volume and cleanliness.
- Predictions reflect the dataset’s distribution; unseen brands/models or very sparse categories may be underrepresented.
- This is not financial advice; use results as decision support and verify listings independently.
