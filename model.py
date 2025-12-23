import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from rapidfuzz import process, fuzz

NUM_COLS = ["Engine Size", "Mileage", "Age", "Mileage_per_year"]
CAT_COLS = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]
TARGET = "Price"


def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    current_year = int(df["Year"].max()) + 1
    df["Age"] = (current_year - df["Year"]).clip(lower=0)
    df["Mileage_per_year"] = df["Mileage"] / df["Age"].replace(0, 1)
    return df


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CAT_COLS),
        ]
    )
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe


def _fuzzy_match(requested: list, candidates: list, threshold: int, topk: int) -> list:
    req = [r for r in (requested or []) if isinstance(r, str)]
    if not req:
        return []
    cand = [c for c in candidates if isinstance(c, str)]
    matched = set()
    for r in req:
        # Use token_set_ratio for robustness to word order
        res = process.extract(r, cand, scorer=fuzz.token_set_ratio, limit=topk)
        for label, score, _ in res:
            if score >= threshold:
                matched.add(label)
    return list(matched)


def _safe_lower_list(xs):
    return [str(x).lower() for x in xs]


def get_model_and_df(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = _feature_engineer(df)

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    pipe = _build_pipeline()
    pipe.fit(X, y)
    return pipe, df


def _apply_filters(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    dff = df.copy()

    # Budget filter uses listed price (Price column)
    budget = constraints.get("budget")
    budget_tol = constraints.get("budget_tolerance_pct", 0.0) or 0.0
    if budget is not None:
        limit = float(budget) * (1.0 + float(budget_tol))
        dff = dff[dff["Price"] <= limit]

    # Year range
    year_min = constraints.get("year_min")
    year_max = constraints.get("year_max")
    if year_min is not None:
        dff = dff[dff["Year"] >= year_min]
    if year_max is not None:
        dff = dff[dff["Year"] <= year_max]

    # Mileage max
    mileage_max = constraints.get("mileage_max")
    if mileage_max is not None:
        dff = dff[dff["Mileage"] <= mileage_max]

    # Brands: only filter if requested names match known brands in data
    brands = constraints.get("brands")
    if brands:
        thresh = int(constraints.get("fuzzy_threshold", 85))
        topk = int(constraints.get("fuzzy_topk", 3))
        cands = list(dff["Brand"].astype(str).unique())
        matched = _fuzzy_match(brands, cands, threshold=thresh, topk=topk)
        # retry with a slightly lower threshold if nothing matched
        if not matched and thresh > 70:
            matched = _fuzzy_match(brands, cands, threshold=max(70, thresh - 15), topk=topk)
        if matched:
            dff = dff[dff["Brand"].astype(str).str.lower().isin(_safe_lower_list(matched))]

    # Models (optional): only filter if requested names match known models
    models = constraints.get("models")
    if models:
        thresh = int(constraints.get("fuzzy_threshold", 85))
        topk = int(constraints.get("fuzzy_topk", 3))
        cands = list(dff["Model"].astype(str).unique())
        matched = _fuzzy_match(models, cands, threshold=thresh, topk=topk)
        if matched:
            dff = dff[dff["Model"].astype(str).str.lower().isin(_safe_lower_list(matched))]

    # Fuel/Transmission/Condition with fuzzy tolerance
    thresh = int(constraints.get("fuzzy_threshold", 85))
    topk = int(constraints.get("fuzzy_topk", 3))

    fuel = constraints.get("fuel")
    if fuel:
        cands = list(dff["Fuel Type"].astype(str).unique())
        matched = _fuzzy_match(fuel, cands, threshold=thresh, topk=topk)
        if matched:
            dff = dff[dff["Fuel Type"].astype(str).str.lower().isin(_safe_lower_list(matched))]

    trans = constraints.get("transmission")
    if trans:
        cands = list(dff["Transmission"].astype(str).unique())
        matched = _fuzzy_match(trans, cands, threshold=thresh, topk=topk)
        if matched:
            dff = dff[dff["Transmission"].astype(str).str.lower().isin(_safe_lower_list(matched))]

    conds = constraints.get("condition")
    if conds:
        cands = list(dff["Condition"].astype(str).unique())
        matched = _fuzzy_match(conds, cands, threshold=thresh, topk=topk)
        if matched:
            dff = dff[dff["Condition"].astype(str).str.lower().isin(_safe_lower_list(matched))]

    return dff


def _format_results(df_res: pd.DataFrame, top_n: int, note: str) -> str:
    if df_res.empty:
        return "No matches found. Try relaxing constraints."

    cols = ["Brand", "Model", "Year", "Fuel Type", "Transmission", "Mileage", "Condition", "Price", "pred_price", "value_score"]
    view = df_res[cols].copy()

    lines = [
        f"**Top {min(top_n, len(view))} value picks**",
        f"{note}",
        "",
    ]

    for i, row in view.head(top_n).iterrows():
        lines.append(
            f"- **{row['Brand']} {row['Model']} {int(row['Year'])}** | {row['Fuel Type']}, {row['Transmission']}, {int(row['Mileage']):,} mi, {row['Condition']} | Listed: ${row['Price']:.0f} | Fair: ${row['pred_price']:.0f} | Value: ${row['value_score']:.0f}"
        )

    return "\n".join(lines)


def recommend(model: Pipeline, df: pd.DataFrame, constraints: dict, top_n: int = 5) -> str:
    # Guard: if query is too vague, ask for clarification instead of showing a generic list
    def _has_signal() -> bool:
        # numeric constraints
        if any(constraints.get(k) is not None for k in ("budget", "mileage_max", "year_min", "year_max")):
            return True
        # categorical constraints present
        if any((constraints.get(k) or []) for k in ("fuel", "transmission", "condition")):
            return True
        # brand/model hints that actually match dataset labels
        qb = []
        for key in ("brands", "models", "prefer_brands"):
            qb.extend(constraints.get(key, []) or [])
        qb = [q for q in qb if isinstance(q, str)]
        if qb:
            cands_b = list(df["Brand"].astype(str).unique())
            cands_m = list(df["Model"].astype(str).unique())
            mb = _fuzzy_match(qb, cands_b, threshold=int(constraints.get("fuzzy_threshold", 85)), topk=int(constraints.get("fuzzy_topk", 3)))
            mm = _fuzzy_match(qb, cands_m, threshold=int(constraints.get("fuzzy_threshold", 85)), topk=int(constraints.get("fuzzy_topk", 3)))
            if mb or mm:
                return True
        return False

    if not _has_signal():
        return (
            "I need a bit more detail. Please include one or more of: budget (e.g., 'under 15000'), year range (e.g., '2018-2021' or 'after 2016'), mileage cap (e.g., '<80k miles'), fuel (petrol/diesel/electric/hybrid), transmission (automatic/manual), condition (used/like new/new), or brand/model (e.g., Toyota Corolla)."
        )

    dff = _apply_filters(df, constraints)

    if dff.empty:
        # Fallback: show related best picks by relaxing non-numeric filters
        df_relax = df.copy()
        # Apply only numeric constraints (budget with tolerance, year range, mileage)
        c_min = {}
        for k in ["budget", "budget_tolerance_pct", "year_min", "year_max", "mileage_max"]:
            if k in constraints:
                c_min[k] = constraints[k]
        df_relax = _apply_filters(df_relax, c_min)
        if df_relax.empty:
            df_relax = df.copy()

        # If user asked for brands, and they exist globally, restrict fallback pool to those brands
        qb = []
        for key in ("brands", "prefer_brands"):
            qb.extend(constraints.get(key, []) or [])
        qb = [q for q in qb if isinstance(q, str)]
        if qb:
            thresh_b = int(constraints.get("fuzzy_threshold", 85))
            topk_b = int(constraints.get("fuzzy_topk", 3))
            cands_all = list(df["Brand"].astype(str).unique())
            matched_global = _fuzzy_match(qb, cands_all, threshold=thresh_b, topk=topk_b)
            if not matched_global and thresh_b > 70:
                matched_global = _fuzzy_match(qb, cands_all, threshold=max(70, thresh_b - 15), topk=topk_b)
            if matched_global:
                df_relax = df_relax[df_relax["Brand"].astype(str).str.lower().isin(_safe_lower_list(matched_global))]

        # Score similarity to provided brand/model and categorical preferences
        bm_hints = []
        for key in ("brands", "models", "prefer_brands"):
            bm_hints.extend(constraints.get(key, []) or [])
        bm_hints = [h for h in bm_hints if isinstance(h, str)]

        fuel_req = constraints.get("fuel") or []
        trans_req = constraints.get("transmission") or []
        cond_req = constraints.get("condition") or []

        def _fuzzy_sim_one(x: str, cands: list[str]) -> float:
            if not cands:
                return 0.0
            best = 0
            for h in cands:
                res = process.extractOne(h, [str(x)], scorer=fuzz.token_set_ratio)
                s = res[1] if res else 0
                if s > best:
                    best = s
            return best / 100.0

        def sim_row(brand: str, model_name: str, fuel: str, trans: str, cond: str) -> float:
            scores = []
            # brand/model similarity
            if bm_hints:
                bm_scores = []
                for h in bm_hints:
                    b = process.extractOne(h, [str(brand)], scorer=fuzz.token_set_ratio)
                    m = process.extractOne(h, [str(model_name)], scorer=fuzz.token_set_ratio)
                    s = max((b[1] if b else 0), (m[1] if m else 0))
                    bm_scores.append(s / 100.0)
                if bm_scores:
                    scores.append(float(np.mean(bm_scores)))
            # categorical similarity
            scores.append(_fuzzy_sim_one(fuel, fuel_req))
            scores.append(_fuzzy_sim_one(trans, trans_req))
            scores.append(_fuzzy_sim_one(cond, cond_req))
            # average of available components
            return float(np.mean([s for s in scores if s is not None])) if scores else 0.0

        Xf = df_relax[NUM_COLS + CAT_COLS]
        pred = model.predict(Xf)
        df_relax = df_relax.assign(pred_price=pred, value_score=pred - df_relax["Price"].values)

        val = df_relax["value_score"].values
        if len(val) > 1 and np.std(val) > 1e-6:
            val_norm = (val - np.mean(val)) / (np.std(val) + 1e-9)
        else:
            val_norm = np.zeros_like(val)

        sim = df_relax.apply(lambda r: sim_row(r["Brand"], r["Model"], r["Fuel Type"], r["Transmission"], r["Condition"]), axis=1).values
        w_value = float(constraints.get("w_value", 1.0))
        w_sim = float(constraints.get("w_pref", 0.5))
        composite = w_value * val_norm + w_sim * sim
        df_relax = df_relax.assign(composite_score=composite)
        df_relax = df_relax.sort_values(["composite_score", "Price"], ascending=[False, True])

        note_parts = ["No exact matches; showing related best picks"]
        if qb:
            note_parts.append("Brand focus: " + ", ".join(qb))
        if constraints.get("budget") is not None:
            note_parts.append(f"Budget <= ${int(constraints['budget'])}")
        if constraints.get("mileage_max") is not None:
            note_parts.append(f"Mileage <= {int(constraints['mileage_max']):,}")
        if constraints.get("year_min") or constraints.get("year_max"):
            ymn = constraints.get("year_min", "-")
            ymx = constraints.get("year_max", "-")
            note_parts.append(f"Year: {ymn}..{ymx}")
        note = " | ".join(note_parts)
        return _format_results(df_relax, top_n=top_n, note=note)

    Xf = dff[NUM_COLS + CAT_COLS]
    pred = model.predict(Xf)
    dff = dff.assign(pred_price=pred, value_score=pred - dff["Price"].values)

    # Composite ranking: value, brand priority, preferences, and nearness to constraints
    val = dff["value_score"].values
    if len(val) > 1 and np.std(val) > 1e-6:
        val_norm = (val - np.mean(val)) / (np.std(val) + 1e-9)
    else:
        val_norm = np.zeros_like(val)

    prefer = set([s.lower() for s in constraints.get("prefer_brands", [])])
    avoid = set([s.lower() for s in constraints.get("avoid_brands", [])])
    brand_l = dff["Brand"].astype(str).str.lower()
    pref_scores = brand_l.apply(lambda b: 1.0 if b in prefer else (-1.0 if b in avoid else 0.0)).values

    # Requested brand handling
    query_brands = []
    for key in ("brands", "prefer_brands"):
        query_brands.extend(constraints.get(key, []) or [])
    query_brands = [qb for qb in query_brands if isinstance(qb, str)]

    # Calculate fuzzy brand match mask within current filtered set
    thresh_b = int(constraints.get("fuzzy_threshold", 85))
    topk_b = int(constraints.get("fuzzy_topk", 3))
    if query_brands:
        brand_cands = list(dff["Brand"].astype(str).unique())
        matched_b = _fuzzy_match(query_brands, brand_cands, threshold=thresh_b, topk=topk_b)
        if not matched_b and thresh_b > 70:
            matched_b = _fuzzy_match(query_brands, brand_cands, threshold=max(70, thresh_b - 15), topk=topk_b)
        brand_mask = dff["Brand"].astype(str).str.lower().isin(_safe_lower_list(matched_b)) if matched_b else pd.Series([False]*len(dff), index=dff.index)
    else:
        brand_mask = pd.Series([False]*len(dff), index=dff.index)

    # If user asked for brands but none exist within constraints, fallback to best from requested brands only
    if query_brands and brand_mask.sum() == 0:
        cands_all = list(df["Brand"].astype(str).unique())
        matched_global = _fuzzy_match(query_brands, cands_all, threshold=thresh_b, topk=topk_b)
        if not matched_global and thresh_b > 70:
            matched_global = _fuzzy_match(query_brands, cands_all, threshold=max(70, thresh_b - 15), topk=topk_b)
        if matched_global:
            df_brand = df[df["Brand"].astype(str).str.lower().isin(_safe_lower_list(matched_global))].copy()
            # Score these by value and nearness to constraints to surface closest fits from requested brands
            Xfb = df_brand[NUM_COLS + CAT_COLS]
            pred_b = model.predict(Xfb)
            df_brand = df_brand.assign(pred_price=pred_b, value_score=pred_b - df_brand["Price"].values)
            # Nearness reuse
            prices_b = df_brand["Price"].values.astype(float)
            miles_b = df_brand["Mileage"].values.astype(float)
            years_b = df_brand["Year"].values.astype(float)
            nearness_b = []
            budget = constraints.get("budget")
            tol = constraints.get("budget_tolerance_pct", 0.0) or 0.0
            mi_max = constraints.get("mileage_max")
            ymn = constraints.get("year_min")
            ymx = constraints.get("year_max")
            for p, m, y in zip(prices_b, miles_b, years_b):
                comps = []
                if budget is not None:
                    if p <= budget:
                        comps.append(1.0)
                    else:
                        limit = budget * (1.0 + max(tol, 1e-9))
                        if p <= limit:
                            comps.append(1.0 - (p - budget) / (limit - budget))
                        else:
                            comps.append(max(0.0, 1.0 - (p - limit) / (limit + 1e3)))
                if mi_max is not None:
                    if m <= mi_max:
                        comps.append(1.0)
                    else:
                        denom = max(mi_max, 1.0)
                        comps.append(max(0.0, 1.0 - (m - mi_max) / denom))
                if ymn is not None:
                    if y >= ymn:
                        comps.append(1.0)
                    else:
                        comps.append(max(0.0, 1.0 - (ymn - y) / 5.0))
                if ymx is not None:
                    if y <= ymx:
                        comps.append(1.0)
                    else:
                        comps.append(max(0.0, 1.0 - (y - ymx) / 5.0))
                nearness_b.append(np.mean(comps) if comps else 0.0)
            nearness_b = np.array(nearness_b)
            val_b = df_brand["value_score"].values
            if len(val_b) > 1 and np.std(val_b) > 1e-6:
                val_norm_b = (val_b - np.mean(val_b)) / (np.std(val_b) + 1e-9)
            else:
                val_norm_b = np.zeros_like(val_b)
            # Strong brand weight in this path
            composite_b = 1.0 * val_norm_b + 1.0 * nearness_b
            df_brand = df_brand.assign(composite_score=composite_b)
            df_brand = df_brand.sort_values(["composite_score", "Price"], ascending=[False, True])
            note = "No requested brand within constraints; showing best from requested brands"
            return _format_results(df_brand, top_n=top_n, note=note)

    nearness = []
    budget = constraints.get("budget")
    tol = constraints.get("budget_tolerance_pct", 0.0) or 0.0
    mi_max = constraints.get("mileage_max")
    ymn = constraints.get("year_min")
    ymx = constraints.get("year_max")
    prices = dff["Price"].values.astype(float)
    miles = dff["Mileage"].values.astype(float)
    years = dff["Year"].values.astype(float)

    for p, m, y in zip(prices, miles, years):
        comps = []
        if budget is not None:
            if p <= budget:
                comps.append(1.0)
            else:
                limit = budget * (1.0 + max(tol, 1e-9))
                if p <= limit:
                    comps.append(1.0 - (p - budget) / (limit - budget))
                else:
                    comps.append(0.0)
        if mi_max is not None:
            if m <= mi_max:
                comps.append(1.0)
            else:
                denom = max(mi_max, 1.0)
                comps.append(max(0.0, 1.0 - (m - mi_max) / denom))
        if ymn is not None:
            if y >= ymn:
                comps.append(1.0)
            else:
                comps.append(max(0.0, 1.0 - (ymn - y) / 5.0))
        if ymx is not None:
            if y <= ymx:
                comps.append(1.0)
            else:
                comps.append(max(0.0, 1.0 - (y - ymx) / 5.0))
        nearness.append(np.mean(comps) if comps else 0.0)
    nearness = np.array(nearness)

    w_value = float(constraints.get("w_value", 1.0))
    w_pref = float(constraints.get("w_pref", 0.5))
    w_near = float(constraints.get("w_nearness", 0.5))
    # strong boost for requested brands inside constrained set
    brand_priority = np.where(brand_mask.values, 1.0, 0.0) if len(dff) else np.array([])
    composite = (
        w_value * val_norm
        + w_pref * pref_scores
        + w_near * nearness
        + 1.2 * brand_priority
    )
    dff = dff.assign(composite_score=composite)

    # Prefer requested brands first, then higher composite score, then lower price
    if len(dff):
        dff = dff.assign(_brand_first=brand_priority)
        dff = dff.sort_values(["_brand_first", "composite_score", "Price"], ascending=[False, False, True])
        dff = dff.drop(columns=["_brand_first"])

    note_parts = []
    if constraints.get("budget") is not None:
        note_parts.append(f"Budget <= ${int(constraints['budget'])}")
    if constraints.get("brands"):
        note_parts.append("Brands: " + ", ".join(constraints["brands"]))
    if constraints.get("fuel"):
        note_parts.append("Fuel: " + ", ".join(constraints["fuel"]))
    if constraints.get("transmission"):
        note_parts.append("Transmission: " + ", ".join(constraints["transmission"]))
    if constraints.get("year_min") or constraints.get("year_max"):
        ymn = constraints.get("year_min", "-")
        ymx = constraints.get("year_max", "-")
        note_parts.append(f"Year: {ymn}..{ymx}")
    if constraints.get("mileage_max") is not None:
        note_parts.append(f"Mileage <= {int(constraints['mileage_max']):,}")

    note = " | ".join(note_parts) if note_parts else "No constraints detected"
    return _format_results(dff, top_n=top_n, note=note)
