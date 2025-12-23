import re
from typing import Dict, List, Optional
from rapidfuzz import process, fuzz

_NUMBER = r"(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?"

FUEL_KEYWORDS = {
    "petrol": ["petrol", "gas", "gasoline"],
    "diesel": ["diesel"],
    "electric": ["electric", "ev"],
    "hybrid": ["hybrid"],
}
TRANS_KEYWORDS = {
    "automatic": ["automatic", "auto"],
    "manual": ["manual", "stick"],
}
COND_KEYWORDS = {
    "new": ["new"],
    "like new": ["like new", "excellent","very good"],
    "used": ["used", "pre-owned", "preowned", "good", "fair"],
}


def _parse_budget(text: str) -> Optional[float]:
    # Keep two versions: raw for currency-symbol checks, and cleaned for numeric parsing
    raw = text
    t = text.lower()
    # Normalize thousands separators and drop currency words/symbols for numeric capture
    t_clean = t.replace(",", " ").replace("usd", "").replace("$", " ")
    # Expand k/m suffixes (works with/without spaces)
    t_clean = re.sub(rf"({_NUMBER})\s*[kK]\b", lambda m: str(float(m.group(1).replace(',', '')) * 1_000), t_clean)
    t_clean = re.sub(rf"({_NUMBER})\s*[mM]\b", lambda m: str(float(m.group(1).replace(',', '')) * 1_000_000), t_clean)

    # Primary patterns: comparators, budget keyword, approximation keywords
    patterns = [
        rf"(?:under|below|<=|less than|max|up to)\s+({_NUMBER})",
        rf"(?:around|about|approx(?:\.|imately)?|near|close to)\s+({_NUMBER})",
        rf"budget\s*(?:is|=|:)?\s*({_NUMBER})",
    ]
    for p in patterns:
        m = re.search(p, t_clean)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except:  # noqa: E722
                pass

    # Currency anywhere, e.g., $20 000 or 20 000$
    cur_patterns = [
        rf"[$]\s*({_NUMBER})",
        rf"({_NUMBER})\s*[$]",
        rf"\b({_NUMBER})\s*usd\b",
    ]
    for p in cur_patterns:
        m = re.search(p, raw, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except:  # noqa: E722
                pass

    # Fallback: a standalone large number likely indicating budget (avoid years ~2000-2025)
    # Choose the first number between 3,000 and 500,000
    nums = [n.replace(',', '') for n in re.findall(_NUMBER, t_clean)]
    for n in nums:
        try:
            val = float(n)
            if 3000 <= val <= 500000:
                return val
        except:  # noqa: E722
            continue
    return None


def _parse_years(text: str):
    # after 2016, >=2018, 2015-2020
    t = text.lower()
    rng = re.search(rf"(\d{{4}})\s*[-to]+\s*(\d{{4}})", t)
    if rng:
        a, b = int(rng.group(1)), int(rng.group(2))
        return min(a, b), max(a, b)
    ge = re.search(r"(?:after|since|>=|from)\s*(\d{4})", t)
    if ge:
        return int(ge.group(1)), None
    le = re.search(r"(?:before|<=|through)\s*(\d{4})", t)
    if le:
        return None, int(le.group(1))
    single = re.search(r"\b(19|20)\d{2}\b", t)
    if single:
        y = int(re.search(r"\b(19|20)\d{2}\b", t).group(0))
        return y, y
    return None, None


def _parse_mileage(text: str) -> Optional[float]:
    t = text.lower().replace(",", "")
    t = re.sub(rf"({_NUMBER})\s*[kK]\b", lambda m: str(float(m.group(1)) * 1_000), t)
    m = re.search(rf"(?:<|<=|under|below|less than|max)\s*({_NUMBER})\s*(?:mi|miles|kms|km)?", t)
    if m:
        val = float(m.group(1))
        # naive km detection
        if "km" in t and "mi" not in t:
            val *= 0.621371
        return val
    return None


def _collect_keywords(text: str, kw_map, threshold: int = 85) -> List[str]:
    t = text.lower()
    found = []

    # 1) Exact containment (handles multi-word like "like new")
    for key, kws in kw_map.items():
        for k in kws:
            if re.search(rf"\b{re.escape(k)}\b", t):
                found.append(key)
                break

    # 2) Fuzzy on single-word tokens for misspellings (e.g., "automatc")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", t)
    variant_to_key = {}
    variants = []
    for key, kws in kw_map.items():
        for k in kws:
            variant_to_key[k] = key
            variants.append(k)

    for tok in tokens:
        # skip common filler words
        if tok in {"and", "or", "with", "without", "like", "new", "used", "the", "a", "an"}:
            continue
        match = process.extractOne(tok, variants, scorer=fuzz.token_set_ratio)
        if match and match[1] >= threshold:
            key = variant_to_key.get(match[0])
            if key:
                found.append(key)

    return list(dict.fromkeys(found))


def _extract_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)
    brand_hints = []
    for tok in tokens:
        if tok.lower() in {"under", "below", "less", "than", "after", "before", "auto", "automatic", "manual", "new", "used", "like"}:
            continue
        brand_hints.append(tok)
    return list(dict.fromkeys(brand_hints))


def parse_user_query(text: str) -> Dict:
    constraints = {}
    constraints["budget"] = _parse_budget(text)
    ymin, ymax = _parse_years(text)
    if ymin is not None:
        constraints["year_min"] = ymin
    if ymax is not None:
        constraints["year_max"] = ymax
    constraints["mileage_max"] = _parse_mileage(text)

    constraints["fuel"] = _collect_keywords(text, FUEL_KEYWORDS)
    constraints["transmission"] = _collect_keywords(text, TRANS_KEYWORDS)
    constraints["condition"] = _collect_keywords(text, COND_KEYWORDS)

    # brand/model hints (downstream matching can be fuzzy)
    brand_hints = _extract_tokens(text)
    constraints["brands"] = brand_hints
    constraints["models"] = brand_hints

    # soft preferences
    prefer = []
    avoid = []
    for m in re.finditer(r"(?:prefer|like|love)\s+([A-Za-z][A-Za-z0-9\-]+)", text, flags=re.IGNORECASE):
        prefer.append(m.group(1))
    for m in re.finditer(r"(?:avoid|not\s+prefer|dislike)\s+([A-Za-z][A-Za-z0-9\-]+)", text, flags=re.IGNORECASE):
        avoid.append(m.group(1))
    constraints["prefer_brands"] = list(dict.fromkeys(prefer))
    constraints["avoid_brands"] = list(dict.fromkeys(avoid))

    # approximation tolerance for budget if user says around/near
    t = text.lower()
    approx = 0.0
    if re.search(r"\b(?:around|about|approx(?:\.|imately)?|near|close to)\b", t):
        approx = 0.1
    constraints["budget_tolerance_pct"] = approx

    return constraints
