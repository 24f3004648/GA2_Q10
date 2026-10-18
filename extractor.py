"""Invoice field extraction: deterministic parser, with an optional local-LLM
(Ollama, OpenAI-compatible) fallback for fields the parser cannot find.

Every LLM answer is checked against the source text before it is accepted, so
a hallucinated value never reaches the response.
"""
import json
import os
import re
import urllib.request
from datetime import datetime
from typing import Optional

CURRENCIES = ["USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY", "SGD", "AED"]
SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
WORDS = {"dollars": "USD", "dollar": "USD", "euros": "EUR", "euro": "EUR", "pounds": "GBP", "pound": "GBP"}

SUFFIX = (
    r"(?:Industries|Ltd|Limited|Inc|LLC|Corp|Corporation|GmbH|Co|Company|Group|Holdings|"
    r"Systems|Solutions|Technologies|Enterprises|Partners|Services|Pvt|PLC|AG|SA|Labs|"
    r"Associates|Trading|Supplies|Manufacturing|Logistics|Consulting|Studios|Works)"
)
VENDOR_RE = re.compile(
    r"\b((?:[A-Z][\w&'-]*\s+){1,5}?" + SUFFIX + r"\b\.?(?:\s+" + SUFFIX + r"\b\.?)*)"
)
LABELED_VENDOR_RE = re.compile(
    r"(?:vendor|supplier|from|billed by|issued by|pay to|payable to|payee|seller)\s*[:\-]?\s*"
    r"([A-Z][^\n,;]*?)(?=\s+(?:invoice|total|amount|due|dated|for|on|is|bill)\b|[\n,;]|$)",
    re.IGNORECASE,
)
LEADING_STOPWORDS = {
    "invoice", "bill", "from", "to", "payment", "please", "pay", "total", "due", "dear",
    "vendor", "supplier", "issued", "by", "billed", "attn", "re", "the", "a", "an",
    "this", "is", "for", "of", "thank", "you",
}

ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
TEXT_DATE_RES = [
    (re.compile(rf"\b((?:{MONTHS})[a-z]*\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})\b", re.I), "mdy"),
    (re.compile(rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})[a-z]*\.?,?\s+\d{{4}})\b", re.I), "dmy"),
]
NUMBER_RE = re.compile(r"(?<![\w#.-])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d|-\w)")
TOTAL_KW_RE = re.compile(
    r"(?<!sub)(?<!sub-)(total(?:\s+amount)?(?:\s+due)?|amount\s+due|balance\s+due|grand\s+total|"
    r"amount\s+payable|please\s+pay|pay)\W{0,12}$",
    re.I,
)
DUE_KW_RE = re.compile(r"(due|pay(?:ment)?\s+by|payable\s+by|pay\s+on|deadline)\W*(?:date|on|by)?\W*$", re.I)


def _clean_vendor(name: str) -> str:
    tokens = name.strip().split()
    while len(tokens) > 1 and tokens[0].strip(":,").lower() in LEADING_STOPWORDS:
        tokens.pop(0)
    return " ".join(tokens).strip(" :,-")


def find_vendor(text: str) -> Optional[str]:
    m = VENDOR_RE.search(text)
    if m:
        return _clean_vendor(m.group(1))
    m = LABELED_VENDOR_RE.search(text)
    if m:
        return _clean_vendor(m.group(1))
    return None


def _norm_text_date(raw: str, order: str) -> Optional[str]:
    raw = re.sub(r"(st|nd|rd|th)\b", "", raw, flags=re.I).replace(",", " ").replace(".", " ")
    parts = raw.split()
    if len(parts) != 3:
        return None
    month_s, day_s, year_s = (parts[0], parts[1], parts[2]) if order == "mdy" else (parts[1], parts[0], parts[2])
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(f"{month_s[:9]} {day_s} {year_s}", fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.strptime(f"{month_s[:3]} {day_s} {year_s}", "%b %d %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def find_date(text: str) -> Optional[str]:
    cands = []  # (position, iso)
    for m in ISO_DATE_RE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            datetime(y, mo, d)
        except ValueError:
            continue
        cands.append((m.start(), m.group(0)))
    if not cands:
        for rx, order in TEXT_DATE_RES:
            for m in rx.finditer(text):
                iso = _norm_text_date(m.group(1), order)
                if iso:
                    cands.append((m.start(), iso))
    if not cands:
        return None
    cands.sort()
    for pos, iso in cands:  # prefer a date introduced by a "due"-style label
        if DUE_KW_RE.search(text[max(0, pos - 30):pos]):
            return iso
    for pos, iso in cands:  # otherwise skip dates labelled as the issue date
        if not re.search(r"(issued?|invoice\s+date|dated|date\s+of\s+issue)\W*$", text[max(0, pos - 25):pos], re.I):
            return iso
    return cands[0][1]


def find_currency(text: str) -> Optional[str]:
    counts = {}
    for code in CURRENCIES:
        n = len(re.findall(rf"\b{code}\b", text))
        if n:
            counts[code] = n
    if counts:
        return max(counts, key=counts.get)
    for sym, code in SYMBOLS.items():
        if sym in text:
            return code
    low = text.lower()
    for word, code in WORDS.items():
        if re.search(rf"\b{word}\b", low):
            return code
    return None


def find_amount(text: str, vendor: Optional[str]) -> Optional[float]:
    scrub = ISO_DATE_RE.sub(" ", text)
    for rx, _ in TEXT_DATE_RES:
        scrub = rx.sub(" ", scrub)
    if vendor:
        scrub = scrub.replace(vendor, " ")
    best, best_score = None, -1
    for m in NUMBER_RE.finditer(scrub):
        before = scrub[max(0, m.start() - 30):m.start()]
        after = scrub[m.end():m.end() + 6]
        score = 0
        if re.search(r"(?:\b(?:%s)\b|[$€£])\s*$" % "|".join(CURRENCIES), before, re.I):
            score += 2
        if re.match(r"\s*(?:%s)\b" % "|".join(CURRENCIES), after, re.I):
            score += 2
        if TOTAL_KW_RE.search(before):
            score += 3
        if score > best_score:  # ties keep the earliest candidate
            try:
                best, best_score = float(m.group(1).replace(",", "")), score
            except ValueError:
                continue
    return best if best_score > 0 else None


def heuristic_extract(text: str) -> dict:
    vendor = find_vendor(text)
    return {
        "vendor": vendor,
        "amount": find_amount(text, vendor),
        "currency": find_currency(text),
        "date": find_date(text),
    }


# ---- optional local LLM (Ollama / any OpenAI-compatible server) -------------
def llm_extract(text: str) -> Optional[dict]:
    base = os.environ.get("OLLAMA_URL", "").rstrip("/")
    if not base:
        return None
    payload = {
        "model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
        "stream": False,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract invoice fields. Reply with JSON only: "
                    '{"vendor": string, "amount": number (total due), '
                    '"currency": 3-letter uppercase code, "date": payment due date as YYYY-MM-DD}.'
                ),
            },
            {"role": "user", "content": text},
        ],
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            body = json.loads(resp.read())
        return json.loads(body["choices"][0]["message"]["content"])
    except Exception:
        return None


def _validated(llm: dict, text: str) -> dict:
    """Keep only LLM values that are grounded in the source text."""
    out = {}
    v = llm.get("vendor")
    if isinstance(v, str) and v.strip() and v.strip().lower() in text.lower():
        out["vendor"] = v.strip()
    try:
        a = float(llm.get("amount"))
        plain = text.replace(",", "")
        if re.search(rf"(?<![\d.]){re.escape(('%f' % a).rstrip('0').rstrip('.'))}(?!\d)", plain):
            out["amount"] = a
    except (TypeError, ValueError):
        pass
    c = llm.get("currency")
    if isinstance(c, str) and re.fullmatch(r"[A-Za-z]{3}", c.strip()) and c.strip().upper() in text.upper():
        out["currency"] = c.strip().upper()
    d = llm.get("date")
    if isinstance(d, str) and ISO_DATE_RE.fullmatch(d.strip()) and d.strip() in text:
        out["date"] = d.strip()
    return out


def extract(text: str) -> Optional[dict]:
    """Return all four fields, or None if they cannot be determined."""
    fields = heuristic_extract(text)
    if any(fields[k] is None for k in fields):
        llm = llm_extract(text)
        if llm:
            for k, v in _validated(llm, text).items():
                if fields.get(k) is None:
                    fields[k] = v
    if any(fields[k] is None for k in fields):
        return None
    fields["currency"] = fields["currency"].upper()
    return fields
