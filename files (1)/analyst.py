"""
Natural Language Spreadsheet Analyst
------------------------------------
messy CSV  ->  auto-clean  ->  English question  ->  LLM writes pandas code
           ->  AST-validated + sandboxed execution  ->  table + chart + 1-line insight

Run from CLI:   python analyst.py data.csv "Top 10 complaint types"
Run the 5 demo questions:   python analyst.py data.csv --eval
Run the UI:     streamlit run app.py
"""
from __future__ import annotations

import argparse
import ast
import builtins
import io
import json
import os
import re
import threading
import types
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# =============================================================================
# 1. ROBUST LOADING + AUTO-CLEANING
# =============================================================================

NULL_TOKENS = {
    "", "na", "n/a", "#n/a", "nan", "null", "none", "nil", "-", "--", "?",
    "missing", "undefined", "#null!", "n.a.", "not available",
}
# columns whose name contains one of these tokens are identifiers -> keep as text
ID_TOKENS = {"id", "key", "zip", "zipcode", "postal", "phone", "code", "fips", "bbl", "number", "no"}
DATE_HINTS = ("date", "time", "created", "closed", "updated", "timestamp", "_at", "dob", "datetime")


def _snake(name: Any, i: int) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
    return s or f"col_{i}"


def _read_bytes(source) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read"):
        data = source.read()
        return data if isinstance(data, bytes) else data.encode("utf-8")
    with open(source, "rb") as f:
        return f.read()


def load_csv(source, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read a CSV from path / bytes / file-like. Survives bad encodings, odd delimiters,
    ragged rows and junk lines. Everything is read as text; types are inferred later."""
    raw = _read_bytes(source)
    if not raw.strip():
        raise ValueError("The file is empty.")
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        for sniff in (False, True):
            try:
                kw = dict(dtype=str, on_bad_lines="skip", nrows=max_rows,
                          skip_blank_lines=True, keep_default_na=False, na_values=[])
                if sniff:
                    kw.update(sep=None, engine="python")  # auto-detect ; | \t delimiters
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, **kw)
                if df.shape[1] > 1 or sniff:
                    # keep_default_na=False -> restore NaN for truly empty cells
                    return df.replace({"": np.nan})
            except Exception as e:  # noqa: BLE001
                last_err = e
    raise ValueError(f"Could not parse the file as CSV: {last_err}")


def _infer_numeric(name: str, s: pd.Series) -> Optional[pd.Series]:
    if any(t in ID_TOKENS for t in name.split("_")):
        return None
    nn = s.dropna()
    if nn.empty:
        return None
    cleaned = nn.str.replace(r"[,$€£₹%\s]", "", regex=True)
    if cleaned.str.match(r"^0\d+").any():  # leading zeros -> codes, not numbers
        return None
    num = pd.to_numeric(cleaned, errors="coerce")
    if num.notna().mean() < 0.9:
        return None
    full = s.str.replace(r"[,$€£₹%\s]", "", regex=True)
    return pd.to_numeric(full, errors="coerce").astype("float64")


def _to_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
        except (TypeError, ValueError):
            out = pd.to_datetime(s, errors="coerce", utc=True)
    return out.dt.tz_localize(None)


def _infer_datetime(name: str, s: pd.Series) -> Optional[pd.Series]:
    nn = s.dropna()
    if nn.empty:
        return None
    # skip obvious non-dates cheaply: dates contain digits and a separator
    sample = nn.sample(min(200, len(nn)), random_state=0)
    hinted = any(h in name for h in DATE_HINTS)
    if sample.str.contains(r"\d").mean() < (0.6 if hinted else 0.9):
        return None
    if _to_datetime(sample).notna().mean() < (0.6 if hinted else 0.9):
        return None
    parsed = _to_datetime(s)
    if parsed[s.notna()].notna().mean() < (0.6 if hinted else 0.9):
        return None
    return parsed


def _normalize_text(s: pd.Series) -> pd.Series:
    """Collapse whitespace and unify case variants (BROOKLYN / Brooklyn / brooklyn)."""
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    nn = s.dropna()
    if 0 < nn.nunique() <= 500:
        canon = nn.groupby(nn.str.lower()).agg(lambda x: x.value_counts().idxmax())
        s = s.str.lower().map(canon)
    return s


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). Never raises on weird values; coerces instead."""
    report: dict[str, Any] = {"rows_in": int(len(df)), "cols_in": int(df.shape[1]), "notes": []}
    df = df.copy().astype("object")

    # column names -> unique snake_case
    seen: dict[str, int] = {}
    new_cols = []
    for i, c in enumerate(df.columns):
        n = _snake(c, i)
        seen[n] = seen.get(n, 0) + 1
        new_cols.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    df.columns = new_cols

    # strip + unify null tokens
    for c in df.columns:
        s = df[c].astype("object").where(df[c].notna(), np.nan)
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        low = s.map(lambda v: v.lower() if isinstance(v, str) else v)
        df[c] = s.where(~low.isin(NULL_TOKENS), np.nan)

    df = df.dropna(how="all").dropna(axis=1, how="all")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        report["notes"].append(f"Removed {before - len(df)} duplicate rows")

    types_found: dict[str, str] = {}
    for c in list(df.columns):
        s = df[c]
        num = _infer_numeric(c, s)
        if num is not None:
            df[c] = num
            types_found[c] = "number"
            continue
        dt = _infer_datetime(c, s)
        if dt is not None:
            bad = int(s.notna().sum() - dt.notna().sum())
            if bad:
                report["notes"].append(f"{c}: {bad} unparseable dates set to null")
            df[c] = dt
            types_found[c] = "datetime"
            continue
        s = _normalize_text(s.astype("object"))
        n_null = int(s.isna().sum())
        if s.nunique() <= 50 and n_null:
            s = s.fillna("Unknown")
            report["notes"].append(f"{c}: {n_null} nulls -> 'Unknown'")
        df[c] = s
        types_found[c] = "text"

    # derived column: resolution time (very useful for 311-style data)
    dt_cols = [c for c, t in types_found.items() if t == "datetime"]
    start = next((c for c in dt_cols if re.search(r"creat|open|start|received", c)), None)
    end = next((c for c in dt_cols if re.search(r"clos|end|resol|complet", c)), None)
    if start and end and start != end:
        hrs = (df[end] - df[start]).dt.total_seconds() / 3600
        neg = int((hrs < 0).sum())
        df["response_time_hours"] = hrs.where(hrs >= 0)
        report["notes"].append(
            f"Added response_time_hours = {end} - {start}"
            + (f" ({neg} negative values nulled)" if neg else "")
        )

    report.update(rows_out=int(len(df)), cols_out=int(df.shape[1]), types=types_found,
                  null_counts={c: int(n) for c, n in df.isna().sum().items() if n})
    return df, report


def describe_schema(df: pd.DataFrame) -> str:
    """Compact description of the cleaned data that goes into the LLM prompt."""
    lines = [f"Rows: {len(df)}", "Columns:"]
    for c in df.columns:
        s = df[c]
        nul = f", {s.isna().mean():.0%} null" if s.isna().any() else ""
        if pd.api.types.is_datetime64_any_dtype(s):
            lines.append(f"- {c} (datetime64) {s.min()} .. {s.max()}{nul}")
        elif pd.api.types.is_numeric_dtype(s):
            lines.append(f"- {c} (number) min={s.min():.4g} max={s.max():.4g} mean={s.mean():.4g}{nul}")
        else:
            nu = s.nunique()
            if nu <= 25:
                vals = ", ".join(f"{k} ({v})" for k, v in s.value_counts().head(25).items())
                lines.append(f"- {c} (text, {nu} unique) values: {vals}{nul}")
            else:
                ex = ", ".join(map(str, s.dropna().unique()[:3]))
                lines.append(f"- {c} (text, {nu} unique) e.g. {ex}{nul}")
    return "\n".join(lines)


# =============================================================================
# 2. SAFE EXECUTION OF LLM-GENERATED CODE
# =============================================================================

class CodeRejected(Exception):
    """Raised when generated code fails the safety / sanity checks."""


FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.FunctionDef,
    ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom, ast.AsyncFor, ast.AsyncWith,
    ast.With, ast.While, ast.Delete, ast.Raise, ast.Try,
)
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr", "breakpoint", "exit", "quit", "help", "dir",
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "pickle", "builtins",
    "type", "object", "memoryview", "bytearray", "print",
}
BANNED_ATTRS = {
    "eval", "query", "pipe", "style", "plot", "hist", "boxplot", "system", "popen", "os", "sys",
    "io", "subprocess", "builtins", "compat", "testing", "read_csv", "to_clipboard",
    "to_csv", "to_excel", "to_json", "to_pickle", "to_sql", "to_parquet", "to_feather",
    "to_hdf", "to_html", "to_latex", "to_markdown", "to_xml", "to_orc", "to_stata", "to_gbq",
    "to_string", "to_records", "mro", "f_globals", "gi_frame", "cr_frame",
}
ALLOWED_BUILTINS = (
    "len range min max sum sorted abs round int float str bool list dict set tuple enumerate "
    "zip any all isinstance reversed map filter slice divmod pow"
).split()

SAFE_BUILTINS = {n: getattr(builtins, n) for n in ALLOWED_BUILTINS}
# whitelist namespaces: the model never receives the real pandas / numpy modules
SAFE_PD = types.SimpleNamespace(**{n: getattr(pd, n) for n in (
    "DataFrame Series to_datetime to_numeric Timestamp Timedelta concat cut qcut isna notna "
    "date_range crosstab pivot_table merge Grouper DateOffset unique NA NaT"
).split()})
SAFE_NP = types.SimpleNamespace(**{n: getattr(np, n) for n in (
    "nan inf mean median sum round where log log1p sqrt abs nanmean nanmedian percentile "
    "arange select isnan floor ceil minimum maximum nan_to_num clip std var"
).split()})


def validate_code(code: str) -> None:
    """Static (AST) safety check. Rejects imports, dunder access, file/network/OS access,
    exec/eval, loops that can hang, and absurdly large numeric constants."""
    if not code or not code.strip():
        raise CodeRejected("No code was produced.")
    if len(code) > 5000:
        raise CodeRejected("Code is too long.")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeRejected(f"Syntax error: {e}") from e

    assigns_result = False
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise CodeRejected(f"Forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("_"):
                raise CodeRejected(f"Forbidden name: {node.id}")
            if node.id == "result" and isinstance(node.ctx, ast.Store):
                assigns_result = True
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in BANNED_ATTRS or node.attr.startswith("read_"):
                raise CodeRejected(f"Forbidden attribute: .{node.attr}")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and "__" in node.value:
                raise CodeRejected("Forbidden string containing '__'")
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool) \
                    and abs(node.value) > 1e8:
                raise CodeRejected("Numeric constant too large")
    if not assigns_result:
        raise CodeRejected("The code must assign the final answer to a variable named `result`.")


def run_sandboxed(code: str, df: pd.DataFrame, timeout: float = 20.0) -> Any:
    """Validate then execute code with restricted builtins, whitelisted pd/np, a copy of df,
    and a wall-clock timeout. Returns the value of `result`."""
    validate_code(code)
    env = {"__builtins__": SAFE_BUILTINS, "pd": SAFE_PD, "np": SAFE_NP, "df": df.copy()}
    box: dict[str, Any] = {}

    def target():
        try:
            exec(compile(code, "<llm_code>", "exec"), env)  # noqa: S102 - validated + restricted
            box["result"] = env.get("result")
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"Code took longer than {timeout:.0f}s.")
    if "error" in box:
        e = box["error"]
        raise RuntimeError(f"{type(e).__name__}: {e}") from e
    return box.get("result")


def to_table(res: Any, max_rows: int = 200) -> pd.DataFrame:
    """Turn whatever the code returned into a tidy DataFrame."""
    if isinstance(res, pd.DataFrame):
        out = res.copy()
    elif isinstance(res, pd.Series):
        name = res.name if res.name is not None else "value"
        out = res.rename(name).reset_index()
    elif isinstance(res, (dict, list, tuple, set)):
        out = pd.DataFrame(list(res) if isinstance(res, set) else res)
        if out.shape[1] == 1 and out.columns[0] == 0:
            out.columns = ["value"]
    elif res is None:
        raise ValueError("`result` is None.")
    else:  # scalar
        out = pd.DataFrame({"value": [res]})

    # flatten MultiIndex columns; surface a meaningful index
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join(map(str, c)).strip("_") for c in out.columns]
    idx = out.index
    if isinstance(idx, pd.MultiIndex) or any(n is not None for n in idx.names) \
            or not pd.api.types.is_integer_dtype(idx):
        out = out.reset_index()
    out.columns = [str(c) if str(c) != "index" else "category" for c in out.columns]
    for c in out.columns:
        if isinstance(out[c].dtype, pd.PeriodDtype):
            out[c] = out[c].dt.to_timestamp()
        elif isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(str)
        elif isinstance(out[c].dtype, pd.IntervalDtype):
            out[c] = out[c].astype(str)
    if out.empty:
        raise ValueError("The result is empty (check filters / exact spelling of values).")
    return out.head(max_rows).reset_index(drop=True)

def validate_result(table: pd.DataFrame) -> None:
    if not isinstance(table, pd.DataFrame):
        raise ValueError("Invalid result format.")
    if table.empty:
        raise ValueError("No matching results found.")
    if not table.columns.is_unique:
        raise ValueError("Duplicate result columns found.")
    if table.isna().all(axis=None):
        raise ValueError("Result contains only missing values.")
    for col in table.select_dtypes(include="number").columns:
        if not np.isfinite(table[col].dropna()).all():
            raise ValueError(f"Invalid numeric values in {col}.")


# =============================================================================
# 3. LLM (Groq or OpenAI, both through the `openai` package)
# =============================================================================

PREFERRED_MODELS = [
    "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
    "llama-3.1-8b-instant", "qwen/qwen3-32b", "meta-llama/llama-4-scout-17b-16e-instruct",
]
_NOT_CHAT = re.compile(r"whisper|guard|tts|orpheus|playai|embed|safeguard|prompt", re.I)


class LLM:
    """Groq (key starts with gsk_) or OpenAI. If the chosen model is unavailable for the key,
    it asks the API which models exist and switches automatically."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from openai import OpenAI

        api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY (free at console.groq.com) or OPENAI_API_KEY.")
        self.groq = api_key.startswith("gsk_")
        self.model = model or os.getenv("LLM_MODEL") or (None if self.groq else "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key,
                             base_url="https://api.groq.com/openai/v1" if self.groq else None)

    def _pick_model(self, exclude: Optional[str] = None) -> str:
        ids = [m.id for m in self.client.models.list().data]
        usable = [i for i in ids if not _NOT_CHAT.search(i) and i != exclude]
        for pref in PREFERRED_MODELS:
            if pref in usable:
                return pref
        if usable:
            return sorted(usable)[0]
        raise RuntimeError("No chat models are available for this API key.")

    def _call(self, messages: list[dict], json_mode: bool) -> str:
        kw: dict[str, Any] = dict(model=self.model, messages=messages, temperature=0)
        if json_mode:
            try:
                r = self.client.chat.completions.create(response_format={"type": "json_object"}, **kw)
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                if self._model_missing(e):
                    raise
        r = self.client.chat.completions.create(**kw)
        return r.choices[0].message.content or ""

    @staticmethod
    def _model_missing(e: Exception) -> bool:
        t = str(e).lower()
        return "model_not_found" in t or "decommission" in t or "does not exist" in t or "404" in t

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        if not self.model:
            self.model = self._pick_model()
        try:
            return self._call(messages, json_mode)
        except Exception as e:  # noqa: BLE001
            if not self._model_missing(e):
                raise
            self.model = self._pick_model(exclude=self.model)  # switch to a model that exists
            return self._call(messages, json_mode)


SYSTEM_PROMPT = """You are an expert pandas data analyst. A cleaned DataFrame named `df` is ALREADY loaded.
`pd` and `np` are available (limited whitelist). Do NOT import anything, read/write files, print, or define functions/classes/loops with while.

DATA SCHEMA:
{schema}

RULES
1. Use the exact column names above. Datetime columns are datetime64 (use .dt). Numeric nulls are NaN.
2. Put the final answer in a variable named `result` (a DataFrame preferred; Series/scalar OK).
3. `result` must be small and presentation-ready: aggregate, give columns clear names, sort rankings
   descending, and keep only the top 10-20 rows unless the user asks for more.
4. For "top N" use .head(N). For percentages, add a column with values 0-100 rounded to 2 decimals.
5. For trends over time group by a period, e.g.
   df.groupby(df['created_date'].dt.to_period('M').dt.to_timestamp()).size().reset_index(name='complaints')
6. For text filters match case-insensitively: df[df['borough'].str.lower() == 'brooklyn'].
   Never use df.query / df.eval / df.pipe / df.plot.
7. Round float results to 2 decimals. Always reset_index() so grouping keys are normal columns.
8. If a needed value is missing in the data, still return the closest sensible aggregate.

OUTPUT: return ONLY a JSON object, no markdown:
{{"code": "<python code, newline-separated, assigns `result`>",
  "chart": {{"type": "bar|line|pie", "x": "<column in result>", "y": "<numeric column in result>", "title": "<short title>"}}}}

CHART TYPE: line = trends over time; pie = share/proportion of a whole with <= 6 categories;
bar = rankings and comparisons (default).

EXAMPLES
Q: How many records per category?
{{"code": "result = df.groupby('category').size().reset_index(name='count').sort_values('count', ascending=False)", "chart": {{"type": "bar", "x": "category", "y": "count", "title": "Records per category"}}}}
Q: Average amount by region?
{{"code": "result = df.groupby('region')['amount'].mean().round(2).reset_index(name='avg_amount').sort_values('avg_amount', ascending=False)", "chart": {{"type": "bar", "x": "region", "y": "avg_amount", "title": "Average amount by region"}}}}
"""


def parse_llm_output(text: str) -> tuple[str, dict]:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            code = data.get("code", "")
            if isinstance(code, list):
                code = "\n".join(code)
            return str(code), data.get("chart") or {}
        except json.JSONDecodeError:
            pass
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)  # fallback: plain code answer
    return (fence.group(1) if fence else text).strip(), {}


# =============================================================================
# 4. CHART GENERATION (Plotly)
# =============================================================================

def _pretty(c: Any) -> str:
    return str(c).replace("_", " ").strip().title()


def build_chart(table: pd.DataFrame, hint: Optional[dict] = None):
    """Pick bar / line / pie from the shape of the result (LLM hint is validated, not trusted)."""
    import plotly.express as px

    hint = hint or {}
    t = table.copy()
    is_num = lambda c: pd.api.types.is_numeric_dtype(t[c]) and not pd.api.types.is_bool_dtype(t[c])  # noqa: E731
    is_dt = lambda c: pd.api.types.is_datetime64_any_dtype(t[c])  # noqa: E731
    for c in t.columns:
        if pd.api.types.is_float_dtype(t[c]):
            t[c] = t[c].round(2)

    num_cols = [c for c in t.columns if is_num(c)]
    if not num_cols:  # text-only result -> chart value counts of the first column
        c0 = t.columns[0]
        vc = t[c0].astype(str).value_counts().head(25)
        t = pd.DataFrame({c0: vc.index, "count": vc.values})
        num_cols = ["count"]

    # single-row results (scalars / one aggregate row): one bar per metric
    if len(t) == 1:
        vals = [float(t[c].iloc[0]) for c in num_cols]
        t = pd.DataFrame({"metric": [_pretty(c) for c in num_cols], "value": vals})
        fig = px.bar(t, x="metric", y="value", text_auto=True,
                     title=hint.get("title") or "Result", template="plotly_white")
        return fig.update_layout(xaxis_title="", yaxis_title="Value", height=420)

    dt_cols = [c for c in t.columns if is_dt(c)]
    cat_cols = [c for c in t.columns if c not in num_cols and c not in dt_cols]

    x = hint.get("x") if hint.get("x") in t.columns else None
    if x is None:
        x = dt_cols[0] if dt_cols else (cat_cols[0] if cat_cols else t.columns[0])
    y = hint.get("y") if hint.get("y") in num_cols and hint.get("y") != x else None
    if y is None:
        y = next((c for c in num_cols if c != x), num_cols[0])

    ctype = str(hint.get("type", "bar")).lower()
    if ctype not in {"bar", "line", "pie"}:
        ctype = "bar"
    if is_dt(x):
        ctype = "line"
    elif ctype == "line" and not is_num(x):
        ctype = "bar"

    color = next((c for c in cat_cols if c != x and 1 < t[c].nunique() <= 10), None)
    n_x = t[x].nunique()
    if ctype == "pie" and (n_x > 8 or color or (t[y] < 0).any()):
        ctype = "bar"

    title = hint.get("title") or f"{_pretty(y)} by {_pretty(x)}"
    labels = {c: _pretty(c) for c in t.columns}

    if ctype == "line":
        fig = px.line(t.sort_values(x), x=x, y=y, color=color, markers=True, labels=labels,
                      title=title, template="plotly_white")
    elif ctype == "pie":
        fig = px.pie(t, names=x, values=y, title=title, template="plotly_white", hole=0.3)
        fig.update_traces(textinfo="percent+label")
    else:
        t = t.head(30)
        avg_len = t[x].astype(str).str.len().mean()
        horizontal = (avg_len > 10 or n_x > 8) and not is_num(x)
        if horizontal:
            fig = px.bar(t, x=y, y=x, color=color, orientation="h", text_auto=True,
                         labels=labels, title=title, template="plotly_white", barmode="group")
            fig.update_yaxes(categoryorder="total ascending")
        else:
            fig = px.bar(t, x=x, y=y, color=color, text_auto=True, labels=labels,
                         title=title, template="plotly_white", barmode="group")
    fig.update_layout(height=max(420, 22 * n_x + 200) if ctype == "bar" and n_x > 12 else 460,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


# =============================================================================
# 5. INSIGHT (1 line)
# =============================================================================

def fallback_insight(table: pd.DataFrame) -> str:
    num = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    if len(table) == 1 or not num:
        row = table.iloc[0]
        return "; ".join(f"{_pretty(c)}: {row[c]}" for c in table.columns[:3])
    other = [c for c in table.columns if c not in num]
    y = num[0]
    lab = other[0] if other else table.columns[0]
    hi, lo = table.loc[table[y].idxmax()], table.loc[table[y].idxmin()]
    return (f"Highest {_pretty(y)}: {hi[lab]} ({hi[y]:,.2f}); lowest: {lo[lab]} ({lo[y]:,.2f}).")


def make_insight(llm, question: str, table: pd.DataFrame) -> str:
    try:
        preview = table.head(15).to_csv(index=False)
        msg = [{"role": "system", "content":
                "You write ONE concise insight sentence (max 25 words) about a query result. "
                "Use only numbers present in the table. No markdown, no preamble."},
               {"role": "user", "content": f"Question: {question}\nResult:\n{preview}"}]
        text = " ".join((llm.chat(msg) or "").split())
        return text if text else fallback_insight(table)
    except Exception:  # noqa: BLE001
        return fallback_insight(table)


# =============================================================================
# 6. THE AGENT
# =============================================================================

@dataclass
class Answer:
    question: str
    insight: str
    table: Optional[pd.DataFrame] = None
    figure: Any = None
    code: str = ""
    attempts: int = 0
    error: Optional[str] = None
    log: list = field(default_factory=list)


class SpreadsheetAgent:
    def __init__(self, source, llm=None, max_rows: Optional[int] = None, max_attempts: int = 3):
        self.llm = llm or LLM()
        self.max_attempts = max_attempts
        raw = load_csv(source, max_rows=max_rows)
        self.df, self.report = clean_dataframe(raw)
        self.schema = describe_schema(self.df)

    def ask(self, question: str) -> Answer:
        """Never raises: on failure returns an Answer with .error set."""
        ans = Answer(question=question, insight="")
        if not question or not question.strip():
            ans.insight, ans.error = "Please type a question.", "empty question"
            return ans
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(schema=self.schema)},
                    {"role": "user", "content": question.strip()}]
        hint: dict = {}
        last_err = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            ans.attempts = attempt
            raw = ""
            try:
                raw = self.llm.chat(messages, json_mode=True)
                code, hint = parse_llm_output(raw)
                ans.code = code
                table = to_table(run_sandboxed(code, self.df))
                validate_result(table)
                ans.table = table
                ans.insight = make_insight(self.llm, question, table)
                try:
                    ans.figure = build_chart(table, hint)
                except Exception as ce:  # noqa: BLE001  chart problems must not lose the table
                    ans.log.append(f"chart error: {ce}")
                ans.error = None
                return ans
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                ans.log.append(f"attempt {attempt}: {last_err}")
                messages += [
                    {"role": "assistant", "content": raw or "(no output)"},
                    {"role": "user", "content":
                        f"That failed: {last_err}\nFix the code and return the same JSON format again."},
                ]
        ans.error = last_err
        ans.insight = "Sorry, I couldn't answer that reliably. Try rephrasing the question."
        return ans


# =============================================================================
# 7. CLI + the 5 demo / evaluation questions (NYC 311 flavoured)
# =============================================================================

EVAL_QUESTIONS = [
    "What are the top 10 complaint types?",
    "Which borough has the most complaints?",
    "What is the average resolution time in hours for each borough?",
    "Show the monthly trend of complaints.",
    "Which 5 agencies handle the most complaints, and what share of the total is each?",
]


def main() -> None:
    p = argparse.ArgumentParser(description="NL spreadsheet analyst")
    p.add_argument("csv")
    p.add_argument("question", nargs="?")
    p.add_argument("--eval", action="store_true", help="run the 5 demo questions")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()

    agent = SpreadsheetAgent(a.csv, max_rows=a.max_rows)
    print("Cleaning report:", json.dumps(agent.report, indent=2, default=str))
    os.makedirs(a.out, exist_ok=True)
    qs = EVAL_QUESTIONS if a.eval else [a.question or input("Question: ")]
    for i, q in enumerate(qs, 1):
        r = agent.ask(q)
        print(f"\n[{i}] {q}\nInsight: {r.insight}")
        if r.table is not None:
            print(r.table.head(15).to_string(index=False))
        if r.figure is not None:
            path = os.path.join(a.out, f"chart_{i}.html")
            r.figure.write_html(path)
            print("Chart saved:", path)
        if r.error:
            print("Error:", r.error)


if __name__ == "__main__":
    main()
