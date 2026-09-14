"""Streaming CSV reader + incremental profiler for large files.

The in-memory path materialises the whole CSV into a single ``list[dict]``
before profiling. For large CSVs that is wasteful: 1M rows become millions
of small dicts. This module instead streams the CSV in bounded line chunks,
parses each chunk with Polars at C speed, accumulates exact running
aggregates (counts, sums, min/max, missing values, top categories,
duplicates) plus per-chunk quantile summaries (bounded memory), and emits a
profile with the same top-level shape as ``app.profile.build_data_profile``
plus a representative first / random / last sample. The LLM is still called
exactly once for the whole file regardless of size.

Note: chunks are aligned on physical lines, so quoted fields that contain a
literal newline are not supported on the streaming path (they are on the
in-memory path). This is a deliberate trade-off for speed and bounded memory.
"""

import io
import math
import random
from collections import Counter
from typing import Any, Iterable

import polars as pl

from .csv_adapter import _coerce
from .file_adapter import _detect_delimiter
from ..profile.semantics import build_semantics

DEFAULT_CHUNK_ROWS = 40_000
FIRST_N = 5
RANDOM_N = 10
LAST_N = 5
STREAM_ROW_THRESHOLD = 25_000
STREAM_CHAR_THRESHOLD = 3_000_000
CAT_CARD_CAP = 500
DUP_HASH_CAP = 50_000
SAMPLE_STR_MAX = 60
MAX_CORR_COLS = 25

_NUM_DTYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)


def should_stream_csv(text: Any, max_row_estimate: int = STREAM_ROW_THRESHOLD, min_chars: int = STREAM_CHAR_THRESHOLD) -> bool:
    """True when a CSV text is large enough to warrant the streaming path.

    Uses cheap signals (length + row count estimate) so a small pasted CSV
    never pays the streaming cost.
    """
    if not isinstance(text, str) or not text:
        return False
    return len(text) >= min_chars or text.count("\n") >= max_row_estimate


def iter_csv_chunks(text: str, delimiter: str = ",", chunk_size: int = DEFAULT_CHUNK_ROWS) -> Iterable[list[dict]]:
    """Yield ``chunk_size`` rows at a time as pure-Python dict lists.

    Kept as a portable fallback; the profiling path uses Polars chunks for
    speed.
    """
    reader = __import__("csv").DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames or not [h for h in reader.fieldnames if h and h.strip()]:
        raise ValueError("CSV must include a header row")

    bucket: list[dict] = []
    for row in reader:
        record = {}
        for header, value in row.items():
            key = (header or "").strip()
            if not key:
                continue
            record[key] = _coerce(value)
        if record:
            bucket.append(record)
            if len(bucket) >= chunk_size:
                yield bucket
                bucket = []
    if bucket:
        yield bucket


def _body_chunks(text: str, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> Iterable[list[str]]:
    """Yield chunks of raw data lines (header excluded)."""
    stream = io.StringIO(text)
    stream.readline()  # skip header row
    lines: list[str] = []
    for line in stream:
        lines.append(line)
        if len(lines) >= chunk_rows:
            yield lines
            lines = []
    if lines:
        yield lines


class _ColAcc:
    """Bounded per-column running statistics (chunk-level aggregates)."""

    __slots__ = (
        "name", "nulls", "num_n", "num_sum", "num_sumsq", "num_min", "num_max",
        "bool_n", "str_n", "str_len_sum", "chunk_quants", "counter", "cat_total",
        "card_capped", "cat_overflow", "constant_cands", "not_constant",
    )

    def __init__(self, name: str):
        self.name = name
        self.nulls = 0
        self.num_n = 0
        self.num_sum = 0.0
        self.num_sumsq = 0.0
        self.num_min = None
        self.num_max = None
        self.bool_n = 0
        self.str_n = 0
        self.str_len_sum = 0.0
        self.chunk_quants: list[tuple[float, float, float]] = []
        self.counter: Counter = Counter()
        self.cat_total = 0
        self.card_capped = False
        self.cat_overflow = False
        self.constant_cands: set = set()
        self.not_constant = False

    def add_quants(self, q25, q50, q75) -> None:
        self.chunk_quants.append((q25, q50, q75))

    def aggregate_quants(self):
        if not self.chunk_quants:
            return None
        q25s = sorted(q[0] for q in self.chunk_quants)
        q50s = sorted(q[1] for q in self.chunk_quants)
        q75s = sorted(q[2] for q in self.chunk_quants)
        med = lambda xs: xs[len(xs) // 2]
        return med(q25s), med(q50s), med(q75s)

    def mean_string_len(self) -> float:
        return self.str_len_sum / self.str_n if self.str_n else 0.0

    def num_mean(self) -> float:
        return self.num_sum / self.num_n if self.num_n else 0.0

    def num_std(self) -> float:
        if self.num_n < 2:
            return 0.0
        var = (self.num_sumsq - self.num_sum * self.num_sum / self.num_n) / (self.num_n - 1)
        return math.sqrt(var) if var > 0 else 0.0


def _accumulate_numeric(acc: _ColAcc, s: pl.Series) -> None:
    n = s.count()
    if n == 0:
        return
    total = s.sum()
    acc.num_n += n
    acc.num_sum += total
    acc.num_sumsq += float((s.cast(pl.Float64) ** 2).sum())
    m = s.min()
    if m is not None and (acc.num_min is None or m < acc.num_min):
        acc.num_min = m
    mx = s.max()
    if mx is not None and (acc.num_max is None or mx > acc.num_max):
        acc.num_max = mx
    acc.add_quants(
        float(s.quantile(0.25, interpolation="nearest")),
        float(s.median()),
        float(s.quantile(0.75, interpolation="nearest")),
    )


def _accumulate_string(acc: _ColAcc, s: pl.Series, first_chunk: bool) -> None:
    n = s.count()
    if n == 0:
        return
    acc.str_n += n
    lengths = s.str.len_chars()
    acc.str_len_sum += float(lengths.sum())
    if acc.card_capped or acc.cat_overflow:
        if acc.card_capped:
            acc.cat_total += n
        return
    vc = s.value_counts(name="count")
    value_series = vc.get_column(vc.columns[0])
    count_series = vc.get_column("count")
    for val, count in zip(value_series, count_series):
        if isinstance(val, str) and len(val) > 40:
            continue
        acc.counter[val] += count
        acc.cat_total += count
        if len(acc.counter) > CAT_CARD_CAP:
            acc.card_capped = True
            acc.counter.clear()
            acc.cat_total = 0
            acc.cat_overflow = True
            return


def _accumulate_constant(acc: _ColAcc, s: pl.Series) -> None:
    nn = s.drop_nulls()
    u = nn.n_unique()
    if u > 1:
        acc.not_constant = True
        acc.constant_cands.clear()
        return
    if u == 1 and not acc.not_constant:
        first = nn.first()
        if first is not None:
            acc.constant_cands.add(first)
            if len(acc.constant_cands) > 2:
                acc.not_constant = True
                acc.constant_cands.clear()


def incremental_profile_csv(
    text: str,
    delimiter: str | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    first_n: int = FIRST_N,
    random_n: int = RANDOM_N,
    last_n: int = LAST_N,
) -> dict:
    """Profile a CSV stream, returning ``{"profile", "sample"}``.

    ``profile`` matches the top-level shape of ``build_data_profile`` so the
    existing prompt/deterministic paths work unchanged. ``sample`` matches the
    shape expected by ``build_profile_prompt`` (first / random / last buckets).
    """
    import csv as _csv

    if delimiter is None:
        delimiter = _detect_delimiter(text)
    elif isinstance(delimiter, str) and delimiter.lower() in ("auto", "tab"):
        delimiter = "\t" if delimiter.lower() == "tab" else _detect_delimiter(text)

    reader = _csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames or not [h for h in reader.fieldnames if h and h.strip()]:
        raise ValueError("CSV must include a header row")
    header = [h.strip() for h in reader.fieldnames if h and h.strip()]

    rows = 0
    col_acc: dict[str, _ColAcc] = {name: _ColAcc(name) for name in header}
    first_rows: list[dict] = []
    last_rows: list[dict] = []
    random_pool: list[dict] = []
    dup_set: set = set()
    dup_count = 0
    dup_capped = False
    schema_inferred = False
    schema = None
    corr_cols: list[str] = []
    pair_sum: dict[tuple[str, str], float] = {}
    pair_n: dict[tuple[str, str], int] = {}

    for ci, lines in enumerate(_body_chunks(text, chunk_rows)):
        body = "".join(lines)
        if schema_inferred:
            frame = pl.read_csv(
                io.StringIO(body),
                has_header=False,
                new_columns=header,
                schema=schema,
                infer_schema_length=0,
            )
        else:
            frame = pl.read_csv(
                io.StringIO(body),
                has_header=False,
                new_columns=header,
                infer_schema_length=200,
            )
            schema_inferred = True
            schema = frame.schema
            numeric_cols = [n for n in header if n in frame.columns and frame.schema.get(n) in _NUM_DTYPES]
            if len(numeric_cols) <= MAX_CORR_COLS:
                corr_cols = numeric_cols

        height = frame.height
        rows += height
        for name in header:
            if name not in frame.columns:
                continue
            s = frame.get_column(name)
            acc = col_acc[name]
            nulls = s.null_count()
            acc.nulls += int(nulls)
            nval = height - nulls
            if nval == 0:
                continue
            dtype = frame.schema[name]
            if dtype in _NUM_DTYPES:
                _accumulate_numeric(acc, s)
            elif dtype == pl.Boolean:
                acc.bool_n += nval
            elif dtype in (pl.Date, pl.Datetime):
                pass  # classified as datetime from dtype, no aggregates needed
            else:
                acc.str_n += nval
                _accumulate_string(acc, s, ci == 0)
            _accumulate_constant(acc, s)

        if corr_cols:
            for i in range(len(corr_cols)):
                a = corr_cols[i]
                sa = frame.get_column(a).cast(pl.Float64)
                for j in range(i + 1, len(corr_cols)):
                    b = corr_cols[j]
                    prod = sa * frame.get_column(b).cast(pl.Float64)
                    key = (a, b)
                    pair_sum[key] = pair_sum.get(key, 0.0) + float(prod.sum())
                    pair_n[key] = pair_n.get(key, 0) + int(prod.count())

        # first / random / last sample rows
        if ci == 0 and first_rows is not None:
            first_rows = frame.head(first_n).to_dicts()
        if not dup_capped and height and header:
            hashes = frame.select(pl.struct(header).hash(seed=42).alias("h")).to_series().to_list()
            for h in hashes:
                if h in dup_set:
                    dup_count += 1
                elif len(dup_set) >= DUP_HASH_CAP:
                    dup_capped = True
                    break
                else:
                    dup_set.add(h)
        tail = frame.tail(last_n).to_dicts()
        last_rows = (last_rows + tail)[-last_n:]
        if height >= 10:
            random_pool.append(frame.sample(n=1, seed=42 + ci).to_dicts()[0])
        elif tail:
            random_pool.append(tail[0])

    # representative random sample from the per-chunk pool
    if random_pool:
        rng = random.Random(42)
        random_rows = rng.sample(random_pool, min(random_n, len(random_pool)))
    else:
        random_rows = []

    profile = _assemble_profile(col_acc, header, rows, dup_count, dup_capped, delimiter, pair_sum, pair_n)
    profile = _count_outliers(profile, col_acc, text, delimiter, header, chunk_rows)
    sample = {
        "total_rows": rows,
        "per_bucket": min(first_n, max(RANDOM_N, LAST_N)),
        "buckets": {
            "first": [_compact_sample_row(r) for r in first_rows],
            "random": [_compact_sample_row(r) for r in random_rows],
            "last": [_compact_sample_row(r) for r in last_rows],
        },
    }
    return {"profile": profile, "sample": sample}


def _count_outliers(profile: dict, col_acc: dict, text: str, delimiter: str, header: list, chunk_rows: int) -> dict:
    """Second streaming pass: count numeric values outside final IQR bounds."""
    import csv as _csv

    stats = {s["column"]: s for s in (profile["statistics"].get("important_numeric_columns") or [])}
    targets = {name: acc for name, acc in col_acc.items() if name in stats}
    if not targets:
        profile["anomalies"] = {"total_potential_outliers": 0, "top_columns": []}
        return profile

    bounds = {}
    for name in targets:
        quants = col_acc[name].aggregate_quants()
        if quants is None:
            continue
        q1, _q2, q3 = quants
        if q1 is None or q3 is None or q3 == q1:
            continue
        iqr = q3 - q1
        bounds[name] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    if not bounds:
        profile["anomalies"] = {"total_potential_outliers": 0, "top_columns": []}
        return profile

    reader = _csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames or not [h for h in reader.fieldnames if h and h.strip()]:
        return profile
    names = set(bounds)
    counts = {n: 0 for n in names}
    for row in reader:
        for name in names:
            raw = row.get(name)
            if raw is None:
                continue
            try:
                v = float(raw.strip())
            except (ValueError, TypeError):
                continue
            lo, hi = bounds[name]
            if v < lo or v > hi:
                counts[name] += 1
    total = sum(counts.values())
    top = sorted(
        ({"column": n, "outliers": c, "share": round(c / max(1, col_acc[n].num_n), 4)} for n, c in counts.items() if c),
        key=lambda x: x["outliers"],
        reverse=True,
    )[:10]
    profile["anomalies"] = {"total_potential_outliers": total, "top_columns": top}
    return profile


def _compact_sample_row(record: dict) -> dict:
    return {
        k: (str(v)[:SAMPLE_STR_MAX] + "…" if isinstance(v, str) and len(v) > SAMPLE_STR_MAX else v)
        for k, v in record.items()
    }


def _classify(acc: _ColAcc) -> str:
    non_null = acc.num_n + acc.bool_n + acc.str_n
    if non_null == 0:
        return "categorical"
    if acc.bool_n == non_null:
        return "boolean"
    if acc.num_n == non_null:
        return "numeric"
    if acc.mean_string_len() > 60:
        return "text"
    if acc.card_capped or acc.cat_overflow:
        return "text"
    return "categorical"


def _assemble_profile(
    col_acc: dict[str, _ColAcc],
    header: list[str],
    rows: int,
    dup_count: int,
    dup_capped: bool,
    delimiter: str,
    pair_sum: dict[tuple[str, str], float] | None = None,
    pair_n: dict[tuple[str, str], int] | None = None,
) -> dict:
    classified = {name: _classify(acc) for name, acc in col_acc.items()}
    ordered = sorted(classified)
    numeric = [n for n in ordered if classified[n] == "numeric"]
    cats = [n for n in ordered if classified[n] == "categorical"]
    texts = [n for n in ordered if classified[n] == "text"]
    datetimes = [n for n in ordered if classified[n] == "datetime"]
    booleans = [n for n in ordered if classified[n] == "boolean"]

    stats = []
    for name in numeric:
        acc = col_acc[name]
        mean = acc.num_mean()
        cv = abs(acc.num_std() / mean) if mean else 0.0
        quants = acc.aggregate_quants()
        stats.append({
            "column": name,
            "min": acc.num_min,
            "max": acc.num_max,
            "mean": round(mean, 6),
            "median": quants[1] if quants else None,
            "std": round(acc.num_std(), 6),
            "cv": cv,
        })
    stats.sort(key=lambda s: (s["cv"], abs(s["mean"] or 0)), reverse=True)
    stats = stats[:10]

    global_min = None
    global_max = None
    for name in numeric:
        acc = col_acc[name]
        if global_min is None or (acc.num_min is not None and acc.num_min < global_min["value"]):
            global_min = {"value": acc.num_min, "column": name}
        if global_max is None or (acc.num_max is not None and acc.num_max > global_max["value"]):
            global_max = {"value": acc.num_max, "column": name}

    pairs = []
    if pair_n:
        for (a, b), n in pair_n.items():
            if n < 2 or a not in numeric or b not in numeric:
                continue
            ca, cb = col_acc[a], col_acc[b]
            sum_a, sum_b = ca.num_sum, cb.num_sum
            sum_a2, sum_b2 = ca.num_sumsq, cb.num_sumsq
            ab = pair_sum.get((a, b), 0.0)
            denom_sq = (n * sum_a2 - sum_a * sum_a) * (n * sum_b2 - sum_b * sum_b)
            if denom_sq <= 0:
                continue
            denom = math.sqrt(denom_sq)
            if not denom:
                continue
            r = (n * ab - sum_a * sum_b) / denom
            pairs.append({"features": [a, b], "value": round(r, 4)})
    pairs.sort(key=lambda p: p["value"], reverse=True)
    top_pos = [p for p in pairs if p["value"] > 0][:5]
    top_neg = [p for p in pairs if p["value"] < 0][:5]

    cat_breakdown = []
    imbalance = []
    for name in cats:
        acc = col_acc[name]
        if acc.card_capped or not acc.counter:
            continue
        top = acc.counter.most_common(5)
        unique = len(acc.counter)
        cat_breakdown.append({
            "column": name,
            "unique": unique,
            "top_values": [{"value": v, "count": c} for v, c in top],
        })
        if unique > 2 and acc.cat_total:
            share = top[0][1] / acc.cat_total
            if share >= 0.8:
                imbalance.append({
                    "column": name,
                    "dominant_class_share": round(share, 4),
                    "suggestion": "Class imbalance — aggregate summaries should flag this skew.",
                })

    missing_total = sum(acc.nulls for acc in col_acc.values())
    missing_by_col = sorted(
        ((name, acc.nulls) for name, acc in col_acc.items() if acc.nulls),
        key=lambda t: t[1],
        reverse=True,
    )[:5]

    constants = [n for n, acc in col_acc.items() if not acc.not_constant and acc.constant_cands]

    bands = {"low": [], "mid": [], "high": []}
    for name in sorted(header):
        lower = name.lower()
        if any(w in lower for w in ("id", "key", "code", "no")):
            bands["low"].append(name)
        elif any(w in lower for w in ("amount", "price", "score", "count", "qty", "total", "sum", "percent", "rate")):
            bands["high"].append(name)
        else:
            bands["mid"].append(name)

    patterns = {
        "frequency_bands": bands,
    }
    domain = {"label": "general", "confidence": 0.0, "evidence": "no strong domain vocabulary"}
    if any(n in header for n in ("customer", "order", "user", "product", "employee", "transaction")):
        domain = {"label": "business", "confidence": 0.7, "evidence": "recognised business-style column names"}
    elif any(n in header for n in ("sensor", "timestamp", "temperature", "pressure", "value")):
        domain = {"label": "iot_telemetry", "confidence": 0.6, "evidence": "recognised telemetry-style column names"}

    return {
        "dataset": {"rows": rows, "columns": len(header)},
        "data_types": {
            "numeric": numeric,
            "categorical": cats,
            "text": texts,
            "datetime": datetimes,
            "boolean": booleans,
        },
        "quality": {
            "missing_values_total": missing_total,
            "missing_by_column_top": [{"column": n, "count": c} for n, c in missing_by_col],
            "duplicate_rows": dup_count if not dup_capped else None,
            "duplicate_count_estimated": dup_capped,
            "constant_columns": constants,
            "unique_value_counts": {n: 1 for n in constants},
        },
        "columns": sorted(header),
        "column_categories": {
            "numeric": numeric,
            "categorical": cats,
            "text": texts,
            "datetime": datetimes,
            "boolean": booleans,
        },
        "semantics": build_semantics(header),
        "statistics": {
            "important_numeric_columns": stats,
            "global_min": global_min,
            "global_max": global_max,
        },
        "correlations": {"top_positive": top_pos, "top_negative": top_neg},
        "categorical_breakdown": cat_breakdown,
        "class_imbalance": imbalance,
        "anomalies": {"total_potential_outliers": 0, "top_columns": []},
        "domain": domain,
        "patterns": patterns,
        "input_type": "csv",
        "delimiter": delimiter,
        "profile_method": "streamed-incremental",
    }