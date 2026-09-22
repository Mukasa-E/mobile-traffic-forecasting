"""
Milan Telecom dataset — memory-efficient ingestion pipeline.

Handles the dataset the way it's actually distributed: ONE tab-separated,
headerless .txt file per day, where each (SquareId, TimeInterval) appears
as MULTIPLE rows (one per CountryCode). We aggregate across country codes
immediately, so we never hold per-country granularity in memory longer
than one chunk.

Columns (per the official dataset description):
    0: SquareId        (int, 1-10000)
    1: TimeInterval     (int64 ms since epoch)
    2: CountryCode      (int)
    3: SMSin
    4: SMSout
    5: CallIn
    6: CallOut
    7: Internet         <- the column we care about for this assignment

Usage:
    python pipeline.py raw/sms-call-internet-mi-2013-12-01.txt

Each call appends one day of aggregated (square, timestamp, internet_total)
rows to a growing Parquet dataset at processed/internet_traffic.parquet,
and prints memory usage before/after so you have real numbers to report
in Section 1 of the report.
"""

import gc
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path("raw")
OUT_DIR = PROJECT_ROOT / "processed"
OUT_DIR.mkdir(exist_ok=True)
PARQUET_PATH = OUT_DIR / "internet_traffic.parquet"

COLUMN_NAMES = [
    "square_id", "time_interval", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet",
]

# Naive dtypes pandas would infer on its own (this is the "before" baseline)
NAIVE_DTYPES = {
    "square_id": "int64",
    "time_interval": "int64",
    "country_code": "float64",  # has NaNs in the real data -> pandas upcasts to float
    "sms_in": "float64", "sms_out": "float64",
    "call_in": "float64", "call_out": "float64",
    "internet": "float64",
}

# Optimised dtypes: square_id fits in uint16 (max 10000), country_code in
# float32 is plenty of precision, all traffic counts are small non-negative
# floats so float32 loses no meaningful information for this use case.
OPTIMISED_DTYPES = {
    "square_id": "uint16",
    "time_interval": "int64",   # keep as-is, needed for exact join on timestamp
    "country_code": "float32",
    "sms_in": "float32", "sms_out": "float32",
    "call_in": "float32", "call_out": "float32",
    "internet": "float32",
}

CHUNK_SIZE = 500_000  # rows per chunk read from disk


def df_memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e6


def process_day_naive(path: Path) -> tuple[pd.DataFrame, float, float]:
    """Baseline: read the whole day file at once with default/naive dtypes.
    Used ONLY to produce the 'before' memory number for the report — do not
    use this path for the full 2-month run, it defeats the point."""
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMN_NAMES,
                      dtype=NAIVE_DTYPES)
    mem = df_memory_mb(df)
    elapsed = time.time() - t0
    return df, mem, elapsed


def process_day_optimised(path: Path) -> tuple[pd.DataFrame, float, float]:
    """Chunked read + downcast dtypes + aggregate across country_code
    immediately, so peak memory is bounded by CHUNK_SIZE, not file size."""
    t0 = time.time()
    agg_parts = []
    peak_chunk_mb = 0.0

    reader = pd.read_csv(
        path, sep="\t", header=None, names=COLUMN_NAMES,
        dtype=OPTIMISED_DTYPES, chunksize=CHUNK_SIZE,
        usecols=["square_id", "time_interval", "internet"],  # drop sms/call, not needed for this task
    )
    for chunk in reader:
        peak_chunk_mb = max(peak_chunk_mb, df_memory_mb(chunk))
        # collapse country-code rows: sum internet traffic per (square, interval)
        part = chunk.groupby(["square_id", "time_interval"], as_index=False)["internet"].sum()
        agg_parts.append(part)
        del chunk

    day_df = pd.concat(agg_parts, ignore_index=True)
    day_df = day_df.groupby(["square_id", "time_interval"], as_index=False)["internet"].sum()
    day_df["internet"] = day_df["internet"].astype("float32")
    elapsed = time.time() - t0
    return day_df, peak_chunk_mb, elapsed


def append_to_parquet(day_df: pd.DataFrame):
    """Append one day's aggregated data to the growing Parquet dataset."""
    day_df["timestamp"] = pd.to_datetime(day_df["time_interval"], unit="ms")
    day_df = day_df.drop(columns=["time_interval"])

    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
        combined = pd.concat([existing, day_df], ignore_index=True)
        del existing
    else:
        combined = day_df

    combined = combined.drop_duplicates(subset=["square_id", "timestamp"])
    combined.to_parquet(PARQUET_PATH, index=False, compression="snappy")
    del combined
    gc.collect()


def run(file_path: str, compare_naive: bool = False):
    path = Path(file_path)
    print(f"\n=== Processing {path.name} ===")

    if compare_naive:
        tracemalloc.start()
        _, naive_mb, naive_t = process_day_naive(path)
        _, naive_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[naive]      in-memory size: {naive_mb:8.1f} MB | "
              f"peak traced: {naive_peak/1e6:8.1f} MB | time: {naive_t:5.2f}s")

    tracemalloc.start()
    day_df, opt_peak_chunk_mb, opt_t = process_day_optimised(path)
    _, opt_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[optimised]  final agg size:  {df_memory_mb(day_df):8.3f} MB | "
          f"peak traced: {opt_peak/1e6:8.1f} MB | largest chunk: {opt_peak_chunk_mb:6.1f} MB | "
          f"time: {opt_t:5.2f}s")

    append_to_parquet(day_df)
    total_rows = pd.read_parquet(PARQUET_PATH, columns=["square_id"]).shape[0]
    print(f"Parquet dataset now has {total_rows:,} rows "
          f"({PARQUET_PATH.stat().st_size/1e6:.1f} MB on disk)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_day_file.txt> [--compare-naive]")
        sys.exit(1)
    compare = "--compare-naive" in sys.argv
    run(sys.argv[1], compare_naive=compare)
