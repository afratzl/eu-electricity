#!/usr/bin/env python3
"""
Quick smoke test for ned_client.py.
Run locally (NED_API_KEY must be set in your environment):

    export NED_API_KEY=your_key_here
    python scripts/smoke_test_ned_client.py

Includes a multi-week test specifically because the original single-day-only
version of this test passed cleanly while silently missing ned.nl's
pagination (a single day is under one page; anything longer wasn't being
tested here at all, and the first real multi-month run went out undetected).
"""

from ned_client import fetch_ned_generation, fetch_ned_generation_monthly
import pandas as pd

# ============================================================
# TEST 1: single day (original test, kept for the fast basic check)
# ============================================================
START = '2026-07-20'
END = '2026-07-21'

print(f"[Test 1] Fetching NL generation for {START} to {END} (single day)...\n")
df = fetch_ned_generation(START, END)

if df.empty:
    print("✗ Got an empty DataFrame -- something's wrong. Check NED_API_KEY and the error output above.")
else:
    print(f"✓ Shape: {df.shape[0]} timestamps x {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Time range: {df.index.min()} to {df.index.max()}\n")

    print("Per-column summary (MW):")
    print(df.describe().loc[['mean', 'max']].T)

    print("\nFirst few rows:")
    print(df.head())

    print("\nAny missing columns from the 10 expected types?")
    expected = {'Solar', 'Wind Onshore', 'Wind Offshore', 'Fossil Gas',
                'Fossil Hard coal', 'Nuclear', 'Biomass', 'Waste',
                'Other', 'Other (WKK)'}
    missing = expected - set(df.columns)
    print(missing if missing else "None -- all 10 present")

    if 'Other (WKK)' in df.columns:
        print(f"\nWKK Total shape check (should vary through the day, not be flat "
              f"like Geothermal was): min={df['Other (WKK)'].min():.0f}, "
              f"max={df['Other (WKK)'].max():.0f} MW")

    # Second call with the same dates should hit the cache and print nothing
    # new (no "Fetching ned.nl data..." line, no repeated requests)
    print("\nCalling again with same dates (should use cache, no new fetch print)...")
    df2 = fetch_ned_generation(START, END)
    print("✓ Cache check passed" if df2.equals(df) else "✗ Cache returned different data!")

expected_rows_test1 = 96  # 1 day at 15-min resolution
if not df.empty:
    status = "✓" if df.shape[0] == expected_rows_test1 else "✗ MISMATCH"
    print(f"\n{status} Row count check: got {df.shape[0]}, expected {expected_rows_test1}")

# ============================================================
# TEST 2: multi-week range, specifically to exercise pagination
# ============================================================
MULTI_START = '2026-06-01'
MULTI_END = '2026-07-21'  # 50 days -- comfortably spans multiple ned.nl pages

print(f"\n\n[Test 2] Fetching NL generation for {MULTI_START} to {MULTI_END} "
      f"(50 days, spans multiple pages)...\n")
df_multi = fetch_ned_generation(MULTI_START, MULTI_END)

if df_multi.empty:
    print("✗ Got an empty DataFrame for the multi-day range -- something's wrong.")
else:
    days = (pd.Timestamp(MULTI_END) - pd.Timestamp(MULTI_START)).days
    expected_rows_test2 = days * 96
    print(f"✓ Shape: {df_multi.shape[0]} timestamps x {df_multi.shape[1]} columns")
    status = "✓" if df_multi.shape[0] == expected_rows_test2 else "✗ MISMATCH -- possible pagination truncation!"
    print(f"{status} Row count check: got {df_multi.shape[0]}, expected {expected_rows_test2} "
          f"({days} days x 96 intervals/day)")
    print(f"Time range: {df_multi.index.min()} to {df_multi.index.max()}")

# ============================================================
# TEST 3: monthly granularity, used by eu_energy_data_collection.py
# ============================================================
print(f"\n\n[Test 3] Fetching NL generation (monthly) for 2026-01-01 to 2026-07-21...\n")
df_monthly = fetch_ned_generation_monthly('2026-01-01', '2026-07-21')

if df_monthly.empty:
    print("✗ Got an empty DataFrame for the monthly fetch -- something's wrong.")
else:
    print(f"✓ Shape: {df_monthly.shape[0]} rows x {df_monthly.shape[1]} columns")
    print(f"✓ Expected roughly 7 rows (Jan-Jul 2026), got {df_monthly.shape[0]}")
    print(df_monthly)
