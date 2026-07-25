#!/usr/bin/env python3
"""
Quick smoke test for ned_client.py.
Run locally (NED_API_KEY must be set in your environment):

    export NED_API_KEY=your_key_here
    python scripts/smoke_test_ned_client.py
"""

from ned_client import fetch_ned_generation

# Pick a recent full day so all 9 types should have data
START = '2026-07-20'
END = '2026-07-21'

print(f"Fetching NL generation for {START} to {END}...\n")
df = fetch_ned_generation(START, END)

if df.empty:
    print("✗ Got an empty DataFrame -- something's wrong. Check NED_API_KEY and the error output above.")
else:
    print(f"✓ Shape: {df.shape[0]} timestamps x {df.shape[1]} columns")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Time range: {df.index.min()} to {df.index.max()}\n")

    print("Per-column summary (MW):")
    print(df.describe().loc[['mean', 'max']].T)

    # Sanity checks worth eyeballing:
    # - Solar max should be roughly in the multi-GW range on a sunny summer day
    # - Nuclear should look close to constant (NL has one reactor, Borssele, ~0.5 GW)
    # - Wind Offshore max should reflect NL's growing offshore fleet (several GW)
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
