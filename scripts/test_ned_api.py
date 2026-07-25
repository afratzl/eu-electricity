#!/usr/bin/env python3
"""
Test script: Compare ned.nl vs ENTSO-E for NL nuclear generation
One day of data, 15-minute resolution, plotted side by side
"""

import os
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================
NED_API_KEY = os.environ.get('NED_API_KEY', '')
ENTSOE_API_KEY = None  # Set via environment or hardcode for test

TEST_DATE = '2026-06-15'  # Change as needed
TEST_DATE_NEXT = '2026-06-16'

# ============================================================
# NED.NL FETCH
# ============================================================

def fetch_ned_nuclear(date_from, date_to):
    """Fetch NL nuclear from ned.nl API, 15-min resolution"""
    url = 'https://api.ned.nl/v1/utilizations'
    headers = {
        'X-AUTH-TOKEN': NED_API_KEY,
        'accept': 'application/json'
    }
    params = {
        'point': 0,           # Netherlands
        'type': 20,           # Nuclear
        'granularity': 4,     # 15 minutes
        'granularitytimezone': 1,  # CET
        'classification': 2,  # Current
        'activity': 1,        # Providing
        'validfrom[after]': date_from,
        'validfrom[strictly_before]': date_to
    }

    print(f"Fetching ned.nl nuclear for {date_from}...")
    response = requests.get(url, headers=headers, params=params)
    print(f"  Status: {response.status_code}")

    if response.status_code != 200:
        print(f"  Error: {response.text[:200]}")
        return None

    data = response.json()
    records = data.get('hydra:member', data) if isinstance(data, dict) else data

    rows = []
    for r in records:
        validfrom = pd.Timestamp(r['validfrom']).tz_convert('Europe/Brussels')
        capacity_kw = r.get('capacity', 0) or 0
        capacity_mw = capacity_kw / 1000
        rows.append({'timestamp': validfrom, 'ned_mw': capacity_mw})

    if not rows:
        print("  No records returned")
        return None

    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    print(f"  ✓ {len(df)} records, range: {df.index[0]} to {df.index[-1]}")
    print(f"  Mean: {df['ned_mw'].mean():.1f} MW, Max: {df['ned_mw'].max():.1f} MW")
    return df


# ============================================================
# ENTSO-E FETCH
# ============================================================

def fetch_entsoe_nuclear(date_from, date_to, api_key):
    """Fetch NL nuclear from ENTSO-E"""
    try:
        from entsoe import EntsoePandasClient
        import entsoe.entsoe
        entsoe.entsoe.URL = 'https://external-api.tp.entsoe.eu/api'

        client = EntsoePandasClient(api_key=api_key)
        start = pd.Timestamp(date_from, tz='Europe/Brussels')
        end = pd.Timestamp(date_to, tz='Europe/Brussels')

        print(f"Fetching ENTSO-E nuclear for NL...")
        data = client.query_generation('NL', start=start, end=end)

        if data.empty:
            print("  No data returned")
            return None

        # Find nuclear columns
        nuclear_cols = [c for c in data.columns if 'Nuclear' in str(c)]
        print(f"  Nuclear columns: {nuclear_cols}")

        if not nuclear_cols:
            print("  No nuclear columns found")
            return None

        nuclear = data[nuclear_cols].sum(axis=1)
        nuclear.name = 'entsoe_mw'

        if nuclear.index.tz is None:
            nuclear.index = nuclear.index.tz_localize('UTC').tz_convert('Europe/Brussels')
        else:
            nuclear.index = nuclear.index.tz_convert('Europe/Brussels')

        print(f"  ✓ {len(nuclear)} records")
        print(f"  Mean: {nuclear.mean():.1f} MW, Max: {nuclear.max():.1f} MW")
        return nuclear.to_frame()

    except Exception as e:
        print(f"  Error: {e}")
        return None


# ============================================================
# MAIN
# ============================================================

def main():
    import os

    # Fetch ned.nl
    ned_df = fetch_ned_nuclear(TEST_DATE, TEST_DATE_NEXT)

    # Fetch ENTSO-E
    entsoe_key = ENTSOE_API_KEY or os.environ.get('ENTSOE_API_KEY')
    entsoe_df = None
    if entsoe_key:
        entsoe_df = fetch_entsoe_nuclear(TEST_DATE, TEST_DATE_NEXT, entsoe_key)
    else:
        print("⚠ ENTSOE_API_KEY not set -- skipping ENTSO-E comparison")

    if ned_df is None:
        print("✗ ned.nl fetch failed")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(ned_df.index, ned_df['ned_mw'], label='ned.nl', color='blue', linewidth=2)

    if entsoe_df is not None:
        ax.plot(entsoe_df.index, entsoe_df['entsoe_mw'], label='ENTSO-E', color='orange',
                linewidth=2, linestyle='--')

        # Print difference stats
        merged = ned_df.join(entsoe_df, how='inner')
        if not merged.empty:
            diff = merged['ned_mw'] - merged['entsoe_mw']
            print(f"\nDifference (ned - ENTSO-E):")
            print(f"  Mean: {diff.mean():.1f} MW")
            print(f"  Max:  {diff.max():.1f} MW")
            print(f"  Min:  {diff.min():.1f} MW")

    ax.set_title(f'NL Nuclear Generation - {TEST_DATE}')
    ax.set_xlabel('Time (Brussels)')
    ax.set_ylabel('Power (MW)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/ned_entsoe_comparison.png', dpi=150)
    print(f"\n✓ Plot saved: plots/ned_entsoe_comparison.png")


if __name__ == '__main__':
    main()
