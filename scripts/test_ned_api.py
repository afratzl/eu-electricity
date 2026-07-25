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


def fetch_ned_solar(date_from, date_to):
    """Fetch NL solar from ned.nl API, 15-min resolution"""
    url = 'https://api.ned.nl/v1/utilizations'
    headers = {
        'X-AUTH-TOKEN': NED_API_KEY,
        'accept': 'application/json'
    }
    params = {
        'point': 0,
        'type': 2,            # Solar
        'granularity': 4,
        'granularitytimezone': 1,
        'classification': 2,
        'activity': 1,
        'validfrom[after]': date_from,
        'validfrom[strictly_before]': date_to
    }

    print(f"Fetching ned.nl solar for {date_from}...")
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
        capacity_mw = (r.get('capacity', 0) or 0) / 1000
        rows.append({'timestamp': validfrom, 'ned_mw': capacity_mw})

    if not rows:
        print("  No records returned")
        return None

    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    print(f"  ✓ {len(df)} records")
    print(f"  Mean: {df['ned_mw'].mean():.1f} MW, Max: {df['ned_mw'].max():.1f} MW")
    return df


def fetch_entsoe_solar(date_from, date_to, api_key):
    """Fetch NL solar from ENTSO-E"""
    try:
        from entsoe import EntsoePandasClient
        import entsoe.entsoe
        entsoe.entsoe.URL = 'https://external-api.tp.entsoe.eu/api'

        client = EntsoePandasClient(api_key=api_key)
        start = pd.Timestamp(date_from, tz='Europe/Brussels')
        end = pd.Timestamp(date_to, tz='Europe/Brussels')

        print(f"Fetching ENTSO-E solar for NL...")
        data = client.query_generation('NL', start=start, end=end)

        if data.empty:
            print("  No data returned")
            return None

        solar_cols = [c for c in data.columns if 'Solar' in str(c)]
        print(f"  Solar columns: {solar_cols}")

        if not solar_cols:
            print("  No solar columns found")
            return None

        solar = data[solar_cols].sum(axis=1)
        solar.name = 'entsoe_mw'

        if solar.index.tz is None:
            solar.index = solar.index.tz_localize('UTC').tz_convert('Europe/Brussels')
        else:
            solar.index = solar.index.tz_convert('Europe/Brussels')

        print(f"  ✓ {len(solar)} records")
        print(f"  Mean: {solar.mean():.1f} MW, Max: {solar.max():.1f} MW")
        return solar.to_frame()

    except Exception as e:
        print(f"  Error: {e}")
        return None

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

    entsoe_key = ENTSOE_API_KEY or os.environ.get('ENTSOE_API_KEY')

    # Fetch nuclear
    ned_nuclear = fetch_ned_nuclear(TEST_DATE, TEST_DATE_NEXT)
    entsoe_nuclear = fetch_entsoe_nuclear(TEST_DATE, TEST_DATE_NEXT, entsoe_key) if entsoe_key else None

    # Fetch solar
    ned_solar = fetch_ned_solar(TEST_DATE, TEST_DATE_NEXT)
    entsoe_solar = fetch_entsoe_solar(TEST_DATE, TEST_DATE_NEXT, entsoe_key) if entsoe_key else None

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Nuclear
    if ned_nuclear is not None:
        ax1.plot(ned_nuclear.index, ned_nuclear['ned_mw'], label='ned.nl', color='blue', linewidth=2)
    if entsoe_nuclear is not None:
        ax1.plot(entsoe_nuclear.index, entsoe_nuclear['entsoe_mw'], label='ENTSO-E', color='orange', linewidth=2, linestyle='--')
    ax1.set_title(f'NL Nuclear - {TEST_DATE}')
    ax1.set_ylabel('Power (MW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Solar
    if ned_solar is not None:
        ax2.plot(ned_solar.index, ned_solar['ned_mw'], label='ned.nl', color='blue', linewidth=2)
    if entsoe_solar is not None:
        ax2.plot(entsoe_solar.index, entsoe_solar['entsoe_mw'], label='ENTSO-E', color='orange', linewidth=2, linestyle='--')
    ax2.set_title(f'NL Solar - {TEST_DATE}')
    ax2.set_xlabel('Time (Brussels)')
    ax2.set_ylabel('Power (MW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Print differences
    for name, ned, entsoe in [('Nuclear', ned_nuclear, entsoe_nuclear), ('Solar', ned_solar, entsoe_solar)]:
        if ned is not None and entsoe is not None:
            merged = ned.join(entsoe, how='inner')
            if not merged.empty:
                diff = merged.iloc[:, 0] - merged.iloc[:, 1]
                print(f"\n{name} difference (ned - ENTSO-E):")
                print(f"  Mean: {diff.mean():.1f} MW")
                print(f"  Max:  {diff.max():.1f} MW")
                print(f"  Min:  {diff.min():.1f} MW")

    plt.tight_layout()
    plt.savefig('plots/ned_entsoe_comparison.png', dpi=150)
    print(f"\n✓ Plot saved: plots/ned_entsoe_comparison.png")


if __name__ == '__main__':
    main()
