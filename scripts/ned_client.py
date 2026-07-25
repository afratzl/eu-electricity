"""
ned.nl API client for the Netherlands.

ENTSO-E massively underreports NL solar and onshore wind generation, so NL
is fetched from ned.nl instead. All other countries keep using ENTSO-E.

Public entry point: fetch_ned_generation(start_date, end_date)

Returns a DataFrame shaped exactly like ENTSO-E's client.query_generation()
output, so it can be dropped into the existing pipeline unchanged:
    - index: timestamps, Europe/Brussels tz, 15-minute resolution
    - columns: source-name strings that match config.SOURCE_KEYWORDS
      substrings ('Solar', 'Wind Onshore', 'Wind Offshore', 'Fossil Gas',
      'Fossil Hard coal', 'Nuclear', 'Biomass', 'Geothermal', 'Waste')
    - values: generation in MW

Notes on the mapping (see ned.nl's own API type table + changelog):
    - No hydro or oil type exists on ned.nl for NL. Both are ~0 for NL in
      reality (no elevation for hydro, negligible oil generation), so these
      columns simply don't appear here -- same net effect as ENTSO-E
      returning 0 for them.
    - Offshore wind uses type 51 (WindOffshoreC), ned.nl's newest offshore
      model (added Feb 2023, changelog v1.2.0). Types 17 (WindOffshore) and
      22 (WindOffshoreB) are earlier/superseded models -- do not use them.
    - 'gas' uses type 18 (FossilGasPower), not 23 (NaturalGas) -- the latter
      is gas *consumption* and had its activity flipped to "consuming" in a
      June 2024 API change, so it wouldn't return generation data anyway.
    - 'biomass' uses type 25 (BiomassPower), not 13 (Biomass), which is a
      different (non-power) series.

Rate limit: ned.nl allows 200 requests per 5 minutes. This module caches
the full per-period fetch (9 requests) so repeated calls for the same
(start_date, end_date) window -- which happen once per atomic source in
intraday_analysis.py's collection loop -- don't refetch from the API.
"""

import os
import time
import requests
import pandas as pd

NED_BASE_URL = 'https://api.ned.nl/v1/utilizations'

# Internal column name (matches SOURCE_KEYWORDS substrings) -> ned.nl type code
NED_TYPE_MAP = {
    'Solar':            2,
    'Wind Onshore':     1,
    'Wind Offshore':    51,   # WindOffshoreC -- current model; not 17 or 22
    'Fossil Gas':       18,   # FossilGasPower
    'Fossil Hard coal': 19,   # FossilHardCoal
    'Nuclear':          20,
    'Biomass':          25,   # BiomassPower
    'Geothermal':       9,
    'Waste':            21,   # WastePower
}

MAX_RETRIES = 4
REQUEST_SLEEP_SECONDS = 1.5  # keep well under 200 req / 5 min

# Module-level cache: {(start_date, end_date): DataFrame}
# Cleared implicitly each process run (intraday_analysis.py / eu_energy_data_collection.py
# are both run-once-per-invocation scripts, so no explicit eviction is needed).
_period_cache = {}


def _get_api_key():
    key = os.environ.get('NED_API_KEY', '')
    if not key:
        raise RuntimeError("NED_API_KEY environment variable is not set")
    return key


def _fetch_ned_type(type_code, date_from, date_to, api_key):
    """
    Fetch one ned.nl type for NL over [date_from, date_to).
    date_from / date_to: 'YYYY-MM-DD' strings (validfrom is exclusive at date_to).
    Returns a pandas Series indexed by Europe/Brussels timestamps, values in MW.
    Returns an empty Series on failure -- callers should treat that like
    ENTSO-E returning no data for a source (i.e. it contributes 0, not None).
    """
    headers = {
        'X-AUTH-TOKEN': api_key,
        'accept': 'application/json',
    }
    params = {
        'point': 0,                  # Netherlands
        'type': type_code,
        'granularity': 4,            # 15-minute
        'granularitytimezone': 1,    # CET
        'classification': 2,         # current (not forecast)
        'activity': 1,               # providing (generation, not consumption)
        'validfrom[after]': date_from,
        'validfrom[strictly_before]': date_to,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(NED_BASE_URL, headers=headers, params=params, timeout=30)

            if response.status_code == 429:
                # Rate limited -- back off harder than a normal retry
                time.sleep(5 * (attempt + 1))
                continue

            if response.status_code != 200:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                print(f"    ✗ ned.nl type {type_code}: HTTP {response.status_code}: {response.text[:200]}")
                return pd.Series(dtype=float)

            data = response.json()
            records = data.get('hydra:member', data) if isinstance(data, dict) else data

            if not records:
                return pd.Series(dtype=float)

            rows = {}
            for r in records:
                ts = pd.Timestamp(r['validfrom']).tz_convert('Europe/Brussels')
                capacity_kw = r.get('capacity', 0) or 0
                rows[ts] = capacity_kw / 1000.0  # kW -> MW

            series = pd.Series(rows).sort_index()
            time.sleep(REQUEST_SLEEP_SECONDS)
            return series

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                print(f"    ✗ ned.nl type {type_code}: {e}")
                return pd.Series(dtype=float)

    return pd.Series(dtype=float)


def fetch_ned_generation(start_date, end_date):
    """
    Fetch all mapped ned.nl generation types for NL over [start_date, end_date).

    start_date / end_date: accepts 'YYYY-MM-DD' strings or anything
    pd.Timestamp(...).strftime('%Y-%m-%d') can normalize (datetime, date,
    pd.Timestamp), so callers can pass the same values they'd pass to
    client.query_generation() without extra conversion.

    Results are cached per (start_date, end_date) pair for the lifetime of
    the process, so calling this repeatedly for the same period (once per
    atomic source, as intraday_analysis.py does) triggers only one round of
    9 ned.nl requests, not one round per call.

    Returns a DataFrame shaped like ENTSO-E's client.query_generation()
    output. Returns an empty DataFrame if nothing could be fetched.
    """
    start_key = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end_key = pd.Timestamp(end_date).strftime('%Y-%m-%d')
    cache_key = (start_key, end_key)

    if cache_key in _period_cache:
        return _period_cache[cache_key]

    api_key = _get_api_key()

    print(f"  Fetching ned.nl data for NL: {start_key} to {end_key} ({len(NED_TYPE_MAP)} types)...")

    columns = {}
    for column_name, type_code in NED_TYPE_MAP.items():
        series = _fetch_ned_type(type_code, start_key, end_key, api_key)
        if not series.empty:
            columns[column_name] = series
        else:
            print(f"    ⚠ No ned.nl data for {column_name} (type {type_code})")

    if not columns:
        df = pd.DataFrame()
    else:
        df = pd.concat(columns, axis=1)

    _period_cache[cache_key] = df
    return df
