"""
ned.nl API client for the Netherlands.

ENTSO-E massively underreports NL solar and onshore wind generation, so NL
is fetched from ned.nl instead. All other countries keep using ENTSO-E.

Two public entry points:
    - fetch_ned_generation(start_date, end_date): 15-minute resolution,
      used by intraday_analysis.py.
    - fetch_ned_generation_monthly(start_date, end_date): monthly
      resolution, used by eu_energy_data_collection.py. See "Why two
      granularities" below for why this exists as a separate function
      rather than just calling the 15-minute one and resampling.

Both return a DataFrame shaped exactly like ENTSO-E's client.query_generation()
output, so either can be dropped into the existing pipeline unchanged:
    - index: timestamps, Europe/Brussels tz
    - columns: source-name strings that match config.SOURCE_KEYWORDS
      substrings ('Solar', 'Wind Onshore', 'Wind Offshore', 'Fossil Gas',
      'Fossil Hard coal', 'Nuclear', 'Biomass', 'Waste', 'Other',
      'Other (WKK)') -- 10 columns, not the full 10 atomic sources + small
      categories ENTSO-E provides (no hydro, oil, or geothermal; see notes
      below on why)
    - values: generation in MW (ned.nl's 'capacity' field, kW -> MW)

Notes on the mapping (see ned.nl's own API type table + changelog):
    - No hydro or oil type exists on ned.nl for NL. Both are ~0 for NL in
      reality (no elevation for hydro, negligible oil generation), so these
      columns simply don't appear here -- same net effect as ENTSO-E
      returning 0 for them.
    - Geothermal (type 9) is deliberately NOT included. ned.nl's own data
      catalog (https://ned.nl/nl/datacatalogus) puts it in the "Overige
      duurzame energieproductie (warmte + elektriciteit)" category -- a
      mixed heat-and-electricity bucket, not the "Elektriciteitsproductie"
      (electricity production) category the other 8 types below belong to.
      There is no geothermal-electricity-only type on ned.nl at all, which
      matches reality: the Netherlands has no grid-connected geothermal
      power plants, only geothermal heat (greenhouses, district heating).
      Including it would silently convert heat energy into phantom
      electricity MW. All 8 types below are confirmed to sit in ned.nl's
      "Elektriciteitsproductie" category, so this asymmetry is specific to
      geothermal, not a sign the others need the same scrutiny.
    - Offshore wind uses type 51 (WindOffshoreC), ned.nl's newest offshore
      model (added Feb 2023, changelog v1.2.0, "used for ned.nl and
      energieopwek.nl"). Types 17 (WindOffshore) and 22 (WindOffshoreB) are
      earlier/superseded models -- do not use them.
    - 'gas' uses type 18 (FossilGasPower), not 23 (NaturalGas) -- the latter
      is gas *consumption* and had its activity flipped to "consuming" in a
      June 2024 API change, so it wouldn't return generation data anyway.
    - 'biomass' uses type 25 (BiomassPower), not 13 (Biomass) -- the latter
      is in the same mixed heat+electricity bucket as Geothermal, per the
      same data catalog. Same logic for 'waste': type 21 (WastePower, in
      the electricity category) not type 11 (Waste, in the mixed bucket).

    - Two more types are added below beyond the "10 atomic sources" list:
      OtherPower (26) and WKK Total (35), ned.nl's electricity-production
      catalog's remaining two categories. Both are folded into a column
      literally named 'Other', which matches the 'Other' keyword already in
      config.py's SMALL_NON_RENEWABLES -- so they sum into all-non-
      renewables the same way ENTSO-E's small leftover categories do for
      every other country, keeping NL's Total Generation denominator
      consistent with the rest of the dashboard.
      OPEN QUESTION, not yet resolved: WKK (combined heat-and-power) is
      described by ned.nl's own page as electricity output specifically
      (confirmed -- not a heat number, unlike Geothermal), but the
      Netherlands' WKK installations can run on natural gas, biogas, or
      biomass. We're bucketing 100% of it as non-renewable, which likely
      overstates non-renewables slightly for whatever fraction is biogas/
      biomass-fired. No ENTSO-E equivalent exists to cross-check against
      (ENTSO-E attributes CHP output to underlying fuel type, not as its
      own category), so this can't be resolved by comparison the way the
      Geothermal question was. Revisit if the ElectricityMix cross-check
      or later scrutiny suggests it matters enough to split.
      Also worth confirming empirically that FossilGasPower (18,
      "gascentrales" / utility-scale gas plants) and WKK Total (35,
      decentralized CHP) are genuinely separate installation classes, not
      double-counted -- ned.nl's own descriptions suggest they're distinct
      ("gascentrales" vs "warmtekrachtkoppeling"), but this is inference
      from category names, same kind of assumption that turned out wrong
      for Geothermal, so worth treating as provisional.

Cross-check: both fetch functions also fetch type 27 (ElectricityMix),
ned.nl's own computed total electricity production for NL, and log a
comparison against the sum of the 10 mapped columns above. This isn't
included as a DataFrame column (it would double-count against the per-
source columns in any downstream sum) -- it's purely an informational
sanity check.

Pagination: ned.nl runs on API Platform (confirmed via its response shape),
which paginates collections by default -- a request spanning enough time
returns only the first page under a 'hydra:member' key, with further pages
reachable via 'hydra:view' -> 'hydra:next'. This was discovered the hard
way: a single-day test (under one page) looked fine, but a ~7-month fetch
silently returned only its first page (144 rows instead of ~19,000) with
no error, just badly undercounted totals. _fetch_ned_type follows
'hydra:next' in a loop until it runs out, logging a warning if
'hydra:totalItems' (when present) doesn't match what was actually
collected.
IMPORTANT: this pagination metadata only appears when the request's Accept
header is 'application/ld+json', matching ned.nl's own documented API
examples. An earlier version of this code used 'application/json' instead,
which returns a flat, unwrapped array with no 'hydra:view'/'hydra:next' at
all -- so a first attempt at pagination-following still silently capped at
one page (its "no wrapper found" fallback path assumed no more pages
existed, when in fact the wrapper just wasn't being requested). Confirmed
fixed by switching the Accept header; do not change it back to plain
'application/json' without re-verifying pagination still works (the 50-day
test in smoke_test_ned_client.py's Test 2 is specifically designed to catch
this regression).

Why two granularities (fetch_ned_generation vs fetch_ned_generation_monthly):
Fetching a full year at 15-minute resolution is ~19,000 rows per type,
which at ned.nl's page size means roughly 130+ paginated requests PER TYPE,
so ~1,300+ requests for one year across all 10 types -- and a multi-year
backfill would run into the tens of thousands of requests, taking hours
even paced safely under the 200-req/5-min limit. eu_energy_data_collection.py
only ever needs MONTHLY totals, so fetch_ned_generation_monthly requests
ned.nl's own monthly granularity (7) directly -- about 12 rows per type per
year, no pagination concerns at all. This mirrors how ned.nl's 'capacity'
field is documented as the average power over the selected granularity
window, so a monthly average MW figure times that month's duration (which
the caller's existing time_diffs-based energy conversion already computes
generically) correctly yields that month's total energy -- same principle
the existing code already relies on for ENTSO-E's variable-resolution data.
intraday_analysis.py's fetch_ned_generation keeps 15-minute resolution
since it genuinely needs it, but its date ranges are short (a single day or
a 7-day window), so pagination there means at most a handful of pages, not
hundreds.

Rate limit: ned.nl allows 200 requests per 5 minutes. Both fetch functions
cache their result per (start_date, end_date, granularity) so repeated
calls for the same period -- which happen once per atomic source in
intraday_analysis.py's collection loop -- don't refetch from the API.
"""

import os
import time
import requests
import pandas as pd

NED_BASE_URL = 'https://api.ned.nl/v1/utilizations'
NED_DOMAIN = 'https://api.ned.nl'

# Internal column name (matches SOURCE_KEYWORDS substrings) -> ned.nl type code
# All 8 confirmed to sit in ned.nl's "Elektriciteitsproductie" category --
# see the module docstring for why Geothermal is deliberately excluded.
NED_TYPE_MAP = {
    'Solar':            2,
    'Wind Onshore':     1,
    'Wind Offshore':    51,   # WindOffshoreC -- current model; not 17 or 22
    'Fossil Gas':       18,   # FossilGasPower
    'Fossil Hard coal': 19,   # FossilHardCoal
    'Nuclear':          20,
    'Biomass':          25,   # BiomassPower
    'Waste':            21,   # WastePower
}

# Small leftover categories, folded into config.py's 'Other' catch-all
# keyword (SMALL_NON_RENEWABLES) so NL's Total Generation denominator stays
# consistent with how every ENTSO-E country's small categories are summed.
# See module docstring for the open question on WKK's renewable/non-
# renewable split, and the double-counting check against Fossil Gas.
NED_OTHER_TYPES = {
    'Other':      26,  # OtherPower
    'Other (WKK)': 35,  # WKK Total -- distinct installations from FossilGasPower per
                         # ned.nl's own category descriptions, but see docstring caveat
}

# ned.nl's own computed total electricity production for NL -- used only
# as an informational cross-check against the sum of NED_TYPE_MAP columns,
# not included as a DataFrame column (see module docstring).
NED_ELECTRICITY_MIX_TYPE = 27

# ned.nl granularity codes (see module docstring for why two are used)
GRANULARITY_15MIN = 4
GRANULARITY_MONTH = 7

MAX_RETRIES = 4
REQUEST_SLEEP_SECONDS = 1.5  # keep well under 200 req / 5 min
MAX_PAGES = 1000  # safety cap against runaway pagination loops

# Module-level caches: {(start_date, end_date): DataFrame}, separate per
# granularity so a 15-min fetch and a monthly fetch for the same nominal
# date range never collide.
# Cleared implicitly each process run (intraday_analysis.py / eu_energy_data_collection.py
# are both run-once-per-invocation scripts, so no explicit eviction is needed).
_period_cache_15min = {}
_period_cache_monthly = {}


def _get_api_key():
    key = os.environ.get('NED_API_KEY', '')
    if not key:
        raise RuntimeError("NED_API_KEY environment variable is not set")
    return key


def _get_json_with_retries(url, params, headers):
    """
    GET a single page and return parsed JSON, or None after MAX_RETRIES
    failures. Handles 429 (rate limit) with extra backoff.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code == 429:
                # Rate limited -- back off harder than a normal retry
                time.sleep(5 * (attempt + 1))
                continue

            if response.status_code != 200:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                print(f"    ✗ ned.nl request failed: HTTP {response.status_code}: {response.text[:200]}")
                return None

            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                print(f"    ✗ ned.nl request failed: {e}")
                return None

    return None


def _fetch_ned_type(type_code, date_from, date_to, api_key, granularity=GRANULARITY_15MIN):
    """
    Fetch one ned.nl type for NL over [date_from, date_to) at the given
    granularity, following Hydra pagination (hydra:view -> hydra:next)
    until every page is retrieved.

    date_from / date_to: 'YYYY-MM-DD' strings (validfrom is exclusive at date_to).
    Returns a pandas Series indexed by Europe/Brussels timestamps, values in MW.
    Returns an empty Series on failure -- callers should treat that like
    ENTSO-E returning no data for a source (i.e. it contributes 0, not None).
    """
    headers = {
        'X-AUTH-TOKEN': api_key,
        'accept': 'application/ld+json',
    }
    params = {
        'point': 0,                  # Netherlands
        'type': type_code,
        'granularity': granularity,
        'granularitytimezone': 1,    # CET
        'classification': 2,         # current (not forecast)
        'activity': 1,               # providing (generation, not consumption)
        'validfrom[after]': date_from,
        'validfrom[strictly_before]': date_to,
    }

    all_records = []
    reported_total = None
    url = NED_BASE_URL
    next_params = params
    page_count = 0

    while url is not None:
        page_count += 1
        if page_count > MAX_PAGES:
            print(f"    ⚠ ned.nl type {type_code}: hit MAX_PAGES ({MAX_PAGES}) safety "
                  f"limit, stopping pagination early -- data will be incomplete")
            break

        data = _get_json_with_retries(url, next_params, headers)
        if data is None:
            # Failed after retries -- return whatever we've accumulated so far
            # rather than discarding partial results, but this page's data is lost.
            break

        if isinstance(data, dict):
            page_records = data.get('hydra:member', None)
            if page_records is None:
                # Not a Hydra collection shape -- treat as a plain list if that's
                # what came back, otherwise no records on this page.
                page_records = data if isinstance(data, list) else []

            if reported_total is None:
                reported_total = data.get('hydra:totalItems')

            view = data.get('hydra:view')
            next_path = view.get('hydra:next') if isinstance(view, dict) else None
        else:
            page_records = data if isinstance(data, list) else []
            next_path = None

        all_records.extend(page_records)

        if next_path:
            url = NED_DOMAIN + next_path
            next_params = None  # next_path already has the full query string
            time.sleep(REQUEST_SLEEP_SECONDS)
        else:
            url = None

    if reported_total is not None and len(all_records) < reported_total:
        print(f"    ⚠ ned.nl type {type_code}: got {len(all_records)} of {reported_total} "
              f"reported records across {page_count} page(s) -- data may be incomplete")

    if not all_records:
        return pd.Series(dtype=float)

    rows = {}
    for r in all_records:
        ts = pd.Timestamp(r['validfrom']).tz_convert('Europe/Brussels')
        capacity_kw = r.get('capacity', 0) or 0
        rows[ts] = capacity_kw / 1000.0  # kW -> MW

    series = pd.Series(rows).sort_index()
    time.sleep(REQUEST_SLEEP_SECONDS)
    return series


def _fetch_ned_generation_impl(start_date, end_date, granularity, cache):
    """
    Shared implementation for fetch_ned_generation and
    fetch_ned_generation_monthly -- same logic, different granularity and
    cache dict so the two never collide.
    """
    start_key = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end_key = pd.Timestamp(end_date).strftime('%Y-%m-%d')
    cache_key = (start_key, end_key)

    if cache_key in cache:
        return cache[cache_key]

    api_key = _get_api_key()

    all_types = {**NED_TYPE_MAP, **NED_OTHER_TYPES}
    granularity_label = 'monthly' if granularity == GRANULARITY_MONTH else '15-min'
    print(f"  Fetching ned.nl data for NL ({granularity_label}): {start_key} to {end_key} "
          f"({len(all_types)} types)...")

    columns = {}
    for column_name, type_code in all_types.items():
        series = _fetch_ned_type(type_code, start_key, end_key, api_key, granularity=granularity)
        if not series.empty:
            columns[column_name] = series
        else:
            print(f"    ⚠ No ned.nl data for {column_name} (type {type_code})")

    if not columns:
        df = pd.DataFrame()
    else:
        df = pd.concat(columns, axis=1)

    # Informational cross-check only -- not merged into df, see docstring.
    mix_series = _fetch_ned_type(NED_ELECTRICITY_MIX_TYPE, start_key, end_key, api_key, granularity=granularity)
    if not df.empty and not mix_series.empty:
        our_sum = df.sum(axis=1)
        aligned_mix = mix_series.reindex(our_sum.index)
        diff = (our_sum - aligned_mix).dropna()
        if not diff.empty:
            mean_our = our_sum.mean()
            mean_mix = aligned_mix.mean()
            pct_gap = (mean_our - mean_mix) / mean_mix * 100 if mean_mix else float('nan')
            print(f"    ℹ ElectricityMix cross-check: our sum mean={mean_our:.0f} MW, "
                  f"ned.nl ElectricityMix mean={mean_mix:.0f} MW ({pct_gap:+.1f}% gap)")

    cache[cache_key] = df
    return df


def fetch_ned_generation(start_date, end_date):
    """
    Fetch all mapped ned.nl generation types for NL at 15-MINUTE resolution
    over [start_date, end_date). Used by intraday_analysis.py, which
    genuinely needs sub-hourly resolution and only ever requests short
    (single-day or ~7-day) windows -- see module docstring for why this
    granularity choice doesn't have the pagination-volume problem that
    full-year fetches would.

    start_date / end_date: accepts 'YYYY-MM-DD' strings or anything
    pd.Timestamp(...).strftime('%Y-%m-%d') can normalize (datetime, date,
    pd.Timestamp), so callers can pass the same values they'd pass to
    client.query_generation() without extra conversion.

    Results are cached per (start_date, end_date) pair for the lifetime of
    the process, so calling this repeatedly for the same period (once per
    atomic source, as intraday_analysis.py does) triggers only one round of
    fetches, not one round per call.

    Returns a DataFrame shaped like ENTSO-E's client.query_generation()
    output. Returns an empty DataFrame if nothing could be fetched.
    """
    return _fetch_ned_generation_impl(start_date, end_date, GRANULARITY_15MIN, _period_cache_15min)


def fetch_ned_generation_monthly(start_date, end_date):
    """
    Fetch all mapped ned.nl generation types for NL at MONTHLY resolution
    over [start_date, end_date). Used by eu_energy_data_collection.py,
    which only ever needs monthly totals -- see module docstring for why
    this avoids the pagination-volume problem a 15-minute full-year fetch
    would hit (~12 rows per type per year instead of ~19,000).

    Same argument conventions and caching behavior as fetch_ned_generation.
    """
    return _fetch_ned_generation_impl(start_date, end_date, GRANULARITY_MONTH, _period_cache_monthly)
