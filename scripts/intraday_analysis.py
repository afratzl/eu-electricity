#!/usr/bin/env python3
"""
Unified Intraday Energy Analysis Script - REFACTORED
Architecture:
  Phase 1: Data Collection - Fetch all atomic sources + aggregates
  Phase 2: Projection & Correction - Apply component-level corrections
  Phase 3: Plot Generation - Create visualizations from corrected data
  Phase 4: Summary Table Update - Update Google Sheets with yesterday/last week data

Key improvements:
- Weekly hourly averages for projection (not daily)
- Component-level aggregate correction
- Proper Total Generation correction using all sources
- Debug output for threshold violations
- Google Sheets integration for summary table
"""

from entsoe import EntsoePandasClient
import entsoe.entsoe
import entsoe.parsers

# Source colors from monthly/trends script (lowercase keys to match intraday)
ENTSOE_COLORS = {
    # Renewables
    'solar': '#FFD700',  # Gold
    'wind': '#228B22',  # Forest Green
    'wind-onshore': '#2E8B57',  # Sea Green
    'wind-offshore': '#008B8B',  # Dark Cyan
    'hydro': '#1E90FF',  # Dodger Blue
    'biomass': '#9ACD32',  # Yellow Green
    'geothermal': '#708090',  # Slate Gray

    # Non-renewables
    'gas': '#FF1493',  # Deep Pink
    'coal': '#8B008B',  # Dark Magenta
    'nuclear': '#8B4513',  # Saddle Brown
    'oil': '#191970',  # Midnight Blue
    'waste': '#808000',  # Olive

    # Totals
    'all-renewables': '#32CD32',  # Lime Green
    'all-non-renewables': '#000000',  # Black
    'total-generation': '#000000',  # Black
}

# CRITICAL: Set new API endpoint (ENTSO-E migration November 2024)
# See: https://github.com/EnergieID/entsoe-py/issues/154
entsoe.entsoe.URL = 'https://external-api.tp.entsoe.eu/api'

# Custom parser to handle new XML format from ENTSO-E API
def _parse_load_timeseries(soup):
    """
    Custom parser for ENTSO-E API load timeseries
    Handles the new XML format after November 2024 API migration
    """
    import pandas as pd
    
    positions = []
    prices = []
    for point in soup.find_all('point'):
        positions.append(int(point.find('position').text))
        prices.append(float(point.find('quantity').text))

    series = pd.Series(index=positions, data=prices)
    series = series.sort_index()

    series.index = [v for i, v in enumerate(entsoe.parsers._parse_datetimeindex(soup)) if i+1 in series.index]

    return series

# Monkey-patch the parser into entsoe module
entsoe.parsers._parse_load_timeseries = _parse_load_timeseries

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import calendar
import warnings
import os
import sys
import argparse
import time
import json
import random
from matplotlib.patches import Rectangle

# Google Drive imports (for plot hosting)
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    print("⚠ Google Drive API not available - plots will not be uploaded to Drive")

warnings.filterwarnings('ignore')

# Force unbuffered output for real-time progress display
import functools
print = functools.partial(print, flush=True)

# Google Sheets imports (lazy loaded to avoid errors if not installed)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("⚠ gspread not available - Google Sheets update will be skipped")

# Create plots directory
os.makedirs('plots', exist_ok=True)

# EU country codes
EU_COUNTRIES = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
    'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]

# Atomic sources (cannot be broken down further)
ATOMIC_SOURCES = ['solar', 'wind', 'hydro', 'biomass', 'geothermal', 
                  'gas', 'coal', 'nuclear', 'oil', 'waste']

# Aggregate sources
AGGREGATE_SOURCES = ['all-renewables', 'all-non-renewables']

# Aggregate definitions
AGGREGATE_DEFINITIONS = {
    'all-renewables': ['solar', 'wind', 'hydro', 'biomass', 'geothermal'],
    'all-non-renewables': ['gas', 'coal', 'nuclear', 'oil', 'waste']
}

# Energy source keyword mapping
SOURCE_KEYWORDS = {
    'solar': ['Solar'],
    'wind': ['Wind Onshore', 'Wind Offshore'],
    'hydro': ['Hydro', 'Hydro Water Reservoir', 'Hydro Run-of-river', 'Hydro Pumped Storage',
              'Water Reservoir', 'Run-of-river', 'Poundage', 'Hydro Run-of-river and poundage'],
    'biomass': ['Biomass', 'Biogas', 'Biofuel'],
    'geothermal': ['Geothermal'],
    'gas': ['Fossil Gas', 'Natural Gas', 'Gas', 'Fossil Coal-derived gas'],
    'coal': ['Fossil Hard coal', 'Fossil Brown coal', 'Fossil Brown coal/Lignite', 
             'Hard Coal', 'Brown Coal', 'Coal', 'Lignite', 'Fossil Peat', 'Peat'],
    'nuclear': ['Nuclear'],
    'oil': ['Fossil Oil', 'Oil', 'Petroleum'],
    'waste': ['Waste', 'Other non-renewable', 'Other'],
    'all-renewables': ['Solar', 'Wind Onshore', 'Wind Offshore',
                       'Hydro', 'Hydro Water Reservoir', 'Hydro Run-of-river', 'Hydro Pumped Storage',
                       'Water Reservoir', 'Run-of-river', 'Poundage', 'Hydro Run-of-river and poundage',
                       'Geothermal', 'Biomass', 'Biogas', 'Biofuel', 'Other renewable'],
    'all-non-renewables': ['Fossil Gas', 'Natural Gas', 'Gas', 'Fossil Coal-derived gas',
                           'Fossil Hard coal', 'Fossil Brown coal', 'Fossil Brown coal/Lignite',
                           'Hard Coal', 'Brown Coal', 'Coal', 'Lignite', 'Fossil Peat', 'Peat',
                           'Nuclear', 'Fossil Oil', 'Oil', 'Petroleum',
                           'Waste', 'Other non-renewable', 'Other']
}

# Display names
DISPLAY_NAMES = {
    'solar': 'Solar',
    'wind': 'Wind',
    'hydro': 'Hydro',
    'biomass': 'Biomass',
    'geothermal': 'Geothermal',
    'gas': 'Gas',
    'coal': 'Coal',
    'nuclear': 'Nuclear',
    'oil': 'Oil',
    'waste': 'Waste',
    'all-renewables': 'All Renewables',
    'all-non-renewables': 'All Non-Renewables'
}


def format_change_percentage(value):
    """
    Format change percentage with smart decimal handling
    - If |value| >= 10: No decimals (e.g., "+180%", "-58%")
    - If |value| < 10: One decimal (e.g., "+5.8%", "+0.3%", "-2.1%")
    """
    if abs(value) >= 10:
        return f"{value:+.0f}%"  # + sign for positive, - for negative
    else:
        return f"{value:+.1f}%"


def get_intraday_data_for_country(country, start_date, end_date, client, data_type='generation', max_retries=3):
    """
    Get intraday data for a specific country and date range with retry logic
    """
    start = pd.Timestamp(start_date, tz='Europe/Brussels')
    end = pd.Timestamp(end_date, tz='Europe/Brussels') + timedelta(hours=1)

    for attempt in range(max_retries):
        try:
            if data_type == 'generation':
                data = client.query_generation(country, start=start, end=end)
            elif data_type == 'load':
                data = client.query_load(country, start=start, end=end)
            else:
                return pd.DataFrame()

            if data.empty:
                return pd.DataFrame()

            # Convert to Brussels timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert('Europe/Brussels')
            elif str(data.index.tz) != 'Europe/Brussels':
                data.index = data.index.tz_convert('Europe/Brussels')

            time.sleep(0.1)
            return data

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 0.1
                time.sleep(wait_time)
            else:
                time.sleep(0.1)
                return pd.DataFrame()

    return pd.DataFrame()


def extract_source_from_generation_data(generation_data, source_keywords):
    """
    Extract energy source data
    """
    relevant_columns = []
    for keyword in source_keywords:
        matching_cols = [col for col in generation_data.columns if keyword in col]
        relevant_columns.extend(matching_cols)
    relevant_columns = list(set(relevant_columns))

    if relevant_columns:
        if len(relevant_columns) == 1:
            energy_series = generation_data[relevant_columns[0]]
        else:
            energy_series = generation_data[relevant_columns].sum(axis=1)
        return energy_series, relevant_columns
    else:
        return pd.Series(0, index=generation_data.index), []


def interpolate_country_data(country_series, country_name, mark_extrapolated=False):
    """
    Interpolate to 15-minute resolution
    """
    if len(country_series) == 0:
        return None

    time_diffs = country_series.index.to_series().diff().dt.total_seconds() / 60
    most_common_interval = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else 15

    start_time = country_series.index.min().floor('15T')
    end_time = country_series.index.max().ceil('15T')
    complete_index = pd.date_range(start_time, end_time, freq='15T')

    last_actual_time = country_series.index.max() if mark_extrapolated else None

    if most_common_interval >= 45:  # Hourly
        interpolated = country_series.reindex(complete_index)
        
        # Try cubic interpolation, fall back to linear if not enough points
        try:
            interpolated = interpolated.interpolate(method='cubic', limit_area='inside')
        except ValueError as e:
            # Cubic needs at least 4 points; fall back to linear for sparse data
            if "derivatives at boundaries" in str(e):
                interpolated = interpolated.interpolate(method='linear', limit_area='inside')
            else:
                raise
        
        interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')

        if mark_extrapolated:
            mask = complete_index > last_actual_time
            interpolated.loc[mask] = np.nan
    else:  # Already 15-min
        interpolated = country_series.reindex(complete_index)
        interpolated = interpolated.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

        if mark_extrapolated:
            mask = complete_index > last_actual_time
            interpolated.loc[mask] = np.nan

    return interpolated


def aggregate_eu_data(countries, start_date, end_date, client, source_keywords, data_type='generation', mark_extrapolated=False):
    """
    Aggregate energy data across EU countries
    Returns: (eu_total, country_data_df, successful_countries)
    """
    all_interpolated_data = []
    successful_countries = []

    for country in countries:
        country_data = get_intraday_data_for_country(country, start_date, end_date, client, data_type)

        if not country_data.empty:
            if data_type == 'generation':
                country_energy, energy_columns = extract_source_from_generation_data(country_data, source_keywords)

                if energy_columns:
                    country_energy.name = country
                    interpolated = interpolate_country_data(country_energy, country, mark_extrapolated=mark_extrapolated)

                    if interpolated is not None:
                        all_interpolated_data.append(interpolated)
                        successful_countries.append(country)

    if not all_interpolated_data:
        return pd.Series(dtype=float), pd.DataFrame(), []

    combined_df = pd.concat(all_interpolated_data, axis=1)
    eu_total = combined_df.sum(axis=1, skipna=True)

    return eu_total, combined_df, successful_countries


# ============================================================================
# PHASE 1: DATA COLLECTION
# ============================================================================

def collect_all_data(api_key):
    """
    Phase 1: Collect ALL data for all atomic sources, aggregates, and total generation
    Returns a structured data object with everything we need
    """
    client = EntsoePandasClient(api_key=api_key)
    
    print("=" * 80)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 80)
    
    # Cache fetch time at start for consistent cutoff across all sources
    fetch_time = pd.Timestamp.now(tz='Europe/Brussels')
    print(f"🕐 Reference fetch time: {fetch_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Define periods
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    week_ago_end = yesterday
    week_ago_start = week_ago_end - timedelta(days=7)
    year_ago_end = datetime(today.year - 1, yesterday.month, yesterday.day)
    year_ago_start = year_ago_end - timedelta(days=7)
    two_years_ago_end = datetime(today.year - 2, yesterday.month, yesterday.day)
    two_years_ago_start = two_years_ago_end - timedelta(days=7)
    
    periods = {
        'today': (today, today + timedelta(days=1)),
        'yesterday': (yesterday, yesterday + timedelta(days=1)),
        'week_ago': (week_ago_start, week_ago_end),
        'year_ago': (year_ago_start, year_ago_end),
        'two_years_ago': (two_years_ago_start, two_years_ago_end)
    }
    
    # Data storage
    data_matrix = {
        'atomic_sources': {},  # source -> period -> country_df
        'aggregates': {},      # source -> period -> eu_total_series
        'total_generation': {} # period -> country_df
    }
    
    # Fetch atomic sources (with country breakdown)
    print("\n📊 Fetching 10 Atomic Sources (with country data)...")
    for source in ATOMIC_SOURCES:
        print(f"\n  {DISPLAY_NAMES[source]}:")
        data_matrix['atomic_sources'][source] = {}
        
        for period_name, (start_date, end_date) in periods.items():
            mark_extrap = (period_name in ['today', 'yesterday'])
            
            eu_total, country_df, countries = aggregate_eu_data(
                EU_COUNTRIES, start_date, end_date, client,
                SOURCE_KEYWORDS[source], 'generation', mark_extrapolated=mark_extrap
            )
            
            if not country_df.empty:
                data_matrix['atomic_sources'][source][period_name] = country_df
                print(f"    {period_name}: ✓ {len(countries)} countries, {len(country_df)} timestamps")
            else:
                print(f"    {period_name}: ✗ No data")
    
    # Fetch aggregates (EU totals only, no country breakdown needed)
    print("\n📊 Fetching 2 Aggregate Sources (EU totals only)...")
    all_gen_keywords = SOURCE_KEYWORDS['all-renewables'] + SOURCE_KEYWORDS['all-non-renewables']
    
    for source in AGGREGATE_SOURCES:
        print(f"\n  {DISPLAY_NAMES[source]}:")
        data_matrix['aggregates'][source] = {}
        
        for period_name, (start_date, end_date) in periods.items():
            mark_extrap = (period_name in ['today', 'yesterday'])
            
            eu_total, _, countries = aggregate_eu_data(
                EU_COUNTRIES, start_date, end_date, client,
                SOURCE_KEYWORDS[source], 'generation', mark_extrapolated=mark_extrap
            )
            
            if not eu_total.empty:
                data_matrix['aggregates'][source][period_name] = eu_total
                print(f"    {period_name}: ✓ {len(eu_total)} timestamps")
            else:
                print(f"    {period_name}: ✗ No data")
    
    # Fetch Total Generation (with country breakdown for denominator correction)
    print("\n📊 Fetching Total Generation (with country data)...")
    for period_name, (start_date, end_date) in periods.items():
        mark_extrap = (period_name in ['today', 'yesterday'])
        
        eu_total, country_df, countries = aggregate_eu_data(
            EU_COUNTRIES, start_date, end_date, client,
            all_gen_keywords, 'generation', mark_extrapolated=mark_extrap
        )
        
        if not country_df.empty:
            data_matrix['total_generation'][period_name] = country_df
            print(f"  {period_name}: ✓ {len(countries)} countries, {len(country_df)} timestamps")
        else:
            print(f"  {period_name}: ✗ No data")
    
    print("\n✓ Data collection complete!")
    return data_matrix, periods, fetch_time


def extract_country_from_raw_data(raw_data_matrix, country_code):
    """
    Extract or aggregate data for a specific country from raw multi-country data
    
    Args:
        raw_data_matrix: Full data matrix with all 27 countries
        country_code: 'EU' for aggregate, 'DE' for Germany, etc.
    
    Returns:
        data_matrix: Same structure but with country-specific data
    """
    print(f"\n📊 Extracting data for: {country_code}")
    
    extracted_data = {
        'atomic_sources': {},
        'aggregates': {},
        'total_generation': {}
    }
    
    # Extract atomic sources
    for source in ATOMIC_SOURCES:
        if source not in raw_data_matrix['atomic_sources']:
            continue
        
        extracted_data['atomic_sources'][source] = {}
        
        for period_name, country_df in raw_data_matrix['atomic_sources'][source].items():
            if country_df is None or country_df.empty:
                continue
            
            if country_code == 'EU':
                # Keep all 27 countries - threshold detection needs individual country data
                # Summing happens AFTER correction in apply_corrections_for_period
                extracted_data['atomic_sources'][source][period_name] = country_df.copy()
            else:
                # Extract single country - keep as DataFrame with country column
                if country_code in country_df.columns:
                    extracted_df = country_df[[country_code]].copy()
                    extracted_data['atomic_sources'][source][period_name] = extracted_df
                else:
                    # Country doesn't have this source - create zero-filled DataFrame
                    # This ensures plots show 0% rather than failing
                    zero_df = pd.DataFrame({
                        country_code: pd.Series(0.0, index=country_df.index)
                    })
                    extracted_data['atomic_sources'][source][period_name] = zero_df
    
    # Extract aggregates (already EU totals in raw data, but handle country-specific)
    for agg_source in AGGREGATE_SOURCES:
        if agg_source not in raw_data_matrix['aggregates']:
            continue
        
        extracted_data['aggregates'][agg_source] = {}
        
        for period_name, eu_series in raw_data_matrix['aggregates'][agg_source].items():
            if eu_series is None or eu_series.empty:
                continue
            
            if country_code == 'EU':
                # Already EU aggregate - keep as Series
                extracted_data['aggregates'][agg_source][period_name] = eu_series
            else:
                # Build aggregate from atomic sources for this country
                components = AGGREGATE_DEFINITIONS[agg_source]
                summed_series = None
                
                for component in components:
                    if component in extracted_data['atomic_sources']:
                        if period_name in extracted_data['atomic_sources'][component]:
                            # Each is a DataFrame with single country column - extract as Series
                            component_series = extracted_data['atomic_sources'][component][period_name].iloc[:, 0]
                            if summed_series is None:
                                summed_series = component_series
                            else:
                                summed_series = summed_series + component_series
                
                if summed_series is not None:
                    extracted_data['aggregates'][agg_source][period_name] = summed_series
    
    # Extract total generation
    for period_name, country_df in raw_data_matrix['total_generation'].items():
        if country_df is None or country_df.empty:
            continue
        
        if country_code == 'EU':
            # Keep all 27 countries - threshold detection needs individual country data
            # Summing happens AFTER correction in apply_corrections_for_period
            extracted_data['total_generation'][period_name] = country_df.copy()
        else:
            # Extract single country - keep as DataFrame with country column
            if country_code in country_df.columns:
                extracted_df = country_df[[country_code]].copy()
                extracted_data['total_generation'][period_name] = extracted_df
            else:
                # Country doesn't have total generation data - create zero-filled DataFrame
                zero_df = pd.DataFrame({
                    country_code: pd.Series(0.0, index=country_df.index)
                })
                extracted_data['total_generation'][period_name] = zero_df
    
    print(f"✓ Extracted data for {country_code}")
    return extracted_data


# ============================================================================
# PHASE 2: PROJECTION & CORRECTION
# ============================================================================

def apply_projections_and_corrections(data_matrix):
    """
    Phase 2: Apply 10% threshold and correct aggregates/total_gen using atomic sources
    Uses weekly hourly averages (e.g., average of all 15:00 times from past week)
    Returns BOTH actual (uncorrected) and projected (corrected) versions for today/yesterday
    """
    print("\n" + "=" * 80)
    print("PHASE 2: PROJECTION & CORRECTION")
    print("=" * 80)
    
    corrected_data = {}
    
    # Process TODAY
    if 'today' in data_matrix['total_generation'] and 'week_ago' in data_matrix['total_generation']:
        print("\n🔧 Processing TODAY...")
        result = apply_corrections_for_period(data_matrix, 'today', 'week_ago')
        
        # Store both actual and corrected
        corrected_data['today'] = result['actual']  # Actual (solid line)
        corrected_data['today_projected'] = result['corrected']  # Projected (dashed line)
    
    # Process YESTERDAY
    if 'yesterday' in data_matrix['total_generation'] and 'week_ago' in data_matrix['total_generation']:
        print("\n🔧 Processing YESTERDAY...")
        result = apply_corrections_for_period(data_matrix, 'yesterday', 'week_ago')
        
        # Store both actual and corrected
        corrected_data['yesterday'] = result['actual']  # Actual (solid line)
        corrected_data['yesterday_projected'] = result['corrected']  # Projected (dashed line)
    
    # Historical periods (no projection needed)
    for period in ['week_ago', 'year_ago', 'two_years_ago']:
        if period in data_matrix['total_generation']:
            print(f"\n📋 Processing {period.upper()} (no projection)...")
            corrected_data[period] = build_period_data_no_projection(data_matrix, period)
    
    print("\n✓ Projection & correction complete!")
    return corrected_data


def apply_corrections_for_period(data_matrix, target_period, reference_period):
    """
    Apply component-level corrections for a specific period
    Projects ALL timestamps from 00:00 to fetch_time for consistency
    Uses hybrid approach for aggregates:
      - Today: safe summation (avoid race condition)
      - Yesterday+: delta correction (preserve untracked sources)
    Returns BOTH actual (uncorrected) and corrected versions
    """
    print(f"  Analyzing {target_period} against {reference_period}...")
    
    # Build weekly hourly averages for each atomic source
    weekly_hourly_avgs = {}
    for source in ATOMIC_SOURCES:
        if source in data_matrix['atomic_sources'] and reference_period in data_matrix['atomic_sources'][source]:
            ref_data = data_matrix['atomic_sources'][source][reference_period]
            
            # Add time column
            ref_data_with_time = ref_data.copy()
            ref_data_with_time['time'] = ref_data_with_time.index.strftime('%H:%M')
            
            # Group by time to get hourly averages across the week
            weekly_hourly_avgs[source] = ref_data_with_time.groupby('time').mean(numeric_only=True)
    
    # Get target period data
    target_atomic = {src: data_matrix['atomic_sources'][src].get(target_period) 
                     for src in ATOMIC_SOURCES if src in data_matrix['atomic_sources']}
    target_total_gen = data_matrix['total_generation'].get(target_period)
    
    if target_total_gen is None:
        return {}
    
    # Get timestamp range for TARGET period (one day: 96 timestamps)
    # Create proper range from 00:00 to 23:45 of target day
    # This ensures we project all time slots, even if some data is missing
    start_date = target_total_gen.index[0].normalize()  # Midnight of target day
    end_date = start_date + pd.Timedelta(days=1)
    full_timestamp_range = pd.date_range(start_date, end_date, freq='15min', inclusive='left', tz=target_total_gen.index.tz)
    
    # Build BOTH corrected and actual (uncorrected) data
    corrected_sources = {}
    actual_sources = {}
    correction_log = []
    
    for source in ATOMIC_SOURCES + AGGREGATE_SOURCES:
        corrected_sources[source] = {}
        actual_sources[source] = {}
    
    # Process ALL timestamps in full range (00:00 to 23:45)
    # This ensures consistent time coverage across all sources
    for timestamp in full_timestamp_range:
        time_str = timestamp.strftime('%H:%M')
        
        # Project atomic sources for this timestamp
        for source in ATOMIC_SOURCES:
            if source not in target_atomic or target_atomic[source] is None:
                continue
            
            # Initialize this timestamp for this source
            if timestamp not in corrected_sources[source]:
                corrected_sources[source][timestamp] = {}
            if timestamp not in actual_sources[source]:
                actual_sources[source][timestamp] = {}
            
            # Check if we have actual data for this timestamp
            if timestamp in target_atomic[source].index:
                source_row = target_atomic[source].loc[timestamp]
                
                # Get weekly hourly average for this source and time
                countries_to_check = set()
                if source in weekly_hourly_avgs and time_str in weekly_hourly_avgs[source].index:
                    # Check all countries that have weekly averages
                    countries_to_check.update(weekly_hourly_avgs[source].columns)
                # Also check countries that are present in current data
                countries_to_check.update(source_row.index)
                
                for country in countries_to_check:
                    # Get actual value if country exists in current timestamp
                    if country in source_row.index:
                        actual_val = source_row[country]
                    else:
                        # Country completely missing from this timestamp
                        actual_val = np.nan
                    
                    # Store actual (uncorrected) value
                    actual_sources[source][timestamp][country] = actual_val if not pd.isna(actual_val) else 0
                    
                    # Default: use actual value
                    corrected_val = actual_val if not pd.isna(actual_val) else 0
                    
                    # Get weekly hourly average for this source-country-time
                    if source in weekly_hourly_avgs and time_str in weekly_hourly_avgs[source].index:
                        if country in weekly_hourly_avgs[source].columns:
                            week_avg = weekly_hourly_avgs[source].loc[time_str, country]
                            
                            if not pd.isna(week_avg) and week_avg > 0:
                                threshold = 0.1 * week_avg
                                
                                # Check if below threshold
                                if pd.isna(actual_val) or actual_val < threshold:
                                    correction_log.append({
                                        'time': time_str,
                                        'source': source,
                                        'country': country,
                                        'actual': actual_val if not pd.isna(actual_val) else 0,
                                        'expected': week_avg,
                                        'threshold': threshold
                                    })
                                    corrected_val = week_avg
                    
                    # Store corrected value
                    corrected_sources[source][timestamp][country] = corrected_val
            
            else:
                # No actual data for this timestamp - use weekly average for ALL countries
                if source in weekly_hourly_avgs and time_str in weekly_hourly_avgs[source].index:
                    for country in weekly_hourly_avgs[source].columns:
                        week_avg = weekly_hourly_avgs[source].loc[time_str, country]
                        
                        if not pd.isna(week_avg) and week_avg > 0:
                            # No actual data - store 0 for actual, weekly avg for projected
                            actual_sources[source][timestamp][country] = 0
                            corrected_sources[source][timestamp][country] = week_avg
                            
                            correction_log.append({
                                'time': time_str,
                                'source': source,
                                'country': country,
                                'actual': 0,
                                'expected': week_avg,
                                'threshold': 0
                            })
    
    # Print correction log
    if correction_log:
        print(f"\n  Detected {len(correction_log)} corrections (missing data + threshold violations):")
        for log in correction_log[:20]:  # Print first 20
            print(f"    {log['time']} | {log['country']}-{log['source']}: "
                  f"{log['actual']:.1f} MW < 10% of {log['expected']:.1f} MW "
                  f"(threshold: {log['threshold']:.1f} MW) → Using {log['expected']:.1f} MW")
        if len(correction_log) > 20:
            print(f"    ... and {len(correction_log) - 20} more corrections")
    else:
        print("  ✓ No corrections needed")
    
    # Build aggregates using HYBRID APPROACH
    # TODAY: Safe summation (avoid 90-minute race condition)
    # YESTERDAY+: Delta correction (preserve untracked sources)
    for agg_source in AGGREGATE_SOURCES:
        components = AGGREGATE_DEFINITIONS[agg_source]
        measured_aggregate = data_matrix['aggregates'].get(agg_source, {}).get(target_period)
        
        if target_period == 'today':
            # SAFE SUMMATION for today (high race condition risk)
            print(f"  Using safe summation for {agg_source} (today)")
            
            for timestamp in full_timestamp_range:
                # Sum actual components
                actual_agg_value = 0
                for component in components:
                    if timestamp in actual_sources[component]:
                        actual_agg_value += sum(actual_sources[component][timestamp].values())
                actual_sources[agg_source][timestamp] = {'EU': actual_agg_value}
                
                # Sum projected components
                projected_agg_value = 0
                for component in components:
                    if timestamp in corrected_sources[component]:
                        projected_agg_value += sum(corrected_sources[component][timestamp].values())
                corrected_sources[agg_source][timestamp] = {'EU': projected_agg_value}
        
        else:
            # DELTA CORRECTION for yesterday+ (stable data, preserve untracked sources)
            print(f"  Using delta correction for {agg_source} ({target_period})")
            
            for timestamp in full_timestamp_range:
                # Actual: use measured aggregate directly
                if measured_aggregate is not None and timestamp in measured_aggregate.index:
                    actual_agg_value = measured_aggregate[timestamp]
                else:
                    actual_agg_value = 0
                actual_sources[agg_source][timestamp] = {'EU': actual_agg_value}
                
                # Projected: start with measured, add component corrections
                corrected_agg_value = actual_agg_value
                
                for component in components:
                    projected_component = 0
                    if timestamp in corrected_sources[component]:
                        projected_component = sum(corrected_sources[component][timestamp].values())
                    
                    actual_component = 0
                    if timestamp in actual_sources[component]:
                        actual_component = sum(actual_sources[component][timestamp].values())
                    
                    correction = projected_component - actual_component
                    corrected_agg_value += correction
                
                corrected_sources[agg_source][timestamp] = {'EU': corrected_agg_value}
    
    # Build corrected and actual total generation
    corrected_total_gen = {}
    actual_total_gen = {}
    for timestamp in full_timestamp_range:
        # Corrected total
        total_corrected = 0
        for source in ATOMIC_SOURCES:
            if timestamp in corrected_sources[source]:
                total_corrected += sum(corrected_sources[source][timestamp].values())
        corrected_total_gen[timestamp] = total_corrected
        
        # Actual total
        total_actual = 0
        for source in ATOMIC_SOURCES:
            if timestamp in actual_sources[source]:
                total_actual += sum(actual_sources[source][timestamp].values())
        actual_total_gen[timestamp] = total_actual
    
    # Return BOTH versions
    result = {
        'corrected': {
            'atomic_sources': corrected_sources,
            'total_generation': corrected_total_gen
        },
        'actual': {
            'atomic_sources': actual_sources,
            'total_generation': actual_total_gen
        }
    }
    
    # Add aggregates at top level for easy access
    for agg_source in AGGREGATE_SOURCES:
        result['corrected'][agg_source] = corrected_sources[agg_source]
        result['actual'][agg_source] = actual_sources[agg_source]
    
    return result


def build_period_data_no_projection(data_matrix, period):
    """
    Build period data without projection (for historical periods)
    Returns structure matching apply_corrections_for_period
    """
    atomic_sources_data = {}
    aggregate_sources_data = {}
    
    # Atomic sources
    for source in ATOMIC_SOURCES:
        if source in data_matrix['atomic_sources'] and period in data_matrix['atomic_sources'][source]:
            source_data = data_matrix['atomic_sources'][source][period]
            atomic_sources_data[source] = {}
            
            for timestamp in source_data.index:
                atomic_sources_data[source][timestamp] = {}
                for country in source_data.columns:
                    val = source_data.loc[timestamp, country]
                    atomic_sources_data[source][timestamp][country] = val if not pd.isna(val) else 0
    
    # Aggregates - build from atomic sources
    for agg_source in AGGREGATE_SOURCES:
        components = AGGREGATE_DEFINITIONS[agg_source]
        aggregate_sources_data[agg_source] = {}
        
        # Get all timestamps from any component
        all_timestamps = set()
        for component in components:
            if component in atomic_sources_data:
                all_timestamps.update(atomic_sources_data[component].keys())
        
        for timestamp in all_timestamps:
            total = 0
            for component in components:
                if component in atomic_sources_data and timestamp in atomic_sources_data[component]:
                    total += sum(atomic_sources_data[component][timestamp].values())
            aggregate_sources_data[agg_source][timestamp] = {'EU': total}
    
    # Total generation from all atomic sources
    total_generation_data = {}
    all_timestamps = set()
    for source in ATOMIC_SOURCES:
        if source in atomic_sources_data:
            all_timestamps.update(atomic_sources_data[source].keys())
    
    for timestamp in all_timestamps:
        total = 0
        for source in ATOMIC_SOURCES:
            if source in atomic_sources_data and timestamp in atomic_sources_data[source]:
                total += sum(atomic_sources_data[source][timestamp].values())
        total_generation_data[timestamp] = total
    
    # Return structure matching apply_corrections_for_period
    result = {
        'atomic_sources': atomic_sources_data,
        'total_generation': total_generation_data
    }
    
    # Add aggregates at top level for easy access
    for agg_source, agg_data in aggregate_sources_data.items():
        result[agg_source] = agg_data
    
    return result


# ============================================================================
# PHASE 3: PLOT GENERATION
# ============================================================================

def convert_corrected_data_to_plot_format(source_type, corrected_data):
    """
    Convert corrected data structure to format expected by plotting functions
    Returns: dict with period -> DataFrame mapping
    
    Now handles properly structured data with 'today', 'today_projected', etc.
    """
    plot_data = {}
    
    for period_name, period_data in corrected_data.items():
        if not period_data:
            continue
        
        # Determine if atomic or aggregate source
        if source_type in ATOMIC_SOURCES:
            if 'atomic_sources' not in period_data or source_type not in period_data['atomic_sources']:
                continue
            source_data = period_data['atomic_sources'][source_type]
        elif source_type in AGGREGATE_SOURCES:
            if source_type not in period_data:
                continue
            source_data = period_data[source_type]
        else:
            continue
        
        total_gen_data = period_data.get('total_generation', {})
        
        # Build DataFrame
        rows = []
        for timestamp in sorted(source_data.keys()):
            # Sum across countries for this source
            energy_prod = sum(source_data[timestamp].values())
            total_gen = total_gen_data.get(timestamp, energy_prod)  # Fallback if missing
            
            if total_gen > 0:
                percentage = np.clip((energy_prod / total_gen) * 100, 0, 100)
            else:
                percentage = 0
            
            rows.append({
                'timestamp': timestamp,
                'energy_production': energy_prod,
                'total_generation': total_gen,
                'energy_percentage': percentage,
                'date': timestamp.strftime('%Y-%m-%d'),
                'time': timestamp.strftime('%H:%M')
            })
        
        if rows:
            plot_data[period_name] = pd.DataFrame(rows)
    
    return plot_data


def create_time_axis():
    """
    Create time axis for 15-minute bins
    """
    times = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            times.append(f"{hour:02d}:{minute:02d}")
    return times


def calculate_daily_statistics(data_dict, fetch_time=None):
    """
    Calculate daily statistics for plotting
    """
    standard_times = create_time_axis()
    stats = {}

    for period_name, df in data_dict.items():
        if df is None or len(df) == 0:
            continue

        if period_name in ['today', 'yesterday', 'today_projected', 'yesterday_projected']:
            time_indexed = df.groupby('time')[['energy_production', 'total_generation', 'energy_percentage']].mean()

            aligned_energy = time_indexed['energy_production'].reindex(standard_times)
            aligned_percentage = time_indexed['energy_percentage'].reindex(standard_times)

            if period_name in ['today', 'today_projected']:
                # Use cached fetch_time for consistent cutoff across all sources
                if fetch_time is None:
                    current_time = pd.Timestamp.now(tz='Europe/Brussels')
                else:
                    current_time = fetch_time
                cutoff_time = current_time - timedelta(hours=2)
                cutoff_time = cutoff_time.floor('15T')

                try:
                    cutoff_time_str = cutoff_time.strftime('%H:%M')
                    cutoff_idx = standard_times.index(cutoff_time_str)
                except ValueError:
                    cutoff_idx = len([t for t in standard_times if t <= cutoff_time_str])

                # Interpolate only up to cutoff
                aligned_energy.iloc[:cutoff_idx] = aligned_energy.iloc[:cutoff_idx].interpolate()
                aligned_percentage.iloc[:cutoff_idx] = aligned_percentage.iloc[:cutoff_idx].interpolate()

                # Set future to NaN
                aligned_energy.iloc[cutoff_idx:] = np.nan
                aligned_percentage.iloc[cutoff_idx:] = np.nan
                
                # Fill past NaN
                aligned_energy.iloc[:cutoff_idx] = aligned_energy.iloc[:cutoff_idx].fillna(0.1)
                aligned_percentage.iloc[:cutoff_idx] = aligned_percentage.iloc[:cutoff_idx].fillna(0)
            else:
                aligned_energy = aligned_energy.interpolate().fillna(0.1)
                aligned_percentage = aligned_percentage.interpolate().fillna(0)

            stats[period_name] = {
                'time_bins': standard_times,
                'energy_mean': aligned_energy.values,
                'energy_std': np.zeros(len(standard_times)),
                'percentage_mean': aligned_percentage.values,
                'percentage_std': np.zeros(len(standard_times)),
            }

        else:
            # Multi-day periods
            unique_dates = df['date'].unique()
            daily_energy_data = []
            daily_percentage_data = []

            for date in unique_dates:
                day_data = df[df['date'] == date]
                if len(day_data) > 0:
                    time_indexed = day_data.set_index('time')[['energy_production', 'energy_percentage']].groupby(level=0).mean()
                    
                    aligned_energy = time_indexed['energy_production'].reindex(standard_times).interpolate().fillna(0.1)
                    aligned_percentage = time_indexed['energy_percentage'].reindex(standard_times).interpolate().fillna(0)

                    daily_energy_data.append(aligned_energy.values)
                    daily_percentage_data.append(aligned_percentage.values)

            if daily_energy_data:
                energy_array = np.array(daily_energy_data)
                percentage_array = np.array(daily_percentage_data)

                stats[period_name] = {
                    'time_bins': standard_times,
                    'energy_mean': np.nanmean(energy_array, axis=0),
                    'energy_std': np.nanstd(energy_array, axis=0),
                    'percentage_mean': np.nanmean(percentage_array, axis=0),
                    'percentage_std': np.nanstd(percentage_array, axis=0),
                }

    return stats


def plot_analysis(stats_data, source_type, output_file_base, country_code='EU'):
    """
    Create two separate plots - percentage and absolute
    Returns tuple of (percentage_file, absolute_file)
    
    Args:
        stats_data: Statistical data for plotting
        source_type: Energy source type
        output_file_base: Base filename for output
        country_code: Country code (EU, DE, MD, etc.)
    """
    if not stats_data:
        print("No data for plotting")
        return None, None
    
    # Country display names
    COUNTRY_DISPLAY_NAMES = {
        'EU': 'European Union',
        'DE': 'Germany',
        'MD': 'Moldova',
        'FR': 'France',
        'ES': 'Spain',
        'IT': 'Italy',
        'PL': 'Poland',
        'NL': 'Netherlands',
        'BE': 'Belgium',
        'AT': 'Austria',
        'SE': 'Sweden',
        'DK': 'Denmark',
        'FI': 'Finland',
        'NO': 'Norway',
        'CH': 'Switzerland',
        'GB': 'United Kingdom',
        'PT': 'Portugal',
        'GR': 'Greece',
        'CZ': 'Czechia',
        'RO': 'Romania',
        'HU': 'Hungary',
        'BG': 'Bulgaria',
        'SK': 'Slovakia',
        'SI': 'Slovenia',
        'HR': 'Croatia',
        'LT': 'Lithuania',
        'LV': 'Latvia',
        'EE': 'Estonia',
        'IE': 'Ireland',
        'LU': 'Luxembourg',
        'MT': 'Malta',
        'CY': 'Cyprus'
    }
    
    # Calculate dynamic years for legend labels
    from datetime import datetime
    current_year = datetime.now().year
    previous_year = current_year - 1
    two_years_ago = current_year - 2
    
    def draw_flag_placeholder(fig, country_code, size=100):
        """
        Draw a placeholder for flag - colored rectangle with country code
        3:2 aspect ratio (standard flag proportion)
        
        Args:
            fig: matplotlib figure
            country_code: 'EU', 'DE', 'MD', etc.
            size: pixel size (width=height for square)
        """
        # Create rectangular flag area - positioned for 12×12 canvas
        # x=0.1, y=0.85, width=0.075, height=0.05
        ax_flag = fig.add_axes([0.1, 0.85, 0.075, 0.05])
        
        # Country-specific colors (or default blue)
        colors_map = {
            'EU': '#003399',    # EU blue
            'DE': '#000000',    # Germany black
            'MD': '#0046AE',    # Moldova blue
            'FR': '#0055A4',    # France blue
            'ES': '#AA151B',    # Spain red
        }
        bg_color = colors_map.get(country_code, '#4A90E2')  # Default blue
        
        # Draw colored rectangle
        ax_flag.add_patch(plt.Rectangle((0, 0), 1, 1, 
                                         facecolor=bg_color, 
                                         edgecolor='#CCCCCC', 
                                         linewidth=2))
        
        # Add country code in white
        ax_flag.text(0.5, 0.5, country_code, 
                    color='white', 
                    fontsize=14,
                    fontweight='bold',
                    ha='center', va='center')
        
        ax_flag.set_xlim(0, 1)
        ax_flag.set_ylim(0, 1)
        ax_flag.axis('off')
        
        return ax_flag
    
    def load_flag(fig, country_code):
        """
        Load real flag PNG or fall back to placeholder
        3:2 aspect ratio (standard flag proportion)
        
        Args:
            fig: matplotlib figure
            country_code: 'EU', 'DE', 'MD', etc.
        
        Returns:
            axes object for the flag
        """
        import os
        from matplotlib import image as mpimg
        
        # Try to load PNG flag (matplotlib compatible)
        flag_path = f'flags/{country_code}.png'
        
        if os.path.exists(flag_path):
            try:
                # Create axes for flag - positioned for 12×12 canvas
                # x=0.1, y=0.85, width=0.075, height=0.05
                ax_flag = fig.add_axes([0.1, 0.85, 0.075, 0.05])
                
                # Load and display PNG
                flag_img = mpimg.imread(flag_path)
                ax_flag.imshow(flag_img, aspect='auto')
                ax_flag.axis('off')
                
                print(f"  ✓ Loaded flag: {flag_path}")
                return ax_flag
            except Exception as e:
                print(f"  ⚠ Could not load {flag_path}: {e}")
                print(f"  → Using placeholder instead")
                return draw_flag_placeholder(fig, country_code)
        else:
            print(f"  ℹ Flag PNG not found: {flag_path}")
            print(f"  → Using placeholder (run GitHub Actions to generate PNGs)")
            return draw_flag_placeholder(fig, country_code)

    colors = {
        'today': '#FF4444',
        'yesterday': '#FF8800',
        'week_ago': '#4444FF',
        'year_ago': '#44AA44',
        'two_years_ago': '#AA44AA',
        'today_projected': '#FF4444',
        'yesterday_projected': '#FF8800'
    }

    linestyles = {
        'today': '-',
        'yesterday': '-',
        'week_ago': '-',
        'year_ago': '-',
        'two_years_ago': '-',
        'today_projected': (0, (2, 2)),  # Equal: 2pt dash, 2pt gap (tighter pattern)
        'yesterday_projected': (0, (2, 2))  # Equal: 2pt dash, 2pt gap
    }

    labels = {
        'today': 'Today',
        'yesterday': 'Yesterday',
        'week_ago': 'Last Week',
        'year_ago': f'Same Week {previous_year}',
        'two_years_ago': f'Same Week {two_years_ago}',
        'today_projected': 'Today (Projected)',
        'yesterday_projected': 'Yesterday (Projected)'
    }

    time_labels = create_time_axis()
    
    # Calculate x-axis tick positions (every 4 hours) + add 24:00 at end
    tick_positions = list(range(0, len(time_labels), 16))  # Every 4 hours (16 * 15min = 4h)
    tick_positions.append(len(time_labels))  # Add position for 24:00
    
    tick_labels_axis = [time_labels[i] if i < len(time_labels) else '' for i in tick_positions[:-1]]
    tick_labels_axis.append('24:00')  # Add 24:00 label at the end
    
    source_name = DISPLAY_NAMES[source_type]
    
    # Shorten aggregate names for plot titles (but keep full names in dropdown)
    if source_name == 'All Renewables':
        source_name = 'Renewables'
    elif source_name == 'All Non-Renewables':
        source_name = 'Non-Renewables'
    
    # Order for 2-column legend:
    # Plot order: historical first (background), then today/yesterday last (foreground on top)
    # This ensures red (today) and orange (yesterday) are clearly visible
    plot_order = ['two_years_ago', 'year_ago', 'week_ago',
                  'yesterday_projected', 'yesterday', 'today_projected', 'today']
    
    # Generate output filenames
    output_file_percentage = output_file_base.replace('.png', '_percentage.png')
    output_file_absolute = output_file_base.replace('.png', '_absolute.png')
    
    # ========================================================================
    # PLOT 1: PERCENTAGE
    # ========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 12))  # Canvas: 12 wide × 12 tall
    
    # Set exact plot area positioning
    plt.subplots_adjust(left=0.22, right=0.9, top=0.80, bottom=0.35)
    
    # Add flag (top-left) - loads real SVG or uses placeholder
    load_flag(fig1, country_code)
    
    # Add country name below flag (figure coordinates)
    country_display = COUNTRY_DISPLAY_NAMES.get(country_code, country_code)
    fig1.text(0.1, 0.843, country_display,
             fontsize=18, fontweight='normal',
             ha='left', va='top',
             color='#333')
    
    # Titles in figure coordinates (not axes coordinates)
    # Main title
    fig1.text(0.55, 0.89, 'Electricity Generation',
             fontsize=30, fontweight='bold',
             ha='center', va='top')
    
    # Subtitle
    fig1.text(0.55, 0.835, f'{source_name} · Fraction of Total',
             fontsize=24, fontweight='normal',
             ha='center', va='top')
    ax1.set_xlabel('Time of Day (Brussels)', fontsize=24, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Electrical Power (%)', fontsize=24, fontweight='bold', labelpad=8)

    max_percentage = 0

    for period_name in plot_order:
        if period_name not in stats_data:
            continue
            
        data = stats_data[period_name]
        if 'percentage_mean' not in data or len(data['percentage_mean']) == 0:
            continue

        color = colors.get(period_name, 'gray')
        linestyle = linestyles.get(period_name, '-')
        label = labels.get(period_name, period_name)

        x_values = np.arange(len(data['percentage_mean']))
        y_values = data['percentage_mean'].copy()
        max_percentage = max(max_percentage, np.nanmax(y_values))

        if period_name in ['today', 'today_projected']:
            mask = ~np.isnan(y_values)
            if np.any(mask):
                x_values = x_values[mask]
                y_values = y_values[mask]
            else:
                continue

        ax1.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=6, label=label, marker='')

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'percentage_std' in data:
            std_values = data['percentage_std'][:len(x_values)]
            upper_bound = y_values + std_values
            lower_bound = y_values - std_values
            max_percentage = max(max_percentage, np.nanmax(upper_bound))
            ax1.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
    ax1.set_ylim(0, max_percentage * 1.20 if max_percentage > 0 else 50)  # 20% headroom
    
    # Set x-axis time labels
    ax1.set_xlim(0, len(time_labels))
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels_axis)
    
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    
    # Reorder legend for 3-2-2 layout:
    # Row 1: Previous Week, Last Year, Two Years Ago
    # Row 2: Yesterday, Yesterday (Projected), [empty]
    # Row 3: Today, Today (Projected), [empty]
    handles, labels_list = ax1.get_legend_handles_labels()
    legend_order = ['week_ago', 'year_ago', 'two_years_ago',
                    'today', 'yesterday',
                    'today_projected', 'yesterday_projected']
    
    # Create ordered handles/labels matching desired legend layout
    ordered_handles = []
    ordered_labels = []
    label_to_handle = dict(zip(labels_list, handles))
    
    for period in legend_order:
        period_label = labels.get(period, period)
        if period_label in label_to_handle:
            ordered_handles.append(label_to_handle[period_label])
            ordered_labels.append(period_label)
    
    ax1.legend(ordered_handles, ordered_labels, 
              loc='upper left', bbox_to_anchor=(0.14, 0.255),  # Figure coordinates
              bbox_transform=fig1.transFigure,  # Use figure coordinate system
              ncol=3, fontsize=18, frameon=False)
    
    # Add watermark (bottom-left) and timestamp (bottom-right)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    
    # Watermark (left) - aligned with plot left edge
    fig1.text(0.2, 0.125, "afratzl.github.io/eu-electricity",
              ha='left', va='top',
              fontsize=12, color='#666',
              style='italic')
    
    # Timestamp (right) - aligned with plot right edge
    fig1.text(0.9, 0.125, f"Generated: {timestamp}",
              ha='right', va='top',
              fontsize=12, color='#666',
              style='italic')
    
    plt.savefig(output_file_percentage, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved percentage plot: {output_file_percentage}")
    plt.close()
    
    # ========================================================================
    # PLOT 2: ABSOLUTE
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 12))  # Canvas: 12 wide × 12 tall
    
    # Set exact plot area positioning
    plt.subplots_adjust(left=0.22, right=0.9, top=0.80, bottom=0.35)
    
    # Add flag (top-left) - loads real SVG or uses placeholder
    load_flag(fig2, country_code)
    
    # Add country name below flag (figure coordinates)
    fig2.text(0.1, 0.843, country_display,
             fontsize=18, fontweight='normal',
             ha='left', va='top',
             color='#333')
    
    # Titles in figure coordinates (not axes coordinates)
    # Main title
    fig2.text(0.55, 0.89, 'Electricity Generation',
             fontsize=30, fontweight='bold',
             ha='center', va='top')
    
    # Subtitle
    fig2.text(0.55, 0.835, f'{source_name} · Absolute Values',
             fontsize=24, fontweight='normal',
             ha='center', va='top')
    ax2.set_xlabel('Time of Day (Brussels)', fontsize=24, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Electrical Power (GW)', fontsize=24, fontweight='bold', labelpad=8)

    max_energy = 0

    for period_name in plot_order:
        if period_name not in stats_data:
            continue
            
        data = stats_data[period_name]
        if 'energy_mean' not in data or len(data['energy_mean']) == 0:
            continue

        color = colors.get(period_name, 'gray')
        linestyle = linestyles.get(period_name, '-')
        label = labels.get(period_name, period_name)

        x_values = np.arange(len(data['energy_mean']))
        # Convert MW to GW
        y_values = data['energy_mean'].copy() / 1000
        max_energy = max(max_energy, np.nanmax(y_values))

        if period_name in ['today', 'today_projected']:
            mask = ~np.isnan(y_values)
            if np.any(mask):
                x_values = x_values[mask]
                y_values = y_values[mask]
            else:
                continue

        ax2.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=6, label=label, marker='')

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'energy_std' in data:
            # Convert MW to GW for std as well
            std_values = data['energy_std'][:len(x_values)] / 1000
            upper_bound = y_values + std_values
            lower_bound = y_values - std_values
            max_energy = max(max_energy, np.nanmax(upper_bound))
            ax2.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
    ax2.set_ylim(0, max_energy * 1.20)  # 20% headroom
    
    # Set x-axis time labels
    ax2.set_xlim(0, len(time_labels))
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels_axis)
    
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    
    # ========================================================================
    # AUTO UNIT CONVERSION: Check y-tick spacing and convert GW→MW→kW if needed
    # ========================================================================
    yticks = ax2.get_yticks()
    if len(yticks) > 1:
        tick_spacing = yticks[1] - yticks[0]
        
        # Determine appropriate unit
        if tick_spacing < 0.00001:  # Less than 0.01 MW = 10 kW
            unit_label = 'Electrical Power (kW)'
            conversion_factor = 1000000  # GW to kW
            unit_name = 'kW'
        elif tick_spacing < 0.01:  # Less than 10 MW
            unit_label = 'Electrical Power (MW)'
            conversion_factor = 1000  # GW to MW
            unit_name = 'MW'
        else:
            unit_label = None  # Keep GW
            conversion_factor = 1
            unit_name = 'GW'
        
        # If conversion needed, clear and replot
        if unit_label is not None:
            print(f"  → Converting to {unit_name} (tick spacing: {tick_spacing:.6f} GW)")
            
            # Clear the plot
            ax2.clear()
            
            # Replot all data with converted units
            max_energy_converted = 0
            for period_name in plot_order:
                if period_name not in stats_data:
                    continue
                    
                data = stats_data[period_name]
                if 'energy_mean' not in data or len(data['energy_mean']) == 0:
                    continue
            
                color = colors.get(period_name, 'gray')
                linestyle = linestyles.get(period_name, '-')
                label = labels.get(period_name, period_name)
            
                x_values = np.arange(len(data['energy_mean']))
                # Convert MW to target unit (kW or MW instead of GW)
                y_values = data['energy_mean'].copy() / 1000 * conversion_factor
                max_energy_converted = max(max_energy_converted, np.nanmax(y_values))
            
                if period_name in ['today', 'today_projected']:
                    mask = ~np.isnan(y_values)
                    if np.any(mask):
                        x_values = x_values[mask]
                        y_values = y_values[mask]
                    else:
                        continue
            
                ax2.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=6, label=label, marker='')
            
                if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'energy_std' in data:
                    # Convert std values to target unit
                    std_values = data['energy_std'][:len(x_values)] / 1000 * conversion_factor
                    upper_bound = y_values + std_values
                    lower_bound = y_values - std_values
                    max_energy_converted = max(max_energy_converted, np.nanmax(upper_bound))
                    ax2.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)
            
            # Reapply formatting
            ax2.set_xlabel('Time of Day (Brussels)', fontsize=24, fontweight='bold', labelpad=8)
            ax2.set_ylabel(unit_label, fontsize=24, fontweight='bold', labelpad=8)
            ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax2.set_ylim(0, max_energy_converted * 1.20)
            ax2.set_xlim(0, len(time_labels))
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels_axis)
            ax2.grid(True, alpha=0.3, linewidth=1.5)
    
    # Reorder legend to match percentage plot (3-2-2 layout)
    handles2, labels_list2 = ax2.get_legend_handles_labels()
    
    ordered_handles2 = []
    ordered_labels2 = []
    label_to_handle2 = dict(zip(labels_list2, handles2))
    
    for period in legend_order:
        period_label = labels.get(period, period)
        if period_label in label_to_handle2:
            ordered_handles2.append(label_to_handle2[period_label])
            ordered_labels2.append(period_label)
    
    ax2.legend(ordered_handles2, ordered_labels2,
              loc='upper left', bbox_to_anchor=(0.14, 0.255),  # Figure coordinates
              bbox_transform=fig2.transFigure,  # Use figure coordinate system
              ncol=3, fontsize=18, frameon=False)
    
    # Add watermark (bottom-left) and timestamp (bottom-right)
    # Watermark (left) - aligned with plot left edge
    fig2.text(0.2, 0.125, "afratzl.github.io/eu-electricity",
              ha='left', va='top',
              fontsize=12, color='#666',
              style='italic')
    
    # Timestamp (right) - aligned with plot right edge
    fig2.text(0.9, 0.125, f"Generated: {timestamp}",
              ha='right', va='top',
              fontsize=12, color='#666',
              style='italic')
    
    plt.savefig(output_file_absolute, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved absolute plot: {output_file_absolute}")
    plt.close()
    
    return output_file_percentage, output_file_absolute


def generate_plot_for_source(source_type, corrected_data, output_file_base, fetch_time=None, country_code='EU'):
    """
    Phase 3: Generate plot for a specific source from corrected data
    """
    print(f"\n" + "=" * 80)
    print(f"PHASE 3: PLOT GENERATION - {DISPLAY_NAMES[source_type].upper()}")
    print("=" * 80)
    
    # Convert corrected data to plot format
    plot_data = convert_corrected_data_to_plot_format(source_type, corrected_data)
    
    if not plot_data:
        print(f"✗ No data available for {source_type} (skipping)")
        return None, None
    
    # Calculate statistics (pass fetch_time for consistent cutoff)
    stats_data = calculate_daily_statistics(plot_data, fetch_time=fetch_time)
    
    # Create plots (returns percentage and absolute files)
    percentage_file, absolute_file = plot_analysis(stats_data, source_type, output_file_base, country_code=country_code)
    
    return percentage_file, absolute_file


# ==============================================================================
# PHASE 4: SUMMARY TABLE UPDATE
# ==============================================================================

def calculate_period_totals(period_data, period_name):
    """
    Calculate total production (GWh) and percentages for a period
    Returns dict: {source_name: {'gwh': value, 'percentage': value}}
    """
    if not period_data:
        return {}
    
    totals = {}
    
    # Get total generation
    total_gen_data = period_data.get('total_generation', {})
    # Convert MW to GWh: MW * 0.25 hours (15-min intervals) / 1000
    total_gen_gwh = sum(total_gen_data.values()) * 0.25 / 1000
    
    # Calculate for atomic sources
    for source in ATOMIC_SOURCES:
        if 'atomic_sources' not in period_data or source not in period_data['atomic_sources']:
            continue
        
        source_data = period_data['atomic_sources'][source]
        
        # Sum all countries, all timestamps
        source_total_mw = 0
        for timestamp, countries in source_data.items():
            source_total_mw += sum(countries.values())
        
        # Convert MW to GWh: MW * hours / 1000
        # For 15-minute intervals, each reading represents 0.25 hours
        source_gwh = source_total_mw * 0.25 / 1000
        percentage = (source_gwh / total_gen_gwh * 100) if total_gen_gwh > 0 else 0
        
        totals[source] = {
            'gwh': source_gwh,
            'percentage': percentage
        }
    
    # Calculate for aggregates
    for agg_source in AGGREGATE_SOURCES:
        if agg_source not in period_data:
            continue
        
        agg_data = period_data[agg_source]
        
        # Sum all timestamps
        agg_total_mw = 0
        for timestamp, countries in agg_data.items():
            agg_total_mw += sum(countries.values())
        
        # Convert MW to GWh: MW * 0.25 hours (15-min intervals) / 1000
        agg_gwh = agg_total_mw * 0.25 / 1000
        percentage = (agg_gwh / total_gen_gwh * 100) if total_gen_gwh > 0 else 0
        
        totals[agg_source] = {
            'gwh': agg_gwh,
            'percentage': percentage
        }
    
    return totals


def get_or_create_country_sheet(gc, drive_service, country_code='EU'):
    """
    Get or create a country-specific Google Sheet and move to correct Drive folder
    Sets permissions: Anyone with link can view
    SAVES sheet ID to drive_links.json for sharing with other scripts
    
    Args:
        gc: gspread client
        drive_service: Google Drive API service
        country_code: 'EU', 'DE', 'ES', etc.
    
    Returns:
        spreadsheet: gspread Spreadsheet object
    """
    import json
    import os
    
    sheet_name = f"{country_code} Electricity Production Data"
    spreadsheet = None
    is_new_sheet = False
    
    # Try to get from JSON first
    drive_links_file = 'plots/drive_links.json'
    if os.path.exists(drive_links_file):
        try:
            with open(drive_links_file, 'r') as f:
                links = json.load(f)
                
            # Check if we have this country's sheet ID
            if country_code in links and 'data_sheet_id' in links[country_code]:
                sheet_id = links[country_code]['data_sheet_id']
                try:
                    spreadsheet = gc.open_by_key(sheet_id)
                    print(f"✓ Opened existing sheet from JSON: {sheet_name}")
                except:
                    print(f"  ⚠ Sheet ID in JSON is invalid, will search by name")
        except:
            pass
    
    # If not found in JSON, try to open by name
    if spreadsheet is None:
        try:
            spreadsheet = gc.open(sheet_name)
            print(f"✓ Opened existing sheet by name: {sheet_name}")
        except gspread.SpreadsheetNotFound:
            # Create new sheet
            spreadsheet = gc.create(sheet_name)
            print(f"✓ Created new sheet: {sheet_name}")
            is_new_sheet = True
            
            # Set permissions: Anyone with link can view
            try:
                permission = {'type': 'anyone', 'role': 'reader'}
                drive_service.permissions().create(
                    fileId=spreadsheet.id,
                    body=permission,
                    fields='id'
                ).execute()
                print(f"  ✓ Set permissions: Anyone with link can view")
            except Exception as e:
                # Permission might already exist
                if 'already exists' not in str(e).lower():
                    print(f"  ⚠ Could not set permissions: {e}")
    
    # Move to correct Drive folder structure (ALWAYS check, not just for new sheets)
    try:
        # Find root folder
        query = "name='EU-Electricity-Plots' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        folders = results.get('files', [])
        
        if folders:
            root_folder_id = folders[0]['id']
            
            # Find or create country folder
            query = f"name='{country_code}' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
            country_folders = results.get('files', [])
            
            if country_folders:
                country_folder_id = country_folders[0]['id']
            else:
                # Create country folder
                file_metadata = {
                    'name': country_code,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [root_folder_id]
                }
                folder = drive_service.files().create(body=file_metadata, fields='id').execute()
                country_folder_id = folder.get('id')
                print(f"  ✓ Created folder: EU-Electricity-Plots/{country_code}/")
            
            # Check if sheet is already in correct folder
            file = drive_service.files().get(fileId=spreadsheet.id, fields='parents').execute()
            current_parents = file.get('parents', [])
            
            if country_folder_id not in current_parents:
                # Move sheet to country folder
                previous_parents = ",".join(current_parents)
                drive_service.files().update(
                    fileId=spreadsheet.id,
                    addParents=country_folder_id,
                    removeParents=previous_parents,
                    fields='id, parents'
                ).execute()
                print(f"  ✓ Moved to: EU-Electricity-Plots/{country_code}/")
            else:
                print(f"  ✓ Already in: EU-Electricity-Plots/{country_code}/")
            
    except Exception as e:
        print(f"  ⚠ Could not move to Drive folder: {e}")
    
    # Save sheet ID to JSON (CRITICAL - this is what was missing!)
    try:
        # Load existing links
        links = {}
        if os.path.exists(drive_links_file):
            with open(drive_links_file, 'r') as f:
                links = json.load(f)
        
        # Update with sheet ID
        if country_code not in links:
            links[country_code] = {}
        links[country_code]['data_sheet_id'] = spreadsheet.id
        
        # Save back
        os.makedirs('plots', exist_ok=True)
        with open(drive_links_file, 'w') as f:
            json.dump(links, f, indent=2)
        
        print(f"  ✓ Saved sheet ID to drive_links.json")
    except Exception as e:
        print(f"  ⚠ Could not save to JSON: {e}")
    
    return spreadsheet


def update_summary_table_worksheet(corrected_data, country_code='EU'):
    """
    Update Google Sheets "Summary Table Data" worksheet with yesterday/last week data
    Uses PROJECTED (corrected) data for accuracy
    
    FIXED: Now uses 22 columns (A-V) matching monthly/trends script
    """
    if not GSPREAD_AVAILABLE:
        print("\n⚠ Skipping Google Sheets update - gspread not available")
        return
    
    print("\n" + "=" * 80)
    print("PHASE 4: UPDATE SUMMARY TABLE (GOOGLE SHEETS)")
    print("=" * 80)
    
    try:
        # Get credentials
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            print("⚠ GOOGLE_CREDENTIALS_JSON not set - skipping Sheets update")
            return
        
        creds_dict = json.loads(google_creds_json)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        gc = gspread.authorize(credentials)
        
        # Initialize drive service for sheet organization
        from googleapiclient.discovery import build
        from google.oauth2.service_account import Credentials as ServiceAccountCredentials
        
        credentials_drive = ServiceAccountCredentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        drive_service = build('drive', 'v3', credentials=credentials_drive)
        
        # Get or create country-specific spreadsheet
        spreadsheet = get_or_create_country_sheet(gc, drive_service, country_code=country_code)
        print(f"✓ Connected to Google Sheets ({country_code}): {spreadsheet.url}")
        
        # Get current date info for dynamic headers
        current_date = datetime.now()
        current_year = current_date.year      # 2026
        previous_year = current_year - 1      # 2025
        two_years_ago = current_year - 2      # 2024
        
        # Get or create worksheet
        try:
            worksheet = spreadsheet.worksheet('Summary Table Data')
            print("✓ Found existing 'Summary Table Data' worksheet")
            
            # Check if worksheet has enough columns (need 22: A-V)
            if worksheet.col_count < 22:
                print(f"  Expanding worksheet from {worksheet.col_count} to 22 columns...")
                worksheet.resize(rows=worksheet.row_count, cols=22)
                
                # Update header row with new columns
                headers = [
                    'Source', 
                    'Yesterday_GWh', 'Yesterday_%', 
                    'LastWeek_GWh', 'LastWeek_%',
                    f'YTD{current_year}_GWh', f'YTD{current_year}_%',
                    f'{previous_year}_GWh', f'{previous_year}_%',
                    'Last_Updated',
                    'Yesterday_Change_2015_%', 'LastWeek_Change_2015_%',
                    f'YTD{current_year}_Change_2015_%', f'{previous_year}_Change_2015_%',
                    f'Yesterday_Change_{previous_year}_%', f'LastWeek_Change_{previous_year}_%',
                    f'YTD{current_year}_Change_{previous_year}_%', f'{previous_year}_Change_{previous_year}_%',
                    f'Yesterday_Change_{two_years_ago}_%', f'LastWeek_Change_{two_years_ago}_%',
                    f'YTD{current_year}_Change_{two_years_ago}_%', f'{previous_year}_Change_{two_years_ago}_%'
                ]
                worksheet.update('A1:V1', [headers])
                worksheet.format('A1:V1', {'textFormat': {'bold': True}})
                print("  ✓ Worksheet expanded and header updated")
                
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title='Summary Table Data', rows=20, cols=25)
            print("✓ Created new 'Summary Table Data' worksheet")
            
            # Add headers (22 columns A-V)
            headers = [
                'Source', 
                'Yesterday_GWh', 'Yesterday_%', 
                'LastWeek_GWh', 'LastWeek_%',
                f'YTD{current_year}_GWh', f'YTD{current_year}_%',
                f'{previous_year}_GWh', f'{previous_year}_%',
                'Last_Updated',
                'Yesterday_Change_2015_%', 'LastWeek_Change_2015_%',
                f'YTD{current_year}_Change_2015_%', f'{previous_year}_Change_2015_%',
                f'Yesterday_Change_{previous_year}_%', f'LastWeek_Change_{previous_year}_%',
                f'YTD{current_year}_Change_{previous_year}_%', f'{previous_year}_Change_{previous_year}_%',
                f'Yesterday_Change_{two_years_ago}_%', f'LastWeek_Change_{two_years_ago}_%',
                f'YTD{current_year}_Change_{two_years_ago}_%', f'{previous_year}_Change_{two_years_ago}_%'
            ]
            worksheet.update('A1:V1', [headers])
            worksheet.format('A1:V1', {'textFormat': {'bold': True}})
        
        # Calculate yesterday totals (using PROJECTED data)
        yesterday_totals = calculate_period_totals(
            corrected_data.get('yesterday_projected', {}), 
            'yesterday'
        )
        
        # Calculate last week totals (no projection needed for historical)
        week_totals = calculate_period_totals(
            corrected_data.get('week_ago', {}),
            'week_ago'
        )
        
        if not yesterday_totals or not week_totals:
            print("⚠ Insufficient data to update summary table")
            return
        
        # Define source order (needed for 2015 data loading)
        source_order = [
            'all-renewables',
            'solar', 'wind', 'hydro', 'biomass', 'geothermal',
            'all-non-renewables',
            'gas', 'coal', 'nuclear', 'oil', 'waste'
        ]
        
        # Load 2015, 2025, and 2024 data for change calculation
        print("  Loading baseline data (2015, 2025, 2024)...")
        data_2015 = {}
        data_2025 = {}
        data_2024 = {}
        
        # Get yesterday's month for baseline (e.g., if yesterday was Jan 5, use January)
        yesterday_date = datetime.now() - timedelta(days=1)
        baseline_month = yesterday_date.month  # e.g., 1 for January
        
        # Map source names to worksheet names
        source_to_worksheet = {
            'solar': 'Solar Monthly Production',
            'wind': 'Wind Monthly Production',
            'hydro': 'Hydro Monthly Production',
            'biomass': 'Biomass Monthly Production',
            'geothermal': 'Geothermal Monthly Production',
            'gas': 'Gas Monthly Production',
            'coal': 'Coal Monthly Production',
            'nuclear': 'Nuclear Monthly Production',
            'oil': 'Oil Monthly Production',
            'waste': 'Waste Monthly Production',
            'all-renewables': 'All Renewables Monthly Production',
            'all-non-renewables': None  # Calculated from Total - Renewables
        }
        
        for source in source_order:
            if source == 'all-non-renewables':
                continue  # Will calculate this separately
            
            worksheet_name = source_to_worksheet.get(source)
            if not worksheet_name:
                continue
            
            try:
                ws_baseline = spreadsheet.worksheet(worksheet_name)
                values = ws_baseline.get_all_values()
                
                if len(values) < 2:
                    continue
                
                # Parse to find baseline data
                df = pd.DataFrame(values[1:], columns=values[0])
                df = df[df['Month'] != 'Total']
                
                # Get the monthly TOTAL for the baseline month
                month_abbr = calendar.month_abbr[baseline_month]
                month_row = df[df['Month'] == month_abbr]
                
                if month_row.empty:
                    continue
                
                # Load 2015 data
                if '2015' in df.columns:
                    monthly_total_2015 = pd.to_numeric(month_row['2015'].iloc[0], errors='coerce')
                    if not pd.isna(monthly_total_2015):
                        data_2015[source] = monthly_total_2015
                
                # Load 2025 data (previous_year)
                if '2025' in df.columns:
                    monthly_total_2025 = pd.to_numeric(month_row['2025'].iloc[0], errors='coerce')
                    if not pd.isna(monthly_total_2025):
                        data_2025[source] = monthly_total_2025
                
                # Load 2024 data (two_years_ago)
                if '2024' in df.columns:
                    monthly_total_2024 = pd.to_numeric(month_row['2024'].iloc[0], errors='coerce')
                    if not pd.isna(monthly_total_2024):
                        data_2024[source] = monthly_total_2024
                    
            except Exception as e:
                print(f"  ⚠ Could not load baseline data for {source}: {e}")
                continue
        
        # Calculate all-non-renewables from Total - Renewables for all 3 years
        if 'all-renewables' in data_2015 or 'all-renewables' in data_2025 or 'all-renewables' in data_2024:
            try:
                ws_total = spreadsheet.worksheet('Total Generation Monthly Production')
                values = ws_total.get_all_values()
                df = pd.DataFrame(values[1:], columns=values[0])
                df = df[df['Month'] != 'Total']
                
                month_abbr = calendar.month_abbr[baseline_month]
                month_row = df[df['Month'] == month_abbr]
                
                if not month_row.empty:
                    # 2015
                    if '2015' in df.columns and 'all-renewables' in data_2015:
                        total_2015_monthly = pd.to_numeric(month_row['2015'].iloc[0], errors='coerce')
                        if not pd.isna(total_2015_monthly):
                            data_2015['all-non-renewables'] = total_2015_monthly - data_2015['all-renewables']
                    
                    # 2025
                    if '2025' in df.columns and 'all-renewables' in data_2025:
                        total_2025_monthly = pd.to_numeric(month_row['2025'].iloc[0], errors='coerce')
                        if not pd.isna(total_2025_monthly):
                            data_2025['all-non-renewables'] = total_2025_monthly - data_2025['all-renewables']
                    
                    # 2024
                    if '2024' in df.columns and 'all-renewables' in data_2024:
                        total_2024_monthly = pd.to_numeric(month_row['2024'].iloc[0], errors='coerce')
                        if not pd.isna(total_2024_monthly):
                            data_2024['all-non-renewables'] = total_2024_monthly - data_2024['all-renewables']
            except:
                pass
        
        print(f"  ✓ Loaded baselines: 2015={len(data_2015)}, 2025={len(data_2025)}, 2024={len(data_2024)} sources")
        
        # Prepare data rows - ONLY columns that intraday owns (B-E, K-L)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        
        # First, update column A (Source names) if needed
        source_names = []
        for source in source_order:
            display_name = DISPLAY_NAMES.get(source, source.title())
            source_names.append([display_name])
        
        worksheet.update('A2:A13', source_names)
        
        # Now update columns B-E (Yesterday, Last Week) and K-L, O-P, S-T (all change columns)
        data_updates_be = []    # Columns B-E
        data_updates_klops = [] # Columns K-L, O-P, S-T (6 columns)
        
        for source in source_order:
            if source not in yesterday_totals or source not in week_totals:
                data_updates_be.append(['', '', '', ''])
                data_updates_klops.append(['', '', '', '', '', ''])
                continue
            
            # Columns B-E (Yesterday & Last Week)
            row_be = [
                f"{yesterday_totals[source]['gwh']:.1f}",      # B: Yesterday_GWh
                f"{yesterday_totals[source]['percentage']:.2f}",  # C: Yesterday_%
                f"{week_totals[source]['gwh']:.1f}",           # D: LastWeek_GWh
                f"{week_totals[source]['percentage']:.2f}"     # E: LastWeek_%
            ]
            data_updates_be.append(row_be)
            
            # Initialize all change values
            yesterday_change_2015 = ''
            lastweek_change_2015 = ''
            yesterday_change_2025 = ''
            lastweek_change_2025 = ''
            yesterday_change_2024 = ''
            lastweek_change_2024 = ''
            
            yesterday_gwh = yesterday_totals[source]['gwh']
            lastweek_gwh = week_totals[source]['gwh']
            
            # Calculate change from 2015 (columns K-L)
            if source in data_2015 and data_2015[source] > 0:
                monthly_total_2015 = data_2015[source]
                days_in_month = calendar.monthrange(2015, baseline_month)[1]
                
                # Yesterday change
                baseline_yesterday = (monthly_total_2015 / days_in_month) * 1
                change_y = (yesterday_gwh - baseline_yesterday) / baseline_yesterday * 100
                yesterday_change_2015 = format_change_percentage(change_y)
                
                # Last week change
                baseline_week = (monthly_total_2015 / days_in_month) * 7
                change_w = (lastweek_gwh - baseline_week) / baseline_week * 100
                lastweek_change_2015 = format_change_percentage(change_w)
            
            # Calculate change from 2025 (columns O-P)
            if source in data_2025 and data_2025[source] > 0:
                monthly_total_2025 = data_2025[source]
                days_in_month = calendar.monthrange(2025, baseline_month)[1]
                
                # Yesterday change
                baseline_yesterday = (monthly_total_2025 / days_in_month) * 1
                change_y = (yesterday_gwh - baseline_yesterday) / baseline_yesterday * 100
                yesterday_change_2025 = format_change_percentage(change_y)
                
                # Last week change
                baseline_week = (monthly_total_2025 / days_in_month) * 7
                change_w = (lastweek_gwh - baseline_week) / baseline_week * 100
                lastweek_change_2025 = format_change_percentage(change_w)
            
            # Calculate change from 2024 (columns S-T)
            if source in data_2024 and data_2024[source] > 0:
                monthly_total_2024 = data_2024[source]
                days_in_month = calendar.monthrange(2024, baseline_month)[1]
                
                # Yesterday change
                baseline_yesterday = (monthly_total_2024 / days_in_month) * 1
                change_y = (yesterday_gwh - baseline_yesterday) / baseline_yesterday * 100
                yesterday_change_2024 = format_change_percentage(change_y)
                
                # Last week change
                baseline_week = (monthly_total_2024 / days_in_month) * 7
                change_w = (lastweek_gwh - baseline_week) / baseline_week * 100
                lastweek_change_2024 = format_change_percentage(change_w)
            
            # Build row with all 6 change columns: K, L, O, P, S, T
            row_klops = [
                yesterday_change_2015, lastweek_change_2015,  # K, L
                yesterday_change_2025, lastweek_change_2025,  # O, P
                yesterday_change_2024, lastweek_change_2024   # S, T
            ]
            data_updates_klops.append(row_klops)
        
        # Update columns B-E (preserves F-I for monthly script)
        if data_updates_be:
            worksheet.update('B2:E13', data_updates_be)
        
        # Update columns K-L, O-P, S-T (all intraday change columns)
        if data_updates_klops:
            # Need to update non-contiguous ranges: K-L, then O-P, then S-T
            # Split the 6-column data into 3 separate 2-column updates
            data_kl = [[row[0], row[1]] for row in data_updates_klops]  # K, L
            data_op = [[row[2], row[3]] for row in data_updates_klops]  # O, P
            data_st = [[row[4], row[5]] for row in data_updates_klops]  # S, T
            
            worksheet.update('K2:L13', data_kl)
            worksheet.update('O2:P13', data_op)
            worksheet.update('S2:T13', data_st)
            
            # Update timestamp in column J
            timestamp_updates = [[timestamp]] * len(source_order)
            worksheet.update('J2:J13', timestamp_updates)
            
            # Format aggregate rows (bold) - ONLY columns that intraday owns
            worksheet.format('A2:E2', {'textFormat': {'bold': True}})  # All Renewables
            worksheet.format('J2:L2', {'textFormat': {'bold': True}})
            worksheet.format('O2:P2', {'textFormat': {'bold': True}})
            worksheet.format('S2:T2', {'textFormat': {'bold': True}})
            
            worksheet.format('A8:E8', {'textFormat': {'bold': True}})  # All Non-Renewables
            worksheet.format('J8:L8', {'textFormat': {'bold': True}})
            worksheet.format('O8:P8', {'textFormat': {'bold': True}})
            worksheet.format('S8:T8', {'textFormat': {'bold': True}})
            
            print(f"✓ Updated {len(source_order)} sources with yesterday/last week data")
            print(f"   Columns updated: B-E (values), K-L (vs 2015), O-P (vs 2025), S-T (vs 2024)")
            print(f"   Historical data (columns F-I, M-N, Q-R, U-V) preserved for monthly script!")
            print(f"   Worksheet: {spreadsheet.url}")
        else:
            print("⚠ No data to update")
    
    except Exception as e:
        print(f"✗ Error updating Google Sheets: {e}")
        import traceback
        traceback.print_exc()

def get_or_create_drive_folder(service, folder_name, parent_id=None, share_with_email=None):
    """
    Get or create a folder in Google Drive
    Optionally shares with specified email
    Returns folder ID
    """
    # Search for existing folder
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    
    folders = results.get('files', [])
    
    if folders:
        folder_id = folders[0]['id']
        folder_already_existed = True
    else:
        # Create folder if it doesn't exist
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]
        
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
        print(f"  Created Drive folder: {folder_name}")
        folder_already_existed = False
    
    # Share with email if provided (do this regardless of whether folder existed)
    if share_with_email:
        try:
            # Check if already shared with this email
            permissions = service.permissions().list(fileId=folder_id, fields='permissions(emailAddress)').execute()
            existing_emails = [p.get('emailAddress') for p in permissions.get('permissions', [])]
            
            if share_with_email not in existing_emails:
                permission = {
                    'type': 'user',
                    'role': 'writer',  # Or 'reader' if you only want view access
                    'emailAddress': share_with_email
                }
                service.permissions().create(
                    fileId=folder_id,
                    body=permission,
                    sendNotificationEmail=False  # Don't spam with emails
                ).execute()
                print(f"  ✓ Shared folder '{folder_name}' with {share_with_email}")
            else:
                if folder_already_existed:
                    print(f"  ✓ Folder '{folder_name}' already shared with {share_with_email}")
        except Exception as e:
            print(f"  ⚠ Could not share folder '{folder_name}': {e}")
    
    return folder_id


def upload_plot_to_drive(file_path, country='EU'):
    """
    Upload a plot to Google Drive with geography-first structure
    Structure: EU-Electricity-Plots/[Country]/Intraday/[plot].png
    
    Returns: Drive file ID or None if failed
    """
    if not GDRIVE_AVAILABLE:
        return None
    
    try:
        # Get credentials from environment
        google_creds_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            return None
        
        creds_dict = json.loads(google_creds_json)
        credentials = ServiceAccountCredentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        
        service = build('drive', 'v3', credentials=credentials)
        
        # Create folder structure: EU-Electricity-Plots/[Country]/Intraday/
        # Get or create root folder (share with owner if email provided)
        owner_email = os.getenv('OWNER_EMAIL')  # Optional: your Gmail address
        root_folder_id = get_or_create_drive_folder(service, 'EU-Electricity-Plots', share_with_email=owner_email)
        
        # Get or create country folder
        country_folder_id = get_or_create_drive_folder(service, country, root_folder_id)
        
        # Get or create Intraday folder
        intraday_folder_id = get_or_create_drive_folder(service, 'Intraday', country_folder_id)
        
        # Print folder URL for easy access
        folder_url = f'https://drive.google.com/drive/folders/{intraday_folder_id}'
        print(f"  📁 Folder: {folder_url}")
        
        # Get filename from path
        filename = os.path.basename(file_path)
        
        # Check if file already exists
        query = f"name='{filename}' and '{intraday_folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        existing_files = results.get('files', [])
        
        if existing_files:
            # Update existing file
            file_id = existing_files[0]['id']
            media = MediaFileUpload(file_path, mimetype='image/png')
            service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
        else:
            # Create new file
            file_metadata = {
                'name': filename,
                'parents': [intraday_folder_id]
            }
            media = MediaFileUpload(file_path, mimetype='image/png')
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            file_id = file.get('id')
        
        # Set permissions to "Anyone with the link can view"
        # Check if permission already exists
        try:
            existing_perms = service.permissions().list(
                fileId=file_id,
                fields='permissions(id,type)'
            ).execute()
            
            # Check if 'anyone' permission exists
            anyone_perm = None
            for perm in existing_perms.get('permissions', []):
                if perm.get('type') == 'anyone':
                    anyone_perm = perm
                    break
            
            if anyone_perm:
                # Update existing permission
                service.permissions().update(
                    fileId=file_id,
                    permissionId=anyone_perm['id'],
                    body={'role': 'reader'}
                ).execute()
            else:
                # Create new permission
                permission = {
                    'type': 'anyone',
                    'role': 'reader'
                }
                service.permissions().create(
                    fileId=file_id,
                    body=permission
                ).execute()
        except Exception as e:
            print(f"  ⚠ Warning: Could not set permissions on {os.path.basename(file_path)}: {e}")
        
        return file_id
        
    except Exception as e:
        print(f"  ⚠ Drive upload failed for {os.path.basename(file_path)}: {e}")
        return None

def create_time_axis():
    """
    Create time axis for 15-minute bins
    """
    times = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            times.append(f"{hour:02d}:{minute:02d}")
    return times


def generate_yesterday_plots(corrected_data, country_code='EU'):
    """
    Generate two yesterday plots (absolute and percentage) showing all 10 sources
    
    Args:
        corrected_data: Dictionary containing yesterday_projected data for all sources
        country_code: Country code (EU, DE, ES, etc.)
    
    Returns:
        tuple: (percentage_file, absolute_file) paths or (None, None) if failed
    """
    import os
    from datetime import datetime, timedelta
    
    # Check if we have yesterday_projected data
    if 'yesterday_projected' not in corrected_data:
        print("  ⚠ No yesterday_projected data available")
        print(f"  Available keys in corrected_data: {list(corrected_data.keys())}")
        return None, None
    
    yesterday_data = corrected_data['yesterday_projected']
    
    # Debug: Show what's actually in yesterday_data
    print(f"  📊 yesterday_projected keys: {list(yesterday_data.keys())}")
    
    # Access atomic_sources (nested structure)
    if 'atomic_sources' not in yesterday_data:
        print("  ⚠ No atomic_sources in yesterday_projected")
        return None, None
    
    atomic_sources = yesterday_data['atomic_sources']
    print(f"  📊 atomic_sources keys: {list(atomic_sources.keys())}")
    
    # Get yesterday's date for title
    yesterday_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Define the 10 sources in the order we want to plot them (MUST match intraday script)
    source_list = [
        'solar', 'wind', 'hydro', 'biomass', 'geothermal',
        'gas', 'coal', 'nuclear', 'oil', 'waste'
    ]
    
    # Country display names
    COUNTRY_DISPLAY_NAMES = {
        'EU': 'European Union',
        'DE': 'Germany',
        'ES': 'Spain',
        'FR': 'France',
        'IT': 'Italy',
        'MD': 'Moldova',
        'PL': 'Poland',
        'NL': 'Netherlands',
        'BE': 'Belgium',
        'AT': 'Austria',
        'SE': 'Sweden',
        'DK': 'Denmark',
        'FI': 'Finland',
        'NO': 'Norway',
        'CH': 'Switzerland',
        'GB': 'United Kingdom',
        'PT': 'Portugal',
        'GR': 'Greece',
        'CZ': 'Czechia',
        'RO': 'Romania',
        'HU': 'Hungary',
        'BG': 'Bulgaria',
        'SK': 'Slovakia',
        'SI': 'Slovenia',
        'HR': 'Croatia',
        'LT': 'Lithuania',
        'LV': 'Latvia',
        'EE': 'Estonia',
        'IE': 'Ireland',
        'LU': 'Luxembourg',
        'MT': 'Malta',
        'CY': 'Cyprus'
    }
    
    country_name = COUNTRY_DISPLAY_NAMES.get(country_code, country_code)
    
    def load_flag(fig, country_code):
        """Load real flag PNG or fall back to placeholder"""
        import os
        from matplotlib import image as mpimg
        
        flag_path = f'flags/{country_code}.png'
        
        if os.path.exists(flag_path):
            try:
                ax_flag = fig.add_axes([0.1, 0.85, 0.075, 0.05])
                flag_img = mpimg.imread(flag_path)
                ax_flag.imshow(flag_img, aspect='auto')
                ax_flag.axis('off')
                return ax_flag
            except:
                pass
        
        # Fallback: colored rectangle with country code
        ax_flag = fig.add_axes([0.1, 0.85, 0.075, 0.05])
        colors_map = {
            'EU': '#003399', 'DE': '#000000', 'ES': '#AA151B',
            'FR': '#0055A4', 'MD': '#0046AE',
        }
        bg_color = colors_map.get(country_code, '#4A90E2')
        ax_flag.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=bg_color, 
                                        edgecolor='#CCCCCC', linewidth=2))
        ax_flag.text(0.5, 0.5, country_code, color='white', 
                    fontsize=14, fontweight='bold', ha='center', va='center')
        ax_flag.set_xlim(0, 1)
        ax_flag.set_ylim(0, 1)
        ax_flag.axis('off')
        return ax_flag
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Extract data for each source (sum across countries for each timestamp)
    source_data = {}
    timestamps = None
    
    print(f"  🔍 Extracting data for {len(source_list)} sources...")
    for source_name in source_list:
        if source_name in atomic_sources:
            source_timestamps = atomic_sources[source_name]
            if source_timestamps:
                # Sum across countries for each timestamp
                time_series = []
                sorted_timestamps = sorted(source_timestamps.keys())
                
                for timestamp in sorted_timestamps:
                    country_values = source_timestamps[timestamp]
                    total = sum(country_values.values())
                    time_series.append(total)
                
                if time_series:
                    source_data[source_name] = np.array(time_series)
                    if timestamps is None:
                        timestamps = sorted_timestamps
                    print(f"    ✓ {source_name}: {len(time_series)} data points")
                else:
                    print(f"    ⚠ {source_name}: no valid timestamps")
            else:
                print(f"    ⚠ {source_name}: empty data")
        else:
            print(f"    ✗ {source_name}: not found in atomic_sources")
    
    if not source_data or timestamps is None:
        print("  ⚠ No valid source data for yesterday")
        print(f"  Sources found: {list(source_data.keys())}")
        return None, None
    
    # Create x-axis values (0-95 for 96 timestamps)
    x_values = np.arange(len(timestamps))
    
    # Create time labels and tick positions (matching intraday script)
    time_labels = create_time_axis()  # 96 labels: 00:00, 00:15, ..., 23:45
    tick_positions = list(range(0, len(time_labels), 16))  # Every 4 hours (16 * 15min = 4h)
    tick_positions.append(len(time_labels))  # Add position for 24:00
    tick_labels_axis = [time_labels[i] if i < len(time_labels) else '' for i in tick_positions[:-1]]
    tick_labels_axis.append('24:00')
    
    # Calculate total generation for percentage calculation
    total_gen = np.zeros(len(x_values))
    for source_name, data in source_data.items():
        if len(data) == len(x_values):
            total_gen += data
    
    # Prevent division by zero
    total_gen = np.where(total_gen == 0, 1, total_gen)
    
    print(f"\n📊 Generating yesterday plots for {country_code}...")
    
    # ========== PLOT 1: PERCENTAGE ==========
    fig1 = plt.figure(figsize=(12, 12))
    plt.subplots_adjust(left=0.22, right=0.9, top=0.80, bottom=0.35)
    ax1 = fig1.add_subplot(111)
    
    # Plot each source
    lines_pct = []
    labels_pct = []
    for source_name in source_list:
        if source_name in source_data:
            data = source_data[source_name]
            if len(data) == len(x_values):
                pct_data = (data / total_gen) * 100
                color = ENTSOE_COLORS.get(source_name, 'black')
                line, = ax1.plot(x_values, pct_data, color=color, linewidth=6, label=source_name)
                lines_pct.append(line)
                labels_pct.append(source_name)
    
    # Formatting
    ax1.set_xlabel('Hour of Day', fontsize=24, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Generation (%)', fontsize=24, fontweight='bold', labelpad=8)
    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.set_xlim(0, len(time_labels))
    ax1.set_ylim(0, max(100, ax1.get_ylim()[1] * 1.2))  # 20% margin
    
    # X-axis ticks
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels_axis)
    
    # Tick parameters
    ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
    
    # Title
    fig1.text(0.525, 0.92, 'Electricity Generation', 
              fontsize=18, fontweight='bold', ha='center')
    fig1.text(0.525, 0.89, f'{country_name} · Yesterday ({yesterday_date}) · All Sources', 
              fontsize=14, ha='center', color='#333')
    
    # Flag
    load_flag(fig1, country_code)
    
    # Legend: 4-column layout with custom reordering
    # Desired order:
    # Wind      | Hydro     | Solar      |
    # Biomass   | Geotherm  | [empty]    |
    # Nuclear   | Gas       | Coal       |
    # Waste     | Oil       | [empty]    |
    
    # Map source names to indices in lines_pct/labels_pct
    source_to_idx = {labels_pct[i]: i for i in range(len(labels_pct))}
    
    # Create reordered handles and labels
    empty = Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)
    
    reordered_handles = []
    reordered_labels = []
    
    # Row 1: Wind, Hydro, Solar
    for name in ['wind', 'hydro', 'solar']:
        if name in source_to_idx:
            idx = source_to_idx[name]
            reordered_handles.append(lines_pct[idx])
            reordered_labels.append(name.title())  # Display as title case
        else:
            reordered_handles.append(empty)
            reordered_labels.append('')
    
    # Row 2: Biomass, Geothermal, [empty]
    for name in ['biomass', 'geothermal']:
        if name in source_to_idx:
            idx = source_to_idx[name]
            reordered_handles.append(lines_pct[idx])
            reordered_labels.append(name.title())  # Display as title case
        else:
            reordered_handles.append(empty)
            reordered_labels.append('')
    reordered_handles.append(empty)
    reordered_labels.append('')
    
    # Row 3: Nuclear, Gas, Coal
    for name in ['nuclear', 'gas', 'coal']:
        if name in source_to_idx:
            idx = source_to_idx[name]
            reordered_handles.append(lines_pct[idx])
            reordered_labels.append(name.title())  # Display as title case
        else:
            reordered_handles.append(empty)
            reordered_labels.append('')
    
    # Row 4: Waste, Oil, [empty]
    for name in ['waste', 'oil']:
        if name in source_to_idx:
            idx = source_to_idx[name]
            reordered_handles.append(lines_pct[idx])
            reordered_labels.append(name.title())  # Display as title case
        else:
            reordered_handles.append(empty)
            reordered_labels.append('')
    reordered_handles.append(empty)
    reordered_labels.append('')
    
    # Add legend with 4-column layout matching trends
    ax1.legend(reordered_handles, reordered_labels,
              loc='upper left', 
              bbox_to_anchor=(0.14, 0.255),
              bbox_transform=fig1.transFigure,
              ncol=4, 
              fontsize=18, 
              frameon=False)
    
    # Save percentage plot
    percentage_file = f'plots/{country_code}_yesterday_all_sources_percentage.png'
    fig1.savefig(percentage_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"  ✓ Saved: {percentage_file}")
    
    # ========== PLOT 2: ABSOLUTE (GW) ==========
    fig2 = plt.figure(figsize=(12, 12))
    plt.subplots_adjust(left=0.22, right=0.9, top=0.80, bottom=0.35)
    ax2 = fig2.add_subplot(111)
    
    # Plot each source
    lines_abs = []
    labels_abs = []
    for source_name in source_list:
        if source_name in source_data:
            data = source_data[source_name]
            if len(data) == len(x_values):
                gw_data = data / 1000  # Convert MW to GW
                color = ENTSOE_COLORS.get(source_name, 'black')
                line, = ax2.plot(x_values, gw_data, color=color, linewidth=6, label=source_name)
                lines_abs.append(line)
                labels_abs.append(source_name)
    
    # Formatting
    ax2.set_xlabel('Hour of Day', fontsize=24, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Generation (GW)', fontsize=24, fontweight='bold', labelpad=8)
    ax2.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.set_xlim(0, len(time_labels))
    
    # Y-axis with 20% margin
    current_max = ax2.get_ylim()[1]
    ax2.set_ylim(0, current_max * 1.2)
    
    # X-axis ticks
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels_axis)
    
    # Tick parameters
    ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
    
    # Title
    fig2.text(0.525, 0.92, 'Electricity Generation', 
              fontsize=18, fontweight='bold', ha='center')
    fig2.text(0.525, 0.89, f'{country_name} · Yesterday ({yesterday_date}) · All Sources', 
              fontsize=14, ha='center', color='#333')
    
    # Flag
    load_flag(fig2, country_code)
    
    # Legend: same reordering as percentage plot
    source_to_idx_abs = {labels_abs[i]: i for i in range(len(labels_abs))}
    
    reordered_handles_abs = []
    reordered_labels_abs = []
    
    # Row 1: Wind, Hydro, Solar
    for name in ['wind', 'hydro', 'solar']:
        if name in source_to_idx_abs:
            idx = source_to_idx_abs[name]
            reordered_handles_abs.append(lines_abs[idx])
            reordered_labels_abs.append(name.title())  # Display as title case
        else:
            reordered_handles_abs.append(empty)
            reordered_labels_abs.append('')
    
    # Row 2: Biomass, Geothermal, [empty]
    for name in ['biomass', 'geothermal']:
        if name in source_to_idx_abs:
            idx = source_to_idx_abs[name]
            reordered_handles_abs.append(lines_abs[idx])
            reordered_labels_abs.append(name.title())  # Display as title case
        else:
            reordered_handles_abs.append(empty)
            reordered_labels_abs.append('')
    reordered_handles_abs.append(empty)
    reordered_labels_abs.append('')
    
    # Row 3: Nuclear, Gas, Coal
    for name in ['nuclear', 'gas', 'coal']:
        if name in source_to_idx_abs:
            idx = source_to_idx_abs[name]
            reordered_handles_abs.append(lines_abs[idx])
            reordered_labels_abs.append(name.title())  # Display as title case
        else:
            reordered_handles_abs.append(empty)
            reordered_labels_abs.append('')
    
    # Row 4: Waste, Oil, [empty]
    for name in ['waste', 'oil']:
        if name in source_to_idx_abs:
            idx = source_to_idx_abs[name]
            reordered_handles_abs.append(lines_abs[idx])
            reordered_labels_abs.append(name.title())  # Display as title case
        else:
            reordered_handles_abs.append(empty)
            reordered_labels_abs.append('')
    reordered_handles_abs.append(empty)
    reordered_labels_abs.append('')
    
    # Add legend with 4-column layout matching trends
    ax2.legend(reordered_handles_abs, reordered_labels_abs,
              loc='upper left', 
              bbox_to_anchor=(0.14, 0.255),
              bbox_transform=fig2.transFigure,
              ncol=4, 
              fontsize=18, 
              frameon=False)
    
    # Save absolute plot
    absolute_file = f'plots/{country_code}_yesterday_all_sources_absolute.png'
    fig2.savefig(absolute_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  ✓ Saved: {absolute_file}")
    
    return percentage_file, absolute_file


def upload_yesterday_plot_to_drive(file_path, country='EU'):
    """
    Upload yesterday plot to Google Drive
    Structure: EU-Electricity-Plots/[Country]/Yesterday/[plot].png
    
    Returns: Drive file ID or None if failed
    """
    import os
    import json
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    
    try:
        # Get credentials from environment
        google_creds_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            print("  ⚠ GOOGLE_CREDENTIALS_JSON not set")
            return None
        
        creds_dict = json.loads(google_creds_json)
        credentials = ServiceAccountCredentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        
        service = build('drive', 'v3', credentials=credentials)
        
        # Helper function to get or create folder
        def get_or_create_folder(folder_name, parent_id=None):
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            
            results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
            folders = results.get('files', [])
            
            if folders:
                return folders[0]['id']
            else:
                file_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                if parent_id:
                    file_metadata['parents'] = [parent_id]
                folder = service.files().create(body=file_metadata, fields='id').execute()
                return folder.get('id')
        
        # Create folder structure: EU-Electricity-Plots/[Country]/Yesterday/
        root_folder_id = get_or_create_folder('EU-Electricity-Plots')
        country_folder_id = get_or_create_folder(country, root_folder_id)
        yesterday_folder_id = get_or_create_folder('Yesterday', country_folder_id)
        
        # Get filename
        filename = os.path.basename(file_path)
        
        # Check if file already exists
        query = f"name='{filename}' and '{yesterday_folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        existing_files = results.get('files', [])
        
        if existing_files:
            # Update existing file
            file_id = existing_files[0]['id']
            media = MediaFileUpload(file_path, mimetype='image/png')
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            # Create new file
            file_metadata = {
                'name': filename,
                'parents': [yesterday_folder_id]
            }
            media = MediaFileUpload(file_path, mimetype='image/png')
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            file_id = file.get('id')
        
        # Set permissions: Anyone with link can view
        try:
            permission = {'type': 'anyone', 'role': 'reader'}
            service.permissions().create(fileId=file_id, body=permission).execute()
        except:
            pass
        
        return file_id
        
    except Exception as e:
        print(f"  ⚠ Drive upload failed for {os.path.basename(file_path)}: {e}")
        return None

def main():
    """
    Main function - orchestrates the 3 phases
    Generates ALL 12 plots by default, or single plot if --source specified
    """
    parser = argparse.ArgumentParser(description='EU Energy Intraday Analysis v2')
    parser.add_argument('--source', 
                       choices=ATOMIC_SOURCES + AGGREGATE_SOURCES,
                       help='Optional: Generate only this source (default: all sources)')
    
    args = parser.parse_args()
    
    if args.source:
        # Single source mode (for testing or backward compatibility)
        print("\n" + "=" * 80)
        print(f"{DISPLAY_NAMES[args.source].upper()} INTRADAY ANALYSIS")
        print("=" * 80)
    else:
        # Batch mode (default)
        print("\n" + "=" * 80)
        print("EU ENERGY INTRADAY ANALYSIS - BATCH MODE")
        print("Generating all 12 source plots from single data collection")
        print("=" * 80)
    
    # Get API key
    api_key = os.environ.get('ENTSOE_API_KEY')
    if not api_key:
        print("ERROR: ENTSOE_API_KEY environment variable not set!")
        sys.exit(1)
    
    try:
        # Phase 1: Collect all data ONCE (all 27 EU countries)
        print("\n⚡ OPTIMIZATION: Fetching all 27 EU countries ONCE for all target countries")
        raw_data_matrix, periods, fetch_time = collect_all_data(api_key)
        
        # Countries to process
        countries_to_process = ['EU', 'DE', 'ES', 'FR']
        total_plots_generated = {}  # Track plots per country
        
        print(f"\n" + "=" * 80)
        print(f"PROCESSING {len(countries_to_process)} COUNTRIES: {', '.join(countries_to_process)}")
        print("=" * 80)
        
        for country_idx, country_code in enumerate(countries_to_process, 1):
            print(f"\n" + "=" * 80)
            print(f"PROCESSING {country_code} ({country_idx}/{len(countries_to_process)})")
            print("=" * 80)
            
            # Extract data for this country from raw data
            data_matrix = extract_country_from_raw_data(raw_data_matrix, country_code)
            
            # Phase 2: Apply projections and corrections for this country
            corrected_data = apply_projections_and_corrections(data_matrix)
            
            # Phase 3: Generate plots
            if args.source:
                # Single plot mode
                print("\n" + "=" * 80)
                print(f"PHASE 3: GENERATING {DISPLAY_NAMES[args.source].upper()} PLOTS")
                print("=" * 80)
                output_file_base = f'plots/{args.source.replace("-", "_")}_analysis.png'
                percentage_file, absolute_file = generate_plot_for_source(args.source, corrected_data, output_file_base, fetch_time=fetch_time, country_code=country_code)
                
                # Upload both to Google Drive (if plots were generated)
                if percentage_file and absolute_file:
                    print(f"\n📤 Uploading to Google Drive...")
                    perc_id = upload_plot_to_drive(percentage_file, country=country_code)
                    abs_id = upload_plot_to_drive(absolute_file, country=country_code)
                    if perc_id and abs_id:
                        total_plots_generated[country_code] = 1
                        print(f"  ✓ Uploaded both plots to {country_code}/Intraday/")
                else:
                    total_plots_generated[country_code] = 0
                    print(f"  ⚠ Skipping upload (no data for {args.source})")
            else:
                # Batch mode - generate all plots
                print("\n" + "=" * 80)
                print("PHASE 3: GENERATING ALL 12 PLOTS (24 files: percentage + absolute)")
                print("=" * 80)
                
                all_sources = ATOMIC_SOURCES + AGGREGATE_SOURCES
                drive_file_ids = {}
                plots_generated = 0  # Track successful plots
                
                # Generate individual source plots
                for i, source in enumerate(all_sources, 1):
                    print(f"\n[{i}/{len(all_sources)}] Processing {DISPLAY_NAMES[source]}...")
                    output_file_base = f'plots/{source.replace("-", "_")}_analysis.png'
                    percentage_file, absolute_file = generate_plot_for_source(source, corrected_data, output_file_base, fetch_time=fetch_time, country_code=country_code)
                    
                    # Upload both to Google Drive (only if plots were generated)
                    if percentage_file and absolute_file:
                        perc_id = upload_plot_to_drive(percentage_file, country=country_code)
                        abs_id = upload_plot_to_drive(absolute_file, country=country_code)
                        if perc_id and abs_id:
                            drive_file_ids[source] = {
                                'percentage': perc_id,
                                'absolute': abs_id
                            }
                            plots_generated += 1
                            print(f"  ✓ Uploaded both plots to Drive: {country_code}/Intraday/")
                    else:
                        print(f"  ⚠ Skipping upload for {source} (no data available)")
                
                total_plots_generated[country_code] = plots_generated
                
                # === GENERATE YESTERDAY ALL-SOURCES PLOTS ===
                print(f"\n" + "=" * 80)
                print(f"GENERATING YESTERDAY ALL-SOURCES PLOTS FOR {country_code}")
                print("=" * 80)
                yesterday_perc, yesterday_abs = generate_yesterday_plots(corrected_data, country_code=country_code)
                
                if yesterday_perc and yesterday_abs:
                    if GDRIVE_AVAILABLE:
                        print(f"  📤 Uploading yesterday plots to Drive...")
                        yesterday_perc_id = upload_yesterday_plot_to_drive(yesterday_perc, country=country_code)
                        yesterday_abs_id = upload_yesterday_plot_to_drive(yesterday_abs, country=country_code)
                        
                        if yesterday_perc_id and yesterday_abs_id:
                            # Store separately (not in drive_file_ids with individual sources)
                            drive_file_ids['__yesterday__'] = {
                                'percentage': yesterday_perc_id,
                                'absolute': yesterday_abs_id
                            }
                            print(f"  ✓ Uploaded yesterday plots to Drive: {country_code}/Yesterday/")
                        else:
                            print(f"  ⚠ Failed to upload yesterday plots to Drive")
                    else:
                        print(f"  ⚠ Drive upload not available")
                else:
                    print(f"  ⚠ Failed to generate yesterday plots")
                # === END YESTERDAY PLOTS ===
                
                # Save Drive file IDs to JSON
                if drive_file_ids:
                    print(f"\n📤 Saving Drive links for {len(drive_file_ids)} items...")
                    individual_sources = [k for k in drive_file_ids.keys() if k != '__yesterday__']
                    print(f"   Individual sources: {', '.join(individual_sources)}")
                    if '__yesterday__' in drive_file_ids:
                        print(f"   Yesterday: all_sources")
                    
                    drive_links_file = 'plots/drive_links.json'
                    drive_links = {}
                    
                    # Load existing links
                    if os.path.exists(drive_links_file):
                        try:
                            with open(drive_links_file, 'r') as f:
                                drive_links = json.load(f)
                        except:
                            pass
                    
                    # Initialize country structure
                    if country_code not in drive_links:
                        drive_links[country_code] = {}
                    if 'Intraday' not in drive_links[country_code]:
                        drive_links[country_code]['Intraday'] = {}
                    
                    # Random thumbnail size to bypass mobile browser cache
                    thumbnail_size = random.choice([1998, 1999, 2000, 2001, 2002])
                    print(f"  📐 Using thumbnail size: w{thumbnail_size} (cache-busting)")
                    
                    # Save individual source plots
                    for source, file_ids in drive_file_ids.items():
                        if source == '__yesterday__':
                            continue  # Handle separately below
                        
                        drive_links[country_code]['Intraday'][source] = {
                            'percentage': {
                                'file_id': file_ids['percentage'],
                                'view_url': f'https://drive.google.com/file/d/{file_ids["percentage"]}/view',
                                'direct_url': f'https://drive.google.com/thumbnail?id={file_ids["percentage"]}&sz=w{thumbnail_size}'
                            },
                            'absolute': {
                                'file_id': file_ids['absolute'],
                                'view_url': f'https://drive.google.com/file/d/{file_ids["absolute"]}/view',
                                'direct_url': f'https://drive.google.com/thumbnail?id={file_ids["absolute"]}&sz=w{thumbnail_size}'
                            },
                            'updated': datetime.now().isoformat()
                        }
                    
                    # Save yesterday plots (separate section)
                    if '__yesterday__' in drive_file_ids:
                        if 'Yesterday' not in drive_links[country_code]:
                            drive_links[country_code]['Yesterday'] = {}
                        
                        yesterday_ids = drive_file_ids['__yesterday__']
                        drive_links[country_code]['Yesterday'] = {
                            'all_sources': {
                                'percentage': {
                                    'file_id': yesterday_ids['percentage'],
                                    'view_url': f'https://drive.google.com/file/d/{yesterday_ids["percentage"]}/view',
                                    'direct_url': f'https://drive.google.com/thumbnail?id={yesterday_ids["percentage"]}&sz=w{thumbnail_size}',
                                    'updated': datetime.now().isoformat()
                                },
                                'absolute': {
                                    'file_id': yesterday_ids['absolute'],
                                    'view_url': f'https://drive.google.com/file/d/{yesterday_ids["absolute"]}/view',
                                    'direct_url': f'https://drive.google.com/thumbnail?id={yesterday_ids["absolute"]}&sz=w{thumbnail_size}',
                                    'updated': datetime.now().isoformat()
                                }
                            }
                        }
                    
                    # Save back to file (atomic write with validation)
                    drive_links_file_path = os.path.abspath(drive_links_file)
                    temp_file = drive_links_file + '.tmp'
                    
                    # Check if file exists and is writable
                    if os.path.exists(drive_links_file):
                        if not os.access(drive_links_file, os.W_OK):
                            print(f"  ⚠ Warning: {drive_links_file} exists but is not writable!")
                        else:
                            print(f"  Overwriting existing file: {drive_links_file}")
                    
                    try:
                        # Write to temporary file first
                        with open(temp_file, 'w') as f:
                            json.dump(drive_links, f, indent=2)
                        
                        # Validate the JSON by reading it back
                        with open(temp_file, 'r') as f:
                            content = f.read()
                            # Check for git conflict markers
                            if '<<<<<<< ' in content or '=======' in content or '>>>>>>> ' in content:
                                raise ValueError("Git conflict markers detected in JSON file!")
                            # Validate it's valid JSON and structure
                            saved_data = json.loads(content)
                        
                        # If validation passes, atomically replace the file
                        os.replace(temp_file, drive_links_file)
                        
                        # Verify structure
                        individual_sources = [k for k in drive_file_ids.keys() if k != '__yesterday__']
                        sample_source = individual_sources[0] if individual_sources else None
                        
                        if sample_source:
                            if country_code in saved_data and 'Intraday' in saved_data[country_code]:
                                if sample_source in saved_data[country_code]['Intraday']:
                                    source_data = saved_data[country_code]['Intraday'][sample_source]
                                    if 'percentage' in source_data and 'absolute' in source_data:
                                        file_size = os.path.getsize(drive_links_file)
                                        print(f"  ✓ Drive links saved to {drive_links_file}")
                                        print(f"     Full path: {drive_links_file_path}")
                                        print(f"     File size: {file_size} bytes")
                                        print(f"     ✓ Verified structure (Intraday + Yesterday)")
                                    else:
                                        print(f"  ⚠ WARNING: OLD structure detected! Missing percentage/absolute")
                                else:
                                    print(f"  ⚠ WARNING: Source {sample_source} not in saved JSON")
                        
                        # Verify yesterday section if it was saved
                        if '__yesterday__' in drive_file_ids:
                            if country_code in saved_data and 'Yesterday' in saved_data[country_code]:
                                if 'all_sources' in saved_data[country_code]['Yesterday']:
                                    print(f"     ✓ Verified Yesterday section")
                                else:
                                    print(f"  ⚠ WARNING: Yesterday section missing all_sources")
                        
                    except ValueError as e:
                        print(f"  ✗ JSON validation error: {e}")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        raise
                    except PermissionError as e:
                        print(f"  ✗ Permission error: {e}")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        raise
                    except Exception as e:
                        print(f"  ✗ Error writing file: {e}")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        raise
                else:
                    print("\n⚠ Warning: No Drive file IDs collected - JSON not updated")
                    print("   Check if uploads succeeded above")
            
            # Phase 4: Update Summary Table in Google Sheets
            update_summary_table_worksheet(corrected_data, country_code=country_code)
            
            print(f"\n✓ {country_code} COMPLETE!")
        
        # Create timestamp file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        with open('plots/last_update.html', 'w') as f:
            f.write(f'<p>Last updated: {timestamp}</p>')
        
        print(f"\n" + "=" * 80)
        print(f"✓ ALL COUNTRIES COMPLETE!")
        print("=" * 80)
        if args.source:
            total_count = sum(total_plots_generated.values())
            print(f"   - {DISPLAY_NAMES[args.source]} plots: {total_count}/{len(countries_to_process)} countries with data")
            for country, count in total_plots_generated.items():
                status = "✓ Generated" if count > 0 else "⚠ No data"
                print(f"     • {country}: {status}")
        else:
            total_count = sum(total_plots_generated.values())
            print(f"   - {total_count} source plots generated across {len(countries_to_process)} countries")
            for country, count in total_plots_generated.items():
                print(f"     • {country}: {count}/12 sources + yesterday")
            print(f"   - Summary tables updated in Google Sheets")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
