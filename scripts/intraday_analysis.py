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


def get_or_create_country_sheet(gc, drive_service, country_code='EU'):
    """
    Get or create country-specific electricity data sheet
    Structure: EU-Electricity-Plots/[Country]/[Country] Electricity Production Data
    
    Args:
        gc: gspread client
        drive_service: Google Drive API service (or None)
        country_code: Country code (EU, DE, FR, etc.)
    
    Returns: gspread Spreadsheet object
    """
    import json
    import os
    
    sheet_name = f'{country_code} Electricity Production Data'
    
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
                    print(f"✓ Opened existing sheet: {sheet_name}")
                    return spreadsheet
                except:
                    print(f"  ⚠ Sheet ID in JSON is invalid, will create new")
        except:
            pass
    
    # Sheet doesn't exist or JSON doesn't have it - create new
    print(f"  Creating new sheet: {sheet_name}")
    spreadsheet = gc.create(sheet_name)
    
    # Move to correct Drive folder structure if drive_service provided
    if drive_service:
        try:
            # Get or create: EU-Electricity-Plots/[Country]/
            root_folder_id = get_or_create_drive_folder(drive_service, 'EU-Electricity-Plots')
            country_folder_id = get_or_create_drive_folder(drive_service, country_code, root_folder_id)
            
            # Move spreadsheet to country folder
            file = drive_service.files().get(fileId=spreadsheet.id, fields='parents').execute()
            previous_parents = ",".join(file.get('parents', []))
            
            drive_service.files().update(
                fileId=spreadsheet.id,
                addParents=country_folder_id,
                removeParents=previous_parents,
                fields='id, parents'
            ).execute()
            
            print(f"  ✓ Moved sheet to: EU-Electricity-Plots/{country_code}/")
            
            # Set permissions: Anyone with link can view
            try:
                permission = {
                    'type': 'anyone',
                    'role': 'reader'
                }
                drive_service.permissions().create(
                    fileId=spreadsheet.id,
                    body=permission,
                    fields='id'
                ).execute()
                print(f"  ✓ Set permissions: Anyone with link can view")
            except Exception as e:
                print(f"  ⚠ Could not set permissions: {e}")
                
        except Exception as e:
            print(f"  ⚠ Could not organize in Drive: {e}")
    
    # Save sheet ID to JSON
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

            time.sleep(0.2)
            return data

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt)
                time.sleep(wait_time)
            else:
                time.sleep(0.5)
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


def aggregate_eu_data(countries, start_date, end_date, client, source_keywords, data_type='generation', mark_extrapolated=False, country_code='EU'):
    """
    Aggregate energy data across EU countries OR extract single country
    
    Args:
        country_code: 
            - 'EU' = sum all countries
            - 'DE' = extract just DE
            - None = return raw dataframe without aggregation
    
    Returns: (total_series, country_data_df, successful_countries)
        - If country_code='EU': total_series is sum of all countries
        - If country_code='DE': total_series is just DE column
        - If country_code=None: total_series is empty, just returns raw country_df
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
    
    # Extract or aggregate based on country_code
    if country_code is None:
        # Raw mode: return empty series, full dataframe
        total = pd.Series(dtype=float)
    elif country_code == 'EU':
        # Aggregate: sum all countries
        total = combined_df.sum(axis=1, skipna=True)
    else:
        # Single country: extract that column
        if country_code in combined_df.columns:
            total = combined_df[country_code]
        else:
            # Country not in data
            total = pd.Series(0, index=combined_df.index if not combined_df.empty else [])

    return total, combined_df, successful_countries


# ============================================================================
# PHASE 1: DATA COLLECTION
# ============================================================================

def collect_all_data(api_key):
    """
    Phase 1: Collect ALL data for all atomic sources and total generation
    Fetches all 27 EU countries ONCE - no aggregation yet
    Returns raw country-level data that can be extracted/aggregated later
    """
    client = EntsoePandasClient(api_key=api_key)
    
    # Cache fetch time at start for consistent cutoff across all sources
    fetch_time = pd.Timestamp.now(tz='Europe/Brussels')
    print(f"🕐 Reference fetch time: {fetch_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    print("=" * 80)
    print("PHASE 1: DATA COLLECTION (ALL COUNTRIES)")
    print("=" * 80)
    
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
    
    # Data storage - raw country dataframes
    data_matrix = {
        'atomic_sources': {},  # source -> period -> country_df (all 27 countries as columns)
        'total_generation': {} # period -> country_df (all 27 countries as columns)
    }
    
    # Fetch atomic sources (with country breakdown)
    print("\n📊 Fetching 10 Atomic Sources (all 27 countries)...")
    for source in ATOMIC_SOURCES:
        print(f"\n  {DISPLAY_NAMES[source]}:")
        data_matrix['atomic_sources'][source] = {}
        
        for period_name, (start_date, end_date) in periods.items():
            mark_extrap = (period_name in ['today', 'yesterday'])
            
            # Fetch WITHOUT aggregation - get raw country dataframe
            _, country_df, countries = aggregate_eu_data(
                EU_COUNTRIES, start_date, end_date, client,
                SOURCE_KEYWORDS[source], 'generation', mark_extrapolated=mark_extrap,
                country_code=None  # None = return raw dataframe without aggregation
            )
            
            if not country_df.empty:
                data_matrix['atomic_sources'][source][period_name] = country_df
                print(f"    {period_name}: ✓ {len(countries)} countries, {len(country_df)} timestamps")
            else:
                print(f"    {period_name}: ✗ No data")
    
    # Fetch Total Generation (with country breakdown)
    print("\n📊 Fetching Total Generation (all 27 countries)...")
    all_gen_keywords = SOURCE_KEYWORDS['all-renewables'] + SOURCE_KEYWORDS['all-non-renewables']
    for period_name, (start_date, end_date) in periods.items():
        mark_extrap = (period_name in ['today', 'yesterday'])
        
        # Fetch WITHOUT aggregation - get raw country dataframe
        _, country_df, countries = aggregate_eu_data(
            EU_COUNTRIES, start_date, end_date, client,
            all_gen_keywords, 'generation', mark_extrapolated=mark_extrap,
            country_code=None  # None = return raw dataframe without aggregation
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
    Extract or aggregate data for a specific country from raw data matrix
    
    Args:
        raw_data_matrix: Result from collect_all_data() - contains all 27 countries
        country_code: 'EU' to aggregate all, 'DE' for Germany, etc.
    
    Returns: data_matrix in same format as old collect_all_data() but for specific country
    """
    print(f"\n📊 Extracting data for: {country_code}")
    
    processed_data = {
        'atomic_sources': {},
        'aggregates': {},
        'total_generation': {}
    }
    
    # Process atomic sources
    for source in ATOMIC_SOURCES:
        processed_data['atomic_sources'][source] = {}
        
        for period_name, country_df in raw_data_matrix['atomic_sources'][source].items():
            if country_df.empty:
                continue
            
            if country_code == 'EU':
                # Sum all countries
                aggregated = country_df.sum(axis=1, skipna=True)
            else:
                # Extract single country
                if country_code in country_df.columns:
                    aggregated = country_df[country_code]
                else:
                    # Country not in data - create zeros
                    aggregated = pd.Series(0, index=country_df.index)
            
            # Store back as single-column dataframe for compatibility
            processed_data['atomic_sources'][source][period_name] = pd.DataFrame({
                country_code: aggregated
            })
    
    # Compute aggregates from atomic sources
    for agg_source in AGGREGATE_SOURCES:
        processed_data['aggregates'][agg_source] = {}
        components = AGGREGATE_DEFINITIONS[agg_source]
        
        for period_name in raw_data_matrix['atomic_sources'][ATOMIC_SOURCES[0]].keys():
            # Sum the components
            agg_total = None
            for component in components:
                if component in processed_data['atomic_sources']:
                    if period_name in processed_data['atomic_sources'][component]:
                        component_data = processed_data['atomic_sources'][component][period_name]
                        if not component_data.empty:
                            if agg_total is None:
                                agg_total = component_data[country_code].copy()
                            else:
                                agg_total += component_data[country_code]
            
            if agg_total is not None:
                processed_data['aggregates'][agg_source][period_name] = agg_total
    
    # Process total generation
    for period_name, country_df in raw_data_matrix['total_generation'].items():
        if country_df.empty:
            continue
        
        if country_code == 'EU':
            # Sum all countries
            total = country_df.sum(axis=1, skipna=True)
        else:
            # Extract single country
            if country_code in country_df.columns:
                total = country_df[country_code]
            else:
                total = pd.Series(0, index=country_df.index)
        
        # Store back as single-column dataframe
        processed_data['total_generation'][period_name] = pd.DataFrame({
            country_code: total
        })
    
    return processed_data


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
    Uses weekly hourly averages for threshold comparison
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
    
    # Build weekly hourly average for total generation
    total_gen_weekly_avg = None
    if reference_period in data_matrix['total_generation']:
        ref_total_gen = data_matrix['total_generation'][reference_period]
        ref_total_gen_with_time = ref_total_gen.copy()
        ref_total_gen_with_time['time'] = ref_total_gen_with_time.index.strftime('%H:%M')
        total_gen_weekly_avg = ref_total_gen_with_time.groupby('time').mean(numeric_only=True)
    
    # Get target period data
    target_atomic = {src: data_matrix['atomic_sources'][src].get(target_period) 
                     for src in ATOMIC_SOURCES if src in data_matrix['atomic_sources']}
    target_total_gen = data_matrix['total_generation'].get(target_period)
    
    if target_total_gen is None:
        return {}
    
    # Build BOTH corrected and actual (uncorrected) data
    corrected_sources = {}
    actual_sources = {}  # NEW: Store uncorrected versions
    correction_log = []
    
    for source in ATOMIC_SOURCES + AGGREGATE_SOURCES:
        corrected_sources[source] = {}
        actual_sources[source] = {}
    
    # Process each timestamp
    for timestamp in target_total_gen.index:
        time_str = timestamp.strftime('%H:%M')
        
        # Correct atomic sources
        for source in ATOMIC_SOURCES:
            if source not in target_atomic or target_atomic[source] is None:
                continue
            
            if timestamp not in target_atomic[source].index:
                continue
            
            source_row = target_atomic[source].loc[timestamp]
            
            # Initialize this timestamp for this source
            if timestamp not in corrected_sources[source]:
                corrected_sources[source][timestamp] = {}
                actual_sources[source][timestamp] = {}
            
            for country in source_row.index:
                actual_val = source_row[country]
                
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
    
    # Print correction log
    if correction_log:
        print(f"\n  🚨 Detected {len(correction_log)} values below 10% threshold:")
        for log in correction_log[:20]:  # Print first 20
            print(f"    {log['time']} | {log['country']}-{log['source']}: "
                  f"{log['actual']:.1f} MW < 10% of {log['expected']:.1f} MW "
                  f"(threshold: {log['threshold']:.1f} MW) → Using {log['expected']:.1f} MW")
        if len(correction_log) > 20:
            print(f"    ... and {len(correction_log) - 20} more corrections")
    else:
        print("  ✓ No corrections needed")
    
    # Build aggregate sources from corrected atomic sources
    for agg_source in AGGREGATE_SOURCES:
        components = AGGREGATE_DEFINITIONS[agg_source]
        
        for timestamp in target_total_gen.index:
            # Corrected aggregate - collect all countries
            corrected_by_country = {}
            for component in components:
                if timestamp in corrected_sources[component]:
                    for country, val in corrected_sources[component][timestamp].items():
                        if country not in corrected_by_country:
                            corrected_by_country[country] = 0
                        corrected_by_country[country] += val
            corrected_sources[agg_source][timestamp] = corrected_by_country
            
            # Actual (uncorrected) aggregate - collect all countries  
            actual_by_country = {}
            for component in components:
                if timestamp in actual_sources[component]:
                    for country, val in actual_sources[component][timestamp].items():
                        if country not in actual_by_country:
                            actual_by_country[country] = 0
                        actual_by_country[country] += val
            actual_sources[agg_source][timestamp] = actual_by_country
    
    # Build corrected and actual total generation
    corrected_total_gen = {}
    actual_total_gen = {}
    for timestamp in target_total_gen.index:
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
            # Collect by country
            by_country = {}
            for component in components:
                if component in atomic_sources_data and timestamp in atomic_sources_data[component]:
                    for country, val in atomic_sources_data[component][timestamp].items():
                        if country not in by_country:
                            by_country[country] = 0
                        by_country[country] += val
            aggregate_sources_data[agg_source][timestamp] = by_country
    
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


def calculate_daily_statistics(data_dict):
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
                current_time = pd.Timestamp.now(tz='Europe/Brussels')
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
                    'energy_mean': np.mean(energy_array, axis=0),
                    'energy_std': np.std(energy_array, axis=0),
                    'percentage_mean': np.mean(percentage_array, axis=0),
                    'percentage_std': np.std(percentage_array, axis=0),
                }

    return stats


def plot_analysis(stats_data, source_type, output_file_base):
    """
    Create two separate plots - percentage and absolute
    Returns tuple of (percentage_file, absolute_file)
    """
    if not stats_data:
        print("No data for plotting")
        return None, None

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
        'today_projected': '--',
        'yesterday_projected': '--'
    }

    labels = {
        'today': 'Today',
        'yesterday': 'Yesterday',
        'week_ago': 'Previous Week',
        'year_ago': 'Last Year',
        'two_years_ago': 'Two Years Ago',
        'today_projected': 'Today (Projected)',
        'yesterday_projected': 'Yesterday (Projected)'
    }

    time_labels = create_time_axis()
    
    # Calculate x-axis tick positions (every 4 hours) for both plots
    tick_positions = list(range(0, len(time_labels), 16))  # Every 4 hours (16 * 15min = 4h)
    tick_labels_axis = [time_labels[i] if i < len(time_labels) else '' for i in tick_positions]
    
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
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    
    fig1.suptitle(f'{source_name} Electricity Generation (EU)', fontsize=34, fontweight='bold', x=0.5, y=0.98, ha="center")
    ax1.set_title('Fraction of Total Generation', fontsize=26, fontweight='normal', pad=15)
    ax1.set_xlabel('Time of Day (Brussels)', fontsize=28, fontweight='bold', labelpad=15)
    ax1.set_ylabel('Electrical Power (%)', fontsize=28, fontweight='bold', labelpad=15)

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

        ax1.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=6, label=label)

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'percentage_std' in data:
            std_values = data['percentage_std'][:len(x_values)]
            upper_bound = y_values + std_values
            lower_bound = y_values - std_values
            max_percentage = max(max_percentage, np.nanmax(upper_bound))
            ax1.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax1.tick_params(axis='both', labelsize=22)
    ax1.set_ylim(0, max_percentage * 1.20 if max_percentage > 0 else 50)  # 20% headroom
    
    # Set x-axis time labels
    ax1.set_xlim(0, len(time_labels))
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels_axis)
    
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    
    # Reorder legend to show today/yesterday first, then historical
    # (even though plotting order is reversed for proper z-layering)
    handles, labels_list = ax1.get_legend_handles_labels()
    legend_order = ['today', 'today_projected', 'yesterday', 'yesterday_projected',
                    'week_ago', 'year_ago', 'two_years_ago']
    
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
              loc='upper center', bbox_to_anchor=(0.5, -0.18), 
              ncol=2, fontsize=20, frameon=False)
    
    plt.tight_layout()
    
    plt.savefig(output_file_percentage, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved percentage plot: {output_file_percentage}")
    plt.close()
    
    # ========================================================================
    # PLOT 2: ABSOLUTE
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    fig2.suptitle(f'{source_name} Electricity Generation (EU)', fontsize=34, fontweight='bold', x=0.5, y=0.98, ha="center")
    ax2.set_title('Absolute Generation', fontsize=26, fontweight='normal', pad=15)
    ax2.set_xlabel('Time of Day (Brussels)', fontsize=28, fontweight='bold', labelpad=15)
    ax2.set_ylabel('Electrical Power (GW)', fontsize=28, fontweight='bold', labelpad=15)

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

        ax2.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=6, label=label)

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'energy_std' in data:
            # Convert MW to GW for std as well
            std_values = data['energy_std'][:len(x_values)] / 1000
            upper_bound = y_values + std_values
            lower_bound = y_values - std_values
            max_energy = max(max_energy, np.nanmax(upper_bound))
            ax2.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax2.tick_params(axis='both', labelsize=22)
    ax2.set_ylim(0, max_energy * 1.20)  # 20% headroom
    
    # Set x-axis time labels
    ax2.set_xlim(0, len(time_labels))
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels_axis)
    
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    
    # Reorder legend to show today/yesterday first, then historical
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
              loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=20, frameon=False)
    
    plt.tight_layout()
    
    plt.savefig(output_file_absolute, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved absolute plot: {output_file_absolute}")
    plt.close()
    
    return output_file_percentage, output_file_absolute


def generate_plot_for_source(source_type, corrected_data, output_file_base):
    """
    Phase 3: Generate plot for a specific source from corrected data
    """
    print(f"\n" + "=" * 80)
    print(f"PHASE 3: PLOT GENERATION - {DISPLAY_NAMES[source_type].upper()}")
    print("=" * 80)
    
    # Convert corrected data to plot format
    plot_data = convert_corrected_data_to_plot_format(source_type, corrected_data)
    
    if not plot_data:
        print(f"✗ No data available for {source_type}")
        return
    
    # Calculate statistics
    stats_data = calculate_daily_statistics(plot_data)
    
    # Create plots (returns percentage and absolute files)
    percentage_file, absolute_file = plot_analysis(stats_data, source_type, output_file_base)
    
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


def update_summary_table_worksheet(corrected_data, country_code='EU'):
    """
    Update Google Sheets "Summary Table Data" worksheet with yesterday/last week data
    Uses PROJECTED (corrected) data for accuracy
    
    Args:
        corrected_data: Processed data from apply_projections_and_corrections
        country_code: Country code for the sheet (EU, DE, etc.)
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
        
        # Get or create country sheet
        spreadsheet = get_or_create_country_sheet(gc, drive_service, country_code=country_code)
        print(f"✓ Connected to spreadsheet ({country_code})")
        
        # Get or create worksheet
        try:
            worksheet = spreadsheet.worksheet('Summary Table Data')
            print("✓ Found existing 'Summary Table Data' worksheet")
            
            # Check if worksheet has enough columns (need 14: A-N)
            if worksheet.col_count < 14:
                print(f"  Expanding worksheet from {worksheet.col_count} to 14 columns...")
                worksheet.resize(rows=worksheet.row_count, cols=14)
                
                # Update header row with new columns
                headers = [
                    'Source', 
                    'Yesterday_GWh', 'Yesterday_%', 
                    'LastWeek_GWh', 'LastWeek_%',
                    'YTD2025_GWh', 'YTD2025_%',
                    'Avg2020_2024_GWh', 'Avg2020_2024_%',
                    'Last_Updated',
                    'Yesterday_Change_2015_%', 'LastWeek_Change_2015_%',
                    'YTD2025_Change_2015_%', 'Avg2020_2024_Change_2015_%'
                ]
                worksheet.update('A1:N1', [headers])
                worksheet.format('A1:N1', {'textFormat': {'bold': True}})
                print("  ✓ Worksheet expanded and header updated")
                
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title='Summary Table Data', rows=20, cols=15)
            print("✓ Created new 'Summary Table Data' worksheet")
            
            # Add headers (now includes K-N for change from 2015)
            headers = [
                'Source', 
                'Yesterday_GWh', 'Yesterday_%', 
                'LastWeek_GWh', 'LastWeek_%',
                'YTD2025_GWh', 'YTD2025_%',
                'Avg2020_2024_GWh', 'Avg2020_2024_%',
                'Last_Updated',
                'Yesterday_Change_2015_%', 'LastWeek_Change_2015_%',
                'YTD2025_Change_2015_%', 'Avg2020_2024_Change_2015_%'
            ]
            worksheet.update('A1:N1', [headers])
            worksheet.format('A1:N1', {'textFormat': {'bold': True}})
        
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
        
        # Load 2015 data for change calculation
        print("  Loading 2015 baseline data...")
        data_2015 = {}
        
        # Get yesterday's month for baseline (e.g., if yesterday was Nov 30, use November 2015)
        yesterday_date = datetime.now() - timedelta(days=1)
        baseline_month = yesterday_date.month  # e.g., 11 for November
        
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
                ws_2015 = spreadsheet.worksheet(worksheet_name)
                values = ws_2015.get_all_values()
                
                if len(values) < 2:
                    continue
                
                # Parse to find 2015 data
                df = pd.DataFrame(values[1:], columns=values[0])
                df = df[df['Month'] != 'Total']
                
                # Check if 2015 column exists
                if '2015' not in df.columns:
                    print(f"  ⚠ No 2015 data for {source}")
                    continue
                
                # Get the monthly TOTAL for the baseline month
                month_abbr = calendar.month_abbr[baseline_month]
                month_row = df[df['Month'] == month_abbr]
                
                if not month_row.empty:
                    monthly_total_2015 = pd.to_numeric(month_row['2015'].iloc[0], errors='coerce')
                    if not pd.isna(monthly_total_2015):
                        # Monthly sheets store MONTHLY TOTALS, so store as-is
                        # We'll convert to daily when comparing
                        data_2015[source] = monthly_total_2015
                        print(f"  {source}: Nov 2015 = {monthly_total_2015:.1f} GWh (monthly total)")
                    
            except Exception as e:
                print(f"  ⚠ Could not load 2015 data for {source}: {e}")
                continue
        
        # Calculate all-non-renewables from Total - Renewables
        if 'all-renewables' in data_2015:
            try:
                ws_total = spreadsheet.worksheet('Total Generation Monthly Production')
                values = ws_total.get_all_values()
                df = pd.DataFrame(values[1:], columns=values[0])
                df = df[df['Month'] != 'Total']
                
                if '2015' in df.columns:
                    month_abbr = calendar.month_abbr[baseline_month]
                    month_row = df[df['Month'] == month_abbr]
                    
                    if not month_row.empty:
                        total_2015_monthly = pd.to_numeric(month_row['2015'].iloc[0], errors='coerce')
                        if not pd.isna(total_2015_monthly):
                            # Store monthly totals
                            data_2015['all-non-renewables'] = total_2015_monthly - data_2015['all-renewables']
            except:
                pass
        
        print(f"  ✓ Loaded 2015 baseline for {len(data_2015)} sources")
        
        # Prepare data rows - ONLY columns that intraday owns
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        
        # First, update column A (Source names) if needed
        source_names = []
        for source in source_order:
            display_name = DISPLAY_NAMES.get(source, source.title())
            source_names.append([display_name])
        
        worksheet.update('A2:A13', source_names)
        
        # Now update columns B-E (Yesterday, Last Week) and K-L (Change from 2015)
        data_updates_be = []  # Columns B-E
        data_updates_kl = []  # Columns K-L
        
        for source in source_order:
            if source not in yesterday_totals or source not in week_totals:
                data_updates_be.append(['', '', '', ''])
                data_updates_kl.append(['', ''])
                continue
            
            # Columns B-E (existing)
            row_be = [
                f"{yesterday_totals[source]['gwh']:.1f}",      # B: Yesterday_GWh
                f"{yesterday_totals[source]['percentage']:.2f}",  # C: Yesterday_%
                f"{week_totals[source]['gwh']:.1f}",           # D: LastWeek_GWh
                f"{week_totals[source]['percentage']:.2f}"     # E: LastWeek_%
            ]
            data_updates_be.append(row_be)
            
            # Columns K-L (change from 2015)
            yesterday_change = ''
            lastweek_change = ''
            
            if source in data_2015 and data_2015[source] > 0:
                monthly_total_2015 = data_2015[source]  # Monthly total in GWh
                days_in_baseline_month = calendar.monthrange(2015, baseline_month)[1]
                
                # Yesterday change: monthly_total / days_in_month * 1 day
                baseline_yesterday = (monthly_total_2015 / days_in_baseline_month) * 1
                yesterday_gwh = yesterday_totals[source]['gwh']
                change_y = (yesterday_gwh - baseline_yesterday) / baseline_yesterday * 100
                yesterday_change = format_change_percentage(change_y)
                
                # Debug: print first source
                if source == source_order[0]:
                    print(f"  DEBUG {source}: yesterday={yesterday_gwh:.1f} GWh, baseline={baseline_yesterday:.1f} GWh (={monthly_total_2015:.1f}/{days_in_baseline_month}), change={change_y:.1f}%")
                
                # Last week change: monthly_total / days_in_month * 7 days
                baseline_week = (monthly_total_2015 / days_in_baseline_month) * 7
                lastweek_gwh = week_totals[source]['gwh']
                change_w = (lastweek_gwh - baseline_week) / baseline_week * 100
                lastweek_change = format_change_percentage(change_w)
            
            row_kl = [yesterday_change, lastweek_change]
            data_updates_kl.append(row_kl)
        
        # Update columns B-E (preserves F-I historical data!)
        if data_updates_be:
            worksheet.update('B2:E13', data_updates_be)
        
        # Update columns K-L (change from 2015)
        if data_updates_kl:
            worksheet.update('K2:L13', data_updates_kl)
            
            # Update timestamp in column J
            timestamp_updates = [[timestamp]] * len(source_order)
            worksheet.update('J2:J13', timestamp_updates)
            
            # Format aggregate rows (bold)
            worksheet.format('A2:N2', {'textFormat': {'bold': True}})  # All Renewables
            worksheet.format('A8:N8', {'textFormat': {'bold': True}})  # All Non-Renewables
            
            print(f"✓ Updated {len(source_order)} sources with yesterday/last week data (columns B-E, K-L)")
            print(f"   Historical data (columns F-I, M-N) preserved!")
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


def main():
    """
    Main function - orchestrates the 3 phases
    Generates ALL 12 plots by default, or single plot if --source specified
    Processes all countries by default, or single country if --country specified
    
    OPTIMIZED: Fetches all 27 EU countries ONCE, then extracts/aggregates for each country
    """
    parser = argparse.ArgumentParser(description='EU Energy Intraday Analysis v2')
    parser.add_argument('--source', 
                       choices=ATOMIC_SOURCES + AGGREGATE_SOURCES,
                       help='Optional: Generate only this source (default: all sources)')
    parser.add_argument('--country', default=None,
                       help='Country code (EU, DE, FR, etc.). If not specified, processes all countries.')
    
    args = parser.parse_args()
    
    # Define countries to process (should match data collection script)
    countries_to_process = ['EU', 'DE']
    
    # If specific country requested, only process that one
    if args.country:
        countries_to_process = [args.country.upper()]
    
    print("\n" + "=" * 80)
    print("ENERGY INTRADAY ANALYSIS - OPTIMIZED")
    print("=" * 80)
    print(f"Countries to process: {', '.join(countries_to_process)}")
    if args.source:
        print(f"Source filter: {DISPLAY_NAMES[args.source]} only")
    else:
        print("Processing: All 12 sources")
    print("=" * 80)
    
    # Get API key
    api_key = os.environ.get('ENTSOE_API_KEY')
    if not api_key:
        print("ERROR: ENTSOE_API_KEY environment variable not set!")
        sys.exit(1)
    
    try:
        # Phase 1: Collect all data ONCE (all 27 countries)
        print(f"\n⚡ OPTIMIZATION: Fetching all 27 EU countries ONCE for all {len(countries_to_process)} target countries")
        raw_data_matrix, periods, fetch_time = collect_all_data(api_key)
        
        # Process each country using the SAME raw data
        for country_code in countries_to_process:
            print(f"\n{'='*80}")
            print(f"PROCESSING {country_code} (extracting from shared data)")
            print(f"{'='*80}")
            
            try:
                # Extract/aggregate this country's data from raw data
                data_matrix = extract_country_from_raw_data(raw_data_matrix, country_code)
        
                # Phase 2: Apply projections and corrections ONCE
                corrected_data = apply_projections_and_corrections(data_matrix)
                
                # Phase 3: Generate plots
                if args.source:
                    # Single plot mode
                    print("\n" + "=" * 80)
                    print(f"PHASE 3: GENERATING {DISPLAY_NAMES[args.source].upper()} PLOTS")
                    print("=" * 80)
                    output_file_base = f'plots/{args.source.replace("-", "_")}_analysis.png'
                    percentage_file, absolute_file = generate_plot_for_source(args.source, corrected_data, output_file_base)
                    
                    # Upload both to Google Drive
                    print(f"\n📤 Uploading to Google Drive...")
                    perc_id = upload_plot_to_drive(percentage_file, country=country_code)
                    abs_id = upload_plot_to_drive(absolute_file, country=country_code)
                    if perc_id and abs_id:
                        print(f"  ✓ Uploaded both plots to {country_code}/Intraday/")
                else:
                    # Batch mode - generate all plots
                    print("\n" + "=" * 80)
                    print("PHASE 3: GENERATING ALL 12 PLOTS (24 files: percentage + absolute)")
                    print("=" * 80)
                    
                    all_sources = ATOMIC_SOURCES + AGGREGATE_SOURCES
                    drive_file_ids = {}
                    
                    for i, source in enumerate(all_sources, 1):
                        print(f"\n[{i}/{len(all_sources)}] Processing {DISPLAY_NAMES[source]}...")
                        output_file_base = f'plots/{source.replace("-", "_")}_analysis.png'
                        percentage_file, absolute_file = generate_plot_for_source(source, corrected_data, output_file_base)
                        
                        # Upload both to Google Drive
                        perc_id = upload_plot_to_drive(percentage_file, country=country_code)
                        abs_id = upload_plot_to_drive(absolute_file, country=country_code)
                        if perc_id and abs_id:
                            drive_file_ids[source] = {
                                'percentage': perc_id,
                                'absolute': abs_id
                            }
                            print(f"  ✓ Uploaded both plots to Drive: {country_code}/Intraday/")
                    
                    # Save Drive file IDs to JSON
                    if drive_file_ids:
                        print(f"\n📤 Saving Drive links for {len(drive_file_ids)} sources...")
                        print(f"   Sources: {', '.join(drive_file_ids.keys())}")
                        drive_links_file = 'plots/drive_links.json'
                        drive_links = {}
                        
                        # Load existing links
                        if os.path.exists(drive_links_file):
                            try:
                                with open(drive_links_file, 'r') as f:
                                    drive_links = json.load(f)
                            except:
                                pass
                        
                        # Update with new file IDs
                        if country_code not in drive_links:
                            drive_links[country_code] = {}
                        if 'Intraday' not in drive_links[country_code]:
                            drive_links[country_code]['Intraday'] = {}
                        
                        # Random thumbnail size to bypass mobile browser cache
                        # Rotates between 5 sizes: each new URL forces browser to fetch fresh image
                        thumbnail_size = random.choice([1998, 1999, 2000, 2001, 2002])
                        print(f"  📐 Using thumbnail size: w{thumbnail_size} (cache-busting)")
                        
                        for source, file_ids in drive_file_ids.items():
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
                            sample_source = list(drive_file_ids.keys())[0] if drive_file_ids else None
                            if sample_source:
                                if country_code in saved_data and 'Intraday' in saved_data[country_code]:
                                    if sample_source in saved_data[country_code]['Intraday']:
                                        source_data = saved_data[country_code]['Intraday'][sample_source]
                                        if 'percentage' in source_data and 'absolute' in source_data:
                                            file_size = os.path.getsize(drive_links_file)
                                            print(f"  ✓ Drive links saved to {drive_links_file}")
                                            print(f"     Full path: {drive_links_file_path}")
                                            print(f"     File size: {file_size} bytes")
                                            print(f"     ✓ Verified NEW structure (percentage/absolute)")
                                        else:
                                            print(f"  ⚠ WARNING: OLD structure detected! Missing percentage/absolute")
                                    else:
                                        print(f"  ⚠ WARNING: Source {sample_source} not in saved JSON")
                            
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
                
                # Create timestamp file
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
                with open('plots/last_update.html', 'w') as f:
                    f.write(f'<p>Last updated: {timestamp}</p>')
                
                # Phase 4: Update Summary Table in Google Sheets
                update_summary_table_worksheet(corrected_data, country_code=country_code)
                
                print(f"\n✓ {country_code} COMPLETE!")
                
            except Exception as e:
                print(f"✗ Error processing {country_code}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final summary
        print(f"\n" + "=" * 80)
        print(f"ALL COUNTRIES COMPLETE!")
        print(f"Processed: {', '.join(countries_to_process)}")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ Fatal error during data collection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
