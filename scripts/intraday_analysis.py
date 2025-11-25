#!/usr/bin/env python3
"""
Unified Intraday Energy Analysis Script - REFACTORED
Architecture:
  Phase 1: Data Collection - Fetch all atomic sources + aggregates
  Phase 2: Projection & Correction - Apply component-level corrections
  Phase 3: Plot Generation - Create visualizations from corrected data

Key improvements:
- Weekly hourly averages for projection (not daily)
- Component-level aggregate correction
- Proper Total Generation correction using all sources
- Debug output for threshold violations
"""

from entsoe import EntsoePandasClient
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import sys
import argparse
import time

warnings.filterwarnings('ignore')

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
        interpolated = interpolated.interpolate(method='cubic', limit_area='inside')
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
    return data_matrix, periods


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
            # Corrected aggregate
            total_corrected = 0
            for component in components:
                if timestamp in corrected_sources[component]:
                    total_corrected += sum(corrected_sources[component][timestamp].values())
            corrected_sources[agg_source][timestamp] = {'EU': total_corrected}
            
            # Actual (uncorrected) aggregate
            total_actual = 0
            for component in components:
                if timestamp in actual_sources[component]:
                    total_actual += sum(actual_sources[component][timestamp].values())
            actual_sources[agg_source][timestamp] = {'EU': total_actual}
    
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


def plot_analysis(stats_data, source_type, output_file):
    """
    Create vertical plots - percentage on top, absolute below
    """
    if not stats_data:
        print("No data for plotting")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))

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
        'week_ago': 'Previous Week (avg ± std)',
        'year_ago': 'Same Period Last Year (avg ± std)',
        'two_years_ago': 'Same Period 2 Years Ago (avg ± std)',
        'today_projected': 'Today (Projected)',
        'yesterday_projected': 'Yesterday (Projected)'
    }

    time_labels = create_time_axis()
    source_name = DISPLAY_NAMES[source_type]
    
    # PLOT 1 (TOP): PERCENTAGE
    fig.suptitle(source_name, fontsize=34, fontweight='bold', x=0.5, y=0.98, ha="center")
    ax1.set_title('Percentage of EU Production', fontsize=26, fontweight='normal', pad=10)
    ax1.set_xlabel('Time of Day (Brussels)', fontsize=28, fontweight='bold', labelpad=15)
    ax1.set_ylabel('Energy production (%)', fontsize=28, fontweight='bold', labelpad=15)

    max_percentage = 0
    plot_order = ['week_ago', 'year_ago', 'two_years_ago', 'yesterday', 'today', 
                  'yesterday_projected', 'today_projected']

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

    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.tick_params(axis='both', labelsize=22)
    ax1.set_xlim(0, len(time_labels))
    ax1.set_ylim(0, max_percentage * 1.05 if max_percentage > 0 else 50)

    # PLOT 2 (BOTTOM): ABSOLUTE VALUES
    ax2.set_title('Absolute Production', fontsize=26, fontweight='normal', pad=10)
    ax2.set_xlabel('Time of Day (Brussels)', fontsize=28, fontweight='bold', labelpad=15)
    ax2.set_ylabel('Energy production (MW)', fontsize=28, fontweight='bold', labelpad=15)

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
        y_values = data['energy_mean'].copy()
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
            std_values = data['energy_std'][:len(x_values)]
            upper_bound = y_values + std_values
            lower_bound = y_values - std_values
            max_energy = max(max_energy, np.nanmax(upper_bound))
            ax2.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.tick_params(axis='both', labelsize=22)
    ax2.set_xlim(0, len(time_labels))
    ax2.set_ylim(0, max_energy * 1.05)

    # X-axis ticks
    tick_positions = np.arange(0, len(time_labels), 8)
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([time_labels[i] for i in tick_positions], rotation=45)

    # Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize=20, frameon=False, columnspacing=1.5)

    plt.tight_layout(rect=[0, 0.05, 1, 0.985])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def generate_plot_for_source(source_type, corrected_data, output_file):
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
    
    # Create plot
    plot_analysis(stats_data, source_type, output_file)
    
    print(f"✓ Plot saved to {output_file}")


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
        # Phase 1: Collect all data ONCE
        data_matrix, periods = collect_all_data(api_key)
        
        # Phase 2: Apply projections and corrections ONCE
        corrected_data = apply_projections_and_corrections(data_matrix)
        
        # Phase 3: Generate plots
        if args.source:
            # Single plot mode
            print("\n" + "=" * 80)
            print(f"PHASE 3: GENERATING {DISPLAY_NAMES[args.source].upper()} PLOT")
            print("=" * 80)
            output_file = f'plots/{args.source.replace("-", "_")}_analysis.png'
            generate_plot_for_source(args.source, corrected_data, output_file)
            print(f"\n✓ Plot saved to {output_file}")
        else:
            # Batch mode - generate all plots
            print("\n" + "=" * 80)
            print("PHASE 3: GENERATING ALL 12 PLOTS")
            print("=" * 80)
            
            all_sources = ATOMIC_SOURCES + AGGREGATE_SOURCES
            for i, source in enumerate(all_sources, 1):
                print(f"\n[{i}/{len(all_sources)}] Processing {DISPLAY_NAMES[source]}...")
                output_file = f'plots/{source.replace("-", "_")}_analysis.png'
                generate_plot_for_source(source, corrected_data, output_file)
        
        # Create timestamp file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        with open('plots/last_update.html', 'w') as f:
            f.write(f'<p>Last updated: {timestamp}</p>')
        
        print(f"\n" + "=" * 80)
        if args.source:
            print(f"✓ COMPLETE! {DISPLAY_NAMES[args.source]} plot generated")
        else:
            print(f"✓ COMPLETE! All 12 plots generated successfully")
            print(f"   - 10 atomic sources")
            print(f"   - 2 aggregates")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
