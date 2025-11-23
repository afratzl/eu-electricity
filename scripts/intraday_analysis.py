#!/usr/bin/env python3
"""
Unified Intraday Energy Analysis Script
Based on the working wind_analysis.py structure
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

# Energy source keyword mapping (EXACT from data collection script)
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

            time.sleep(0.2)  # Rate limiting
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
    Extract energy source data using EXACT logic from working script
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
    Interpolate to 15-minute resolution (EXACT from working script)
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

            elif data_type == 'load':
                if isinstance(country_data, pd.Series):
                    country_series = country_data
                elif isinstance(country_data, pd.DataFrame) and len(country_data.columns) == 1:
                    country_series = country_data.iloc[:, 0]
                else:
                    country_series = country_data.sum(axis=1)

                country_series.name = country
                interpolated = interpolate_country_data(country_series, country, mark_extrapolated=mark_extrapolated)

                if interpolated is not None:
                    all_interpolated_data.append(interpolated)
                    successful_countries.append(country)

    if not all_interpolated_data:
        return pd.DataFrame(), pd.DataFrame(), []

    combined_df = pd.concat(all_interpolated_data, axis=1)
    eu_total = combined_df.sum(axis=1, skipna=True)

    return eu_total, combined_df, successful_countries


def load_intraday_data(source_type, api_key):
    """
    Load 15-minute data from ENTSO-E (EXACT structure from working script)
    """
    client = EntsoePandasClient(api_key=api_key)
    source_keywords = SOURCE_KEYWORDS[source_type]
    
    print("=" * 60)
    print(f"LOADING {DISPLAY_NAMES[source_type].upper()} DATA")
    print("=" * 60)

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

    all_data = {}
    all_country_data = {}

    for period_name, (start_date, end_date) in periods.items():
        print(f"\n{period_name.upper()}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        mark_extrap = (period_name in ['today', 'yesterday'])

        energy_data, energy_by_country, energy_countries = aggregate_eu_data(
            EU_COUNTRIES, start_date, end_date, client, source_keywords, 'generation', mark_extrapolated=mark_extrap
        )
        load_data, load_by_country, load_countries = aggregate_eu_data(
            EU_COUNTRIES, start_date, end_date, client, source_keywords, 'load', mark_extrapolated=mark_extrap
        )

        if not energy_data.empty and not load_data.empty:
            common_times = energy_data.index.intersection(load_data.index)

            if len(common_times) > 0:
                period_df = pd.DataFrame({
                    'timestamp': common_times,
                    'energy_production': energy_data.loc[common_times].values,
                    'total_load': load_data.loc[common_times].values
                })

                period_df = period_df[
                    (period_df['timestamp'] >= pd.Timestamp(start_date, tz='Europe/Brussels')) &
                    (period_df['timestamp'] < pd.Timestamp(end_date, tz='Europe/Brussels'))
                ]

                period_df['energy_percentage'] = np.clip(
                    (period_df['energy_production'] / period_df['total_load']) * 100, 0, 100
                )

                period_df['date'] = period_df['timestamp'].dt.strftime('%Y-%m-%d')
                period_df['time'] = period_df['timestamp'].dt.strftime('%H:%M')

                all_data[period_name] = period_df

                if not energy_by_country.empty:
                    all_country_data[period_name] = energy_by_country.loc[period_df['timestamp']]
                if not load_by_country.empty:
                    all_country_data[f'{period_name}_load'] = load_by_country.loc[period_df['timestamp']]

                avg_pct = period_df['energy_percentage'].mean()
                print(f"  ✓ {len(period_df)} points, avg {DISPLAY_NAMES[source_type]}: {avg_pct:.1f}%")

    # Create projected values
    if 'week_ago' in all_country_data and 'today' in all_country_data and 'today' in all_data:
        if 'today_load' in all_country_data and 'week_ago_load' in all_country_data:
            print("\nCreating projected values for TODAY...")
            all_data['today_projected'] = create_projected_data(
                all_data['today'],
                all_country_data['today'], all_country_data['week_ago'],
                all_country_data['today_load'], all_country_data['week_ago_load'],
                'today'
            )

    if 'week_ago' in all_country_data and 'yesterday' in all_country_data and 'yesterday' in all_data:
        if 'yesterday_load' in all_country_data and 'week_ago_load' in all_country_data:
            print("Creating projected values for YESTERDAY...")
            all_data['yesterday_projected'] = create_projected_data(
                all_data['yesterday'],
                all_country_data['yesterday'], all_country_data['week_ago'],
                all_country_data['yesterday_load'], all_country_data['week_ago_load'],
                'yesterday'
            )

    return all_data


def create_projected_data(actual_df, actual_energy_countries, week_ago_energy_countries,
                          actual_load_countries, week_ago_load_countries, period_name='today'):
    """
    Create projected values (EXACT from working script)
    """
    print(f"  Creating projected data for {period_name}...")

    week_ago_energy_list = set(week_ago_energy_countries.columns)
    today_energy_list = set(actual_energy_countries.columns)

    week_ago_load_list = set(week_ago_load_countries.columns)
    today_load_list = set(actual_load_countries.columns)

    completely_missing_energy = week_ago_energy_list - today_energy_list
    completely_missing_load = week_ago_load_list - today_load_list

    week_ago_energy_with_time = week_ago_energy_countries.copy()
    week_ago_energy_with_time['time'] = week_ago_energy_with_time.index.strftime('%H:%M')
    week_ago_energy_avg = week_ago_energy_with_time.groupby('time').mean(numeric_only=True)

    week_ago_load_with_time = week_ago_load_countries.copy()
    week_ago_load_with_time['time'] = week_ago_load_with_time.index.strftime('%H:%M')
    week_ago_load_avg = week_ago_load_with_time.groupby('time').mean(numeric_only=True)

    projected_df = actual_df.copy()
    missing_by_time = {}

    for i in range(len(actual_df)):
        time_str = actual_df.iloc[i]['time']
        timestamp = actual_df.iloc[i]['timestamp']

        missing_countries = []
        projected_energy_total = 0
        projected_load_total = 0
        has_missing_data = False

        if timestamp in actual_energy_countries.index:
            actual_energy_row = actual_energy_countries.loc[timestamp]

            for country in today_energy_list:
                actual_val = actual_energy_row[country]

                if pd.isna(actual_val) or actual_val < 1:
                    missing_countries.append(country)
                    has_missing_data = True
                    if time_str in week_ago_energy_avg.index and country in week_ago_energy_avg.columns:
                        proj_val = week_ago_energy_avg.loc[time_str, country]
                        if not pd.isna(proj_val):
                            projected_energy_total += proj_val
                else:
                    projected_energy_total += actual_val

        for country in completely_missing_energy:
            if country not in missing_countries:
                missing_countries.append(country)
            has_missing_data = True
            if time_str in week_ago_energy_avg.index and country in week_ago_energy_avg.columns:
                proj_val = week_ago_energy_avg.loc[time_str, country]
                if not pd.isna(proj_val):
                    projected_energy_total += proj_val

        if timestamp in actual_load_countries.index:
            actual_load_row = actual_load_countries.loc[timestamp]

            for country in today_load_list:
                actual_val = actual_load_row[country]

                if pd.isna(actual_val) or actual_val < 1:
                    has_missing_data = True
                    if time_str in week_ago_load_avg.index and country in week_ago_load_avg.columns:
                        proj_val = week_ago_load_avg.loc[time_str, country]
                        if not pd.isna(proj_val):
                            projected_load_total += proj_val
                else:
                    projected_load_total += actual_val

        for country in completely_missing_load:
            has_missing_data = True
            if time_str in week_ago_load_avg.index and country in week_ago_load_avg.columns:
                proj_val = week_ago_load_avg.loc[time_str, country]
                if not pd.isna(proj_val):
                    projected_load_total += proj_val

        if has_missing_data:
            if missing_countries:
                missing_by_time[time_str] = sorted(set(missing_countries))

            projected_df.iloc[i, projected_df.columns.get_loc('energy_production')] = projected_energy_total
            projected_df.iloc[i, projected_df.columns.get_loc('total_load')] = projected_load_total

            if projected_load_total > 0:
                projected_df.iloc[i, projected_df.columns.get_loc('energy_percentage')] = np.clip(
                    (projected_energy_total / projected_load_total) * 100, 0, 100
                )

    if missing_by_time:
        all_missing = set()
        for countries in missing_by_time.values():
            all_missing.update(countries)
        print(f"  Missing: {', '.join(sorted(all_missing))}")

    return projected_df


def calculate_daily_statistics(data_dict):
    """
    Calculate daily statistics (EXACT from working script)
    """
    standard_times = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            standard_times.append(f"{hour:02d}:{minute:02d}")

    stats = {}

    for period_name, df in data_dict.items():
        if len(df) == 0:
            continue

        if period_name in ['today', 'yesterday', 'today_projected', 'yesterday_projected']:
            time_indexed = df.groupby('time')[['energy_production', 'total_load', 'energy_percentage']].mean()

            aligned_energy = time_indexed['energy_production'].reindex(standard_times)
            aligned_load = time_indexed['total_load'].reindex(standard_times)
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

                aligned_energy.iloc[:cutoff_idx] = aligned_energy.iloc[:cutoff_idx].interpolate()
                aligned_load.iloc[:cutoff_idx] = aligned_load.iloc[:cutoff_idx].interpolate()
                aligned_percentage.iloc[:cutoff_idx] = aligned_percentage.iloc[:cutoff_idx].interpolate()

                aligned_energy.iloc[cutoff_idx:] = np.nan
                aligned_load.iloc[cutoff_idx:] = np.nan
                aligned_percentage.iloc[cutoff_idx:] = np.nan
            else:
                aligned_energy = aligned_energy.interpolate()
                aligned_load = aligned_load.interpolate()
                aligned_percentage = aligned_percentage.interpolate()

            aligned_energy = aligned_energy.fillna(0.1)
            aligned_load = aligned_load.fillna(method='ffill').fillna(method='bfill').fillna(50000)
            aligned_percentage = aligned_percentage.fillna(0)

            stats[period_name] = {
                'time_bins': standard_times,
                'energy_mean': aligned_energy.values,
                'energy_std': np.zeros(len(standard_times)),
                'percentage_mean': aligned_percentage.values,
                'percentage_std': np.zeros(len(standard_times)),
            }

        else:
            unique_dates = df['date'].unique()

            daily_energy_data = []
            daily_percentage_data = []

            for date in unique_dates:
                day_data = df[df['date'] == date]

                if len(day_data) > 0:
                    time_indexed = day_data.set_index('time')
                    time_indexed = time_indexed[['energy_production', 'energy_percentage']].groupby(
                        time_indexed.index).mean()

                    aligned_energy = time_indexed['energy_production'].reindex(standard_times).interpolate()
                    aligned_percentage = time_indexed['energy_percentage'].reindex(standard_times).interpolate()

                    aligned_energy = aligned_energy.fillna(0.1)
                    aligned_percentage = aligned_percentage.fillna(0)

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


def create_time_axis():
    """
    Create time axis for 15-minute bins
    """
    times = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            times.append(f"{hour:02d}:{minute:02d}")
    return times


def plot_analysis(stats_data, source_type, output_file):
    """
    Create plots - Separate plots stacked vertically with larger fonts
    """
    if not stats_data:
        print("No data for plotting")
        return None

    print("\nCreating plots...")

    # Create 2 separate figures for better mobile viewing
    source_name = DISPLAY_NAMES[source_type]
    
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
    plot_order = ['week_ago', 'year_ago', 'two_years_ago', 'yesterday', 'today', 
                  'yesterday_projected', 'today_projected']

    # FIGURE 1: Percentage (top plot for mobile)
    fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
    
    ax1.set_title(f'EU {source_name} as Percentage of Total EU Production',
                  fontsize=24, fontweight='bold', pad=20)
    ax1.set_xlabel('Time of Day (Brussels)', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=20, fontweight='bold')

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

        ax1.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=4, label=label)

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'percentage_std' in data:
            std_values = data['percentage_std']
            if period_name == 'today':
                std_values = std_values[mask] if np.any(mask) else std_values

            upper_bound = y_values + std_values[:len(x_values)]
            lower_bound = y_values - std_values[:len(x_values)]
            max_percentage = max(max_percentage, np.nanmax(upper_bound))

            ax1.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.tick_params(axis='both', labelsize=18, width=2, length=8)
    ax1.set_xlim(0, len(time_labels))
    
    if max_percentage > 0:
        ax1.set_ylim(0, max_percentage * 1.05)
    else:
        ax1.set_ylim(0, 50)

    tick_positions = np.arange(0, len(time_labels), 8)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([time_labels[i] for i in tick_positions], rotation=45, ha='right')

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='upper left', fontsize=16, frameon=True, 
               fancybox=True, shadow=True)

    plt.tight_layout()
    
    percentage_file = output_file.replace('_analysis.png', '_percentage.png')
    plt.savefig(percentage_file, dpi=150, bbox_inches='tight')
    print(f"Saved percentage plot to {percentage_file}")
    plt.close()

    # FIGURE 2: Absolute values (bottom plot for mobile)
    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 10))
    
    ax2.set_title(f'EU {source_name} Energy Production',
                  fontsize=24, fontweight='bold', pad=20)
    ax2.set_ylabel('Energy Production (MW)', fontsize=20, fontweight='bold')
    ax2.set_xlabel('Time of Day (Brussels)', fontsize=20, fontweight='bold')

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

        ax2.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=4, label=label)

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'energy_std' in data:
            std_values = data['energy_std']
            if period_name == 'today':
                std_values = std_values[mask] if np.any(mask) else std_values

            upper_bound = y_values + std_values[:len(x_values)]
            lower_bound = y_values - std_values[:len(x_values)]
            max_energy = max(max_energy, np.nanmax(upper_bound))

            ax2.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.tick_params(axis='both', labelsize=18, width=2, length=8)
    ax2.set_xlim(0, len(time_labels))
    ax2.set_ylim(0, max_energy * 1.05)

    tick_positions = np.arange(0, len(time_labels), 8)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([time_labels[i] for i in tick_positions], rotation=45, ha='right')

    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2, loc='upper left', fontsize=16, frameon=True,
               fancybox=True, shadow=True)

    plt.tight_layout()

    absolute_file = output_file.replace('_analysis.png', '_absolute.png')
    plt.savefig(absolute_file, dpi=150, bbox_inches='tight')
    print(f"Saved absolute plot to {absolute_file}")
    plt.close()

    return percentage_file, absolute_file


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='EU Energy Intraday Analysis')
    parser.add_argument('--source', required=True, 
                       choices=list(SOURCE_KEYWORDS.keys()),
                       help='Energy source to analyze')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"{DISPLAY_NAMES[args.source].upper()} INTRADAY ANALYSIS")
    print("=" * 60)
    
    # Get API key
    api_key = os.environ.get('ENTSOE_API_KEY')
    if not api_key:
        print("ERROR: ENTSOE_API_KEY environment variable not set!")
        sys.exit(1)
    
    try:
        # Load data
        raw_data = load_intraday_data(args.source, api_key)
        
        if not raw_data:
            print(f"✗ No data")
            sys.exit(1)
        
        # Calculate statistics
        stats_data = calculate_daily_statistics(raw_data)
        
        # Create plots (now returns 2 files)
        percentage_file, absolute_file = plot_analysis(stats_data, args.source, 
                                                       f'plots/{args.source.replace("-", "_")}_analysis.png')
        
        # Create timestamp file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        with open('plots/last_update.html', 'w') as f:
            f.write(f'<p>Last updated: {timestamp}</p>')
        
        print(f"\n✓ COMPLETE!")
        print(f"  Percentage plot: {percentage_file}")
        print(f"  Absolute plot: {absolute_file}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
