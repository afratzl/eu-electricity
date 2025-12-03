from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import calendar
import os
import json
import sys
import argparse

# EU country codes (most countries in the EU)
eu_countries = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
    'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]

# Define energy source categories to analyze
energy_sources = {
    'Solar': ['Solar'],
    'Wind Onshore': ['Wind Onshore'],
    'Wind Offshore': ['Wind Offshore'],
    'Wind': ['Wind Onshore', 'Wind Offshore'],
    'Hydro': [
        'Hydro', 'Hydro Water Reservoir', 'Hydro Run-of-river', 'Hydro Pumped Storage',
        'Water Reservoir', 'Run-of-river', 'Poundage', 'Hydro Run-of-river and poundage'
    ],
    'Biomass': ['Biomass', 'Biogas', 'Biofuel'],
    'Geothermal': ['Geothermal'],
    'Gas': ['Fossil Gas', 'Natural Gas', 'Gas', 'Fossil Coal-derived gas'],
    'Coal': ['Fossil Hard coal', 'Fossil Brown coal', 'Fossil Brown coal/Lignite', 'Hard Coal', 'Brown Coal', 'Coal',
             'Lignite', 'Fossil Peat', 'Peat'],
    'Nuclear': ['Nuclear'],
    'Oil': ['Fossil Oil', 'Oil', 'Petroleum'],
    'Waste': ['Waste', 'Other non-renewable', 'Other'],
    'All Renewables': [
        'Solar', 'Wind Onshore', 'Wind Offshore',
        'Hydro', 'Hydro Water Reservoir', 'Hydro Run-of-river', 'Hydro Pumped Storage',
        'Water Reservoir', 'Run-of-river', 'Poundage', 'Hydro Run-of-river and poundage',
        'Geothermal', 'Biomass', 'Biogas', 'Biofuel', 'Other renewable'
    ],
    'Total Generation': 'ALL'  # Special case for total generation
}


def generate_10day_chunks(year, current_year, current_date):
    """
    Generate 10-day chunks for a year to avoid ENTSO-E's 1000-row API limit
    Returns list of (start, end, month) tuples
    
    Strategy:
    - 31-day months: 10 + 10 + 10 + 1 day chunks (4 chunks)
    - 30-day months: 10 + 10 + 10 day chunks (3 chunks)
    - 28/29-day months: 10 + 10 + 8/9 day chunks (3 chunks)
    
    For current year, only generates chunks up to current date
    """
    from datetime import datetime, timedelta
    import calendar
    
    chunks = []
    
    for month in range(1, 13):
        # Skip future months in current year
        if year == current_year and month > current_date.month:
            break
            
        days_in_month = calendar.monthrange(year, month)[1]
        
        # For current month in current year, only go up to current day
        if year == current_year and month == current_date.month:
            days_in_month = current_date.day
        
        # Generate chunks for this month
        day = 1
        while day <= days_in_month:
            chunk_start = datetime(year, month, day)
            
            # Determine chunk end
            if days_in_month - day + 1 >= 10:
                # Normal 10-day chunk
                chunk_end = datetime(year, month, min(day + 9, days_in_month))
                day += 10
            else:
                # Remaining days (1-9 days)
                chunk_end = datetime(year, month, days_in_month)
                day = days_in_month + 1
            
            chunks.append((chunk_start, chunk_end, month))
    
    return chunks


def get_all_energy_data_for_country_year(client, country, year):
    """
    Get all energy generation data for a specific country and year with single API call
    Returns structured data for all energy types
    Robust retry logic with exponential backoff
    """
    import time

    # Get current date for comparison
    current_date = datetime.now()
    current_year = current_date.year
    
    # Determine max_month for later use
    if year == current_year:
        max_month = current_date.month
    else:
        max_month = 12

    print(f"  Querying {country} for {year}...")

    # Generate 10-day chunks to avoid ENTSO-E's 1000-row API limit
    chunks = generate_10day_chunks(year, current_year, current_date)
    print(f"    Fetching {len(chunks)} chunks to avoid API row limit...")

    all_chunk_data = []

    # Fetch each chunk with retry logic
    for chunk_idx, (chunk_start, chunk_end, chunk_month) in enumerate(chunks, 1):
        max_retries = 4
        chunk_success = False
        
        for attempt in range(max_retries):
            try:
                # Convert to pandas Timestamp with timezone
                start_ts = pd.Timestamp(chunk_start, tz='Europe/Brussels')
                
                # IMPORTANT: ENTSO-E API treats end as EXCLUSIVE
                # To get data for chunk_end day, we need to request chunk_end + 1 day
                from datetime import timedelta
                end_ts = pd.Timestamp(chunk_end + timedelta(days=1), tz='Europe/Brussels')
                
                # Fetch this chunk
                chunk_data = client.query_generation(country, start=start_ts, end=end_ts)
                
                if not chunk_data.empty:
                    all_chunk_data.append(chunk_data)
                    print(f"    ✓ Chunk {chunk_idx}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()} ({chunk_data.shape[0]} rows, {chunk_data.shape[1]} types)")
                else:
                    print(f"    ⚠ Chunk {chunk_idx}/{len(chunks)}: No data for {chunk_start.date()} to {chunk_end.date()}")
                
                chunk_success = True
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"    ⚠ Chunk {chunk_idx} attempt {attempt + 1} failed: {e}")
                    print(f"      Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"    ✗ Chunk {chunk_idx} failed after {max_retries} attempts: {e}")
                    # Don't return None yet - continue with other chunks
        
        if not chunk_success:
            print(f"    ⚠ Skipping failed chunk {chunk_idx}")

    # Combine all chunks
    if not all_chunk_data:
        print(f"    ⚠ No data returned for {country} {year}")
        return None

    generation_data = pd.concat(all_chunk_data, axis=0).sort_index()
    print(f"    ✓ Combined {len(all_chunk_data)} chunks: {generation_data.shape[0]} total rows with {generation_data.shape[1]} generation types")

    # Now process the combined data (single try block for processing)
    try:
        # Calculate time differences for variable resolution handling
        if len(generation_data) > 1:
            time_diffs = generation_data.index.to_series().diff().dt.total_seconds() / 3600
            time_diffs = time_diffs.fillna(time_diffs.median())

            # Debug info for Spain in 2022 (to track the resolution issue we fixed)
            if country == 'ES' and year == 2022:
                unique_intervals = time_diffs.unique()
                print(f"      Spain 2022 time intervals: {sorted(unique_intervals)} hours")
        else:
            time_diffs = pd.Series([1.0] * len(generation_data), index=generation_data.index)

        # Initialize result structure for this country-year
        country_year_data = {}

        # Process each energy source type
        for source_name, source_keywords in energy_sources.items():
            # Find relevant columns
            if source_keywords == 'ALL':
                relevant_columns = generation_data.columns.tolist()
                energy_series = generation_data.sum(axis=1)
            else:
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
                else:
                    energy_series = pd.Series(0, index=generation_data.index)

            # Convert power to energy using actual time intervals
            energy_data = energy_series * time_diffs

            # Calculate monthly totals
            monthly_data = {month: 0 for month in range(1, 13)}
            total_gwh = 0

            for month in range(1, max_month + 1):
                month_mask = energy_data.index.month == month
                if month_mask.any():
                    monthly_sum = energy_data[month_mask].sum() / 1000  # Convert to GWh
                    monthly_data[month] = monthly_sum
                    total_gwh += monthly_sum

            # Store results for this energy source
            country_year_data[source_name] = {
                'monthly': monthly_data,
                'total': total_gwh,
                'columns_used': relevant_columns
            }

            if total_gwh > 1:  # Only print if significant production
                print(f"      {source_name}: {total_gwh:.1f} GWh ({len(relevant_columns)} columns)")

        # Add debugging for missing columns (for specific countries/years)
        debug_countries = {
            'CZ': 2024,  # Czech Republic 2024+
            'DE': 2015  # Germany 2015+
        }

        if country in debug_countries and year >= debug_countries[country]:
            print(f"      === DEBUG: All available columns for {country} {year} ===")
            for i, col in enumerate(generation_data.columns):
                print(f"        {i + 1:2d}. {col}")

            # Find unmatched columns
            all_matched_columns = set()
            for source_name, source_keywords in energy_sources.items():
                if source_keywords != 'ALL':
                    for keyword in source_keywords:
                        matching_cols = [col for col in generation_data.columns if keyword in col]
                        all_matched_columns.update(matching_cols)

            unmatched_columns = set(generation_data.columns) - all_matched_columns
            if unmatched_columns:
                print(f"      === UNMATCHED COLUMNS (potential missing sources) ===")
                unmatched_with_totals = []

                for col in unmatched_columns:
                    # Calculate total energy for this unmatched column
                    col_data = generation_data[col] * time_diffs
                    col_total = 0
                    for month in range(1, max_month + 1):
                        month_mask = col_data.index.month == month
                        if month_mask.any():
                            col_total += col_data[month_mask].sum() / 1000
                    if col_total > 10:  # Only show significant unmatched sources
                        unmatched_with_totals.append((col, col_total))

                # Sort by energy total (largest first)
                unmatched_with_totals.sort(key=lambda x: x[1], reverse=True)

                renewable_keywords = ['Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal', 'renewable']

                print(f"      --- MISSING RENEWABLES ---")
                renewable_total = 0
                for col, total in unmatched_with_totals:
                    col_str = str(col).lower()
                    if any(keyword.lower() in col_str for keyword in renewable_keywords):
                        print(f"        {col}: {total:.1f} GWh")
                        renewable_total += total
                print(f"      Total missing renewables: {renewable_total:.1f} GWh")

                print(f"      --- MISSING NON-RENEWABLES ---")
                non_renewable_total = 0
                for col, total in unmatched_with_totals:
                    col_str = str(col).lower()
                    if not any(keyword.lower() in col_str for keyword in renewable_keywords):
                        print(f"        {col}: {total:.1f} GWh")
                        non_renewable_total += total
                print(f"      Total missing non-renewables: {non_renewable_total:.1f} GWh")

            print(f"      === END DEBUG ===")

        # Success! Return the processed data
        return country_year_data

    except Exception as e:
        print(f"    ✗ Error processing data for {country} {year}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_all_countries_and_years(client, years_to_analyze):
    """
    Process all countries and years with optimized API calls
    Returns structured data: {energy_source: {year: {monthly_data}, country_data: {year: {country: total}}}}
    """
    print(f"\nProcessing {len(eu_countries)} countries for {len(years_to_analyze)} years...")
    print(
        f"Total API calls needed: {len(eu_countries) * len(years_to_analyze)} (instead of {len(eu_countries) * len(years_to_analyze) * len(energy_sources)})")

    # Initialize result structure
    all_results = {}
    for source_name in energy_sources.keys():
        all_results[source_name] = {
            'year_data': {year: {month: 0 for month in range(1, 13)} for year in years_to_analyze},
            'country_data': {year: {} for year in years_to_analyze}
        }

    # Process each country-year combination
    total_calls = len(eu_countries) * len(years_to_analyze)
    call_count = 0

    for country in eu_countries:
        print(f"\nProcessing {country}...")

        for year in years_to_analyze:
            call_count += 1
            print(f"  Progress: {call_count}/{total_calls} ({call_count / total_calls * 100:.1f}%)")

            # Single API call for all energy data
            country_year_data = get_all_energy_data_for_country_year(client, country, year)

            if country_year_data:
                # Distribute the data to appropriate structures
                for source_name in energy_sources.keys():
                    source_data = country_year_data[source_name]

                    # Add to country totals
                    all_results[source_name]['country_data'][year][country] = source_data['total']

                    # Add to EU totals (monthly aggregation)
                    for month in range(1, 13):
                        all_results[source_name]['year_data'][year][month] += source_data['monthly'][month]
            else:
                # Handle missing data
                for source_name in energy_sources.keys():
                    all_results[source_name]['country_data'][year][country] = 0

    return all_results


def save_all_data_to_google_sheets_with_merge(all_data, month_names):
    """
    Save all energy source data to Google Sheets with merge capability
    Only updates the years being processed, preserves existing data
    Uses environment variables for credentials
    """
    try:
        # Get Google credentials from environment variable
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set!")
        
        # Parse the JSON credentials
        creds_dict = json.loads(google_creds_json)
        
        import gspread
        from google.oauth2.service_account import Credentials
        
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        gc = gspread.authorize(credentials)

        try:
            spreadsheet = gc.open('EU Electricity Production Data')
            print(f"✓ Connected to existing spreadsheet")
        except gspread.SpreadsheetNotFound:
            spreadsheet = gc.create('EU Electricity Production Data')
            # Note: You may want to share with your email - but we don't hardcode emails
            print(f"✓ Created new spreadsheet")

        for source_name, source_data in all_data.items():
            year_data = source_data['year_data']

            worksheet_title = f'{source_name} Monthly Production'

            try:
                worksheet = spreadsheet.worksheet(worksheet_title)
                print(f"  Loading existing worksheet: {worksheet_title}")

                # Get existing data
                existing_values = worksheet.get_all_values()
                print(f"    Found {len(existing_values)} rows in existing sheet")

                if len(existing_values) >= 2:
                    # Parse existing headers and data
                    headers = existing_values[0]
                    existing_df = pd.DataFrame(existing_values[1:], columns=headers)
                    print(f"    Existing headers: {headers}")

                    # Remove 'Total' row if it exists for processing
                    existing_df = existing_df[existing_df['Month'] != 'Total'].copy()
                else:
                    # Create new structure
                    existing_df = pd.DataFrame({'Month': month_names})
                    headers = ['Month']
                    print(f"    No existing data, creating new structure")

            except gspread.WorksheetNotFound:
                print(f"  Creating new worksheet: {worksheet_title}")
                worksheet = spreadsheet.add_worksheet(title=worksheet_title, rows=100, cols=20)
                existing_df = pd.DataFrame({'Month': month_names})
                headers = ['Month']

            # Smart merge: Update existing years in-place, add new years in chronological order
            existing_years = set()
            for col in headers:
                if col != 'Month' and col.isdigit():
                    existing_years.add(int(col))

            processing_years = set(year_data.keys())
            new_years = processing_years - existing_years  # Years not in spreadsheet yet
            updating_years = processing_years & existing_years  # Years that exist and are being updated

            print(f"    Existing years in sheet: {sorted(existing_years, reverse=True) if existing_years else 'None'}")
            print(f"    Processing years: {sorted(processing_years, reverse=True)}")
            print(
                f"    Years being updated in-place: {sorted(updating_years, reverse=True) if updating_years else 'None'}")
            print(f"    New years to add: {sorted(new_years, reverse=True) if new_years else 'None'}")

            # Create final year list: preserve existing order, add new years at appropriate positions
            if existing_years:
                # Start with existing years in their current order (from headers)
                existing_year_order = [int(col) for col in headers if col != 'Month' and col.isdigit()]
                final_year_list = existing_year_order.copy()

                # Add new years in chronological order
                for new_year in sorted(new_years, reverse=True):
                    print(f"      Inserting year {new_year} into {final_year_list}")
                    # Insert new year in the right chronological position
                    inserted = False
                    for i, existing_year in enumerate(final_year_list):
                        if new_year > existing_year:
                            # Insert before this existing year (new_year is more recent)
                            print(f"        {new_year} > {existing_year}, inserting at position {i}")
                            final_year_list.insert(i, new_year)
                            inserted = True
                            break
                    if not inserted:
                        # New year is older than all existing years, add at the end
                        print(f"        {new_year} is oldest, adding at end")
                        final_year_list.append(new_year)
                    print(f"      Result: {final_year_list}")
            else:
                # No existing data, just use processing years
                final_year_list = sorted(processing_years, reverse=True)

            final_headers = ['Month'] + [str(year) for year in final_year_list]

            print(f"    Final column order: {[col for col in final_headers if col != 'Month']}")

            # Build the final data structure
            final_rows = [final_headers]

            for month_idx, month_name in enumerate(month_names):
                month_num = month_idx + 1
                row = [month_name]

                for year_str in final_headers[1:]:  # Skip 'Month'
                    year = int(year_str)

                    if year in processing_years:
                        # Update with new data (whether it's an existing year or new year)
                        value = year_data[year].get(month_num, 0)
                        row.append(f"{value:.2f}")
                    else:
                        # Preserve existing data for years NOT being processed
                        existing_value = "0.00"
                        if not existing_df.empty:
                            month_rows = existing_df[existing_df['Month'] == month_name]
                            if not month_rows.empty and year_str in existing_df.columns:
                                existing_value = str(month_rows[year_str].iloc[0])
                                if existing_value == '' or existing_value == 'nan' or pd.isna(existing_value):
                                    existing_value = "0.00"
                        row.append(existing_value)

                final_rows.append(row)

            # Add total row with same logic
            total_row = ['Total']
            for year_str in final_headers[1:]:
                year = int(year_str)
                if year in processing_years:
                    # Calculate total from new data
                    total = sum(year_data[year].values())
                    total_row.append(f"{total:.2f}")
                else:
                    # Preserve existing total
                    try:
                        if not existing_df.empty and year_str in existing_df.columns:
                            year_values = pd.to_numeric(existing_df[year_str], errors='coerce').fillna(0)
                            total = float(year_values.sum())
                            total_row.append(f"{total:.2f}")
                        else:
                            total_row.append("0.00")
                    except Exception as ex:
                        print(f"      Warning: Error calculating total for {year_str}: {ex}")
                        total_row.append("0.00")

            final_rows.append(total_row)

            # Clear and update the worksheet
            worksheet.clear()
            worksheet.update(final_rows)

            # Format the sheet
            worksheet.format('A1:Z1', {'textFormat': {'bold': True}})
            worksheet.format('A:A', {'textFormat': {'bold': True}})

            # Report what was updated
            new_years_str = ', '.join([str(year) for year in sorted(new_years, reverse=True)]) if new_years else 'None'
            updated_years_str = ', '.join(
                [str(year) for year in sorted(updating_years, reverse=True)]) if updating_years else 'None'
            print(f"  ✓ Updated {source_name}: Added years [{new_years_str}], Updated years [{updated_years_str}]")

        print(f"\n✓ All data saved to Google Sheets: 'EU Electricity Production Data'")
        print(f"URL: {spreadsheet.url}")
        return spreadsheet.url

    except Exception as e:
        print(f"✗ Error saving to Google Sheets: {e}")
        return None


def main():
    """
    Main function with smart update strategy
    """
    print("=" * 80)
    print("EU ENERGY DATA COLLECTION (SECURED & FUTURE-PROOF)")
    print("=" * 80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect EU energy data from ENTSO-E')
    parser.add_argument('--update-type', choices=['daily', 'full'], default=None,
                       help='Update type: daily (current year only) or full (all years 2015-present)')
    args = parser.parse_args()
    
    # Verify environment variables are set
    api_key = os.environ.get('ENTSOE_API_KEY')
    if not api_key:
        print("\n⚠️  ERROR: ENTSOE_API_KEY environment variable not set!")
        print("   Please set this variable before running the script.")
        sys.exit(1)
    
    google_creds = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    if not google_creds:
        print("\n⚠️  ERROR: GOOGLE_CREDENTIALS_JSON environment variable not set!")
        print("   Please set this variable before running the script.")
        sys.exit(1)
    
    print("✓ Environment variables verified")
    
    # Initialize the client
    client = EntsoePandasClient(api_key=api_key)
    
    # Get current date and year (future-proof)
    current_date = datetime.now()
    current_year = current_date.year
    current_day = current_date.day
    
    # Determine update strategy
    if args.update_type:
        # Command line argument overrides automatic detection
        update_type = args.update_type
        print(f"\n📋 Update type specified via argument: {update_type}")
    else:
        # Automatic detection: 1st of month = full, otherwise = daily
        if current_day == 1:
            update_type = 'full'
            print(f"\n📅 It's the 1st of the month - Running FULL refresh")
        else:
            update_type = 'daily'
            print(f"\n📅 Running DAILY update (current year only)")
    
    # Set years to analyze based on update type
    if update_type == 'full':
        years_to_analyze = range(2015, current_year + 1)
        print(f"Years to process: 2015-{current_year} ({len(list(years_to_analyze))} years)")
        print(f"Estimated API calls: ~{len(eu_countries) * len(list(years_to_analyze))}")
    else:  # daily
        years_to_analyze = [current_year]
        print(f"Years to process: {current_year} only")
        print(f"Estimated API calls: ~{len(eu_countries)}")
    
    # Define month names in order (January to December)
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
    
    # Process all data with optimized API calls
    print("\n" + "=" * 80)
    print("STARTING DATA COLLECTION")
    print("=" * 80)
    
    all_data = process_all_countries_and_years(client, years_to_analyze)
    
    # Print basic statistics
    print("\n" + "=" * 80)
    print("DATA COLLECTION SUMMARY")
    print("=" * 80)
    
    for source_name in energy_sources.keys():
        year_data = all_data[source_name]['year_data']
    
        print(f"\n{source_name.upper()}:")
        for year in sorted(year_data.keys()):
            total_production = sum(year_data[year].values())
            print(f"  {year}: {total_production:.0f} GWh ({total_production / 1000:.1f} TWh)")
    
    # Save to Google Sheets with merge capability
    print("\n" + "=" * 80)
    print("SAVING TO GOOGLE SHEETS (WITH MERGE)")
    print("=" * 80)
    
    sheet_url = save_all_data_to_google_sheets_with_merge(all_data, month_names)
    
    print(f"\n" + "=" * 80)
    print("DATA COLLECTION COMPLETE!")
    print("=" * 80)
    print(f"Update type: {update_type}")
    print(f"Processed years: {list(years_to_analyze)}")
    if sheet_url:
        print(f"Google Sheets URL: {sheet_url}")
    print("\nUse the plotting script (eu_energy_plotting.py) to generate charts from this data.")


if __name__ == "__main__":
    main()
