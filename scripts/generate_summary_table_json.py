#!/usr/bin/env python3
"""
Generate Summary Table JSON from Google Sheets (Multi-Country)
Keeps original code structure, just wraps in country loop
"""

import os
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# Countries to process
COUNTRIES = ['EU', 'DE']

def generate_summary_json():
    """
    Read Summary Table Data worksheets from Google Sheets and generate nested JSON
    """
    print("=" * 60)
    print("GENERATING SUMMARY TABLE JSON (MULTI-COUNTRY)")
    print("=" * 60)
    
    try:
        # Get credentials
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            print("✗ GOOGLE_CREDENTIALS_JSON environment variable not set!")
            return False
        
        creds_dict = json.loads(google_creds_json)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        gc = gspread.authorize(credentials)
        
        # Open spreadsheet
        spreadsheet = gc.open('EU Electricity Production Data')
        print(f"✓ Connected to spreadsheet: {spreadsheet.url}")
        
        # Build nested JSON structure - ONE ENTRY PER COUNTRY
        final_json = {}
        
        # Process each country
        for country_code in COUNTRIES:
            print(f"\n--- Processing {country_code} ---")
            
            # Try to find worksheet for this country
            worksheet = None
            worksheet_name = f'Summary Table Data - {country_code}'
            
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
                print(f"✓ Found '{worksheet_name}' worksheet")
            except gspread.WorksheetNotFound:
                # Backward compatibility: try old format for EU
                if country_code == 'EU':
                    try:
                        worksheet_name = 'Summary Table Data'
                        worksheet = spreadsheet.worksheet(worksheet_name)
                        print(f"✓ Found '{worksheet_name}' worksheet (using for EU)")
                    except gspread.WorksheetNotFound:
                        pass
            
            # If no worksheet found, add placeholder
            if not worksheet:
                print(f"✗ No worksheet found for {country_code}")
                final_json[country_code] = {
                    "last_updated": "Data not available",
                    "sources": []
                }
                continue
            
            # Get all data from worksheet
            all_values = worksheet.get_all_values()
            
            if len(all_values) < 2:
                print(f"✗ No data found in worksheet!")
                final_json[country_code] = {
                    "last_updated": "No data available",
                    "sources": []
                }
                continue
            
            # Parse header row
            headers = all_values[0]
            print(f"✓ Found headers: {headers[:5]}...")
            
            # Parse data rows
            data_rows = all_values[1:]
            
            # Define expected source order
            expected_sources = [
                'All Renewables',
                'Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal',
                'All Non-Renewables',
                'Gas', 'Coal', 'Nuclear', 'Oil', 'Waste'
            ]
            
            # Build sources list for THIS COUNTRY
            sources_list = []
            
            for row in data_rows:
                if len(row) < 14:  # Need at least 14 columns (A-N)
                    continue
                
                source_name = row[0]
                
                if not source_name or source_name not in expected_sources:
                    continue
                
                # Determine category
                if source_name == 'All Renewables':
                    category = 'aggregate-renewable'
                elif source_name == 'All Non-Renewables':
                    category = 'aggregate-non-renewable'
                elif source_name in ['Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal']:
                    category = 'renewable'
                else:
                    category = 'non-renewable'
                
                # Parse values (handle empty strings)
                def safe_float(val):
                    try:
                        return float(val) if val else 0.0
                    except ValueError:
                        return 0.0
                
                def safe_string(val):
                    """Return string value or empty string"""
                    return str(val) if val else ""
                
                yesterday_gwh = safe_float(row[1])
                yesterday_pct = safe_float(row[2])
                lastweek_gwh = safe_float(row[3])
                lastweek_pct = safe_float(row[4])
                ytd2025_gwh = safe_float(row[5])
                ytd2025_pct = safe_float(row[6])
                year2024_gwh = safe_float(row[7])
                year2024_pct = safe_float(row[8])
                
                # Change from 2015 (columns K-N)
                yesterday_change = safe_string(row[10]) if len(row) > 10 else ""
                lastweek_change = safe_string(row[11]) if len(row) > 11 else ""
                ytd2025_change = safe_string(row[12]) if len(row) > 12 else ""
                year2024_change = safe_string(row[13]) if len(row) > 13 else ""
                
                # Change from 2024 (columns O-R)
                yesterday_change_2024 = safe_string(row[14]) if len(row) > 14 else ""
                lastweek_change_2024 = safe_string(row[15]) if len(row) > 15 else ""
                ytd2025_change_2024 = safe_string(row[16]) if len(row) > 16 else ""
                year2024_change_2024 = safe_string(row[17]) if len(row) > 17 else ""
                
                # Change from 2023 (columns S-V)
                yesterday_change_2023 = safe_string(row[18]) if len(row) > 18 else ""
                lastweek_change_2023 = safe_string(row[19]) if len(row) > 19 else ""
                ytd2025_change_2023 = safe_string(row[20]) if len(row) > 20 else ""
                year2024_change_2023 = safe_string(row[21]) if len(row) > 21 else ""
                
                # Convert GWh to TWh
                yesterday_twh = yesterday_gwh / 1000
                lastweek_twh = lastweek_gwh / 1000
                ytd2025_twh = ytd2025_gwh / 1000
                year2024_twh = year2024_gwh / 1000
                
                source_data = {
                    "source": source_name,
                    "category": category,
                    "yesterday": {
                        "gwh": round(yesterday_gwh, 1),
                        "twh": round(yesterday_twh, 2),
                        "percentage": round(yesterday_pct, 2),
                        "change_from_2015": yesterday_change,
                        "change_from_2024": yesterday_change_2024,
                        "change_from_2023": yesterday_change_2023
                    },
                    "last_week": {
                        "gwh": round(lastweek_gwh, 1),
                        "twh": round(lastweek_twh, 2),
                        "percentage": round(lastweek_pct, 2),
                        "change_from_2015": lastweek_change,
                        "change_from_2024": lastweek_change_2024,
                        "change_from_2023": lastweek_change_2023
                    },
                    "ytd_2025": {
                        "gwh": round(ytd2025_gwh, 1),
                        "twh": round(ytd2025_twh, 2),
                        "percentage": round(ytd2025_pct, 2),
                        "change_from_2015": ytd2025_change,
                        "change_from_2024": ytd2025_change_2024,
                        "change_from_2023": ytd2025_change_2023
                    },
                    "year_2024": {
                        "gwh": round(year2024_gwh, 1),
                        "twh": round(year2024_twh, 2),
                        "percentage": round(year2024_pct, 2),
                        "change_from_2015": year2024_change,
                        "change_from_2024": year2024_change_2024,
                        "change_from_2023": year2024_change_2023
                    }
                }
                
                sources_list.append(source_data)
            
            # Define source order by contribution
            source_order_map = {
                'All Renewables': 0,
                'Wind': 1,
                'Hydro': 2,
                'Solar': 3,
                'Biomass': 4,
                'Geothermal': 5,
                'All Non-Renewables': 6,
                'Nuclear': 7,
                'Gas': 8,
                'Coal': 9,
                'Waste': 10,
                'Oil': 11
            }
            
            # Sort sources
            sources_list.sort(key=lambda x: source_order_map.get(x["source"], 999))
            
            # Add to final JSON
            final_json[country_code] = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "sources": sources_list
            }
            
            print(f"✓ Parsed {len(sources_list)} sources for {country_code}")
        
        # Write JSON file
        output_path = 'plots/energy_summary_table.json'
        os.makedirs('plots', exist_ok=True)
        
        # Write to temporary file first
        temp_path = output_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(final_json, f, indent=2)
        
        # Validate the JSON by reading it back
        try:
            with open(temp_path, 'r') as f:
                content = f.read()
                # Check for git conflict markers
                if '<<<<<<< ' in content or '=======' in content or '>>>>>>> ' in content:
                    raise ValueError("Git conflict markers detected in JSON file!")
                # Validate it's valid JSON
                validated_data = json.loads(content)
                
                # Extra validation: check we have the data we expect
                for country in COUNTRIES:
                    if country in validated_data:
                        sources_count = len(validated_data[country].get('sources', []))
                        print(f"  Validation: {country} has {sources_count} sources in temp file")
            
            # If validation passes, move temp file to final location
            os.replace(temp_path, output_path)
            print(f"✓ Moved temp file to final location: {output_path}")
            
        except Exception as e:
            print(f"ERROR: JSON validation failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        
        print(f"\n✓ Generated and validated JSON for {len(final_json)} countries")
        print(f"✓ Output: {output_path}")
        
        # Print summary
        print(f"\nCountry Summary:")
        for country, data in final_json.items():
            print(f"  {country}: {len(data['sources'])} sources, updated {data['last_updated']}")
        
        print(f"\n" + "=" * 60)
        print("COMPLETE!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error generating JSON: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_summary_json()
    exit(0 if success else 1)
