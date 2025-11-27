#!/usr/bin/env python3
"""
Generate Summary Table JSON from Google Sheets
Reads "Summary Table Data" worksheet and creates JSON for frontend display
"""

import os
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

def generate_summary_json():
    """
    Read Summary Table Data worksheet from Google Sheets and generate JSON
    """
    print("=" * 60)
    print("GENERATING SUMMARY TABLE JSON")
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
        
        # Get Summary Table Data worksheet
        try:
            worksheet = spreadsheet.worksheet('Summary Table Data')
            print("✓ Found 'Summary Table Data' worksheet")
        except gspread.WorksheetNotFound:
            print("✗ 'Summary Table Data' worksheet not found!")
            print("   Run intraday_analysis.py and eu_energy_plotting.py first")
            return False
        
        # Get all data
        all_values = worksheet.get_all_values()
        
        if len(all_values) < 2:
            print("✗ No data found in worksheet!")
            return False
        
        # Parse header row
        headers = all_values[0]
        print(f"✓ Found headers: {headers}")
        
        # Parse data rows
        data_rows = all_values[1:]
        
        # Define expected source order
        expected_sources = [
            'All Renewables',
            'Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal',
            'All Non-Renewables',
            'Gas', 'Coal', 'Nuclear', 'Oil', 'Waste'
        ]
        
        # Build JSON structure
        json_data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "sources": []
        }
        
        for row in data_rows:
            if len(row) < 10:  # Need at least 10 columns
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
            
            yesterday_gwh = safe_float(row[1])
            yesterday_pct = safe_float(row[2])
            lastweek_gwh = safe_float(row[3])
            lastweek_pct = safe_float(row[4])
            ytd2025_gwh = safe_float(row[5])
            ytd2025_pct = safe_float(row[6])
            avg2020_2024_gwh = safe_float(row[7])
            avg2020_2024_pct = safe_float(row[8])
            last_updated = row[9] if len(row) > 9 else ""
            
            # Convert GWh to TWh for better readability
            yesterday_twh = yesterday_gwh / 1000
            lastweek_twh = lastweek_gwh / 1000
            ytd2025_twh = ytd2025_gwh / 1000
            avg2020_2024_twh = avg2020_2024_gwh / 1000
            
            source_data = {
                "source": source_name,
                "category": category,
                "yesterday": {
                    "gwh": round(yesterday_gwh, 1),
                    "twh": round(yesterday_twh, 2),
                    "percentage": round(yesterday_pct, 2)
                },
                "last_week": {
                    "gwh": round(lastweek_gwh, 1),
                    "twh": round(lastweek_twh, 2),
                    "percentage": round(lastweek_pct, 2)
                },
                "ytd_2025": {
                    "gwh": round(ytd2025_gwh, 1),
                    "twh": round(ytd2025_twh, 2),
                    "percentage": round(ytd2025_pct, 2)
                },
                "avg_2020_2024": {
                    "gwh": round(avg2020_2024_gwh, 1),
                    "twh": round(avg2020_2024_twh, 2),
                    "percentage": round(avg2020_2024_pct, 2)
                }
            }
            
            json_data["sources"].append(source_data)
        
        # Write JSON file
        output_path = 'plots/energy_summary_table.json'
        os.makedirs('plots', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n✓ Generated JSON with {len(json_data['sources'])} sources")
        print(f"✓ Output: {output_path}")
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
