import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
from datetime import datetime
import os
import json

# ENTSO-E COLOR PALETTE
ENTSOE_COLORS = {
    # Renewables
    'Solar': '#FFD700',  # Gold
    'Wind': '#228B22',  # Forest Green
    'Wind Onshore': '#2E8B57',  # Sea Green
    'Wind Offshore': '#008B8B',  # Dark Cyan
    'Hydro': '#1E90FF',  # Dodger Blue
    'Biomass': '#9ACD32',  # Yellow Green
    'Geothermal': '#708090',  # Slate Gray

    # Non-renewables
    'Gas': '#FF1493',  # Deep Pink
    'Coal': '#8B008B',  # Dark Magenta
    'Nuclear': '#8B4513',  # Saddle Brown
    'Oil': '#191970',  # Midnight Blue
    'Waste': '#808000',  # Olive

    # Totals
    'All Renewables': '#00CED1',  # Dark Turquoise
    'All Non-Renewables': '#000000'  # Black
}


def load_data_from_google_sheets():
    """
    Load all energy data from Google Sheets using environment variables
    """
    try:
        # Get Google credentials from environment variable
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set!")
        
        # Parse the JSON credentials
        creds_dict = json.loads(google_creds_json)
        
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        gc = gspread.authorize(credentials)

        spreadsheet = gc.open('EU Energy Production Data')
        print(f"✓ Connected to Google Sheets: {spreadsheet.url}")

        worksheets = spreadsheet.worksheets()
        print(f"✓ Found {len(worksheets)} worksheets")

        all_data = {}

        for worksheet in worksheets:
            sheet_name = worksheet.title

            if 'Monthly Production' not in sheet_name:
                continue

            source_name = sheet_name.replace(' Monthly Production', '')
            print(f"  Loading {source_name} data...")

            values = worksheet.get_all_values()

            if len(values) < 2:
                print(f"    ⚠ No data found in {sheet_name}")
                continue

            df = pd.DataFrame(values[1:], columns=values[0])
            df = df[df['Month'] != 'Total']

            year_columns = [col for col in df.columns if col != 'Month' and col.isdigit()]
            for col in year_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            year_data = {}
            for year_str in year_columns:
                year = int(year_str)
                year_data[year] = {}

                for idx, row in df.iterrows():
                    month_name = row['Month']
                    try:
                        month_num = list(calendar.month_abbr).index(month_name)
                        year_data[year][month_num] = float(row[year_str])
                    except (ValueError, KeyError):
                        continue

            all_data[source_name] = {'year_data': year_data}

            print(f"    ✓ Loaded {len(year_columns)} years of data for {source_name}")

        print(f"\n✓ Successfully loaded data for {len(all_data)} energy sources")
        return all_data

    except Exception as e:
        print(f"✗ Error loading from Google Sheets: {e}")
        return None


def create_all_charts(all_data):
    """
    Create all charts from the loaded data - MOBILE OPTIMIZED
    """
    if not all_data:
        print("No data available for plotting")
        return

    print("\n" + "=" * 60)
    print("CREATING MOBILE-OPTIMIZED CHARTS")
    print("=" * 60)

    first_source = list(all_data.keys())[0]
    years_available = sorted(all_data[first_source]['year_data'].keys())
    print(f"Years available: {years_available}")

    month_names = [calendar.month_abbr[i] for i in range(1, 13)]

    # Color gradient for years
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    n_years = 20  # Buffer for future years

    cmap = LinearSegmentedColormap.from_list('distinct_gradient',
                                             ['#006400', '#228B22', '#00CED1', '#00BFFF',
                                              '#0000FF', '#4B0082', '#8B008B', '#FF00FF',
                                              '#FF1493', '#DC143C', '#FF0000', '#B22222'])
    year_colors = [mcolors.rgb2hex(cmap(i / (n_years - 1))) for i in range(n_years)]

    # Calculate Non-Renewables
    print("\n" + "=" * 60)
    print("CALCULATING NON-RENEWABLES")
    print("=" * 60)

    if 'All Renewables' in all_data and 'Total Generation' in all_data:
        print(f"  Creating All Non-Renewables...")

        renewables_data = all_data['All Renewables']['year_data']
        total_data = all_data['Total Generation']['year_data']

        overlapping_years = set(renewables_data.keys()) & set(total_data.keys())

        all_non_renewables_data = {'year_data': {}}

        for year in overlapping_years:
            all_non_renewables_data['year_data'][year] = {}

            for month in range(1, 13):
                total_gen = total_data[year].get(month, 0)
                renewables_gen = renewables_data[year].get(month, 0)
                non_renewables_gen = max(0, total_gen - renewables_gen)
                all_non_renewables_data['year_data'][year][month] = non_renewables_gen

        all_data['All Non-Renewables'] = all_non_renewables_data
        print(f"  ✓ All Non-Renewables calculated")

    # Individual sources for plotting
    individual_sources = [
        'Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal',
        'Gas', 'Coal', 'Nuclear', 'Oil', 'Waste'
    ]
    
    total_sources = ['All Renewables', 'All Non-Renewables']
    
    sources_to_plot = individual_sources + total_sources

    # Calculate max values
    max_abs_individual = 0
    max_pct_individual = 0
    max_abs_totals = 0
    max_pct_totals = 0

    for source_name in individual_sources:
        if source_name in all_data and 'Total Generation' in all_data:
            year_data = all_data[source_name]['year_data']
            total_data = all_data['Total Generation']['year_data']

            for year in years_available:
                if year in year_data:
                    monthly_data = year_data[year]
                    for month in range(1, 13):
                        val = monthly_data.get(month, 0)
                        max_abs_individual = max(max_abs_individual, val / 1000)

                        if year in total_data:
                            total_val = total_data[year].get(month, 0)
                            if total_val > 0:
                                pct = (val / total_val) * 100
                                max_pct_individual = max(max_pct_individual, pct)

    for source_name in total_sources:
        if source_name in all_data and 'Total Generation' in all_data:
            year_data = all_data[source_name]['year_data']
            total_data = all_data['Total Generation']['year_data']

            for year in years_available:
                if year in year_data:
                    monthly_data = year_data[year]
                    for month in range(1, 13):
                        val = monthly_data.get(month, 0)
                        max_abs_totals = max(max_abs_totals, val / 1000)

                        if year in total_data:
                            total_val = total_data[year].get(month, 0)
                            if total_val > 0:
                                pct = (val / total_val) * 100
                                max_pct_totals = max(max_pct_totals, pct)

    # Set limits with margin
    max_abs_individual = max(100, max_abs_individual * 1.1)
    max_pct_individual = max(35, max_pct_individual * 1.1)
    max_abs_totals = max_abs_totals * 1.1
    max_pct_totals = max_pct_totals * 1.1
    
    print(f"\nY-axis limits:")
    print(f"  Individual: {max_abs_individual:.1f} TWh, {max_pct_individual:.1f}%")
    print(f"  Totals: {max_abs_totals:.1f} TWh, {max_pct_totals:.1f}%")

    # Create plots for each source
    print("\n" + "=" * 60)
    print("CREATING INDIVIDUAL SOURCE PLOTS")
    print("=" * 60)

    for source_name in sources_to_plot:
        if source_name not in all_data or 'Total Generation' not in all_data:
            print(f"  ⚠ Skipping {source_name}")
            continue

        print(f"\nCreating plots for {source_name}...")

        year_data = all_data[source_name]['year_data']
        total_data = all_data['Total Generation']['year_data']

        # PLOT 1: Percentage
        fig1, ax1 = plt.subplots(figsize=(10, 8))

        for i, year in enumerate(years_available):
            if year not in year_data:
                continue

            monthly_data = year_data[year]
            current_date = datetime.now()
            current_year = current_date.year

            if year == current_year:
                months_to_show = range(1, current_date.month + 1)
            else:
                months_to_show = range(1, 13)

            months = [month_names[month - 1] for month in months_to_show]
            values_gwh = [monthly_data.get(month, 0) for month in months_to_show]

            if year in total_data:
                total_monthly = total_data[year]
                percentages = []
                for month in months_to_show:
                    source_val = values_gwh[months_to_show.index(month)]
                    total_val = total_monthly.get(month, 0)
                    if total_val > 0:
                        percentages.append((source_val / total_val) * 100)
                    else:
                        percentages.append(0)

                color = year_colors[i % len(year_colors)]
                ax1.plot(months, percentages, marker='o', color=color, 
                        linewidth=3.5, markersize=8, label=str(year))

        ax1.set_title(f'{source_name} % of Total Generation', 
                     fontsize=32, fontweight='bold', pad=20)
        ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Percentage (%)', fontsize=28, fontweight='bold', labelpad=15)
        
        if source_name in total_sources:
            ax1.set_ylim(0, max_pct_totals)
        else:
            ax1.set_ylim(0, max_pct_individual)
            
        ax1.tick_params(axis='both', labelsize=24)
        ax1.grid(True, alpha=0.3, linewidth=1.5)

        # Legend below - 5 columns, no frame
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=5, fontsize=22, frameon=False)

        plt.tight_layout()

        percentage_filename = f'plots/eu_monthly_{source_name.lower().replace(" ", "_")}_percentage_10years.png'
        plt.savefig(percentage_filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {percentage_filename}")
        plt.close()

        # PLOT 2: Absolute
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        for i, year in enumerate(years_available):
            if year not in year_data:
                continue

            monthly_data = year_data[year]
            current_date = datetime.now()
            current_year = current_date.year

            if year == current_year:
                months_to_show = range(1, current_date.month + 1)
            else:
                months_to_show = range(1, 13)

            months = [month_names[month - 1] for month in months_to_show]
            values_gwh = [monthly_data.get(month, 0) for month in months_to_show]
            values_twh = [val / 1000 for val in values_gwh]

            color = year_colors[i % len(year_colors)]
            ax2.plot(months, values_twh, marker='o', color=color,
                    linewidth=3.5, markersize=8, label=str(year))

        ax2.set_title(f'{source_name} Production (TWh)', 
                     fontsize=32, fontweight='bold', pad=20)
        ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Energy (TWh)', fontsize=28, fontweight='bold', labelpad=15)
        
        if source_name in total_sources:
            ax2.set_ylim(0, max_abs_totals)
        else:
            ax2.set_ylim(0, max_abs_individual)
            
        ax2.tick_params(axis='both', labelsize=24)
        ax2.grid(True, alpha=0.3, linewidth=1.5)

        # Legend below - 5 columns, no frame
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=5, fontsize=22, frameon=False)

        plt.tight_layout()

        absolute_filename = f'plots/eu_monthly_{source_name.lower().replace(" ", "_")}_absolute_10years.png'
        plt.savefig(absolute_filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {absolute_filename}")
        plt.close()

    print("\n" + "=" * 60)
    print("ALL MOBILE-OPTIMIZED PLOTS GENERATED")
    print("=" * 60)


def main():
    """
    Main function
    """
    print("=" * 60)
    print("EU ENERGY PLOTTER - MOBILE OPTIMIZED")
    print("=" * 60)
    print("\nFEATURES:")
    print("  ✓ TWO separate plots per source (percentage + absolute)")
    print("  ✓ Much larger fonts (32px titles, 28px labels, 24px ticks)")
    print("  ✓ Taller plots (12x10) for better mobile viewing")
    print("  ✓ Legends below plots in 5 columns")
    print("  ✓ Normalized y-axes for individual vs totals")
    print("=" * 60)

    # Verify environment variable
    if not os.environ.get('GOOGLE_CREDENTIALS_JSON'):
        print("\n⚠️  WARNING: GOOGLE_CREDENTIALS_JSON not set!")
        return

    all_data = load_data_from_google_sheets()

    if not all_data:
        print("Failed to load data.")
        return

    create_all_charts(all_data)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
