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
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set!")
        
        creds_dict = json.loads(google_creds_json)
        
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        gc = gspread.authorize(credentials)

        spreadsheet = gc.open('EU Electricity Production Data')
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
    UPDATED: Larger fonts, thicker lines, clearer titles, no Y-axis restrictions
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

    n_years = 20

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

    # Create plots for each source - NO Y-AXIS RESTRICTIONS for individual sources
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
        fig1, ax1 = plt.subplots(figsize=(12, 10))

        max_pct_value = 0
        
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
                        pct = (source_val / total_val) * 100
                        percentages.append(pct)
                        max_pct_value = max(max_pct_value, pct)
                    else:
                        percentages.append(0)

                color = year_colors[i % len(year_colors)]
                ax1.plot(months, percentages, marker='o', color=color, 
                        linewidth=6, markersize=13, label=str(year))

        # Title with bold source name ONLY (no confusing subtitle)
        fig1.suptitle(source_name, fontsize=34, fontweight='bold', x=0.5, y=0.98, ha="center")
        ax1.set_title('Percentage of EU Production', fontsize=26, fontweight='normal', pad=15)
        ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Electricity production (%)', fontsize=28, fontweight='bold', labelpad=15)
        
        # NO RESTRICTION - let it scale to data
        ax1.set_ylim(0, max_pct_value * 1.1 if max_pct_value > 0 else 10)
            
        ax1.tick_params(axis='both', labelsize=22)
        ax1.grid(True, alpha=0.3, linewidth=1.5)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=5, fontsize=20, frameon=False)

        plt.tight_layout()

        percentage_filename = f'plots/eu_monthly_{source_name.lower().replace(" ", "_")}_percentage_10years.png'
        plt.savefig(percentage_filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {percentage_filename}")
        plt.close()

        # PLOT 2: Absolute
        fig2, ax2 = plt.subplots(figsize=(12, 10))

        max_abs_value = 0
        
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
            
            max_abs_value = max(max_abs_value, max(values_twh) if values_twh else 0)

            color = year_colors[i % len(year_colors)]
            ax2.plot(months, values_twh, marker='o', color=color,
                    linewidth=6, markersize=13, label=str(year))

        # Title with bold source name ONLY (no confusing subtitle)
        fig2.suptitle(source_name, fontsize=34, fontweight='bold', x=0.5, y=0.98, ha="center")
        ax2.set_title('Absolute Production', fontsize=26, fontweight='normal', pad=15)
        ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Electricity production (TWh)', fontsize=28, fontweight='bold', labelpad=15)
        
        # NO RESTRICTION - let it scale to data
        ax2.set_ylim(0, max_abs_value * 1.1 if max_abs_value > 0 else 10)
            
        ax2.tick_params(axis='both', labelsize=22)
        ax2.grid(True, alpha=0.3, linewidth=1.5)

        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=5, fontsize=20, frameon=False)

        plt.tight_layout()

        absolute_filename = f'plots/eu_monthly_{source_name.lower().replace(" ", "_")}_absolute_10years.png'
        plt.savefig(absolute_filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {absolute_filename}")
        plt.close()

    # Monthly Mean Charts by Period
    print("\n" + "=" * 60)
    print("CREATING MONTHLY MEAN CHARTS BY PERIOD")
    print("=" * 60)

    all_energy_sources = ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal', 'Oil', 'Waste', 'Nuclear', 'Geothermal', 'Biomass']
    available_sources = [source for source in all_energy_sources if source in all_data]

    periods = [
        {'name': '2015-2019', 'start': 2015, 'end': 2019},
        {'name': '2020-2024', 'start': 2020, 'end': 2024},
        {'name': '2025-2029', 'start': 2025, 'end': 2029}
    ]

    if available_sources and 'Total Generation' in all_data:
        months = [calendar.month_abbr[i] for i in range(1, 13)]

        # Calculate max values for consistent y-axis
        max_abs_all_periods = 0
        max_pct_all_periods = 0

        for period in periods:
            period_years = [year for year in years_available if period['start'] <= year <= period['end']]
            if not period_years:
                continue

            for source_name in available_sources:
                source_data = all_data[source_name]['year_data']
                total_data = all_data['Total Generation']['year_data']

                for year in period_years:
                    if year in source_data and year in total_data:
                        source_monthly = source_data[year]
                        total_monthly = total_data[year]

                        for month in range(1, 13):
                            source_val = source_monthly.get(month, 0)
                            total_val = total_monthly.get(month, 0)

                            max_abs_all_periods = max(max_abs_all_periods, source_val / 1000)

                            if total_val > 0:
                                percentage = (source_val / total_val) * 100
                                max_pct_all_periods = max(max_pct_all_periods, percentage)

        max_abs_all_periods *= 1.1
        max_pct_all_periods *= 1.1

        for period in periods:
            print(f"\nCreating Monthly Mean chart for {period['name']}...")

            period_years = [year for year in years_available if period['start'] <= year <= period['end']]
            if not period_years:
                continue

            monthly_absolute = {}
            monthly_percentages = {}

            for source_name in available_sources:
                monthly_absolute[source_name] = {}
                monthly_percentages[source_name] = {}
                source_data = all_data[source_name]['year_data']
                total_data = all_data['Total Generation']['year_data']

                for month in range(1, 13):
                    monthly_absolute[source_name][month] = []
                    monthly_percentages[source_name][month] = []

                for year in period_years:
                    if year in source_data and year in total_data:
                        source_monthly = source_data[year]
                        total_monthly = total_data[year]

                        for month in range(1, 13):
                            source_val = source_monthly.get(month, 0)
                            total_val = total_monthly.get(month, 0)

                            monthly_absolute[source_name][month].append(source_val)

                            if total_val > 0:
                                percentage = (source_val / total_val) * 100
                                monthly_percentages[source_name][month].append(percentage)

            monthly_means_abs = {}
            monthly_means_pct = {}
            for source_name in available_sources:
                monthly_means_abs[source_name] = []
                monthly_means_pct[source_name] = []
                for month in range(1, 13):
                    absolute_vals = monthly_absolute[source_name][month]
                    if absolute_vals:
                        monthly_means_abs[source_name].append(np.mean(absolute_vals))
                    else:
                        monthly_means_abs[source_name].append(0)

                    percentages = monthly_percentages[source_name][month]
                    if percentages:
                        monthly_means_pct[source_name].append(np.mean(percentages))
                    else:
                        monthly_means_pct[source_name].append(0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
            # Increase vertical spacing between subplots for better readability
            fig.subplots_adjust(hspace=1.0)

            for source_name in available_sources:
                color = ENTSOE_COLORS.get(source_name, 'black')

                # ax1 = PERCENTAGE (top)
                ax1.plot(months, monthly_means_pct[source_name], marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)
                
                # ax2 = ABSOLUTE (bottom)
                values_twh = [val / 1000 for val in monthly_means_abs[source_name]]
                ax2.plot(months, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)

            # Bold main title above top subplot
            fig.text(0.5, 0.98, f'All Electricity Sources: {period["name"]}', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for top plot
            ax1.set_title('Percentage of EU Production', fontsize=26, fontweight='normal', pad=60)
            ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity production (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, max_pct_all_periods)
            ax1.tick_params(axis='both', labelsize=22)
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Bold main title above bottom subplot  
            fig.text(0.5, 0.44, f'All Electricity Sources: {period["name"]}', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for bottom plot
            ax2.set_title('Absolute Production', fontsize=26, fontweight='normal', pad=60)
            ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity production (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(0, max_abs_all_periods)
            ax2.tick_params(axis='both', labelsize=22)
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Double legend - one per subplot, positioned lower for more space
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
                       fontsize=20, frameon=False)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
                       fontsize=20, frameon=False)

            # No main suptitle needed since we repeat title on each subplot

            plt.tight_layout(rect=[0, 0.02, 1, 0.985])

            period_name_clean = period['name'].replace('-', '_')
            filename = f'plots/eu_monthly_energy_sources_mean_{period_name_clean}_combined.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")
            plt.close()

    # Renewable vs Non-Renewable by Period
    print("\n" + "=" * 60)
    print("CREATING RENEWABLE VS NON-RENEWABLE CHARTS")
    print("=" * 60)

    if 'All Renewables' in all_data and 'All Non-Renewables' in all_data and 'Total Generation' in all_data:
        month_names_abbr = [calendar.month_abbr[i] for i in range(1, 13)]

        max_abs_renewable_periods = 0

        for period in periods:
            period_years = [year for year in years_available if period['start'] <= year <= period['end']]
            if not period_years:
                continue

            for category_name in ['All Renewables', 'All Non-Renewables']:
                category_data = all_data[category_name]['year_data']

                for year in period_years:
                    if year in category_data:
                        category_monthly = category_data[year]

                        for month in range(1, 13):
                            category_val = category_monthly.get(month, 0)
                            max_abs_renewable_periods = max(max_abs_renewable_periods, category_val / 1000)

        max_abs_renewable_periods *= 1.1

        for period in periods:
            print(f"\nCreating Renewable vs Non-Renewable chart for {period['name']}...")

            period_years = [year for year in years_available if period['start'] <= year <= period['end']]
            if not period_years:
                continue

            monthly_absolute = {}
            monthly_percentages = {}

            for category_name in ['All Renewables', 'All Non-Renewables']:
                monthly_absolute[category_name] = {}
                monthly_percentages[category_name] = {}
                category_data = all_data[category_name]['year_data']
                total_data = all_data['Total Generation']['year_data']

                for month in range(1, 13):
                    monthly_absolute[category_name][month] = []
                    monthly_percentages[category_name][month] = []

                for year in period_years:
                    if year in category_data and year in total_data:
                        category_monthly = category_data[year]
                        total_monthly = total_data[year]

                        for month in range(1, 13):
                            category_val = category_monthly.get(month, 0)
                            total_val = total_monthly.get(month, 0)

                            monthly_absolute[category_name][month].append(category_val)

                            if total_val > 0:
                                percentage = (category_val / total_val) * 100
                                monthly_percentages[category_name][month].append(percentage)

            monthly_means_abs = {}
            monthly_means_pct = {}
            for category_name in ['All Renewables', 'All Non-Renewables']:
                monthly_means_abs[category_name] = []
                monthly_means_pct[category_name] = []
                for month in range(1, 13):
                    absolute_vals = monthly_absolute[category_name][month]
                    if absolute_vals:
                        monthly_means_abs[category_name].append(np.mean(absolute_vals))
                    else:
                        monthly_means_abs[category_name].append(0)

                    percentages = monthly_percentages[category_name][month]
                    if percentages:
                        monthly_means_pct[category_name].append(np.mean(percentages))
                    else:
                        monthly_means_pct[category_name].append(0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
            # Increase vertical spacing between subplots for better readability
            fig.subplots_adjust(hspace=1.0)

            for category_name in ['All Renewables', 'All Non-Renewables']:
                color = ENTSOE_COLORS[category_name]

                # ax1 = PERCENTAGE (top)
                ax1.plot(month_names_abbr, monthly_means_pct[category_name], marker='o', color=color,
                         linewidth=6, markersize=13, label=category_name)
                
                # ax2 = ABSOLUTE (bottom)
                values_twh = [val / 1000 for val in monthly_means_abs[category_name]]
                ax2.plot(month_names_abbr, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=category_name)

            # Bold main title above top subplot
            fig.text(0.5, 0.98, f'Renewables vs Non-Renewables: {period["name"]}', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for top plot
            ax1.set_title('Percentage of EU Production', fontsize=26, fontweight='normal', pad=60)
            ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity production (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='both', labelsize=22)
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Bold main title above bottom subplot
            fig.text(0.5, 0.44, f'Renewables vs Non-Renewables: {period["name"]}', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for bottom plot
            ax2.set_title('Absolute Production', fontsize=26, fontweight='normal', pad=60)
            ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity production (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(0, max_abs_renewable_periods)
            ax2.tick_params(axis='both', labelsize=22)
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Double legend - one per subplot, positioned lower for more space
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                       fontsize=22, frameon=False)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                       fontsize=22, frameon=False)

            # No main suptitle needed since we use fig.text

            plt.tight_layout(rect=[0, 0.02, 1, 0.985])

            period_name_clean = period['name'].replace('-', '_')
            filename = f'plots/eu_monthly_renewable_vs_nonrenewable_mean_{period_name_clean}_combined.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")
            plt.close()

    # Annual Trend Charts
    print("\n" + "=" * 60)
    print("CREATING ANNUAL TREND CHARTS")
    print("=" * 60)

    annual_totals = {}

    renewable_sources = ['Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal']
    non_renewable_sources = ['Gas', 'Coal', 'Nuclear', 'Oil', 'Waste']
    total_sources = ['All Renewables', 'All Non-Renewables']

    available_renewables = [s for s in renewable_sources if s in all_data]
    available_non_renewables = [s for s in non_renewable_sources if s in all_data]
    available_totals = [s for s in total_sources if s in all_data]

    all_sources = available_renewables + available_non_renewables + available_totals

    for source_name in all_sources:
        annual_totals[source_name] = {}
        year_data = all_data[source_name]['year_data']

        for year in years_available:
            if year in year_data:
                annual_total = sum(year_data[year].get(month, 0) for month in range(1, 13))
                annual_totals[source_name][year] = annual_total

    if 'Total Generation' in all_data:
        annual_totals['Total Generation'] = {}
        total_year_data = all_data['Total Generation']['year_data']

        for year in years_available:
            if year in total_year_data:
                annual_total = sum(total_year_data[year].get(month, 0) for month in range(1, 13))
                annual_totals['Total Generation'][year] = annual_total

    # Calculate max values
    max_annual_twh = 0
    max_annual_pct = 0

    for source_name in available_renewables + available_non_renewables:
        if source_name in annual_totals and 'Total Generation' in annual_totals:
            years_list = sorted(annual_totals[source_name].keys())
            for year in years_list:
                val_twh = annual_totals[source_name][year] / 1000
                max_annual_twh = max(max_annual_twh, val_twh)

                source_value = annual_totals[source_name][year]
                total_value = annual_totals['Total Generation'][year]
                if total_value > 0:
                    percentage = (source_value / total_value) * 100
                    max_annual_pct = max(max_annual_pct, percentage)

    max_annual_twh *= 1.1
    max_annual_pct *= 1.1

    # Chart: Renewable Trends
    if available_renewables and 'Total Generation' in annual_totals:
        print("\nCreating Annual Renewable Trends...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
        # Add more vertical spacing between subplots
        fig.subplots_adjust(hspace=1.0)

        lines_plotted = 0
        for source_name in available_renewables:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())

                color = ENTSOE_COLORS.get(source_name, 'black')
                
                # ax1 = PERCENTAGE (top)
                source_years = set(annual_totals[source_name].keys())
                total_years = set(annual_totals['Total Generation'].keys())
                overlapping_years = source_years & total_years & set(years_list)

                if overlapping_years:
                    pct_years = sorted(overlapping_years)
                    percentages = []
                    for year in pct_years:
                        source_value = annual_totals[source_name][year]
                        total_value = annual_totals['Total Generation'][year]
                        if total_value > 0:
                            percentage = (source_value / total_value) * 100
                            percentages.append(percentage)
                        else:
                            percentages.append(0)

                    ax1.plot(pct_years, percentages, marker='o', color=color,
                             linewidth=6, markersize=13, label=source_name)
                
                # ax2 = ABSOLUTE (bottom)
                values_twh = [annual_totals[source_name][year] / 1000 for year in years_list]
                ax2.plot(years_list, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)

                lines_plotted += 1

        if lines_plotted > 0:
            # Bold main title above top subplot
            fig.text(0.5, 0.98, 'Annual Renewable Trends', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for top plot
            ax1.set_title('Percentage of EU Production', fontsize=26, fontweight='normal', pad=60)
            ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity production (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, max_annual_pct)
            ax1.tick_params(axis='both', labelsize=22)
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Bold main title above bottom subplot
            fig.text(0.5, 0.44, 'Annual Renewable Trends', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for bottom plot
            ax2.set_title('Absolute Production', fontsize=26, fontweight='normal', pad=60)
            ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity production (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(0, max_annual_twh)
            ax2.tick_params(axis='both', labelsize=22)
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Double legend - one per subplot, positioned lower for more space
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(available_renewables),
                       fontsize=20, frameon=False)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(available_renewables),
                       fontsize=20, frameon=False)

            # No main suptitle needed since we repeat title on each subplot

            plt.tight_layout(rect=[0, 0.02, 1, 0.985])

            filename = 'plots/eu_annual_renewable_trends_combined.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")
            plt.close()

    # Chart: Non-Renewable Trends
    if available_non_renewables and 'Total Generation' in annual_totals:
        print("\nCreating Annual Non-Renewable Trends...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
        # Add more vertical spacing between subplots
        fig.subplots_adjust(hspace=1.0)

        lines_plotted = 0
        for source_name in available_non_renewables:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())

                color = ENTSOE_COLORS.get(source_name, 'black')
                
                # ax1 = PERCENTAGE (top)
                source_years = set(annual_totals[source_name].keys())
                total_years = set(annual_totals['Total Generation'].keys())
                overlapping_years = source_years & total_years & set(years_list)

                if overlapping_years:
                    pct_years = sorted(overlapping_years)
                    percentages = []
                    for year in pct_years:
                        source_value = annual_totals[source_name][year]
                        total_value = annual_totals['Total Generation'][year]
                        if total_value > 0:
                            percentage = (source_value / total_value) * 100
                            percentages.append(percentage)
                        else:
                            percentages.append(0)

                    ax1.plot(pct_years, percentages, marker='o', color=color,
                             linewidth=6, markersize=13, label=source_name)
                
                # ax2 = ABSOLUTE (bottom)
                values_twh = [annual_totals[source_name][year] / 1000 for year in years_list]
                ax2.plot(years_list, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)

                lines_plotted += 1

        if lines_plotted > 0:
            # Bold main title above top subplot
            fig.text(0.5, 0.98, 'Annual Non-Renewable Trends', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for top plot
            ax1.set_title('Percentage of EU Production', fontsize=26, fontweight='normal', pad=60)
            ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity production (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, max_annual_pct)
            ax1.tick_params(axis='both', labelsize=22)
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Bold main title above bottom subplot
            fig.text(0.5, 0.44, 'Annual Non-Renewable Trends', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for bottom plot
            ax2.set_title('Absolute Production', fontsize=26, fontweight='normal', pad=60)
            ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity production (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(0, max_annual_twh)
            ax2.tick_params(axis='both', labelsize=22)
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Double legend - one per subplot, positioned lower for more space
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=len(available_non_renewables), fontsize=20, frameon=False)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=len(available_non_renewables), fontsize=20, frameon=False)

            # No main suptitle needed since we repeat title on each subplot

            plt.tight_layout(rect=[0, 0.02, 1, 0.985])

            filename = 'plots/eu_annual_non_renewable_trends_combined.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")
            plt.close()

    # Chart: Renewables vs Non-Renewables Totals
    if available_totals and 'Total Generation' in annual_totals:
        print("\nCreating Annual Renewables vs Non-Renewables...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
        # Add more vertical spacing between subplots
        fig.subplots_adjust(hspace=1.0)

        lines_plotted = 0
        for source_name in available_totals:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())

                color = ENTSOE_COLORS[source_name]
                
                # ax1 = PERCENTAGE (top)
                source_years = set(annual_totals[source_name].keys())
                total_years = set(annual_totals['Total Generation'].keys())
                overlapping_years = source_years & total_years & set(years_list)

                if overlapping_years:
                    pct_years = sorted(overlapping_years)
                    percentages = []
                    for year in pct_years:
                        source_value = annual_totals[source_name][year]
                        total_value = annual_totals['Total Generation'][year]
                        if total_value > 0:
                            percentage = (source_value / total_value) * 100
                            percentages.append(percentage)
                        else:
                            percentages.append(0)

                    ax1.plot(pct_years, percentages, marker='o', color=color,
                             linewidth=6, markersize=13, label=source_name)
                
                # ax2 = ABSOLUTE (bottom)
                values_twh = [annual_totals[source_name][year] / 1000 for year in years_list]
                ax2.plot(years_list, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)

                lines_plotted += 1

        if lines_plotted > 0:
            # Bold main title above top subplot
            fig.text(0.5, 0.98, 'Renewables vs Non-Renewables', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for top plot
            ax1.set_title('Percentage of EU Production', fontsize=26, fontweight='normal', pad=60)
            ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity production (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='both', labelsize=22)
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Bold main title above bottom subplot
            fig.text(0.5, 0.44, 'Renewables vs Non-Renewables', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for bottom plot
            ax2.set_title('Absolute Production', fontsize=26, fontweight='normal', pad=60)
            ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity production (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(bottom=0)
            ax2.tick_params(axis='both', labelsize=22)
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Double legend - one per subplot, positioned lower for more space
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                       fontsize=22, frameon=False)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                       fontsize=22, frameon=False)

            # No main suptitle needed since we repeat title on each subplot

            plt.tight_layout(rect=[0, 0.02, 1, 0.985])

            filename = 'plots/eu_annual_renewable_vs_non_renewable_combined.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")
            plt.close()

    # Year-over-Year Change vs 2015 Baseline
    print("\n" + "=" * 60)
    print("CREATING YOY CHANGE VS 2015 BASELINE")
    print("=" * 60)

    if annual_totals:
        print("\nCreating YoY change vs 2015 baseline chart...")

        baseline_year = 2015

        all_sources_for_yoy = available_renewables + available_non_renewables
        totals_for_yoy = ['All Renewables', 'All Non-Renewables']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
        # Add more vertical spacing between subplots
        fig.subplots_adjust(hspace=1.0)

        all_yoy_values = []

        # Top plot: All individual sources
        lines_plotted = 0
        for source_name in all_sources_for_yoy:
            if source_name in annual_totals and baseline_year in annual_totals[source_name]:
                baseline_value = annual_totals[source_name][baseline_year]

                if baseline_value > 0:
                    years_list = sorted(annual_totals[source_name].keys())

                    yoy_changes = []
                    for year in years_list:
                        if year >= baseline_year:
                            current_value = annual_totals[source_name][year]
                            pct_change = ((current_value - baseline_value) / baseline_value) * 100
                            yoy_changes.append(pct_change)
                            all_yoy_values.append(pct_change)

                    years_to_plot = [year for year in years_list if year >= baseline_year]

                    if len(years_to_plot) > 0:
                        color = ENTSOE_COLORS.get(source_name, 'black')
                        ax1.plot(years_to_plot, yoy_changes, marker='o', color=color,
                                 linewidth=6, markersize=13, label=source_name)
                        lines_plotted += 1

        # Bottom plot: Just totals
        for category_name in totals_for_yoy:
            if category_name in annual_totals and baseline_year in annual_totals[category_name]:
                baseline_value = annual_totals[category_name][baseline_year]

                if baseline_value > 0:
                    years_list = sorted(annual_totals[category_name].keys())

                    yoy_changes = []
                    for year in years_list:
                        if year >= baseline_year:
                            current_value = annual_totals[category_name][year]
                            pct_change = ((current_value - baseline_value) / baseline_value) * 100
                            yoy_changes.append(pct_change)
                            all_yoy_values.append(pct_change)

                    years_to_plot = [year for year in years_list if year >= baseline_year]

                    if len(years_to_plot) > 0:
                        color = ENTSOE_COLORS[category_name]
                        ax2.plot(years_to_plot, yoy_changes, marker='o', color=color,
                                 linewidth=6, markersize=13, label=category_name)

        if lines_plotted > 0:
            if all_yoy_values:
                y_min = min(all_yoy_values)
                y_max = max(all_yoy_values)
                y_margin = (y_max - y_min) * 0.1
                y_min_limit = y_min - y_margin
                y_max_limit = y_max + y_margin
            else:
                y_min_limit = -50
                y_max_limit = 100

            # Bold main title above top subplot
            fig.text(0.5, 0.98, 'Year-over-Year Change vs 2015', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for top plot
            ax1.set_title('All Electricity Sources', fontsize=26, fontweight='normal', pad=60)
            ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('% Change from 2015', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(y_min_limit, y_max_limit)
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax1.tick_params(axis='both', labelsize=22)
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Bold main title above bottom subplot
            fig.text(0.5, 0.44, 'Year-over-Year Change vs 2015', 
                    ha='center', fontsize=34, fontweight='bold')
            
            # Subtitle for bottom plot
            ax2.set_title('Renewables vs Non-Renewables', fontsize=26, fontweight='normal', pad=60)
            ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('% Change from 2015', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(y_min_limit, y_max_limit)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.tick_params(axis='both', labelsize=22)
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Double legend - one per subplot, positioned lower for more space
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
                       fontsize=18, frameon=False)
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                       fontsize=22, frameon=False)

            # No main suptitle needed since we repeat title on each subplot

            plt.tight_layout(rect=[0, 0.02, 1, 0.985])

            filename = 'plots/eu_annual_yoy_change_vs_2015.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")
            plt.close()

    print("\n" + "=" * 60)
    print("ALL MOBILE-OPTIMIZED PLOTS GENERATED")
    print("=" * 60)


def update_summary_table_historical_data(all_data):
    """
    Update Google Sheets "Summary Table Data" with YTD 2025 and 2020-2024 average data
    This fills in the columns that the intraday script leaves empty
    """
    print("\n" + "=" * 60)
    print("UPDATING SUMMARY TABLE (HISTORICAL DATA)")
    print("=" * 60)
    
    try:
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            print("⚠ GOOGLE_CREDENTIALS_JSON not set - skipping update")
            return
        
        creds_dict = json.loads(google_creds_json)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        gc = gspread.authorize(credentials)
        
        spreadsheet = gc.open('EU Electricity Production Data')
        print("✓ Connected to spreadsheet")
        
        # Get the Summary Table Data worksheet
        try:
            worksheet = spreadsheet.worksheet('Summary Table Data')
            print("✓ Found 'Summary Table Data' worksheet")
        except gspread.WorksheetNotFound:
            print("⚠ 'Summary Table Data' worksheet not found - run intraday analysis first")
            return
        
        # Get current date info
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        # Calculate YTD 2025 and 2020-2024 averages for each source
        source_mapping = {
            'Solar': 'Solar',
            'Wind': 'Wind',
            'Hydro': 'Hydro',
            'Biomass': 'Biomass',
            'Geothermal': 'Geothermal',
            'Gas': 'Gas',
            'Coal': 'Coal',
            'Nuclear': 'Nuclear',
            'Oil': 'Oil',
            'Waste': 'Waste',
            'All Renewables': 'All Renewables',
            'All Non-Renewables': 'All Non-Renewables'
        }
        
        # Prepare updates
        updates = []
        
        for row_idx, (display_name, data_key) in enumerate(source_mapping.items(), start=2):
            if data_key not in all_data:
                continue
            
            year_data = all_data[data_key]['year_data']
            
            # Calculate YTD 2025
            ytd_2025_gwh = 0
            if 2025 in year_data:
                for month in range(1, current_month + 1):
                    ytd_2025_gwh += year_data[2025].get(month, 0)
            
            # Calculate 2020-2024 average (annual)
            period_totals = []
            for year in range(2020, 2025):
                if year in year_data:
                    year_total = sum(year_data[year].values())
                    period_totals.append(year_total)
            
            avg_2020_2024_gwh = sum(period_totals) / len(period_totals) if period_totals else 0
            
            # Calculate total generation for percentages
            if 'Total Generation' in all_data:
                total_year_data = all_data['Total Generation']['year_data']
                
                # YTD 2025 percentage
                ytd_2025_total = 0
                if 2025 in total_year_data:
                    for month in range(1, current_month + 1):
                        ytd_2025_total += total_year_data[2025].get(month, 0)
                
                ytd_2025_pct = (ytd_2025_gwh / ytd_2025_total * 100) if ytd_2025_total > 0 else 0
                
                # 2020-2024 average percentage
                period_total_gen = []
                for year in range(2020, 2025):
                    if year in total_year_data:
                        year_total_gen = sum(total_year_data[year].values())
                        period_total_gen.append(year_total_gen)
                
                avg_total_gen = sum(period_total_gen) / len(period_total_gen) if period_total_gen else 0
                avg_2020_2024_pct = (avg_2020_2024_gwh / avg_total_gen * 100) if avg_total_gen > 0 else 0
            else:
                ytd_2025_pct = 0
                avg_2020_2024_pct = 0
            
            # Add to updates list (columns F, G, H, I)
            updates.append({
                'range': f'F{row_idx}:I{row_idx}',
                'values': [[
                    f"{ytd_2025_gwh:.1f}",
                    f"{ytd_2025_pct:.2f}",
                    f"{avg_2020_2024_gwh:.1f}",
                    f"{avg_2020_2024_pct:.2f}"
                ]]
            })
        
        # Batch update all rows at once
        if updates:
            for update in updates:
                worksheet.update(update['range'], update['values'])
            
            print(f"✓ Updated {len(updates)} sources with YTD 2025 and 2020-2024 data")
            
            # Update timestamp in last column
            timestamp = current_date.strftime('%Y-%m-%d %H:%M UTC')
            # Update column J for all data rows
            for row_idx in range(2, len(source_mapping) + 2):
                worksheet.update(f'J{row_idx}', [[timestamp]])
            
            print(f"   Worksheet: {spreadsheet.url}")
        else:
            print("⚠ No data to update")
    
    except Exception as e:
        print(f"✗ Error updating summary table: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function
    """
    print("=" * 60)
    print("EU ENERGY PLOTTER - MOBILE OPTIMIZED + ALL CHARTS")
    print("=" * 60)
    print("\nFEATURES:")
    print("  ✓ ALL plots are VERTICAL (2 rows, 1 column)")
    print("  ✓ Individual source plots (titles IN the PNG)")
    print("  ✓ Monthly mean by period charts")
    print("  ✓ Renewable vs non-renewable by period")
    print("  ✓ Annual trend charts")
    print("  ✓ YoY change vs 2015 baseline")
    print("  ✓ LARGER fonts and THICKER lines for mobile")
    print("  ✓ CLEARER titles (no restrictions on Y-axis)")
    print("=" * 60)

    if not os.environ.get('GOOGLE_CREDENTIALS_JSON'):
        print("\n⚠️  WARNING: GOOGLE_CREDENTIALS_JSON not set!")
        return

    all_data = load_data_from_google_sheets()

    if not all_data:
        print("Failed to load data.")
        return

    create_all_charts(all_data)
    
    # Update summary table with historical data
    update_summary_table_historical_data(all_data)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
