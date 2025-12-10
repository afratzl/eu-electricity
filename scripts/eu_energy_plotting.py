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

        # Title and labels - match intraday format
        fig1.suptitle(f'{source_name} Electricity Generation (EU)', fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax1.set_title('Fraction of Total Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
        
        # NO RESTRICTION - let it scale to data
        ax1.set_ylim(0, max_pct_value * 1.2 if max_pct_value > 0 else 10)  # 20% margin
            
        ax1.tick_params(axis='both', labelsize=22)
        ax1.grid(True, alpha=0.3, linewidth=1.5)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), 
                  ncol=5, fontsize=20, frameon=False)

        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        fig1.text(0.93, 0.02, f"Generated: {timestamp}",
                  ha='right', va='bottom',
                  fontsize=11, color='#666',
                  style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

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

        # Title and labels - match intraday format
        fig2.suptitle(f'{source_name} Electricity Generation (EU)', fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax2.set_title('Absolute Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
        
        # NO RESTRICTION - let it scale to data
        ax2.set_ylim(0, max_abs_value * 1.2 if max_abs_value > 0 else 10)  # 20% margin
            
        ax2.tick_params(axis='both', labelsize=22)
        ax2.grid(True, alpha=0.3, linewidth=1.5)

        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
                  ncol=5, fontsize=20, frameon=False)

        # Add timestamp (reuse from percentage plot)
        fig2.text(0.93, 0.02, f"Generated: {timestamp}",
                  ha='right', va='bottom',
                  fontsize=11, color='#666',
                  style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

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

        max_abs_all_periods *= 1.2  # 20% margin
        max_pct_all_periods *= 1.2  # 20% margin

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

            # Generate timestamp once for both plots
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
            
            # PERCENTAGE PLOT
            fig1, ax1 = plt.subplots(figsize=(12, 10))

            # Plot sources and track handles for legend reordering
            from matplotlib.lines import Line2D
            line_handles = {}
            
            for source_name in available_sources:
                color = ENTSOE_COLORS.get(source_name, 'black')
                line, = ax1.plot(months, monthly_means_pct[source_name], marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)
                line_handles[source_name] = line

            # Title and labels - clean format
            fig1.suptitle('Electricity Generation (EU)', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax1.set_title(f'Fraction of Total Generation ({period["name"]})', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, max_pct_all_periods)
            ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax1.grid(True, alpha=0.3, linewidth=1.5)

            # Create legend with spacers for centered row 3 (renewables left, non-renewables right)
            spacer = Line2D([0], [0], color='none', label=' ')
            # ncol=4 fills by COLUMN, so order: col1, col2, col3, col4
            legend_order = [
                'Wind', 'Solar', '_SPACER_',              # Column 1: renewables
                'Hydro', 'Biomass', 'Geothermal',         # Column 2: renewables
                'Nuclear', 'Coal', 'Oil',                 # Column 3: non-renewables
                'Gas', 'Waste', '_SPACER_'                # Column 4: non-renewables
            ]
            
            legend_handles = []
            legend_labels = []
            for item in legend_order:
                if item == '_SPACER_':
                    legend_handles.append(spacer)
                    legend_labels.append(' ')
                elif item in line_handles:
                    legend_handles.append(line_handles[item])
                    legend_labels.append(item)
            
            ax1.legend(legend_handles, legend_labels,
                       loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=4,
                       fontsize=20, frameon=False)

            # Add timestamp
            fig1.text(0.93, 0.02, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            period_name_clean = period['name'].replace('-', '_')
            filename_pct = f'plots/eu_monthly_energy_sources_mean_{period_name_clean}_percentage.png'
            plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved percentage: {filename_pct}")
            plt.close()

            # ABSOLUTE PLOT
            fig2, ax2 = plt.subplots(figsize=(12, 10))

            # Plot sources and track handles for legend reordering
            line_handles = {}
            
            for source_name in available_sources:
                color = ENTSOE_COLORS.get(source_name, 'black')
                values_twh = [val / 1000 for val in monthly_means_abs[source_name]]
                line, = ax2.plot(months, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)
                line_handles[source_name] = line

            # Title and labels - clean format
            fig2.suptitle('Electricity Generation (EU)', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax2.set_title(f'Absolute Generation ({period["name"]})', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(0, max_abs_all_periods)
            ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax2.grid(True, alpha=0.3, linewidth=1.5)

            # Create legend with spacers for centered row 3 (renewables left, non-renewables right)
            spacer = Line2D([0], [0], color='none', label=' ')
            # ncol=4 fills by COLUMN, so order: col1, col2, col3, col4
            legend_order = [
                'Wind', 'Solar', '_SPACER_',              # Column 1: renewables
                'Hydro', 'Biomass', 'Geothermal',         # Column 2: renewables
                'Nuclear', 'Coal', 'Oil',                 # Column 3: non-renewables
                'Gas', 'Waste', '_SPACER_'                # Column 4: non-renewables
            ]
            
            legend_handles = []
            legend_labels = []
            for item in legend_order:
                if item == '_SPACER_':
                    legend_handles.append(spacer)
                    legend_labels.append(' ')
                elif item in line_handles:
                    legend_handles.append(line_handles[item])
                    legend_labels.append(item)
            
            ax2.legend(legend_handles, legend_labels,
                       loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=4,
                       fontsize=20, frameon=False)

            # Add timestamp (reuse same timestamp)
            fig2.text(0.93, 0.02, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename_abs = f'plots/eu_monthly_energy_sources_mean_{period_name_clean}_absolute.png'
            plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved absolute: {filename_abs}")
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

        max_abs_renewable_periods *= 1.2  # 20% margin

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

            # Generate timestamp once for both plots
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
            
            # PERCENTAGE PLOT
            fig1, ax1 = plt.subplots(figsize=(12, 10))

            for category_name in ['All Renewables', 'All Non-Renewables']:
                color = ENTSOE_COLORS[category_name]
                ax1.plot(month_names_abbr, monthly_means_pct[category_name], marker='o', color=color,
                         linewidth=6, markersize=13, label=category_name)

            # Title and labels - clean format
            fig1.suptitle('Electricity Generation (EU)', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax1.set_title(f'Fraction of Total Generation ({period["name"]})', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax1.grid(True, alpha=0.3, linewidth=1.5)

            ax1.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                       fontsize=22, frameon=False)

            # Add timestamp
            fig1.text(0.93, 0.02, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            period_name_clean = period['name'].replace('-', '_')
            filename_pct = f'plots/eu_monthly_renewable_vs_nonrenewable_mean_{period_name_clean}_percentage.png'
            plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved percentage: {filename_pct}")
            plt.close()

            # ABSOLUTE PLOT
            fig2, ax2 = plt.subplots(figsize=(12, 10))

            for category_name in ['All Renewables', 'All Non-Renewables']:
                color = ENTSOE_COLORS[category_name]
                values_twh = [val / 1000 for val in monthly_means_abs[category_name]]
                ax2.plot(month_names_abbr, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=category_name)

            # Title and labels - clean format
            fig2.suptitle('Electricity Generation (EU)', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax2.set_title(f'Absolute Generation ({period["name"]})', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(0, max_abs_renewable_periods)
            ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax2.grid(True, alpha=0.3, linewidth=1.5)

            ax2.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                       fontsize=22, frameon=False)

            # Add timestamp (reuse same timestamp)
            fig2.text(0.93, 0.02, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename_abs = f'plots/eu_monthly_renewable_vs_nonrenewable_mean_{period_name_clean}_absolute.png'
            plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved absolute: {filename_abs}")
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

    max_annual_twh *= 1.2  # 20% margin
    max_annual_pct *= 1.2  # 20% margin

    # Chart: All Sources Annual Trends (combines renewables + non-renewables)
    all_sources = available_renewables + available_non_renewables
    if all_sources and 'Total Generation' in annual_totals:
        print("\nCreating Annual Trends: All Sources...")

        # Generate timestamp once for both plots
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        
        # PERCENTAGE PLOT
        fig1, ax1 = plt.subplots(figsize=(12, 10))

        lines_plotted = 0
        line_handles = {}
        
        for source_name in all_sources:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())
                color = ENTSOE_COLORS.get(source_name, 'black')
                
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

                    line, = ax1.plot(pct_years, percentages, marker='o', color=color,
                             linewidth=6, markersize=13, label=source_name)
                    line_handles[source_name] = line
                    lines_plotted += 1

        if lines_plotted > 0:
            # Title and labels - clean format
            fig1.suptitle('Electricity Generation (EU)', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax1.set_title('Fraction of Total Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, max_annual_pct)
            ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax1.grid(True, alpha=0.3, linewidth=1.5)

            # Create legend with spacers for centered row 3 (renewables left, non-renewables right)
            from matplotlib.lines import Line2D
            spacer = Line2D([0], [0], color='none', label=' ')
            # ncol=4 fills by COLUMN, so order: col1, col2, col3, col4
            legend_order = [
                'Wind', 'Solar', '_SPACER_',              # Column 1: renewables
                'Hydro', 'Biomass', 'Geothermal',         # Column 2: renewables
                'Nuclear', 'Coal', 'Oil',                 # Column 3: non-renewables
                'Gas', 'Waste', '_SPACER_'                # Column 4: non-renewables
            ]
            
            legend_handles = []
            legend_labels = []
            for item in legend_order:
                if item == '_SPACER_':
                    legend_handles.append(spacer)
                    legend_labels.append(' ')
                elif item in line_handles:
                    legend_handles.append(line_handles[item])
                    legend_labels.append(item)
            
            ax1.legend(legend_handles, legend_labels,
                       loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=4,
                       fontsize=20, frameon=False)

            # Add timestamp
            fig1.text(0.93, 0.02, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename_pct = 'plots/eu_annual_all_sources_percentage.png'
            plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved percentage: {filename_pct}")
            plt.close()

        # ABSOLUTE PLOT
        fig2, ax2 = plt.subplots(figsize=(12, 10))

        lines_plotted = 0
        line_handles = {}
        
        for source_name in all_sources:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())
                color = ENTSOE_COLORS.get(source_name, 'black')
                
                values_twh = [annual_totals[source_name][year] / 1000 for year in years_list]
                line, = ax2.plot(years_list, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)
                line_handles[source_name] = line
                lines_plotted += 1

        if lines_plotted > 0:
            # Title and labels - clean format
            fig2.suptitle('Electricity Generation (EU)', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax2.set_title('Absolute Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(0, max_annual_twh)
            ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax2.grid(True, alpha=0.3, linewidth=1.5)

            # Create legend with spacers for centered row 3 (renewables left, non-renewables right)
            spacer = Line2D([0], [0], color='none', label=' ')
            # ncol=4 fills by COLUMN, so order: col1, col2, col3, col4
            legend_order = [
                'Wind', 'Solar', '_SPACER_',              # Column 1: renewables
                'Hydro', 'Biomass', 'Geothermal',         # Column 2: renewables
                'Nuclear', 'Coal', 'Oil',                 # Column 3: non-renewables
                'Gas', 'Waste', '_SPACER_'                # Column 4: non-renewables
            ]
            
            legend_handles = []
            legend_labels = []
            for item in legend_order:
                if item == '_SPACER_':
                    legend_handles.append(spacer)
                    legend_labels.append(' ')
                elif item in line_handles:
                    legend_handles.append(line_handles[item])
                    legend_labels.append(item)
            
            ax2.legend(legend_handles, legend_labels,
                       loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=4,
                       fontsize=20, frameon=False)

            # Add timestamp (reuse same timestamp)
            fig2.text(0.93, 0.02, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename_abs = 'plots/eu_annual_all_sources_absolute.png'
            plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved absolute: {filename_abs}")
            plt.close()

    # Chart: Aggregates Annual Trends (Renewables vs Non-Renewables)
    if available_totals and 'Total Generation' in annual_totals:
        print("\nCreating Annual Trends: Aggregates...")

        # Generate timestamp once for both plots
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        
        # PERCENTAGE PLOT
        fig1, ax1 = plt.subplots(figsize=(12, 10))

        lines_plotted = 0
        for source_name in available_totals:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())
                color = ENTSOE_COLORS[source_name]
                
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
                    lines_plotted += 1

        if lines_plotted > 0:
            # Title and labels - clean format
            fig1.suptitle('Electricity Generation (EU)', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax1.set_title('Fraction of Total Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax1.grid(True, alpha=0.3, linewidth=1.5)

            ax1.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                       fontsize=22, frameon=False)

            # Add timestamp
            fig1.text(0.93, 0.02, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename_pct = 'plots/eu_annual_renewable_vs_nonrenewable_percentage.png'
            plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved percentage: {filename_pct}")
            plt.close()

        # ABSOLUTE PLOT
        fig2, ax2 = plt.subplots(figsize=(12, 10))

        lines_plotted = 0
        for source_name in available_totals:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())
                color = ENTSOE_COLORS[source_name]
                
                values_twh = [annual_totals[source_name][year] / 1000 for year in years_list]
                ax2.plot(years_list, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)
                lines_plotted += 1

        if lines_plotted > 0:
            # Title and labels - clean format
            fig2.suptitle('Electricity Generation (EU)', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax2.set_title('Absolute Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(bottom=0)
            ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax2.grid(True, alpha=0.3, linewidth=1.5)

            ax2.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                       fontsize=22, frameon=False)

            # Add timestamp (reuse same timestamp)
            fig2.text(0.93, 0.02, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename_abs = 'plots/eu_annual_renewable_vs_nonrenewable_absolute.png'
            plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved absolute: {filename_abs}")
            plt.close()

    # Year-over-Year Change vs 2015 Baseline
    print("\n" + "=" * 60)
    print("CREATING YOY CHANGE VS 2015 BASELINE")
    print("=" * 60)

    if annual_totals:
        baseline_year = 2015
        all_sources_for_yoy = available_renewables + available_non_renewables
        totals_for_yoy = ['All Renewables', 'All Non-Renewables']
        
        # Generate timestamp once for both plots
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

        # PLOT 1: ALL SOURCES YoY
        print("\nCreating YoY All Sources vs 2015...")
        
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        
        all_yoy_values = []
        lines_plotted = 0
        line_handles = {}
        
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
                        line, = ax1.plot(years_to_plot, yoy_changes, marker='o', color=color,
                                 linewidth=6, markersize=13, label=source_name)
                        line_handles[source_name] = line
                        lines_plotted += 1

        if lines_plotted > 0:
            # Calculate y-axis limits
            if all_yoy_values:
                y_min = min(all_yoy_values)
                y_max = max(all_yoy_values)
                y_margin = (y_max - y_min) * 0.1
                y_min_limit = y_min - y_margin
                y_max_limit = y_max + y_margin
            else:
                y_min_limit = -50
                y_max_limit = 100

            # Title and labels - clean format
            fig1.suptitle('Electricity Generation (EU)', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax1.set_title('Year-over-Year Change since 2015', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Change vs 2015 (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(y_min_limit, y_max_limit)
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax1.grid(True, alpha=0.3, linewidth=1.5)

            # Create legend with spacers for centered row 3 (renewables left, non-renewables right)
            from matplotlib.lines import Line2D
            spacer = Line2D([0], [0], color='none', label=' ')
            # ncol=4 fills by COLUMN, so order: col1, col2, col3, col4
            legend_order = [
                'Wind', 'Solar', '_SPACER_',              # Column 1: renewables
                'Hydro', 'Biomass', 'Geothermal',         # Column 2: renewables
                'Nuclear', 'Coal', 'Oil',                 # Column 3: non-renewables
                'Gas', 'Waste', '_SPACER_'                # Column 4: non-renewables
            ]
            
            legend_handles = []
            legend_labels = []
            for item in legend_order:
                if item == '_SPACER_':
                    legend_handles.append(spacer)
                    legend_labels.append(' ')
                elif item in line_handles:
                    legend_handles.append(line_handles[item])
                    legend_labels.append(item)
            
            ax1.legend(legend_handles, legend_labels,
                       loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=4,
                       fontsize=18, frameon=False)

            # Add timestamp
            fig1.text(0.93, 0.02, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename_yoy_all = 'plots/eu_annual_yoy_all_sources_vs_2015.png'
            plt.savefig(filename_yoy_all, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {filename_yoy_all}")
            plt.close()

        # PLOT 2: AGGREGATES YoY
        print("\nCreating YoY Aggregates vs 2015...")
        
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        
        agg_yoy_values = []
        
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
                            agg_yoy_values.append(pct_change)

                    years_to_plot = [year for year in years_list if year >= baseline_year]

                    if len(years_to_plot) > 0:
                        color = ENTSOE_COLORS[category_name]
                        ax2.plot(years_to_plot, yoy_changes, marker='o', color=color,
                                 linewidth=6, markersize=13, label=category_name)

        # Calculate y-axis limits
        if agg_yoy_values:
            y_min = min(agg_yoy_values)
            y_max = max(agg_yoy_values)
            y_margin = (y_max - y_min) * 0.1
            y_min_limit = y_min - y_margin
            y_max_limit = y_max + y_margin
        else:
            y_min_limit = -50
            y_max_limit = 100

        # Title and labels - clean format
        fig2.suptitle('Electricity Generation (EU)', 
                     fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax2.set_title('Year-over-Year Change since 2015', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Change vs 2015 (%)', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylim(y_min_limit, y_max_limit)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax2.grid(True, alpha=0.3, linewidth=1.5)

        ax2.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                   fontsize=22, frameon=False)

        # Add timestamp (reuse same timestamp)
        fig2.text(0.93, 0.02, f"Generated: {timestamp}",
                 ha='right', va='bottom', fontsize=11, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename_yoy_agg = 'plots/eu_annual_yoy_aggregates_vs_2015.png'
        plt.savefig(filename_yoy_agg, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {filename_yoy_agg}")
        plt.close()

    print("\n" + "=" * 60)
    print("ALL MOBILE-OPTIMIZED PLOTS GENERATED")
    print("=" * 60)


def update_summary_table_historical_data(all_data):
    """
    Update Google Sheets "Summary Table Data" with current year YTD and previous year data
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
        
        # Get current date info (needed for headers)
        current_date = datetime.now()
        current_year = current_date.year
        previous_year = current_year - 1
        two_years_ago = current_year - 2
        current_month = current_date.month
        
        # Get the Summary Table Data worksheet
        try:
            worksheet = spreadsheet.worksheet('Summary Table Data')
            print("✓ Found 'Summary Table Data' worksheet")
            
            # Check if worksheet has enough columns (need 22: A-V)
            if worksheet.col_count < 22:
                print(f"  Expanding worksheet from {worksheet.col_count} to 22 columns...")
                worksheet.resize(rows=worksheet.row_count, cols=22)
                print("  ✓ Worksheet expanded")
            
            # Always update header row to ensure consistency
            print("  Updating header row...")
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
            print("  ✓ Header row updated")
                
        except gspread.WorksheetNotFound:
            print("⚠ 'Summary Table Data' worksheet not found - run intraday analysis first")
            return
        
        # Define source categories
        renewables = ['Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal']
        non_renewables = ['Gas', 'Coal', 'Nuclear', 'Oil', 'Waste']
        all_sources = renewables + non_renewables
        
        # Calculate totals for each source
        source_calcs = {}
        
        for source_name in all_sources:
            if source_name not in all_data:
                continue
            
            year_data = all_data[source_name]['year_data']
            
            # Calculate YTD current year
            ytd_current_gwh = 0
            if current_year in year_data:
                for month in range(1, current_month + 1):
                    month_value = year_data[current_year].get(month, 0)
                    if month < current_month:
                        # Full month - sheet already has monthly total
                        ytd_current_gwh += month_value
                    else:
                        # Partial month - current month
                        current_day = current_date.day
                        days_in_month = calendar.monthrange(current_year, month)[1]
                        ytd_current_gwh += month_value * (current_day / days_in_month)
            
            # Calculate previous year full year
            year_previous_gwh = 0
            if previous_year in year_data:
                for month in range(1, 13):
                    month_value = year_data[previous_year].get(month, 0)
                    # Sheet already has monthly total, just sum
                    year_previous_gwh += month_value
            
            # Calculate current month from previous year (e.g., December 2024)
            # This is needed for accurate "Change from Previous Year" calculations
            current_month_previous_year_gwh = 0
            if previous_year in year_data:
                current_month_previous_year_gwh = year_data[previous_year].get(current_month, 0)
            
            source_calcs[source_name] = {
                'ytd_current_gwh': ytd_current_gwh,
                'year_previous_gwh': year_previous_gwh,
                'current_month_previous_year_gwh': current_month_previous_year_gwh
            }
        
        # Calculate 2015 baselines for change calculation
        baselines_2015 = {}
        
        for source_name in all_sources:
            if source_name not in all_data:
                continue
            
            year_data = all_data[source_name]['year_data']
            
            if 2015 not in year_data:
                continue
            
            # YTD 2025 baseline: Same period in 2015 (Jan-current_month, with same days)
            ytd_baseline = 0
            for month in range(1, current_month + 1):
                month_value = year_data[2015].get(month, 0)
                if month < current_month:
                    # Full month - sheet already has monthly total
                    ytd_baseline += month_value
                else:
                    # Partial month (up to current day)
                    current_day = current_date.day
                    days_in_month = calendar.monthrange(2015, month)[1]
                    ytd_baseline += month_value * (current_day / days_in_month)
            
            # 2015 full year baseline (for comparing to 2024)
            year_2015_total = 0
            for month in range(1, 13):
                month_value = year_data[2015].get(month, 0)
                # Sheet already has monthly total, just sum
                year_2015_total += month_value
            
            baselines_2015[source_name] = {
                'ytd': ytd_baseline,
                'year': year_2015_total
            }
        
        # Calculate aggregates: Use "All Renewables" from sheets if it exists, 
        # otherwise sum individual sources
        if 'All Renewables' in all_data:
            # Read directly from Google Sheets
            renewables_year_data = all_data['All Renewables']['year_data']
            
            renewables_ytd = 0
            if current_year in renewables_year_data:
                for month in range(1, current_month + 1):
                    month_value = renewables_year_data[current_year].get(month, 0)
                    if month < current_month:
                        # Full month - sheet already has monthly total
                        renewables_ytd += month_value
                    else:
                        # Partial month - current month
                        current_day = current_date.day
                        days_in_month = calendar.monthrange(current_year, month)[1]
                        renewables_ytd += month_value * (current_day / days_in_month)
            
            # Get previous year full year
            renewables_previous = 0
            if previous_year in renewables_year_data:
                for month in range(1, 13):
                    month_value = renewables_year_data[previous_year].get(month, 0)
                    # Sheet already has monthly total, just sum
                    renewables_previous += month_value
            
            # Get current month from previous year
            renewables_current_month_prev = 0
            if previous_year in renewables_year_data:
                renewables_current_month_prev = renewables_year_data[previous_year].get(current_month, 0)
        else:
            # Fallback: sum individual sources
            renewables_ytd = sum(source_calcs[s]['ytd_current_gwh'] for s in renewables if s in source_calcs)
            renewables_previous = sum(source_calcs[s]['year_previous_gwh'] for s in renewables if s in source_calcs)
            renewables_current_month_prev = sum(source_calcs[s]['current_month_previous_year_gwh'] for s in renewables if s in source_calcs)
        
        # Calculate All Non-Renewables from Total Generation - All Renewables
        # This ensures they sum to exactly 100%
        if 'Total Generation' in all_data:
            total_year_data = all_data['Total Generation']['year_data']
            
            # YTD current year total
            total_ytd = 0
            if current_year in total_year_data:
                for month in range(1, current_month + 1):
                    month_value = total_year_data[current_year].get(month, 0)
                    if month < current_month:
                        # Full month - sheet already has monthly total
                        total_ytd += month_value
                    else:
                        # Partial month - current month
                        current_day = current_date.day
                        days_in_month = calendar.monthrange(current_year, month)[1]
                        total_ytd += month_value * (current_day / days_in_month)
            
            non_renewables_ytd = total_ytd - renewables_ytd
            
            # Get previous year full year
            total_previous = 0
            if previous_year in total_year_data:
                for month in range(1, 13):
                    month_value = total_year_data[previous_year].get(month, 0)
                    # Sheet already has monthly total, just sum
                    total_previous += month_value
            
            # Get current month from previous year
            total_current_month_prev = 0
            if previous_year in total_year_data:
                total_current_month_prev = total_year_data[previous_year].get(current_month, 0)
            
            non_renewables_previous = total_previous - renewables_previous
            non_renewables_current_month_prev = total_current_month_prev - renewables_current_month_prev
        else:
            # Fallback: sum individual sources
            non_renewables_ytd = sum(source_calcs[s]['ytd_current_gwh'] for s in non_renewables if s in source_calcs)
            non_renewables_previous = sum(source_calcs[s]['year_previous_gwh'] for s in non_renewables if s in source_calcs)
            non_renewables_current_month_prev = sum(source_calcs[s]['current_month_previous_year_gwh'] for s in non_renewables if s in source_calcs)
        
        source_calcs['All Renewables'] = {
            'ytd_current_gwh': renewables_ytd,
            'year_previous_gwh': renewables_previous,
            'current_month_previous_year_gwh': renewables_current_month_prev
        }
        
        source_calcs['All Non-Renewables'] = {
            'ytd_current_gwh': non_renewables_ytd,
            'year_previous_gwh': non_renewables_previous,
            'current_month_previous_year_gwh': non_renewables_current_month_prev
        }
        
        # Add 2015 baselines for aggregates
        if 'All Renewables' in all_data:
            renewables_year_data = all_data['All Renewables']['year_data']
            if 2015 in renewables_year_data:
                # YTD baseline
                ytd_baseline = 0
                for month in range(1, current_month + 1):
                    month_value = renewables_year_data[2015].get(month, 0)
                    if month < current_month:
                        # Full month - sheet already has monthly total
                        ytd_baseline += month_value
                    else:
                        # Partial month
                        current_day = current_date.day
                        days_in_month = calendar.monthrange(2015, month)[1]
                        ytd_baseline += month_value * (current_day / days_in_month)
                
                # Full year baseline
                year_2015_total = 0
                for month in range(1, 13):
                    month_value = renewables_year_data[2015].get(month, 0)
                    # Sheet already has monthly total, just sum
                    year_2015_total += month_value
                
                baselines_2015['All Renewables'] = {
                    'ytd': ytd_baseline,
                    'year': year_2015_total
                }
        
        # All Non-Renewables 2015 baseline from Total - Renewables
        if 'Total Generation' in all_data and 'All Renewables' in baselines_2015:
            total_year_data = all_data['Total Generation']['year_data']
            if 2015 in total_year_data:
                # YTD baseline
                total_ytd_2015 = 0
                for month in range(1, current_month + 1):
                    month_value = total_year_data[2015].get(month, 0)
                    if month < current_month:
                        # Full month - sheet already has monthly total
                        total_ytd_2015 += month_value
                    else:
                        # Partial month
                        current_day = current_date.day
                        days_in_month = calendar.monthrange(2015, month)[1]
                        total_ytd_2015 += month_value * (current_day / days_in_month)
                
                # Full year baseline
                total_year_2015 = 0
                for month in range(1, 13):
                    month_value = total_year_data[2015].get(month, 0)
                    # Sheet already has monthly total, just sum
                    total_year_2015 += month_value
                
                baselines_2015['All Non-Renewables'] = {
                    'ytd': total_ytd_2015 - baselines_2015['All Renewables']['ytd'],
                    'year': total_year_2015 - baselines_2015['All Renewables']['year']
                }
        
        # Prepare updates with correct order
        source_order = [
            'All Renewables',
            'Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal',
            'All Non-Renewables',
            'Gas', 'Coal', 'Nuclear', 'Oil', 'Waste'
        ]
        
        # Prepare updates
        updates = []
        
        for row_idx, source_name in enumerate(source_order, start=2):
            if source_name not in source_calcs:
                continue
            
            ytd_current_gwh = source_calcs[source_name]['ytd_current_gwh']
            year_previous_gwh = source_calcs[source_name]['year_previous_gwh']
            
            # Calculate total generation for percentages
            ytd_current_pct = 0
            year_previous_pct = 0
            
            if 'Total Generation' in all_data:
                total_year_data = all_data['Total Generation']['year_data']
                
                # YTD current year percentage
                ytd_current_total = 0
                if current_year in total_year_data:
                    for month in range(1, current_month + 1):
                        month_value = total_year_data[current_year].get(month, 0)
                        if month < current_month:
                            # Full month - sheet already has monthly total
                            ytd_current_total += month_value
                        else:
                            # Partial month
                            current_day = current_date.day
                            days_in_month = calendar.monthrange(current_year, month)[1]
                            ytd_current_total += month_value * (current_day / days_in_month)
                
                ytd_current_pct = (ytd_current_gwh / ytd_current_total * 100) if ytd_current_total > 0 else 0
                
                # Previous year full year percentage
                year_previous_total = 0
                if previous_year in total_year_data:
                    for month in range(1, 13):
                        month_value = total_year_data[previous_year].get(month, 0)
                        # Sheet already has monthly total, just sum
                        year_previous_total += month_value
                
                year_previous_pct = (year_previous_gwh / year_previous_total * 100) if year_previous_total > 0 else 0
            
            # Calculate change from 2015
            ytd_change_2015 = ''
            year_previous_change_2015 = ''
            
            if source_name in baselines_2015:
                # YTD current year change from 2015
                baseline_ytd = baselines_2015[source_name]['ytd']
                if baseline_ytd > 0:
                    change = (ytd_current_gwh - baseline_ytd) / baseline_ytd * 100
                    ytd_change_2015 = format_change_percentage(change)
                
                # Previous year full year change from 2015
                baseline_year = baselines_2015[source_name]['year']
                if baseline_year > 0:
                    change = (year_previous_gwh - baseline_year) / baseline_year * 100
                    year_previous_change_2015 = format_change_percentage(change)
            
            # Add to updates list (columns F, G, H, I, M, N)
            updates.append({
                'range_fghi': f'F{row_idx}:I{row_idx}',
                'values_fghi': [[
                    f"{ytd_current_gwh:.1f}",
                    f"{ytd_current_pct:.2f}",
                    f"{year_previous_gwh:.1f}",
                    f"{year_previous_pct:.2f}"
                ]],
                'range_mn': f'M{row_idx}:N{row_idx}',
                'values_mn': [[ytd_change_2015, year_previous_change_2015]]
            })
        
        # Batch update all rows at once
        if updates:
            # Prepare batch updates
            batch_updates = []
            for update in updates:
                batch_updates.append({
                    'range': update['range_fghi'],
                    'values': update['values_fghi']
                })
                batch_updates.append({
                    'range': update['range_mn'],
                    'values': update['values_mn']
                })
            
            worksheet.batch_update(batch_updates)
            print(f"✓ Updated {len(updates)} sources with YTD {current_year} and {previous_year} data (columns F-I, M-N)")
            
            # Update timestamp in last column (batch update)
            timestamp = current_date.strftime('%Y-%m-%d %H:%M UTC')
            timestamp_updates = []
            for row_idx in range(2, 14):
                timestamp_updates.append({
                    'range': f'J{row_idx}',
                    'values': [[timestamp]]
                })
            worksheet.batch_update(timestamp_updates)
            print(f"✓ Updated timestamps")
            
            # ===================================================================
            # Calculate changes from previous year (e.g., 2024)
            # Read Yesterday/LastWeek values that were written by intraday script
            # ===================================================================
            print(f"\nCalculating changes from {previous_year}...")
            
            # Read back Yesterday and LastWeek columns (B-E)
            summary_data = worksheet.get('A2:E13')  # Source, Yesterday_GWh, Yesterday_%, LastWeek_GWh, LastWeek_%
            
            # Prepare updates for change from previous year columns
            change_updates = []
            
            for row_idx, row_data in enumerate(summary_data, start=2):
                source_name = row_data[0]
                
                try:
                    yesterday_gwh = float(row_data[1]) if len(row_data) > 1 and row_data[1] else 0
                    lastweek_gwh = float(row_data[3]) if len(row_data) > 3 and row_data[3] else 0
                except (ValueError, IndexError):
                    yesterday_gwh = 0
                    lastweek_gwh = 0
                
                # Get YTD and full year values from source_calcs
                if source_name not in source_calcs:
                    change_updates.append({
                        'range': f'O{row_idx}:R{row_idx}',
                        'values': [['', '', '', '']]
                    })
                    continue
                
                ytd_current_gwh = source_calcs[source_name]['ytd_current_gwh']
                year_previous_gwh = source_calcs[source_name]['year_previous_gwh']
                
                # Get December (current month) from previous year from monthly sheets
                current_month_prev_year_gwh = 0
                if source_name in ['All Renewables', 'All Non-Renewables']:
                    # Use pre-calculated aggregate value
                    current_month_prev_year_gwh = source_calcs[source_name].get('current_month_previous_year_gwh', 0)
                elif source_name in all_data:
                    # Get from monthly data
                    year_data = all_data[source_name]['year_data']
                    if previous_year in year_data:
                        current_month_prev_year_gwh = year_data[previous_year].get(current_month, 0)
                
                # Calculate changes
                yesterday_change_prev = ''
                lastweek_change_prev = ''
                ytd_change_prev = ''
                year_prev_change_prev = '—'  # Previous year compared to itself is always "—"
                
                # Yesterday change: compare to average day in December previous_year
                if current_month_prev_year_gwh > 0:
                    days_in_current_month = calendar.monthrange(previous_year, current_month)[1]
                    avg_day_prev_year = current_month_prev_year_gwh / days_in_current_month
                    if avg_day_prev_year > 0 and yesterday_gwh > 0:
                        change = (yesterday_gwh - avg_day_prev_year) / avg_day_prev_year * 100
                        yesterday_change_prev = format_change_percentage(change)
                
                # LastWeek change: compare to average week in December previous_year
                if current_month_prev_year_gwh > 0:
                    days_in_current_month = calendar.monthrange(previous_year, current_month)[1]
                    avg_week_prev_year = (current_month_prev_year_gwh / days_in_current_month) * 7
                    if avg_week_prev_year > 0 and lastweek_gwh > 0:
                        change = (lastweek_gwh - avg_week_prev_year) / avg_week_prev_year * 100
                        lastweek_change_prev = format_change_percentage(change)
                
                # YTD change: compare to same period in previous year
                if previous_year in all_data.get(source_name, {}).get('year_data', {}):
                    ytd_prev_year = 0
                    year_data = all_data[source_name]['year_data']
                    for month in range(1, current_month + 1):
                        month_value = year_data[previous_year].get(month, 0)
                        if month < current_month:
                            ytd_prev_year += month_value
                        else:
                            # Partial month
                            current_day = current_date.day
                            days_in_month = calendar.monthrange(previous_year, month)[1]
                            ytd_prev_year += month_value * (current_day / days_in_month)
                    
                    if ytd_prev_year > 0 and ytd_current_gwh > 0:
                        change = (ytd_current_gwh - ytd_prev_year) / ytd_prev_year * 100
                        ytd_change_prev = format_change_percentage(change)
                
                # For aggregates, handle YTD calculation differently
                if source_name in ['All Renewables', 'All Non-Renewables']:
                    # Need to calculate YTD for previous year from aggregates
                    if source_name == 'All Renewables' and 'All Renewables' in all_data:
                        renewables_year_data = all_data['All Renewables']['year_data']
                        if previous_year in renewables_year_data:
                            ytd_prev_year = 0
                            for month in range(1, current_month + 1):
                                month_value = renewables_year_data[previous_year].get(month, 0)
                                if month < current_month:
                                    ytd_prev_year += month_value
                                else:
                                    current_day = current_date.day
                                    days_in_month = calendar.monthrange(previous_year, month)[1]
                                    ytd_prev_year += month_value * (current_day / days_in_month)
                            
                            if ytd_prev_year > 0:
                                change = (ytd_current_gwh - ytd_prev_year) / ytd_prev_year * 100
                                ytd_change_prev = format_change_percentage(change)
                    
                    elif source_name == 'All Non-Renewables' and 'Total Generation' in all_data:
                        # Calculate from Total - Renewables
                        total_year_data = all_data['Total Generation']['year_data']
                        renewables_year_data = all_data.get('All Renewables', {}).get('year_data', {})
                        
                        if previous_year in total_year_data and previous_year in renewables_year_data:
                            ytd_total_prev = 0
                            ytd_renewables_prev = 0
                            
                            for month in range(1, current_month + 1):
                                month_total = total_year_data[previous_year].get(month, 0)
                                month_renewables = renewables_year_data[previous_year].get(month, 0)
                                
                                if month < current_month:
                                    ytd_total_prev += month_total
                                    ytd_renewables_prev += month_renewables
                                else:
                                    current_day = current_date.day
                                    days_in_month = calendar.monthrange(previous_year, month)[1]
                                    ytd_total_prev += month_total * (current_day / days_in_month)
                                    ytd_renewables_prev += month_renewables * (current_day / days_in_month)
                            
                            ytd_prev_year = ytd_total_prev - ytd_renewables_prev
                            if ytd_prev_year > 0:
                                change = (ytd_current_gwh - ytd_prev_year) / ytd_prev_year * 100
                                ytd_change_prev = format_change_percentage(change)
                
                # Add to change updates list
                change_updates.append({
                    'range': f'O{row_idx}:R{row_idx}',
                    'values': [[yesterday_change_prev, lastweek_change_prev, ytd_change_prev, year_prev_change_prev]]
                })
            
            # Write all change from previous year values in a single batch
            if change_updates:
                worksheet.batch_update(change_updates)
                print(f"✓ Updated {len(change_updates)} sources with changes from {previous_year} (columns O-R)")
            
            # ===================================================================
            # Calculate changes from 2 years ago (e.g., 2023)
            # Same logic as previous year changes
            # ===================================================================
            print(f"\nCalculating changes from {two_years_ago}...")
            
            # Prepare updates for change from 2 years ago columns
            change_2ya_updates = []
            
            for row_idx, row_data in enumerate(summary_data, start=2):
                source_name = row_data[0]
                
                try:
                    yesterday_gwh = float(row_data[1]) if len(row_data) > 1 and row_data[1] else 0
                    lastweek_gwh = float(row_data[3]) if len(row_data) > 3 and row_data[3] else 0
                except (ValueError, IndexError):
                    yesterday_gwh = 0
                    lastweek_gwh = 0
                
                # Get YTD and full year values from source_calcs
                if source_name not in source_calcs:
                    change_2ya_updates.append({
                        'range': f'S{row_idx}:V{row_idx}',
                        'values': [['', '', '', '']]
                    })
                    continue
                
                ytd_current_gwh = source_calcs[source_name]['ytd_current_gwh']
                year_previous_gwh = source_calcs[source_name]['year_previous_gwh']
                
                # Get December (current month) from 2 years ago from monthly sheets
                current_month_2ya_gwh = 0
                if source_name in ['All Renewables', 'All Non-Renewables']:
                    # For aggregates, need to calculate
                    if source_name == 'All Renewables' and 'All Renewables' in all_data:
                        renewables_year_data = all_data['All Renewables']['year_data']
                        if two_years_ago in renewables_year_data:
                            current_month_2ya_gwh = renewables_year_data[two_years_ago].get(current_month, 0)
                    elif source_name == 'All Non-Renewables' and 'Total Generation' in all_data:
                        total_year_data = all_data['Total Generation']['year_data']
                        renewables_year_data = all_data.get('All Renewables', {}).get('year_data', {})
                        if two_years_ago in total_year_data and two_years_ago in renewables_year_data:
                            total_month = total_year_data[two_years_ago].get(current_month, 0)
                            renewables_month = renewables_year_data[two_years_ago].get(current_month, 0)
                            current_month_2ya_gwh = total_month - renewables_month
                elif source_name in all_data:
                    # Get from monthly data
                    year_data = all_data[source_name]['year_data']
                    if two_years_ago in year_data:
                        current_month_2ya_gwh = year_data[two_years_ago].get(current_month, 0)
                
                # Get full year 2 years ago
                year_2ya_gwh = 0
                if source_name in ['All Renewables', 'All Non-Renewables']:
                    # For aggregates
                    if source_name == 'All Renewables' and 'All Renewables' in all_data:
                        renewables_year_data = all_data['All Renewables']['year_data']
                        if two_years_ago in renewables_year_data:
                            for month in range(1, 13):
                                year_2ya_gwh += renewables_year_data[two_years_ago].get(month, 0)
                    elif source_name == 'All Non-Renewables' and 'Total Generation' in all_data:
                        total_year_data = all_data['Total Generation']['year_data']
                        renewables_year_data = all_data.get('All Renewables', {}).get('year_data', {})
                        if two_years_ago in total_year_data and two_years_ago in renewables_year_data:
                            total_2ya = sum(total_year_data[two_years_ago].get(m, 0) for m in range(1, 13))
                            renewables_2ya = sum(renewables_year_data[two_years_ago].get(m, 0) for m in range(1, 13))
                            year_2ya_gwh = total_2ya - renewables_2ya
                elif source_name in all_data:
                    year_data = all_data[source_name]['year_data']
                    if two_years_ago in year_data:
                        for month in range(1, 13):
                            year_2ya_gwh += year_data[two_years_ago].get(month, 0)
                
                # Calculate changes
                yesterday_change_2ya = ''
                lastweek_change_2ya = ''
                ytd_change_2ya = ''
                year_prev_change_2ya = ''
                
                # Yesterday change: compare to average day in December 2 years ago
                if current_month_2ya_gwh > 0:
                    days_in_current_month = calendar.monthrange(two_years_ago, current_month)[1]
                    avg_day_2ya = current_month_2ya_gwh / days_in_current_month
                    if avg_day_2ya > 0 and yesterday_gwh > 0:
                        change = (yesterday_gwh - avg_day_2ya) / avg_day_2ya * 100
                        yesterday_change_2ya = format_change_percentage(change)
                
                # LastWeek change: compare to average week in December 2 years ago
                if current_month_2ya_gwh > 0:
                    days_in_current_month = calendar.monthrange(two_years_ago, current_month)[1]
                    avg_week_2ya = (current_month_2ya_gwh / days_in_current_month) * 7
                    if avg_week_2ya > 0 and lastweek_gwh > 0:
                        change = (lastweek_gwh - avg_week_2ya) / avg_week_2ya * 100
                        lastweek_change_2ya = format_change_percentage(change)
                
                # YTD change: compare to same period in 2 years ago
                ytd_2ya = 0
                if source_name in ['All Renewables', 'All Non-Renewables']:
                    if source_name == 'All Renewables' and 'All Renewables' in all_data:
                        renewables_year_data = all_data['All Renewables']['year_data']
                        if two_years_ago in renewables_year_data:
                            for month in range(1, current_month + 1):
                                month_value = renewables_year_data[two_years_ago].get(month, 0)
                                if month < current_month:
                                    ytd_2ya += month_value
                                else:
                                    current_day = current_date.day
                                    days_in_month = calendar.monthrange(two_years_ago, month)[1]
                                    ytd_2ya += month_value * (current_day / days_in_month)
                    elif source_name == 'All Non-Renewables' and 'Total Generation' in all_data:
                        total_year_data = all_data['Total Generation']['year_data']
                        renewables_year_data = all_data.get('All Renewables', {}).get('year_data', {})
                        if two_years_ago in total_year_data and two_years_ago in renewables_year_data:
                            ytd_total_2ya = 0
                            ytd_renewables_2ya = 0
                            for month in range(1, current_month + 1):
                                month_total = total_year_data[two_years_ago].get(month, 0)
                                month_renewables = renewables_year_data[two_years_ago].get(month, 0)
                                if month < current_month:
                                    ytd_total_2ya += month_total
                                    ytd_renewables_2ya += month_renewables
                                else:
                                    current_day = current_date.day
                                    days_in_month = calendar.monthrange(two_years_ago, month)[1]
                                    ytd_total_2ya += month_total * (current_day / days_in_month)
                                    ytd_renewables_2ya += month_renewables * (current_day / days_in_month)
                            ytd_2ya = ytd_total_2ya - ytd_renewables_2ya
                elif source_name in all_data:
                    year_data = all_data[source_name]['year_data']
                    if two_years_ago in year_data:
                        for month in range(1, current_month + 1):
                            month_value = year_data[two_years_ago].get(month, 0)
                            if month < current_month:
                                ytd_2ya += month_value
                            else:
                                current_day = current_date.day
                                days_in_month = calendar.monthrange(two_years_ago, month)[1]
                                ytd_2ya += month_value * (current_day / days_in_month)
                
                if ytd_2ya > 0 and ytd_current_gwh > 0:
                    change = (ytd_current_gwh - ytd_2ya) / ytd_2ya * 100
                    ytd_change_2ya = format_change_percentage(change)
                
                # Previous year (2024) change: compare 2024 to 2023
                if year_2ya_gwh > 0 and year_previous_gwh > 0:
                    change = (year_previous_gwh - year_2ya_gwh) / year_2ya_gwh * 100
                    year_prev_change_2ya = format_change_percentage(change)
                
                # Add to change updates list
                change_2ya_updates.append({
                    'range': f'S{row_idx}:V{row_idx}',
                    'values': [[yesterday_change_2ya, lastweek_change_2ya, ytd_change_2ya, year_prev_change_2ya]]
                })
            
            # Write all change from 2 years ago values in a single batch
            if change_2ya_updates:
                worksheet.batch_update(change_2ya_updates)
                print(f"✓ Updated {len(change_2ya_updates)} sources with changes from {two_years_ago} (columns S-V)")
            
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
