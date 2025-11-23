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
    'Wind Total': '#228B22',  # Forest Green
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
    'All Renewables': '#00CED1',  # Dark Turquoise - darker blue-green
    'All Non-Renewables': '#000000'  # Black
}


def get_screen_dimensions():
    """
    Get screen dimensions in inches for matplotlib figsize
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()

        screen_width_px = root.winfo_screenwidth()
        screen_height_px = root.winfo_screenheight()
        dpi = root.winfo_fpixels('1i')

        root.destroy()

        width_inches = (screen_width_px / dpi) * 0.95
        height_inches = (screen_height_px / dpi) * 0.90

        print(
            f"  Screen: {screen_width_px}x{screen_height_px}px, DPI: {dpi:.0f}, Figure: {width_inches:.1f}x{height_inches:.1f}in")

        return width_inches, height_inches
    except:
        return 19.2, 10.8


def maximize_figure():
    """
    Attempt to maximize the current figure window
    """
    try:
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend()

        if backend == 'TkAgg':
            try:
                manager.window.state('zoomed')
            except:
                try:
                    manager.window.attributes('-zoomed', True)
                except:
                    pass
        elif backend == 'wxAgg':
            try:
                manager.frame.Maximize(True)
            except:
                pass
        elif backend == 'Qt5Agg' or backend == 'Qt4Agg':
            try:
                manager.window.showMaximized()
            except:
                pass
        else:
            try:
                manager.window.showMaximized()
            except:
                try:
                    manager.frame.Maximize(True)
                except:
                    try:
                        manager.window.state('zoomed')
                    except:
                        pass
    except:
        pass


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
    Create all charts from the loaded data
    """
    if not all_data:
        print("No data available for plotting")
        return

    print("\n" + "=" * 60)
    print("CREATING CHARTS FROM GOOGLE SHEETS DATA")
    print("=" * 60)

    fig_width, fig_height = get_screen_dimensions()

    first_source = list(all_data.keys())[0]
    years_available = sorted(all_data[first_source]['year_data'].keys())
    print(f"Years available: {years_available}")

    month_names = [calendar.month_abbr[i] for i in range(1, 13)]

    created_figures = []

    # Create color gradient for years - distinct gradient with future buffer
    # Green → Cyan → Blue → Magenta → Red
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    n_years = 20  # Ensure we have colors for at least 2015-2034

    # Distinct gradient: green → cyan → blue → purple → magenta → red
    cmap = LinearSegmentedColormap.from_list('distinct_gradient',
                                             ['#006400',  # Dark green
                                              '#228B22',  # Forest green
                                              '#00CED1',  # Dark turquoise
                                              '#00BFFF',  # Deep sky blue
                                              '#0000FF',  # Blue
                                              '#4B0082',  # Indigo
                                              '#8B008B',  # Dark magenta
                                              '#FF00FF',  # Magenta
                                              '#FF1493',  # Deep pink
                                              '#DC143C',  # Crimson
                                              '#FF0000',  # Red
                                              '#B22222'])  # Fire brick
    year_colors = [mcolors.rgb2hex(cmap(i / (n_years - 1))) for i in range(n_years)]

    # COMBINED CHARTS: Solar, Wind, Hydro (Absolute + Percentage side-by-side)
    sources_to_plot = ['Solar', 'Wind Total', 'Hydro']

    # First pass: calculate max values for consistent y-axis ranges
    max_abs_value = 0
    max_pct_value = 0

    for source_name in sources_to_plot:
        if source_name in all_data and 'Total Generation' in all_data:
            year_data = all_data[source_name]['year_data']
            total_data = all_data['Total Generation']['year_data']

            for year in years_available:
                if year in year_data:
                    monthly_data = year_data[year]
                    for month in range(1, 13):
                        val = monthly_data.get(month, 0)
                        max_abs_value = max(max_abs_value, val)

                        if year in total_data:
                            total_val = total_data[year].get(month, 0)
                            if total_val > 0:
                                pct = (val / total_val) * 100
                                max_pct_value = max(max_pct_value, pct)

    # Add 10% margin
    max_abs_value *= 1.1
    max_pct_value *= 1.1

    for source_name in sources_to_plot:
        if source_name not in all_data or 'Total Generation' not in all_data:
            print(f"  ⚠ Skipping {source_name}")
            continue

        print(f"\nCreating combined chart for {source_name} (Absolute + Percentage)...")

        year_data = all_data[source_name]['year_data']
        total_data = all_data['Total Generation']['year_data']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        created_figures.append(fig)

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
            values = [monthly_data.get(month, 0) for month in months_to_show]

            color = year_colors[i % len(year_colors)]
            ax1.plot(months, values, marker='o', color=color, linewidth=3, markersize=9, label=str(year))

            if year in total_data:
                total_monthly = total_data[year]
                percentages = []
                for month in months_to_show:
                    source_val = monthly_data.get(month, 0)
                    total_val = total_monthly.get(month, 0)
                    if total_val > 0:
                        percentage = (source_val / total_val) * 100
                        percentages.append(percentage)
                    else:
                        percentages.append(0)

                ax2.plot(months, percentages, marker='o', color=color, linewidth=3, markersize=9, label=str(year))

        ax1.set_title(f'{source_name} Production (GWh)', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Month', fontsize=16)
        ax1.set_ylabel('Energy (GWh)', fontsize=16)
        ax1.set_ylim(0, max_abs_value)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.set_title(f'{source_name} % of Total Generation', fontsize=18, fontweight='bold')
        ax2.set_xlabel('Month', fontsize=16)
        ax2.set_ylabel('Percentage (%)', fontsize=16)
        ax2.set_ylim(0, max_pct_value)
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(years_available),
                   fontsize=12, frameon=False)

        fig.suptitle(f'Monthly {source_name} Energy Production Across EU (10-Year Comparison)',
                     fontsize=20, fontweight='bold', y=0.97)

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        maximize_figure()

        filename = f'plots/eu_monthly_{source_name.lower().replace(" ", "_")}_combined_10years.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Chart saved as: {filename}")

    # COMBINED CHART: Total Renewables (Absolute + Percentage)
    print("\n" + "=" * 60)
    print("CREATING TOTAL RENEWABLE CHART (COMBINED)")
    print("=" * 60)

    if 'All Renewables' in all_data and 'Total Generation' in all_data:
        print("\nCreating combined Total Renewable chart...")

        renewables_data = all_data['All Renewables']['year_data']
        total_data = all_data['Total Generation']['year_data']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        created_figures.append(fig)

        for i, year in enumerate(years_available):
            if year not in renewables_data or year not in total_data:
                continue

            renewables_monthly = renewables_data[year]
            total_monthly = total_data[year]

            current_date = datetime.now()
            current_year = current_date.year

            if year == current_year:
                months_to_show = range(1, current_date.month + 1)
            else:
                months_to_show = range(1, 13)

            months = [month_names[month - 1] for month in months_to_show]
            values = [renewables_monthly.get(month, 0) for month in months_to_show]

            color = year_colors[i % len(year_colors)]
            ax1.plot(months, values, marker='o', color=color, linewidth=3, markersize=9, label=str(year))

            percentages = []
            for month in months_to_show:
                renewables_val = renewables_monthly.get(month, 0)
                total_val = total_monthly.get(month, 0)
                if total_val > 0:
                    percentage = (renewables_val / total_val) * 100
                    percentages.append(percentage)
                else:
                    percentages.append(0)

            ax2.plot(months, percentages, marker='o', color=color, linewidth=3, markersize=9, label=str(year))

        ax1.set_title('Total Renewable Production (GWh)', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Month', fontsize=16)
        ax1.set_ylabel('Energy (GWh)', fontsize=16)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.set_title('Renewables % of Total Generation', fontsize=18, fontweight='bold')
        ax2.set_xlabel('Month', fontsize=16)
        ax2.set_ylabel('Percentage (%)', fontsize=16)
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(years_available),
                   fontsize=12, frameon=False)

        fig.suptitle('Monthly Total Renewable Energy Across EU (10-Year Comparison)',
                     fontsize=20, fontweight='bold', y=0.97)

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        maximize_figure()

        filename = 'plots/eu_monthly_total_renewable_combined_10years.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Chart saved as: {filename}")

    # Monthly Mean Charts by Period (COMBINED: Absolute + Percentage)
    print("\n" + "=" * 60)
    print("CREATING MONTHLY MEAN CHARTS BY PERIOD")
    print("=" * 60)

    all_energy_sources = ['Solar', 'Wind Total', 'Hydro', 'Gas', 'Coal', 'Oil', 'Waste', 'Nuclear', 'Geothermal',
                          'Biomass']
    available_sources = [source for source in all_energy_sources if source in all_data]

    periods = [
        {'name': '2015-2019', 'start': 2015, 'end': 2019},
        {'name': '2020-2024', 'start': 2020, 'end': 2024},
        {'name': '2025-2029', 'start': 2025, 'end': 2029}
    ]

    if available_sources and 'Total Generation' in all_data:
        months = [calendar.month_abbr[i] for i in range(1, 13)]

        # First pass: calculate max values across all periods for consistent y-axis
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

                            max_abs_all_periods = max(max_abs_all_periods, source_val)

                            if total_val > 0:
                                percentage = (source_val / total_val) * 100
                                max_pct_all_periods = max(max_pct_all_periods, percentage)

        # Add 10% margin
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

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
            created_figures.append(fig)

            for source_name in available_sources:
                color = ENTSOE_COLORS.get(source_name, 'black')

                ax1.plot(months, monthly_means_abs[source_name], marker='o', color=color,
                         linewidth=3, markersize=9, label=source_name)
                ax2.plot(months, monthly_means_pct[source_name], marker='o', color=color,
                         linewidth=3, markersize=9, label=source_name)

            ax1.set_title('Production (GWh)', fontsize=18, fontweight='bold')
            ax1.set_xlabel('Month', fontsize=16)
            ax1.set_ylabel('Energy (GWh)', fontsize=16)
            ax1.set_ylim(0, max_abs_all_periods)
            ax1.tick_params(axis='both', labelsize=14)
            ax1.grid(True, linestyle='--', alpha=0.7)

            ax2.set_title('% of Total Generation', fontsize=18, fontweight='bold')
            ax2.set_xlabel('Month', fontsize=16)
            ax2.set_ylabel('Percentage (%)', fontsize=16)
            ax2.set_ylim(0, max_pct_all_periods)
            ax2.tick_params(axis='both', labelsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=5,
                       fontsize=12, frameon=False)

            fig.suptitle(f'Monthly Mean Energy Sources by Period: {period["name"]}',
                         fontsize=20, fontweight='bold', y=0.97)

            plt.tight_layout(rect=[0, 0.06, 1, 0.96])
            maximize_figure()

            period_name_clean = period['name'].replace('-', '_')
            filename = f'plots/eu_monthly_energy_sources_mean_{period_name_clean}_combined.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")

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

    # Renewable vs Non-Renewable by Period (COMBINED: Absolute + Percentage)
    print("\n" + "=" * 60)
    print("CREATING RENEWABLE VS NON-RENEWABLE CHARTS")
    print("=" * 60)

    if 'All Renewables' in all_data and 'All Non-Renewables' in all_data and 'Total Generation' in all_data:
        month_names_abbr = [calendar.month_abbr[i] for i in range(1, 13)]

        # First pass: calculate max values across all periods for consistent y-axis
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
                            max_abs_renewable_periods = max(max_abs_renewable_periods, category_val)

        # Add 10% margin
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

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
            created_figures.append(fig)

            for category_name in ['All Renewables', 'All Non-Renewables']:
                color = ENTSOE_COLORS[category_name]

                ax1.plot(month_names_abbr, monthly_means_abs[category_name], marker='o', color=color,
                         linewidth=3, markersize=9, label=category_name)
                ax2.plot(month_names_abbr, monthly_means_pct[category_name], marker='o', color=color,
                         linewidth=3, markersize=9, label=category_name)

            ax1.set_title('Production (GWh)', fontsize=18, fontweight='bold')
            ax1.set_xlabel('Month', fontsize=16)
            ax1.set_ylabel('Energy (GWh)', fontsize=16)
            ax1.set_ylim(0, max_abs_renewable_periods)
            ax1.tick_params(axis='both', labelsize=14)
            ax1.grid(True, linestyle='--', alpha=0.7)

            ax2.set_title('% of Total Generation', fontsize=18, fontweight='bold')
            ax2.set_xlabel('Month', fontsize=16)
            ax2.set_ylabel('Percentage (%)', fontsize=16)
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='both', labelsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2,
                       fontsize=14, frameon=False)

            fig.suptitle(f'Monthly Mean Renewable vs Non-Renewable: {period["name"]}',
                         fontsize=20, fontweight='bold', y=0.97)

            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            maximize_figure()

            period_name_clean = period['name'].replace('-', '_')
            filename = f'plots/eu_monthly_renewable_vs_nonrenewable_mean_{period_name_clean}_combined.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")

    # Annual Trend Charts
    print("\n" + "=" * 60)
    print("CREATING ANNUAL TREND CHARTS")
    print("=" * 60)

    annual_totals = {}

    renewable_sources = ['Solar', 'Wind Total', 'Hydro', 'Biomass', 'Geothermal']
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

    # Chart: Renewable Trends (COMBINED: Absolute + Percentage)
    if available_renewables and 'Total Generation' in annual_totals:
        print("\nCreating Annual Renewable Trends (Combined)...")

        # Calculate max value for both renewable and non-renewable plots to match y-axis
        max_annual_twh = 0
        max_annual_pct = 0

        for source_name in available_renewables + available_non_renewables:
            if source_name in annual_totals and 'Total Generation' in annual_totals:
                years_list = sorted(annual_totals[source_name].keys())
                for year in years_list:
                    val_twh = annual_totals[source_name][year] / 1000
                    max_annual_twh = max(max_annual_twh, val_twh)

                    # Calculate percentage
                    source_value = annual_totals[source_name][year]
                    total_value = annual_totals['Total Generation'][year]
                    if total_value > 0:
                        percentage = (source_value / total_value) * 100
                        max_annual_pct = max(max_annual_pct, percentage)

        # Add 10% margin
        max_annual_twh *= 1.1
        max_annual_pct *= 1.1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        created_figures.append(fig)

        lines_plotted = 0
        for source_name in available_renewables:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())

                values_twh = [annual_totals[source_name][year] / 1000 for year in years_list]
                color = ENTSOE_COLORS.get(source_name, 'black')
                ax1.plot(years_list, values_twh, marker='o', color=color,
                         linewidth=3, markersize=9, label=source_name)

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

                    ax2.plot(pct_years, percentages, marker='o', color=color,
                             linewidth=3, markersize=9, label=source_name)

                lines_plotted += 1

        if lines_plotted > 0:
            ax1.set_title('Production (TWh)', fontsize=18, fontweight='bold')
            ax1.set_xlabel('Year', fontsize=16)
            ax1.set_ylabel('Energy Production (TWh)', fontsize=16)
            ax1.set_ylim(0, max_annual_twh)
            ax1.tick_params(axis='both', labelsize=14)
            ax1.grid(True, linestyle='--', alpha=0.7)

            ax2.set_title('% of Total Generation', fontsize=18, fontweight='bold')
            ax2.set_xlabel('Year', fontsize=16)
            ax2.set_ylabel('Percentage (%)', fontsize=16)
            ax2.set_ylim(0, max_annual_pct)
            ax2.tick_params(axis='both', labelsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(available_renewables),
                       fontsize=12, frameon=False)

            fig.suptitle('Annual EU Renewable Energy Production Trends',
                         fontsize=20, fontweight='bold', y=0.97)

            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            maximize_figure()

            filename = 'plots/eu_annual_renewable_trends_combined.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")

    # Chart: Non-Renewable Trends (COMBINED)
    if available_non_renewables and 'Total Generation' in annual_totals:
        print("\nCreating Annual Non-Renewable Trends (Combined)...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        created_figures.append(fig)

        lines_plotted = 0
        for source_name in available_non_renewables:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())

                values_twh = [annual_totals[source_name][year] / 1000 for year in years_list]
                color = ENTSOE_COLORS.get(source_name, 'black')
                ax1.plot(years_list, values_twh, marker='o', color=color,
                         linewidth=3, markersize=9, label=source_name)

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

                    ax2.plot(pct_years, percentages, marker='o', color=color,
                             linewidth=3, markersize=9, label=source_name)

                lines_plotted += 1

        if lines_plotted > 0:
            ax1.set_title('Production (TWh)', fontsize=18, fontweight='bold')
            ax1.set_xlabel('Year', fontsize=16)
            ax1.set_ylabel('Energy Production (TWh)', fontsize=16)
            ax1.set_ylim(0, max_annual_twh)
            ax1.tick_params(axis='both', labelsize=14)
            ax1.grid(True, linestyle='--', alpha=0.7)

            ax2.set_title('% of Total Generation', fontsize=18, fontweight='bold')
            ax2.set_xlabel('Year', fontsize=16)
            ax2.set_ylabel('Percentage (%)', fontsize=16)
            ax2.set_ylim(0, max_annual_pct)
            ax2.tick_params(axis='both', labelsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                       ncol=len(available_non_renewables), fontsize=12, frameon=False)

            fig.suptitle('Annual EU Non-Renewable Energy Production Trends',
                         fontsize=20, fontweight='bold', y=0.97)

            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            maximize_figure()

            filename = 'plots/eu_annual_non_renewable_trends_combined.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")

    # Chart: Renewables vs Non-Renewables Totals (COMBINED)
    if available_totals and 'Total Generation' in annual_totals:
        print("\nCreating Annual Renewables vs Non-Renewables (Combined)...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        created_figures.append(fig)

        lines_plotted = 0
        for source_name in available_totals:
            if source_name in annual_totals and len(annual_totals[source_name]) > 0:
                years_list = sorted(annual_totals[source_name].keys())

                values_twh = [annual_totals[source_name][year] / 1000 for year in years_list]
                color = ENTSOE_COLORS[source_name]
                ax1.plot(years_list, values_twh, marker='o', color=color,
                         linewidth=3, markersize=9, label=source_name)

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

                    ax2.plot(pct_years, percentages, marker='o', color=color,
                             linewidth=3, markersize=9, label=source_name)

                lines_plotted += 1

        if lines_plotted > 0:
            ax1.set_title('Production (TWh)', fontsize=18, fontweight='bold')
            ax1.set_xlabel('Year', fontsize=16)
            ax1.set_ylabel('Energy Production (TWh)', fontsize=16)
            ax1.set_ylim(bottom=0)
            ax1.tick_params(axis='both', labelsize=14)
            ax1.grid(True, linestyle='--', alpha=0.7)

            ax2.set_title('% of Total Generation', fontsize=18, fontweight='bold')
            ax2.set_xlabel('Year', fontsize=16)
            ax2.set_ylabel('Percentage (%)', fontsize=16)
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='both', labelsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2,
                       fontsize=14, frameon=False)

            fig.suptitle('Annual EU Energy Transition: Renewables vs Non-Renewables',
                         fontsize=20, fontweight='bold', y=0.97)

            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            maximize_figure()

            filename = 'plots/eu_annual_renewable_vs_non_renewable_combined.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")

    # Year-over-Year Change vs 2015 Baseline
    print("\n" + "=" * 60)
    print("CREATING YOY CHANGE VS 2015 BASELINE")
    print("=" * 60)

    if annual_totals:
        print("\nCreating YoY change vs 2015 baseline chart...")

        # Calculate baseline year (2015)
        baseline_year = 2015

        # Left plot: All sources
        all_sources_for_yoy = available_renewables + available_non_renewables

        # Right plot: Just totals
        totals_for_yoy = ['All Renewables', 'All Non-Renewables']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        created_figures.append(fig)

        # Track min/max for shared y-axis
        all_yoy_values = []

        # Left plot: All individual sources
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
                                 linewidth=3, markersize=9, label=source_name)
                        lines_plotted += 1

        # Right plot: Just All Renewables and All Non-Renewables
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
                                 linewidth=3, markersize=9, label=category_name)

        if lines_plotted > 0:
            # Calculate shared y-axis limits
            if all_yoy_values:
                y_min = min(all_yoy_values)
                y_max = max(all_yoy_values)
                y_margin = (y_max - y_min) * 0.1
                y_min_limit = y_min - y_margin
                y_max_limit = y_max + y_margin
            else:
                y_min_limit = -50
                y_max_limit = 100

            ax1.set_title('All Energy Sources', fontsize=18, fontweight='bold')
            ax1.set_xlabel('Year', fontsize=16)
            ax1.set_ylabel('% Change from 2015', fontsize=16)
            ax1.set_ylim(y_min_limit, y_max_limit)
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax1.tick_params(axis='both', labelsize=14)
            ax1.grid(True, linestyle='--', alpha=0.7)

            ax2.set_title('Renewables vs Non-Renewables', fontsize=18, fontweight='bold')
            ax2.set_xlabel('Year', fontsize=16)
            ax2.set_ylabel('% Change from 2015', fontsize=16)
            ax2.set_ylim(y_min_limit, y_max_limit)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.tick_params(axis='both', labelsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()

            # Combine legends at bottom
            all_handles = handles1 + handles2
            all_labels = labels1 + labels2
            fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                       ncol=6, fontsize=11, frameon=False)

            fig.suptitle('Year-over-Year Change in EU Energy Production vs 2015 Baseline',
                         fontsize=20, fontweight='bold', y=0.97)

            plt.tight_layout(rect=[0, 0.06, 1, 0.96])
            maximize_figure()

            filename = 'plots/eu_annual_yoy_change_vs_2015.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Chart saved as: {filename}")

    return created_figures


def main():
    """
    Main function
    """
    print("=" * 60)
    print("EU ENERGY DATA PLOTTER (SECURED)")
    print("=" * 60)
    print("\nFEATURES:")
    print("  ✓ Secure: Uses environment variables for credentials")
    print("  ✓ UNIFIED sizes: ALL plots use linewidth=3, markersize=9")
    print("  ✓ DISTINCT gradient: green → cyan → blue → magenta → red")
    print("  ✓ All colors highly saturated")
    print("  ✓ Combined absolute + percentage charts")
    print("  ✓ Includes 2025 data in annual trends")
    print("  ✓ YoY change vs 2015 baseline")
    print("  ✓ Full-screen display")
    print("  ✓ ENTSO-E colors")
    print("=" * 60)

    # Verify environment variables are set
    if not os.environ.get('GOOGLE_CREDENTIALS_JSON'):
        print("\n⚠️  WARNING: GOOGLE_CREDENTIALS_JSON environment variable not set!")
        print("   Please set this variable before running the script.")
        return

    all_data = load_data_from_google_sheets()

    if not all_data:
        print("Failed to load data.")
        return

    created_figures = create_all_charts(all_data)

    if created_figures:
        print(f"\n{'=' * 60}")
        print(f"Created {len(created_figures)} figures")
        print(f"{'=' * 60}")
        print("\nDisplaying charts...")
        plt.show()
    else:
        print("No charts created.")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
