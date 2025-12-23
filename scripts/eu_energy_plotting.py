import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
from datetime import datetime
import os
import json

# Google Drive imports (for plot hosting)
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    print("⚠ Google Drive API not available - plots will not be uploaded to Drive")

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


# === GOOGLE DRIVE UPLOAD FUNCTIONS ===

def get_or_create_drive_folder(service, folder_name, parent_id=None):
    """Get or create a folder in Google Drive. Returns folder ID."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    existing_folders = results.get('files', [])
    
    if existing_folders:
        return existing_folders[0]['id']
    
    # Create new folder
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        file_metadata['parents'] = [parent_id]
    
    folder = service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')


def upload_plot_to_drive(service, file_path, plot_type='Monthly', country='EU'):
    """
    Upload a plot to Google Drive with geography-first structure.
    Structure: EU-Electricity-Plots/[Country]/[Monthly|Trends]/[plot].png
    Returns: dict with file_id, view_url, direct_url or None if failed
    """
    if not GDRIVE_AVAILABLE:
        return None
    
    try:
        # Create folder structure: EU-Electricity-Plots/[Country]/[Monthly|Trends]/
        root_folder_id = get_or_create_drive_folder(service, 'EU-Electricity-Plots')
        country_folder_id = get_or_create_drive_folder(service, country, root_folder_id)
        type_folder_id = get_or_create_drive_folder(service, plot_type, country_folder_id)
        
        filename = os.path.basename(file_path)
        
        # Check if file already exists
        query = f"name='{filename}' and '{type_folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        existing_files = results.get('files', [])
        
        if existing_files:
            # Update existing file
            file_id = existing_files[0]['id']
            media = MediaFileUpload(file_path, mimetype='image/png')
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            # Create new file
            file_metadata = {'name': filename, 'parents': [type_folder_id]}
            media = MediaFileUpload(file_path, mimetype='image/png')
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            file_id = file.get('id')
        
        # Set permissions to "Anyone with the link can view"
        try:
            existing_perms = service.permissions().list(
                fileId=file_id,
                fields='permissions(id,type)'
            ).execute()
            
            anyone_perm = None
            for perm in existing_perms.get('permissions', []):
                if perm.get('type') == 'anyone':
                    anyone_perm = perm
                    break
            
            if anyone_perm:
                service.permissions().update(
                    fileId=file_id,
                    permissionId=anyone_perm['id'],
                    body={'role': 'reader'}
                ).execute()
            else:
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
        
        return {
            'file_id': file_id,
            'view_url': f'https://drive.google.com/file/d/{file_id}/view',
            'direct_url': f'https://drive.google.com/thumbnail?id={file_id}&sz=w2000',
            'updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"  ⚠ Drive upload failed for {os.path.basename(file_path)}: {e}")
        return None


def initialize_drive_service():
    """Initialize Google Drive service using credentials from environment."""
    if not GDRIVE_AVAILABLE:
        return None
    
    try:
        google_creds_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            print("  ⚠ GOOGLE_CREDENTIALS_JSON not set")
            return None
        
        creds_dict = json.loads(google_creds_json)
        credentials = ServiceAccountCredentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"  ⚠ Failed to initialize Drive service: {e}")
        return None


def format_change_percentage(value):
    """
    Format change percentage with smart decimal handling
    - If |value| >= 10: No decimals (e.g., "+180%", "-58%")
    - If |value| < 10: One decimal (e.g., "+5.8%", "+0.3%", "-2.1%")
    """
    if abs(value) >= 10:
        return f"{value:+.0f}%"
    else:
        return f"{value:+.1f}%"


def get_or_create_country_sheet(country_code, gc, drive_service):
    """
    Get or create country-specific Google Sheet, and ensure it's in correct Drive folder.
    Mirror of intraday script logic.
    Returns: gspread spreadsheet object
    """
    sheet_name = f'{country_code} Electricity Production Data'
    
    try:
        spreadsheet = gc.open(sheet_name)
        print(f"✓ Opened existing sheet: {sheet_name}")
    except gspread.SpreadsheetNotFound:
        print(f"✗ Sheet not found: {sheet_name}")
        print(f"  Note: Monthly data sheets are created by the data fetching process.")
        return None
    
    # Move to correct Drive folder structure (runs every time)
    try:
        query = "name='EU-Electricity-Plots' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        folders = results.get('files', [])
        
        if folders:
            root_folder_id = folders[0]['id']
            query = f"name='{country_code}' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
            country_folders = results.get('files', [])
            
            country_folder_id = country_folders[0]['id'] if country_folders else None
            if not country_folder_id:
                file_metadata = {
                    'name': country_code,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [root_folder_id]
                }
                folder = drive_service.files().create(body=file_metadata, fields='id').execute()
                country_folder_id = folder.get('id')
                print(f"  ✓ Created folder: EU-Electricity-Plots/{country_code}/")
            
            file = drive_service.files().get(fileId=spreadsheet.id, fields='parents').execute()
            current_parents = file.get('parents', [])
            
            if country_folder_id not in current_parents:
                previous_parents = ",".join(current_parents)
                drive_service.files().update(
                    fileId=spreadsheet.id,
                    addParents=country_folder_id,
                    removeParents=previous_parents,
                    fields='id, parents'
                ).execute()
                print(f"  ✓ Moved to: EU-Electricity-Plots/{country_code}/")
            else:
                print(f"  ✓ Already in: EU-Electricity-Plots/{country_code}/")
    except Exception as e:
        print(f"  ⚠ Could not move to Drive folder: {e}")
    
    return spreadsheet


def load_data_from_google_sheets(country_code='EU'):
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

        # Initialize Drive service
        drive_service = initialize_drive_service()
        
        # Get or create sheet and ensure it's in correct folder
        spreadsheet = get_or_create_country_sheet(country_code, gc, drive_service)
        if not spreadsheet:
            return None
        
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



def create_all_charts(all_data, country_code='EU'):
    """
    Create all charts from the loaded data - MOBILE OPTIMIZED
    UPDATED: Larger fonts, thicker lines, clearer titles, no Y-axis restrictions
    """
    if not all_data:
        print("No data available for plotting")
        return None, None

    print("\n" + "=" * 60)
    print(f"CREATING MOBILE-OPTIMIZED CHARTS FOR {country_code}")
    print("=" * 60)
    
    # Initialize Google Drive service for uploading plots
    drive_service = initialize_drive_service()
    plot_links = {'Monthly': {}, 'Trends': {}}

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
                    source_val = values_gwh[list(months_to_show).index(month)]
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
        display_name = source_name
        if source_name == 'All Renewables':
            display_name = 'Renewable'
        elif source_name == 'All Non-Renewables':
            display_name = 'Non-Renewable'
        
        fig1.suptitle(f'{display_name} Electricity Generation ({country_code})', fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax1.set_title('Fraction of Total Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
        
        ax1.set_ylim(0, max_pct_value * 1.2 if max_pct_value > 0 else 10)
            
        ax1.tick_params(axis='both', labelsize=22)
        ax1.grid(True, alpha=0.3, linewidth=1.5)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), 
                  ncol=5, fontsize=20, frameon=False)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        fig1.text(0.93, 0.04, f"Generated: {timestamp}",
                  ha='right', va='bottom',
                  fontsize=11, color='#666',
                  style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        percentage_filename = f'plots/{country_code.lower()}_monthly_{source_name.lower().replace(" ", "_")}_percentage_10years.png'
        plt.savefig(percentage_filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {percentage_filename}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, percentage_filename, plot_type='Monthly', country=country_code)
            if result:
                if source_name not in plot_links['Monthly']:
                    plot_links['Monthly'][source_name] = {}
                plot_links['Monthly'][source_name]['percentage'] = result
        
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

        display_name = source_name
        if source_name == 'All Renewables':
            display_name = 'Renewable'
        elif source_name == 'All Non-Renewables':
            display_name = 'Non-Renewable'
        
        fig2.suptitle(f'{display_name} Electricity Generation ({country_code})', fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax2.set_title('Absolute Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
        
        ax2.set_ylim(0, max_abs_value * 1.2 if max_abs_value > 0 else 10)
            
        ax2.tick_params(axis='both', labelsize=22)
        ax2.grid(True, alpha=0.3, linewidth=1.5)

        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
                  ncol=5, fontsize=20, frameon=False)

        fig2.text(0.93, 0.04, f"Generated: {timestamp}",
                  ha='right', va='bottom',
                  fontsize=11, color='#666',
                  style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        absolute_filename = f'plots/{country_code.lower()}_monthly_{source_name.lower().replace(" ", "_")}_absolute_10years.png'
        plt.savefig(absolute_filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {absolute_filename}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, absolute_filename, plot_type='Monthly', country=country_code)
            if result:
                if source_name not in plot_links['Monthly']:
                    plot_links['Monthly'][source_name] = {}
                plot_links['Monthly'][source_name]['absolute'] = result
        
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

        max_abs_all_periods *= 1.2
        max_pct_all_periods *= 1.2

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

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
            
            # PERCENTAGE PLOT
            fig1, ax1 = plt.subplots(figsize=(12, 10))

            from matplotlib.lines import Line2D
            line_handles = {}
            
            for source_name in available_sources:
                color = ENTSOE_COLORS.get(source_name, 'black')
                line, = ax1.plot(months, monthly_means_pct[source_name], marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)
                line_handles[source_name] = line

            fig1.suptitle(f'Electricity Generation ({country_code})', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax1.set_title(f'Fraction of Total Generation ({period["name"]})', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, max_pct_all_periods)
            ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax1.grid(True, alpha=0.3, linewidth=1.5)

            spacer = Line2D([0], [0], color='none', label=' ')
            legend_order = [
                'Wind', 'Solar', '_SPACER_',
                'Hydro', 'Biomass', 'Geothermal',
                'Nuclear', 'Coal', 'Oil',
                'Gas', 'Waste', '_SPACER_'
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

            fig1.text(0.93, 0.04, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            period_name_clean = period['name'].replace('-', '_')
            filename_pct = f'plots/{country_code.lower()}_monthly_energy_sources_mean_{period_name_clean}_percentage.png'
            plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved percentage: {filename_pct}")
            
            # Upload to Drive
            if drive_service:
                result = upload_plot_to_drive(drive_service, filename_pct, plot_type='Monthly', country=country_code)
                if result:
                    plot_key = f'energy_sources_mean_{period_name_clean}'
                    if plot_key not in plot_links['Monthly']:
                        plot_links['Monthly'][plot_key] = {}
                    plot_links['Monthly'][plot_key]['percentage'] = result
            
            plt.close()

            # ABSOLUTE PLOT
            fig2, ax2 = plt.subplots(figsize=(12, 10))

            line_handles = {}
            
            for source_name in available_sources:
                color = ENTSOE_COLORS.get(source_name, 'black')
                values_twh = [val / 1000 for val in monthly_means_abs[source_name]]
                line, = ax2.plot(months, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)
                line_handles[source_name] = line

            fig2.suptitle(f'Electricity Generation ({country_code})', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax2.set_title(f'Absolute Generation ({period["name"]})', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(0, max_abs_all_periods)
            ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax2.grid(True, alpha=0.3, linewidth=1.5)

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

            fig2.text(0.93, 0.04, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename_abs = f'plots/{country_code.lower()}_monthly_energy_sources_mean_{period_name_clean}_absolute.png'
            plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved absolute: {filename_abs}")
            
            # Upload to Drive
            if drive_service:
                result = upload_plot_to_drive(drive_service, filename_abs, plot_type='Monthly', country=country_code)
                if result:
                    plot_key = f'energy_sources_mean_{period_name_clean}'
                    if plot_key not in plot_links['Monthly']:
                        plot_links['Monthly'][plot_key] = {}
                    plot_links['Monthly'][plot_key]['absolute'] = result
            
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

        max_abs_renewable_periods *= 1.2

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

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
            
            # PERCENTAGE PLOT
            fig1, ax1 = plt.subplots(figsize=(12, 10))

            for category_name in ['All Renewables', 'All Non-Renewables']:
                color = ENTSOE_COLORS[category_name]
                ax1.plot(month_names_abbr, monthly_means_pct[category_name], marker='o', color=color,
                         linewidth=6, markersize=13, label=category_name)

            fig1.suptitle(f'Electricity Generation ({country_code})', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax1.set_title(f'Fraction of Total Generation ({period["name"]})', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax1.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax1.grid(True, alpha=0.3, linewidth=1.5)

            ax1.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                       fontsize=22, frameon=False)

            fig1.text(0.93, 0.04, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            period_name_clean = period['name'].replace('-', '_')
            filename_pct = f'plots/{country_code.lower()}_monthly_renewable_vs_nonrenewable_mean_{period_name_clean}_percentage.png'
            plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved percentage: {filename_pct}")
            
            # Upload to Drive
            if drive_service:
                result = upload_plot_to_drive(drive_service, filename_pct, plot_type='Monthly', country=country_code)
                if result:
                    plot_key = f'renewable_vs_nonrenewable_mean_{period_name_clean}'
                    if plot_key not in plot_links['Monthly']:
                        plot_links['Monthly'][plot_key] = {}
                    plot_links['Monthly'][plot_key]['percentage'] = result
            
            plt.close()

            # ABSOLUTE PLOT
            fig2, ax2 = plt.subplots(figsize=(12, 10))

            for category_name in ['All Renewables', 'All Non-Renewables']:
                color = ENTSOE_COLORS[category_name]
                values_twh = [val / 1000 for val in monthly_means_abs[category_name]]
                ax2.plot(month_names_abbr, values_twh, marker='o', color=color,
                         linewidth=6, markersize=13, label=category_name)

            fig2.suptitle(f'Electricity Generation ({country_code})', 
                         fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
            ax2.set_title(f'Absolute Generation ({period["name"]})', fontsize=26, fontweight='normal', pad=10, ha='center')
            ax2.set_xlabel('Month', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
            ax2.set_ylim(0, max_abs_renewable_periods)
            ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
            ax2.grid(True, alpha=0.3, linewidth=1.5)

            ax2.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                       fontsize=22, frameon=False)

            fig2.text(0.93, 0.04, f"Generated: {timestamp}",
                     ha='right', va='bottom', fontsize=11, color='#666', style='italic')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            filename_abs = f'plots/{country_code.lower()}_monthly_renewable_vs_nonrenewable_mean_{period_name_clean}_absolute.png'
            plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved absolute: {filename_abs}")
            
            # Upload to Drive
            if drive_service:
                result = upload_plot_to_drive(drive_service, filename_abs, plot_type='Monthly', country=country_code)
                if result:
                    plot_key = f'renewable_vs_nonrenewable_mean_{period_name_clean}'
                    if plot_key not in plot_links['Monthly']:
                        plot_links['Monthly'][plot_key] = {}
                    plot_links['Monthly'][plot_key]['absolute'] = result
            
            plt.close()


    # Annual Trends - All Sources
    print("\n" + "=" * 60)
    print("CREATING ANNUAL TRENDS - ALL SOURCES")
    print("=" * 60)

    if available_sources and 'Total Generation' in all_data:
        annual_totals_abs = {}
        annual_totals_pct = {}

        for source_name in available_sources:
            annual_totals_abs[source_name] = []
            annual_totals_pct[source_name] = []

            source_data = all_data[source_name]['year_data']
            total_data = all_data['Total Generation']['year_data']

            for year in years_available:
                if year in source_data and year in total_data:
                    yearly_sum = sum(source_data[year].values())
                    total_yearly_sum = sum(total_data[year].values())

                    annual_totals_abs[source_name].append(yearly_sum)

                    if total_yearly_sum > 0:
                        percentage = (yearly_sum / total_yearly_sum) * 100
                        annual_totals_pct[source_name].append(percentage)
                    else:
                        annual_totals_pct[source_name].append(0)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

        # PERCENTAGE PLOT
        fig1, ax1 = plt.subplots(figsize=(12, 10))

        from matplotlib.lines import Line2D
        line_handles = {}

        for source_name in available_sources:
            color = ENTSOE_COLORS.get(source_name, 'black')
            line, = ax1.plot(years_available, annual_totals_pct[source_name], marker='o', color=color,
                     linewidth=6, markersize=13, label=source_name)
            line_handles[source_name] = line

        fig1.suptitle(f'Electricity Generation ({country_code})', 
                     fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax1.set_title('Annual Trends - Fraction of Total Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
        ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax1.grid(True, alpha=0.3, linewidth=1.5)

        spacer = Line2D([0], [0], color='none', label=' ')
        legend_order = [
            'Wind', 'Solar', '_SPACER_',
            'Hydro', 'Biomass', 'Geothermal',
            'Nuclear', 'Coal', 'Oil',
            'Gas', 'Waste', '_SPACER_'
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

        fig1.text(0.93, 0.04, f"Generated: {timestamp}",
                 ha='right', va='bottom', fontsize=11, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename_pct = f'plots/{country_code.lower()}_annual_all_sources_percentage.png'
        plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved percentage: {filename_pct}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, filename_pct, plot_type='Trends', country=country_code)
            if result:
                if 'all_sources' not in plot_links['Trends']:
                    plot_links['Trends']['all_sources'] = {}
                plot_links['Trends']['all_sources']['percentage'] = result
        
        plt.close()

        # ABSOLUTE PLOT
        fig2, ax2 = plt.subplots(figsize=(12, 10))

        line_handles = {}

        for source_name in available_sources:
            color = ENTSOE_COLORS.get(source_name, 'black')
            values_twh = [val / 1000 for val in annual_totals_abs[source_name]]
            line, = ax2.plot(years_available, values_twh, marker='o', color=color,
                     linewidth=6, markersize=13, label=source_name)
            line_handles[source_name] = line

        fig2.suptitle(f'Electricity Generation ({country_code})', 
                     fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax2.set_title('Annual Trends - Absolute Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
        ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax2.grid(True, alpha=0.3, linewidth=1.5)

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

        fig2.text(0.93, 0.04, f"Generated: {timestamp}",
                 ha='right', va='bottom', fontsize=11, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename_abs = f'plots/{country_code.lower()}_annual_all_sources_absolute.png'
        plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved absolute: {filename_abs}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, filename_abs, plot_type='Trends', country=country_code)
            if result:
                if 'all_sources' not in plot_links['Trends']:
                    plot_links['Trends']['all_sources'] = {}
                plot_links['Trends']['all_sources']['absolute'] = result
        
        plt.close()

    # Annual Trends - Renewable vs Non-Renewable
    print("\n" + "=" * 60)
    print("CREATING ANNUAL TRENDS - RENEWABLE VS NON-RENEWABLE")
    print("=" * 60)

    if 'All Renewables' in all_data and 'All Non-Renewables' in all_data and 'Total Generation' in all_data:
        annual_renewable = []
        annual_nonrenewable = []
        annual_renewable_pct = []
        annual_nonrenewable_pct = []

        renewables_data = all_data['All Renewables']['year_data']
        nonrenewables_data = all_data['All Non-Renewables']['year_data']
        total_data = all_data['Total Generation']['year_data']

        for year in years_available:
            if year in renewables_data and year in nonrenewables_data and year in total_data:
                yearly_renewable = sum(renewables_data[year].values())
                yearly_nonrenewable = sum(nonrenewables_data[year].values())
                yearly_total = sum(total_data[year].values())

                annual_renewable.append(yearly_renewable)
                annual_nonrenewable.append(yearly_nonrenewable)

                if yearly_total > 0:
                    annual_renewable_pct.append((yearly_renewable / yearly_total) * 100)
                    annual_nonrenewable_pct.append((yearly_nonrenewable / yearly_total) * 100)
                else:
                    annual_renewable_pct.append(0)
                    annual_nonrenewable_pct.append(0)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

        # PERCENTAGE PLOT
        fig1, ax1 = plt.subplots(figsize=(12, 10))

        ax1.plot(years_available, annual_renewable_pct, marker='o', 
                color=ENTSOE_COLORS['All Renewables'],
                linewidth=6, markersize=13, label='All Renewables')
        ax1.plot(years_available, annual_nonrenewable_pct, marker='o',
                color=ENTSOE_COLORS['All Non-Renewables'],
                linewidth=6, markersize=13, label='All Non-Renewables')

        fig1.suptitle(f'Electricity Generation ({country_code})', 
                     fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax1.set_title('Annual Trends - Fraction of Total Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Electricity Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax1.grid(True, alpha=0.3, linewidth=1.5)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                   fontsize=22, frameon=False)

        fig1.text(0.93, 0.04, f"Generated: {timestamp}",
                 ha='right', va='bottom', fontsize=11, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename_pct = f'plots/{country_code.lower()}_annual_renewable_vs_nonrenewable_percentage.png'
        plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved percentage: {filename_pct}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, filename_pct, plot_type='Trends', country=country_code)
            if result:
                if 'renewable_vs_nonrenewable' not in plot_links['Trends']:
                    plot_links['Trends']['renewable_vs_nonrenewable'] = {}
                plot_links['Trends']['renewable_vs_nonrenewable']['percentage'] = result
        
        plt.close()

        # ABSOLUTE PLOT
        fig2, ax2 = plt.subplots(figsize=(12, 10))

        renewable_twh = [val / 1000 for val in annual_renewable]
        nonrenewable_twh = [val / 1000 for val in annual_nonrenewable]

        ax2.plot(years_available, renewable_twh, marker='o',
                color=ENTSOE_COLORS['All Renewables'],
                linewidth=6, markersize=13, label='All Renewables')
        ax2.plot(years_available, nonrenewable_twh, marker='o',
                color=ENTSOE_COLORS['All Non-Renewables'],
                linewidth=6, markersize=13, label='All Non-Renewables')

        fig2.suptitle(f'Electricity Generation ({country_code})', 
                     fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax2.set_title('Annual Trends - Absolute Generation', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Electricity Generation (TWh)', fontsize=28, fontweight='bold', labelpad=15)
        ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax2.grid(True, alpha=0.3, linewidth=1.5)

        ax2.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                   fontsize=22, frameon=False)

        fig2.text(0.93, 0.04, f"Generated: {timestamp}",
                 ha='right', va='bottom', fontsize=11, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename_abs = f'plots/{country_code.lower()}_annual_renewable_vs_nonrenewable_absolute.png'
        plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved absolute: {filename_abs}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, filename_abs, plot_type='Trends', country=country_code)
            if result:
                if 'renewable_vs_nonrenewable' not in plot_links['Trends']:
                    plot_links['Trends']['renewable_vs_nonrenewable'] = {}
                plot_links['Trends']['renewable_vs_nonrenewable']['absolute'] = result
        
        plt.close()


    # YoY Change - All Sources vs 2015
    print("\n" + "=" * 60)
    print("CREATING YOY CHANGE CHARTS - ALL SOURCES VS 2015")
    print("=" * 60)

    if 2015 in years_available and available_sources and 'Total Generation' in all_data:
        yoy_change_abs = {}
        yoy_change_pct = {}

        baseline_year = 2015
        comparison_years = [year for year in years_available if year > baseline_year]

        for source_name in available_sources:
            yoy_change_abs[source_name] = []
            yoy_change_pct[source_name] = []

            source_data = all_data[source_name]['year_data']
            total_data = all_data['Total Generation']['year_data']

            if baseline_year not in source_data or baseline_year not in total_data:
                continue

            baseline_value_abs = sum(source_data[baseline_year].values())
            baseline_total = sum(total_data[baseline_year].values())
            baseline_value_pct = (baseline_value_abs / baseline_total * 100) if baseline_total > 0 else 0

            for year in comparison_years:
                if year in source_data and year in total_data:
                    current_value_abs = sum(source_data[year].values())
                    current_total = sum(total_data[year].values())

                    change_abs = ((current_value_abs - baseline_value_abs) / baseline_value_abs * 100) if baseline_value_abs > 0 else 0
                    yoy_change_abs[source_name].append(change_abs)

                    current_value_pct = (current_value_abs / current_total * 100) if current_total > 0 else 0
                    change_pct = current_value_pct - baseline_value_pct
                    yoy_change_pct[source_name].append(change_pct)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

        # PERCENTAGE CHANGE PLOT
        fig1, ax1 = plt.subplots(figsize=(12, 10))

        from matplotlib.lines import Line2D
        line_handles = {}

        for source_name in available_sources:
            if yoy_change_pct[source_name]:
                color = ENTSOE_COLORS.get(source_name, 'black')
                line, = ax1.plot(comparison_years, yoy_change_pct[source_name], marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)
                line_handles[source_name] = line

        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)

        fig1.suptitle(f'Electricity Generation ({country_code})', 
                     fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax1.set_title(f'YoY Change in Share vs {baseline_year} (Percentage Points)', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Change in Share (Percentage Points)', fontsize=28, fontweight='bold', labelpad=15)
        ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax1.grid(True, alpha=0.3, linewidth=1.5)

        spacer = Line2D([0], [0], color='none', label=' ')
        legend_order = [
            'Wind', 'Solar', '_SPACER_',
            'Hydro', 'Biomass', 'Geothermal',
            'Nuclear', 'Coal', 'Oil',
            'Gas', 'Waste', '_SPACER_'
        ]
        
        legend_handles_list = []
        legend_labels = []
        for item in legend_order:
            if item == '_SPACER_':
                legend_handles_list.append(spacer)
                legend_labels.append(' ')
            elif item in line_handles:
                legend_handles_list.append(line_handles[item])
                legend_labels.append(item)
        
        ax1.legend(legend_handles_list, legend_labels,
                   loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=4,
                   fontsize=20, frameon=False)

        fig1.text(0.93, 0.04, f"Generated: {timestamp}",
                 ha='right', va='bottom', fontsize=11, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename_pct = f'plots/{country_code.lower()}_annual_yoy_all_sources_vs_{baseline_year}_percentage.png'
        plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved percentage: {filename_pct}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, filename_pct, plot_type='Trends', country=country_code)
            if result:
                if 'yoy_all_sources_vs_2015' not in plot_links['Trends']:
                    plot_links['Trends']['yoy_all_sources_vs_2015'] = {}
                plot_links['Trends']['yoy_all_sources_vs_2015']['percentage'] = result
        
        plt.close()

        # ABSOLUTE CHANGE PLOT
        fig2, ax2 = plt.subplots(figsize=(12, 10))

        line_handles = {}

        for source_name in available_sources:
            if yoy_change_abs[source_name]:
                color = ENTSOE_COLORS.get(source_name, 'black')
                line, = ax2.plot(comparison_years, yoy_change_abs[source_name], marker='o', color=color,
                         linewidth=6, markersize=13, label=source_name)
                line_handles[source_name] = line

        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)

        fig2.suptitle(f'Electricity Generation ({country_code})', 
                     fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax2.set_title(f'YoY Change in Absolute Generation vs {baseline_year} (%)', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Change in Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
        ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax2.grid(True, alpha=0.3, linewidth=1.5)

        legend_handles_list = []
        legend_labels = []
        for item in legend_order:
            if item == '_SPACER_':
                legend_handles_list.append(spacer)
                legend_labels.append(' ')
            elif item in line_handles:
                legend_handles_list.append(line_handles[item])
                legend_labels.append(item)
        
        ax2.legend(legend_handles_list, legend_labels,
                   loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=4,
                   fontsize=20, frameon=False)

        fig2.text(0.93, 0.04, f"Generated: {timestamp}",
                 ha='right', va='bottom', fontsize=11, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename_abs = f'plots/{country_code.lower()}_annual_yoy_all_sources_vs_{baseline_year}_absolute.png'
        plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved absolute: {filename_abs}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, filename_abs, plot_type='Trends', country=country_code)
            if result:
                if 'yoy_all_sources_vs_2015' not in plot_links['Trends']:
                    plot_links['Trends']['yoy_all_sources_vs_2015'] = {}
                plot_links['Trends']['yoy_all_sources_vs_2015']['absolute'] = result
        
        plt.close()

    # YoY Change - Aggregates vs 2015
    print("\n" + "=" * 60)
    print("CREATING YOY CHANGE CHARTS - AGGREGATES VS 2015")
    print("=" * 60)

    if 2015 in years_available and 'All Renewables' in all_data and 'All Non-Renewables' in all_data and 'Total Generation' in all_data:
        baseline_year = 2015
        comparison_years = [year for year in years_available if year > baseline_year]

        yoy_renewable_abs = []
        yoy_nonrenewable_abs = []
        yoy_renewable_pct = []
        yoy_nonrenewable_pct = []

        renewables_data = all_data['All Renewables']['year_data']
        nonrenewables_data = all_data['All Non-Renewables']['year_data']
        total_data = all_data['Total Generation']['year_data']

        baseline_renewable_abs = sum(renewables_data[baseline_year].values())
        baseline_nonrenewable_abs = sum(nonrenewables_data[baseline_year].values())
        baseline_total = sum(total_data[baseline_year].values())

        baseline_renewable_pct = (baseline_renewable_abs / baseline_total * 100) if baseline_total > 0 else 0
        baseline_nonrenewable_pct = (baseline_nonrenewable_abs / baseline_total * 100) if baseline_total > 0 else 0

        for year in comparison_years:
            if year in renewables_data and year in nonrenewables_data and year in total_data:
                current_renewable_abs = sum(renewables_data[year].values())
                current_nonrenewable_abs = sum(nonrenewables_data[year].values())
                current_total = sum(total_data[year].values())

                change_renewable_abs = ((current_renewable_abs - baseline_renewable_abs) / baseline_renewable_abs * 100) if baseline_renewable_abs > 0 else 0
                change_nonrenewable_abs = ((current_nonrenewable_abs - baseline_nonrenewable_abs) / baseline_nonrenewable_abs * 100) if baseline_nonrenewable_abs > 0 else 0

                yoy_renewable_abs.append(change_renewable_abs)
                yoy_nonrenewable_abs.append(change_nonrenewable_abs)

                current_renewable_pct = (current_renewable_abs / current_total * 100) if current_total > 0 else 0
                current_nonrenewable_pct = (current_nonrenewable_abs / current_total * 100) if current_total > 0 else 0

                change_renewable_pct = current_renewable_pct - baseline_renewable_pct
                change_nonrenewable_pct = current_nonrenewable_pct - baseline_nonrenewable_pct

                yoy_renewable_pct.append(change_renewable_pct)
                yoy_nonrenewable_pct.append(change_nonrenewable_pct)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

        # PERCENTAGE CHANGE PLOT
        fig1, ax1 = plt.subplots(figsize=(12, 10))

        ax1.plot(comparison_years, yoy_renewable_pct, marker='o',
                color=ENTSOE_COLORS['All Renewables'],
                linewidth=6, markersize=13, label='All Renewables')
        ax1.plot(comparison_years, yoy_nonrenewable_pct, marker='o',
                color=ENTSOE_COLORS['All Non-Renewables'],
                linewidth=6, markersize=13, label='All Non-Renewables')

        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)

        fig1.suptitle(f'Electricity Generation ({country_code})', 
                     fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax1.set_title(f'YoY Change in Share vs {baseline_year} (Percentage Points)', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax1.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Change in Share (Percentage Points)', fontsize=28, fontweight='bold', labelpad=15)
        ax1.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax1.grid(True, alpha=0.3, linewidth=1.5)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                   fontsize=22, frameon=False)

        fig1.text(0.93, 0.04, f"Generated: {timestamp}",
                 ha='right', va='bottom', fontsize=11, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename_pct = f'plots/{country_code.lower()}_annual_yoy_aggregates_vs_{baseline_year}_percentage.png'
        plt.savefig(filename_pct, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved percentage: {filename_pct}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, filename_pct, plot_type='Trends', country=country_code)
            if result:
                if 'yoy_aggregates_vs_2015' not in plot_links['Trends']:
                    plot_links['Trends']['yoy_aggregates_vs_2015'] = {}
                plot_links['Trends']['yoy_aggregates_vs_2015']['percentage'] = result
        
        plt.close()

        # ABSOLUTE CHANGE PLOT
        fig2, ax2 = plt.subplots(figsize=(12, 10))

        ax2.plot(comparison_years, yoy_renewable_abs, marker='o',
                color=ENTSOE_COLORS['All Renewables'],
                linewidth=6, markersize=13, label='All Renewables')
        ax2.plot(comparison_years, yoy_nonrenewable_abs, marker='o',
                color=ENTSOE_COLORS['All Non-Renewables'],
                linewidth=6, markersize=13, label='All Non-Renewables')

        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)

        fig2.suptitle(f'Electricity Generation ({country_code})', 
                     fontsize=34, fontweight='bold', x=0.55, y=0.96, ha='center')
        ax2.set_title(f'YoY Change in Absolute Generation vs {baseline_year} (%)', fontsize=26, fontweight='normal', pad=10, ha='center')
        ax2.set_xlabel('Year', fontsize=28, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Change in Generation (%)', fontsize=28, fontweight='bold', labelpad=15)
        ax2.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax2.grid(True, alpha=0.3, linewidth=1.5)

        ax2.legend(loc='upper center', bbox_to_anchor=(0.45, -0.20), ncol=2,
                   fontsize=22, frameon=False)

        fig2.text(0.93, 0.04, f"Generated: {timestamp}",
                 ha='right', va='bottom', fontsize=11, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename_abs = f'plots/{country_code.lower()}_annual_yoy_aggregates_vs_{baseline_year}_absolute.png'
        plt.savefig(filename_abs, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved absolute: {filename_abs}")
        
        # Upload to Drive
        if drive_service:
            result = upload_plot_to_drive(drive_service, filename_abs, plot_type='Trends', country=country_code)
            if result:
                if 'yoy_aggregates_vs_2015' not in plot_links['Trends']:
                    plot_links['Trends']['yoy_aggregates_vs_2015'] = {}
                plot_links['Trends']['yoy_aggregates_vs_2015']['absolute'] = result
        
        plt.close()

    print("\n" + "=" * 60)
    print("ALL MOBILE-OPTIMIZED PLOTS GENERATED")
    print("=" * 60)
    
    return plot_links, drive_service



def update_summary_table_historical_data(all_data, country_code='EU'):
    """
    Update Google Sheets "Summary Table Data" with current year YTD and previous year data
    This fills in the columns that the intraday script leaves empty
    """
    print("\n" + "=" * 60)
    print(f"UPDATING SUMMARY TABLE (HISTORICAL DATA) FOR {country_code}")
    print("=" * 60)

    try:
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            print("  ⚠ GOOGLE_CREDENTIALS_JSON not set")
            return

        creds_dict = json.loads(google_creds_json)
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        gc = gspread.authorize(credentials)

        sheet_name = f'{country_code} Electricity Production Data'
        spreadsheet = gc.open(sheet_name)
        print(f"✓ Connected to spreadsheet: {sheet_name}")

        try:
            worksheet = spreadsheet.worksheet('Summary Table Data')
        except gspread.WorksheetNotFound:
            print("  ⚠ 'Summary Table Data' worksheet not found")
            return

        print("✓ Found 'Summary Table Data' worksheet")

        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        previous_year = current_year - 1

        # Read existing data
        all_values = worksheet.get_all_values()
        if len(all_values) < 2:
            print("  ⚠ Summary table is empty")
            return

        headers = all_values[0]
        data_rows = all_values[1:]

        # Find column indices
        try:
            source_col_idx = headers.index('Source')
            ytd_current_col_idx = headers.index('YTD_Current_Year_GWh')
            ytd_current_pct_idx = headers.index('YTD_Current_Year_%')
            prev_year_col_idx = headers.index('Previous_Year_GWh')
            prev_year_pct_idx = headers.index('Previous_Year_%')
        except ValueError as e:
            print(f"  ⚠ Required column not found: {e}")
            return

        # Build dictionary of row numbers by source
        source_row_map = {}
        for idx, row in enumerate(data_rows):
            if row[source_col_idx]:
                source_name = row[source_col_idx].strip()
                source_row_map[source_name] = idx + 2  # +2 because sheets are 1-indexed and we skip header

        print(f"  Found {len(source_row_map)} sources in summary table")

        # Calculate values
        updates = []

        # YTD Current Year (sum months 1 to current_month in current_year)
        ytd_totals = {}
        ytd_percentages = {}

        # Previous Year (sum all 12 months of previous_year)
        prev_year_totals = {}
        prev_year_percentages = {}

        # Calculate totals first (for percentage calculations)
        total_ytd_current = 0
        total_prev_year = 0

        if 'Total Generation' in all_data:
            total_data = all_data['Total Generation']['year_data']

            if current_year in total_data:
                for month in range(1, current_month + 1):
                    total_ytd_current += total_data[current_year].get(month, 0)

            if previous_year in total_data:
                for month in range(1, 13):
                    total_prev_year += total_data[previous_year].get(month, 0)

        print(f"  Total YTD {current_year}: {total_ytd_current:.2f} GWh")
        print(f"  Total {previous_year}: {total_prev_year:.2f} GWh")

        # Calculate for each source
        for source_name, source_info in all_data.items():
            if source_name == 'Total Generation':
                continue

            year_data = source_info['year_data']

            # YTD Current Year
            ytd_sum = 0
            if current_year in year_data:
                for month in range(1, current_month + 1):
                    ytd_sum += year_data[current_year].get(month, 0)

            ytd_totals[source_name] = ytd_sum

            if total_ytd_current > 0:
                ytd_percentages[source_name] = (ytd_sum / total_ytd_current) * 100
            else:
                ytd_percentages[source_name] = 0

            # Previous Year
            prev_year_sum = 0
            if previous_year in year_data:
                for month in range(1, 13):
                    prev_year_sum += year_data[previous_year].get(month, 0)

            prev_year_totals[source_name] = prev_year_sum

            if total_prev_year > 0:
                prev_year_percentages[source_name] = (prev_year_sum / total_prev_year) * 100
            else:
                prev_year_percentages[source_name] = 0

        # Prepare updates
        for source_name in ytd_totals:
            if source_name not in source_row_map:
                continue

            row_num = source_row_map[source_name]

            # YTD Current Year
            ytd_value = ytd_totals[source_name]
            ytd_pct = ytd_percentages[source_name]

            # Previous Year
            prev_value = prev_year_totals[source_name]
            prev_pct = prev_year_percentages[source_name]

            updates.append({
                'range': f'{chr(65 + ytd_current_col_idx)}{row_num}',
                'values': [[round(ytd_value, 2)]]
            })
            updates.append({
                'range': f'{chr(65 + ytd_current_pct_idx)}{row_num}',
                'values': [[round(ytd_pct, 2)]]
            })
            updates.append({
                'range': f'{chr(65 + prev_year_col_idx)}{row_num}',
                'values': [[round(prev_value, 2)]]
            })
            updates.append({
                'range': f'{chr(65 + prev_year_pct_idx)}{row_num}',
                'values': [[round(prev_pct, 2)]]
            })

        # Batch update
        if updates:
            worksheet.batch_update(updates)
            print(f"✓ Updated {len(updates)} cells in Summary Table")
        else:
            print("  ⚠ No updates to make")

        # Update timestamp
        try:
            timestamp_col_idx = headers.index('Last_Updated')
            timestamp_value = current_date.strftime('%Y-%m-%d %H:%M:%S UTC')

            # Update timestamp for all rows
            timestamp_updates = []
            for row_num in source_row_map.values():
                timestamp_updates.append({
                    'range': f'{chr(65 + timestamp_col_idx)}{row_num}',
                    'values': [[timestamp_value]]
                })

            if timestamp_updates:
                worksheet.batch_update(timestamp_updates)
                print(f"✓ Updated timestamps")

        except ValueError:
            print("  ⚠ Last_Updated column not found")

    except Exception as e:
        print(f"✗ Error updating summary table: {e}")
        import traceback
        traceback.print_exc()



def main():
    """
    Main function - process multiple countries
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
    print("  ✓ MULTI-COUNTRY SUPPORT")
    print("=" * 60)

    if not os.environ.get('GOOGLE_CREDENTIALS_JSON'):
        print("\n⚠️  WARNING: GOOGLE_CREDENTIALS_JSON not set!")
        return

    # Countries to process (expandable list)
    countries_to_process = ['EU', 'DE']
    
    # Store all plot links by country
    all_plot_links = {}
    
    for country_code in countries_to_process:
        print("\n" + "=" * 80)
        print(f"PROCESSING COUNTRY: {country_code}")
        print("=" * 80)
        
        # Load data for this country
        all_data = load_data_from_google_sheets(country_code=country_code)
        
        if not all_data:
            print(f"✗ Failed to load data for {country_code} - skipping")
            continue
        
        # Create all charts for this country
        plot_links, drive_service = create_all_charts(all_data, country_code=country_code)
        
        # Store links by country
        all_plot_links[country_code] = plot_links
        
        # Update summary table with historical data
        update_summary_table_historical_data(all_data, country_code=country_code)
    
    # Save combined drive links JSON
    drive_links_file = 'plots/drive_links_monthly_trends.json'
    with open(drive_links_file, 'w') as f:
        json.dump(all_plot_links, f, indent=2)
    print(f"\n✓ Drive links saved to: {drive_links_file}")
    
    # Write timestamp file for HTML to read
    timestamp_file = 'plots/last_update_monthly_trends.html'
    generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    with open(timestamp_file, 'w') as f:
        f.write(f'<p>Plots generated: {generation_time}</p>')
    print(f"✓ Timestamp written to {timestamp_file}: {generation_time}")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"   - Processed {len(countries_to_process)} countries: {', '.join(countries_to_process)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
