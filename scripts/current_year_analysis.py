"""
current_year_analysis.py

Standalone script: generates monthly plots for the current year (completed months only).
- Reads data from Google Sheets (same source as eu_energy_plotting.py)
- Plots Jan-Dec x-axis; completed months have data, future months are NaN
- Uploads to Google Drive under CurrentYear/ folder
- Saves links to plots/drive_links.json under 'CurrentYear' section
- Style matches existing monthly/trends plots exactly

Trigger: manually via GitHub Actions
"""

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import calendar
from datetime import datetime
import os
import json
import time
import argparse

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    print("⚠ Google Drive API not available")

# ============================================================
# COLORS & DISPLAY CONFIG (copied from eu_energy_plotting.py)
# ============================================================

ENTSOE_COLORS = {
    'Solar':      '#FFD700',
    'Wind':       '#228B22',
    'Hydro':      '#1E90FF',
    'Biomass':    '#9ACD32',
    'Geothermal': '#708090',
    'Gas':        '#FF1493',
    'Coal':       '#8B008B',
    'Nuclear':    '#8B4513',
    'Oil':        '#191970',
    'Waste':      '#808000',
    'All Renewables':     '#00CED1',
    'All Non-Renewables': '#000000',
}

COUNTRY_DISPLAY_NAMES = {
    'EU': 'European Union', 'DE': 'Germany', 'FR': 'France', 'ES': 'Spain',
    'IT': 'Italy', 'PL': 'Poland', 'NL': 'Netherlands', 'BE': 'Belgium',
    'SE': 'Sweden', 'AT': 'Austria', 'CZ': 'Czechia', 'RO': 'Romania',
    'PT': 'Portugal', 'GR': 'Greece', 'DK': 'Denmark', 'FI': 'Finland',
    'SK': 'Slovakia', 'IE': 'Ireland', 'HR': 'Croatia', 'BG': 'Bulgaria',
    'LT': 'Lithuania', 'SI': 'Slovenia', 'HU': 'Hungary', 'LV': 'Latvia',
    'EE': 'Estonia', 'LU': 'Luxembourg', 'CY': 'Cyprus', 'MT': 'Malta',
    'GB': 'United Kingdom', 'NO': 'Norway', 'CH': 'Switzerland', 'MD': 'Moldova',
}

ALL_SOURCES = ['Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal',
               'Nuclear', 'Gas', 'Coal', 'Oil', 'Waste']

# ============================================================
# GOOGLE DRIVE HELPERS (copied from eu_energy_plotting.py)
# ============================================================

def initialize_drive_service():
    if not GDRIVE_AVAILABLE:
        return None
    try:
        creds_dict = json.loads(os.getenv('GOOGLE_CREDENTIALS_JSON', '{}'))
        credentials = ServiceAccountCredentials.from_service_account_info(
            creds_dict, scopes=['https://www.googleapis.com/auth/drive.file']
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"  ⚠ Failed to initialize Drive service: {e}")
        return None


def get_or_create_drive_folder(service, folder_name, parent_id=None):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    folders = results.get('files', [])
    if folders:
        return folders[0]['id']
    file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        file_metadata['parents'] = [parent_id]
    folder = service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')


def upload_plot_to_drive(service, file_path, country='EU'):
    """
    Upload plot to EU-Electricity-Plots/[Country]/CurrentYear/
    Returns dict with file_id, view_url, direct_url or None.
    """
    if not GDRIVE_AVAILABLE or service is None:
        return None
    try:
        root_id    = get_or_create_drive_folder(service, 'EU-Electricity-Plots')
        country_id = get_or_create_drive_folder(service, country, root_id)
        folder_id  = get_or_create_drive_folder(service, 'CurrentYear', country_id)

        filename = os.path.basename(file_path)

        # Update if exists, create otherwise
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        existing = service.files().list(q=query, spaces='drive', fields='files(id)').execute().get('files', [])

        media = MediaFileUpload(file_path, mimetype='image/png')
        if existing:
            file_id = existing[0]['id']
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            meta = {'name': filename, 'parents': [folder_id]}
            file_id = service.files().create(body=meta, media_body=media, fields='id').execute().get('id')

        # Set public read permission
        try:
            perms = service.permissions().list(fileId=file_id, fields='permissions(id,type)').execute()
            anyone = next((p for p in perms.get('permissions', []) if p.get('type') == 'anyone'), None)
            if anyone:
                service.permissions().update(fileId=file_id, permissionId=anyone['id'],
                                              body={'role': 'reader'}).execute()
            else:
                service.permissions().create(fileId=file_id,
                                              body={'type': 'anyone', 'role': 'reader'}).execute()
        except Exception as e:
            print(f"  ⚠ Could not set permissions: {e}")

        return {
            'file_id':    file_id,
            'view_url':   f'https://drive.google.com/file/d/{file_id}/view',
            'direct_url': f'https://drive.google.com/thumbnail?id={file_id}&sz=w2000',
            'updated':    datetime.now().isoformat()
        }
    except Exception as e:
        print(f"  ⚠ Drive upload failed for {os.path.basename(file_path)}: {e}")
        return None


def save_drive_links(country_code, percentage_result, absolute_result):
    """
    Save CurrentYear drive links to plots/drive_links.json,
    preserving all other sections (Intraday, Monthly, Trends, etc.)
    """
    drive_links_file = 'plots/drive_links.json'
    links = {}
    if os.path.exists(drive_links_file):
        try:
            with open(drive_links_file, 'r') as f:
                links = json.load(f)
        except Exception:
            pass

    if country_code not in links:
        links[country_code] = {}

    links[country_code]['CurrentYear'] = {
        'percentage': percentage_result,
        'absolute':   absolute_result,
    }

    os.makedirs('plots', exist_ok=True)
    with open(drive_links_file, 'w') as f:
        json.dump(links, f, indent=2)
    print(f"  ✓ drive_links.json updated for {country_code} / CurrentYear")


# ============================================================
# GOOGLE SHEETS LOADER (copied from eu_energy_plotting.py)
# ============================================================

def load_data_from_google_sheets(country_code='EU'):
    """Load energy data from Google Sheets. Returns all_data dict."""
    try:
        creds_dict = json.loads(os.environ.get('GOOGLE_CREDENTIALS_JSON', ''))
        scope = ['https://www.googleapis.com/auth/spreadsheets',
                 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        gc = gspread.authorize(credentials)

        sheet_name = f'{country_code} Electricity Production Data'

        # Try JSON first
        drive_links_file = 'plots/drive_links.json'
        spreadsheet = None
        if os.path.exists(drive_links_file):
            try:
                with open(drive_links_file, 'r') as f:
                    links = json.load(f)
                if country_code in links and 'data_sheet_id' in links[country_code]:
                    spreadsheet = gc.open_by_key(links[country_code]['data_sheet_id'])
            except Exception:
                pass

        if spreadsheet is None:
            spreadsheet = gc.open(sheet_name)

        print(f"  ✓ Connected to: {sheet_name}")

        worksheets = spreadsheet.worksheets()
        time.sleep(2)

        all_data = {}
        for ws in worksheets:
            if 'Monthly Production' not in ws.title:
                continue
            source_name = ws.title.replace(' Monthly Production', '')
            time.sleep(10)
            values = ws.get_all_values()
            if len(values) < 2:
                continue
            df = pd.DataFrame(values[1:], columns=values[0])
            df = df[df['Month'] != 'Total']
            year_cols = [c for c in df.columns if c != 'Month' and c.isdigit()]
            for col in year_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            year_data = {}
            for yr_str in year_cols:
                yr = int(yr_str)
                year_data[yr] = {}
                for _, row in df.iterrows():
                    try:
                        month_num = list(calendar.month_abbr).index(row['Month'])
                        year_data[yr][month_num] = float(row[yr_str])
                    except (ValueError, KeyError):
                        continue
            all_data[source_name] = {'year_data': year_data}
            print(f"    ✓ {source_name}: {len(year_cols)} years")

        return all_data

    except Exception as e:
        print(f"  ✗ Error loading data for {country_code}: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================
# PLOT HELPERS (copied from eu_energy_plotting.py)
# ============================================================

def add_flag_and_labels(fig, country_code, main_title, subtitle):
    flag_path = f'flags/{country_code}.png'
    if os.path.exists(flag_path):
        try:
            ax_flag = fig.add_axes([0.06, 0.905, 0.09, 0.06])
            flag_img = mpimg.imread(flag_path)
            ax_flag.imshow(flag_img, aspect='auto')
            ax_flag.axis('off')
            ax_flag.set_zorder(10)
        except Exception:
            pass

    country_display = COUNTRY_DISPLAY_NAMES.get(country_code, country_code)
    fig.text(0.105, 0.897, country_display, fontsize=16, fontweight='normal',
             ha='center', va='top', color='#333')
    fig.text(0.55, 0.965, main_title, fontsize=36, fontweight='bold',
             ha='center', va='top')
    fig.text(0.55, 0.91, subtitle, fontsize=28, fontweight='normal',
             ha='center', va='top')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    fig.text(0.15, 0.035, "afratzl.github.io/eu-electricity",
             ha='left', va='top', fontsize=12, color='#666', style='italic')
    fig.text(0.94, 0.035, f"Generated: {timestamp}",
             ha='right', va='top', fontsize=12, color='#666', style='italic')


def make_banner(fig):
    from matplotlib.patches import Rectangle
    banner = Rectangle((0, 0.86), 1.0, 0.14,
                        transform=fig.transFigure,
                        facecolor='#EBEBEB', edgecolor='none', zorder=0)
    fig.patches.append(banner)


def legend_reorder(ax, handles, labels):
    """
    Reorder 10-source legend into 4-column layout matching existing plots:
    Row 1: Wind, Solar, Hydro
    Row 2: Biomass, Geothermal, (empty)
    Row 3: Gas, Coal, Nuclear
    Row 4: Oil, Waste, (empty)
    """
    from matplotlib.patches import Rectangle
    empty = Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)

    # Build label -> handle map
    lh = dict(zip(labels, handles))

    order = ['Wind', 'Solar', 'Hydro',
             'Biomass', 'Geothermal', None,
             'Gas', 'Coal', 'Nuclear',
             'Oil', 'Waste', None]

    reordered_handles = [lh.get(n, empty) for n in order]
    reordered_labels  = [n if n else '' for n in order]
    return reordered_handles, reordered_labels


# ============================================================
# CORE PLOT GENERATION
# ============================================================

def generate_current_year_plots(all_data, country_code='EU'):
    """
    Generate percentage and absolute monthly plots for the current year.
    Completed months: actual values. Future months: NaN (line stops cleanly).
    Returns (percentage_file, absolute_file) or (None, None).
    """
    current_date  = datetime.now()
    current_year  = current_date.year
    last_complete = current_date.month - 1   # e.g. April -> 3 (March complete... wait)

    # Actually: if today is May 1st, April is just completed
    # We want the last FULLY completed month
    # If current day > 1 we treat previous month as complete
    # If current day == 1 we treat month-1 as complete (same result since month already ticked)
    # Simple rule: last_complete = current_month - 1
    # If last_complete == 0, nothing to plot yet (we're in January)
    if last_complete == 0:
        print(f"  ⚠ No completed months yet for {current_year}, skipping")
        return None, None

    month_names = [calendar.month_abbr[i] for i in range(1, 13)]

    # Build per-source monthly arrays (12 values, NaN for future months)
    if 'Total Generation' not in all_data:
        print(f"  ⚠ No Total Generation data for {country_code}, skipping")
        return None, None

    total_year_data = all_data['Total Generation']['year_data']
    if current_year not in total_year_data:
        print(f"  ⚠ No {current_year} data for {country_code}, skipping")
        return None, None

    # Total generation array (12 months)
    total_gen = np.array([
        total_year_data[current_year].get(m, np.nan) if m <= last_complete else np.nan
        for m in range(1, 13)
    ])

    source_arrays = {}
    for source_name in ALL_SOURCES:
        if source_name not in all_data:
            source_arrays[source_name] = np.full(12, np.nan)
            continue
        year_data = all_data[source_name]['year_data']
        if current_year not in year_data:
            source_arrays[source_name] = np.full(12, np.nan)
            continue
        arr = np.array([
            year_data[current_year].get(m, np.nan) if m <= last_complete else np.nan
            for m in range(1, 13)
        ])
        source_arrays[source_name] = arr

    # Percentage arrays
    with np.errstate(invalid='ignore', divide='ignore'):
        pct_arrays = {
            s: np.where(total_gen > 0, source_arrays[s] / total_gen * 100, np.nan)
            for s in ALL_SOURCES
        }

    # TWh arrays
    twh_arrays = {s: source_arrays[s] / 1000 for s in ALL_SOURCES}

    os.makedirs('plots', exist_ok=True)

    subtitle_months = calendar.month_abbr[last_complete]
    subtitle = f'{current_year} Jan\u2013{subtitle_months} · {{mode}}'

    def make_fig(mode, arrays, ylabel):
        fig, ax = plt.subplots(figsize=(12, 12))
        make_banner(fig)
        plt.subplots_adjust(left=0.15, right=0.94, top=0.83, bottom=0.25)

        max_val = 0
        # Force all 12 month categories to appear on x-axis
        ax.plot(month_names, [0] * 12, alpha=0)
        for source_name in ALL_SOURCES:
            color = ENTSOE_COLORS.get(source_name, 'black')
            y = arrays[source_name]
            valid_max = np.nanmax(y) if not np.all(np.isnan(y)) else 0
            max_val = max(max_val, valid_max)
            ax.plot(month_names, y, marker='o', color=color,
                    linewidth=6, markersize=13, label=source_name)

        add_flag_and_labels(fig, country_code,
                            'Electricity Generation',
                            subtitle.format(mode=mode))

        ax.set_xlabel('Month', fontsize=24, fontweight='bold', labelpad=10)
        ax.set_ylabel(ylabel, fontsize=24, fontweight='bold', labelpad=10)
        ax.set_ylim(0, max_val * 1.2 if max_val > 0 else 10)
        ax.tick_params(axis='both', labelsize=22, length=8, pad=8)
        ax.grid(True, linestyle='--', alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        rh, rl = legend_reorder(ax, handles, labels)
        ax.legend(rh, rl, loc='upper left', bbox_to_anchor=(0.2, 0.165),
                  bbox_transform=fig.transFigure, ncol=4,
                  fontsize=18, frameon=False)
        return fig

    # --- Percentage plot ---
    fig_pct = make_fig('Fraction of Total', pct_arrays,
                       'Electricity Generation (%)')
    pct_file = f'plots/{country_code.lower()}_current_year_percentage.png'
    fig_pct.savefig(pct_file, dpi=150)
    plt.close(fig_pct)
    print(f"  ✓ Saved: {pct_file}")

    # --- Absolute plot ---
    fig_abs = make_fig('Absolute Values', twh_arrays,
                       'Electricity Generation (TWh)')
    abs_file = f'plots/{country_code.lower()}_current_year_absolute.png'
    fig_abs.savefig(abs_file, dpi=150)
    plt.close(fig_abs)
    print(f"  ✓ Saved: {abs_file}")

    return pct_file, abs_file


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate current-year monthly plots')
    parser.add_argument('--country', default=None,
                        help='Single country code (e.g. DE). Default: all countries.')
    args = parser.parse_args()

    if not os.environ.get('GOOGLE_CREDENTIALS_JSON'):
        print("⚠ GOOGLE_CREDENTIALS_JSON not set")
        return

    countries = (
        [args.country.upper()] if args.country else
        ['EU', 'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
         'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
         'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE',
         'NO', 'CH', 'GB', 'MD']
    )

    drive_service = initialize_drive_service()
    current_year  = datetime.now().year

    print("=" * 60)
    print(f"CURRENT YEAR ANALYSIS ({current_year})")
    print(f"Countries: {', '.join(countries)}")
    print("=" * 60)

    for country_code in countries:
        print(f"\n--- {country_code} ---")

        all_data = load_data_from_google_sheets(country_code=country_code)
        if not all_data:
            print(f"  ✗ Could not load data, skipping")
            continue

        # Derive All Non-Renewables = Total - All Renewables (same as plotting script)
        if 'All Renewables' in all_data and 'Total Generation' in all_data:
            ren  = all_data['All Renewables']['year_data']
            tot  = all_data['Total Generation']['year_data']
            non_ren = {}
            for yr in set(ren) & set(tot):
                non_ren[yr] = {m: max(0, tot[yr].get(m, 0) - ren[yr].get(m, 0))
                               for m in range(1, 13)}
            all_data['All Non-Renewables'] = {'year_data': non_ren}

        pct_file, abs_file = generate_current_year_plots(all_data, country_code)

        if pct_file and abs_file and drive_service:
            print(f"  📤 Uploading to Drive...")
            pct_result = upload_plot_to_drive(drive_service, pct_file, country=country_code)
            abs_result = upload_plot_to_drive(drive_service, abs_file, country=country_code)
            if pct_result and abs_result:
                save_drive_links(country_code, pct_result, abs_result)
                print(f"  ✓ Drive upload complete for {country_code}")
            else:
                print(f"  ⚠ Drive upload failed for {country_code}")

        # Small delay between countries to avoid Sheets rate limits
        if country_code != countries[-1]:
            time.sleep(5)

    # Write timestamp
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    with open('plots/last_update_current_year.html', 'w') as f:
        f.write(f'<p>Current year plots generated: {ts}</p>')

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
