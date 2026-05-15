"""
generate_maps.py

Standalone script: generates Europe electricity generation maps.
- Reads percentage data from Google Sheets
- Generates one map per source per time range
- Uploads to Google Drive under Maps/Yesterday/, Maps/LastMonth/, Maps/Annual/YYYY/
- Saves links to plots/drive_links.json under 'Maps' section

Usage:
    python scripts/generate_maps.py                          # all sources, yesterday
    python scripts/generate_maps.py --period last_month      # all sources, last month
    python scripts/generate_maps.py --period annual          # all sources, all years
    python scripts/generate_maps.py --source solar           # single source, yesterday
    python scripts/generate_maps.py --scale dynamic          # dynamic scale
"""

import os
import json
import time
import argparse
import calendar
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.affinity import translate
from shapely.geometry import box
from shapely.ops import unary_union
import pyproj
import gspread
from google.oauth2.service_account import Credentials

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    print("⚠ Google Drive API not available")

# ============================================================
# CONSTANTS
# ============================================================

ENTSOE_COUNTRIES = [
    'AT','BE','BG','HR','CY','CZ','DK','EE','FI','FR',
    'DE','GR','HU','IE','IT','LV','LT','LU','NL',
    'PL','PT','RO','SK','SI','ES','SE','NO','CH','GB','MD'
]

ENTSOE_COLORS = {
    'solar':              '#FFD700',
    'wind':               '#228B22',
    'hydro':              '#1E90FF',
    'biomass':            '#9ACD32',
    'geothermal':         '#708090',
    'gas':                '#FF1493',
    'coal':               '#8B008B',
    'nuclear':            '#8B4513',
    'oil':                '#191970',
    'waste':              '#808000',
    'all-renewables':     '#00CED1',
    'all-non-renewables': '#000000',
}

DISPLAY_NAMES = {
    'solar':              'Solar',
    'wind':               'Wind',
    'hydro':              'Hydro',
    'biomass':            'Biomass',
    'geothermal':         'Geothermal',
    'gas':                'Gas',
    'coal':               'Coal',
    'nuclear':            'Nuclear',
    'oil':                'Oil',
    'waste':              'Waste',
    'all-renewables':     'All Renewables',
    'all-non-renewables': 'All Non-Renewables',
}

# Worksheet name mapping (Google Sheets)
WORKSHEET_NAMES = {
    'solar':              'Solar Monthly Production',
    'wind':               'Wind Monthly Production',
    'hydro':              'Hydro Monthly Production',
    'biomass':            'Biomass Monthly Production',
    'geothermal':         'Geothermal Monthly Production',
    'gas':                'Gas Monthly Production',
    'coal':               'Coal Monthly Production',
    'nuclear':            'Nuclear Monthly Production',
    'oil':                'Oil Monthly Production',
    'waste':              'Waste Monthly Production',
    'all-renewables':     'All Renewables Monthly Production',
    'total':              'Total Generation Monthly Production',
}

FIXED_SCALE_MAX = {
    'solar':              40,
    'wind':               70,
    'hydro':              80,
    'biomass':            30,
    'geothermal':         30,
    'gas':                70,
    'coal':               60,
    'nuclear':            80,
    'oil':                15,
    'waste':              10,
    'all-renewables':     100,
    'all-non-renewables': 100,
}

# Label display overrides
DISPLAY_LABEL = {'GB': 'UK'}

# Label position offsets in EPSG:3035 metres
LABEL_OFFSETS = {
    'CY': (-180000, 0),
    'AT': (80000, 0),
    'GB': (30000, -80000),
    'SE': (-130000, -300000),
    'DE': (80000, 0),
    'HR': (80000, 0),
    'CH': (80000, 0),
}

CONTEXT_LABEL_OFFSETS = {
    'Ukraine':    (0, -120000),
    'Montenegro': (-40000, 0),
    'Kosovo':     (40000, 0),
}

# Malta position (too small for 110m shapefile)
MT_X, MT_Y = 4721805, 1408134

# Context countries: shown with light fill, solid white border, labeled
CONTEXT_COUNTRIES = [
    'Russia', 'Belarus', 'Turkey', 'Ukraine', 'Serbia', 'Albania',
    'Bosnia and Herz.', 'North Macedonia', 'Montenegro', 'Kosovo'
]

# label, fontweight, fontsize, color
CONTEXT_LABELS = {
    'Russia':            ('RU', 'normal', 11, '#888888'),
    'Belarus':           ('BY', 'normal', 11, '#888888'),
    'Turkey':            ('TR', 'normal', 11, '#888888'),
    'Ukraine':           ('UA', 'bold',   14, 'black'),
    'Serbia':            ('RS', 'bold',   14, 'black'),
    'Albania':           ('AL', 'bold',   14, 'black'),
    'Bosnia and Herz.':  ('BA', 'bold',   14, 'black'),
    'North Macedonia':   ('MK', 'bold',   14, 'black'),
    'Montenegro':        ('ME', 'bold',   14, 'black'),
    'Kosovo':            ('XK', 'bold',   14, 'black'),
}

# Europe clip box in EPSG:3035 (removes overseas territories)
EUROPE_CLIP_BOX = box(1200000, 900000, 7500000, 5900000)


# ============================================================
# MAP SETUP (load once, reuse for all maps)
# ============================================================

def load_world_geodata(shapefile_path):
    """
    Load and prepare world geodata:
    - Assign correct ISO2 codes
    - Fix Crimea to Ukraine
    - Reproject to EPSG:3035
    - Build label coordinate transformer
    """
    print("📂 Loading shapefile...")
    world = gpd.read_file(shapefile_path)
    world['iso2'] = world['ISO_A2_EH'].where(world['ISO_A2'] == '-99', world['ISO_A2'])

    # Fix Crimea: reassign from Russia to Ukraine
    crimea_box = box(32.5, 44.3, 36.5, 46.2)
    russia_idx = world[world['iso2'] == 'RU'].index[0]
    ukraine_idx = world[world['iso2'] == 'UA'].index[0]
    russia_geom = world.loc[russia_idx, 'geometry']
    ukraine_geom = world.loc[ukraine_idx, 'geometry']
    crimea_geom = russia_geom.intersection(crimea_box)
    world.loc[russia_idx, 'geometry'] = russia_geom.difference(crimea_box)
    world.loc[ukraine_idx, 'geometry'] = unary_union([ukraine_geom, crimea_geom])
    print("  ✓ Crimea reassigned to Ukraine")

    # Reproject
    world_proj = world.to_crs('EPSG:3035')

    # Clip all geometries to European bounding box
    world_proj_clipped = world_proj.copy()
    world_proj_clipped['geometry'] = world_proj_clipped['geometry'].intersection(EUROPE_CLIP_BOX)

    # Transformer for label coordinates WGS84 -> EPSG:3035
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)

    # Build Europe background (excludes Iceland and context countries)
    europe_proj = world_proj[
        (world_proj['CONTINENT'] == 'Europe') | (world_proj['iso2'].isin(ENTSOE_COUNTRIES))
    ]
    europe_proj = europe_proj[~europe_proj['NAME'].isin(['Iceland'] + CONTEXT_COUNTRIES)].copy()
    europe_proj['geometry'] = europe_proj['geometry'].intersection(EUROPE_CLIP_BOX)
    europe_proj = europe_proj[~europe_proj['geometry'].is_empty]

    # Iceland relocated above UK
    iceland_row = world_proj[world_proj['NAME'] == 'Iceland']
    iceland_shifted = None
    if not iceland_row.empty:
        iceland_geom = iceland_row.geometry.iloc[0].intersection(EUROPE_CLIP_BOX)
        uk_centroid = world_proj[world_proj['iso2'] == 'GB'].geometry.iloc[0].centroid
        target_x = uk_centroid.x - 300000
        target_y = uk_centroid.y + 1500000
        iceland_shifted = translate(
            iceland_geom,
            xoff=target_x - iceland_geom.centroid.x,
            yoff=target_y - iceland_geom.centroid.y
        )

    # Compute tight map bounds from ENTSO-E countries
    our_countries = world_proj_clipped[world_proj_clipped['iso2'].isin(ENTSOE_COUNTRIES)]
    our_countries = our_countries[~our_countries['geometry'].is_empty]
    minx, miny, maxx, maxy = our_countries.total_bounds
    pad_x = (maxx - minx) * 0.03
    pad_y = (maxy - miny) * 0.03

    print(f"  ✓ Map bounds computed, {len(our_countries)} ENTSO-E countries loaded")

    return {
        'world': world,
        'world_proj': world_proj,
        'world_proj_clipped': world_proj_clipped,
        'europe_proj': europe_proj,
        'iceland_shifted': iceland_shifted,
        'transformer': transformer,
        'bounds': (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y),
    }


def get_label_pos(geodata, iso2=None, name=None):
    """Get Natural Earth label position in EPSG:3035"""
    world = geodata['world']
    transformer = geodata['transformer']
    if iso2:
        row = world[world['iso2'] == iso2]
    else:
        row = world[world['NAME'] == name]
    if row.empty:
        return None, None
    lx = float(row['LABEL_X'].iloc[0])
    ly = float(row['LABEL_Y'].iloc[0])
    x, y = transformer.transform(lx, ly)
    return x, y


# ============================================================
# MAP GENERATION
# ============================================================

def generate_map(geodata, values_by_country, source, date_str, scale='fixed'):
    """
    Generate a single map for one source and one date.

    Args:
        geodata: dict from load_world_geodata
        values_by_country: dict {country_iso2: percentage_value}
        source: e.g. 'solar'
        date_str: e.g. '14 May 2026'
        scale: 'fixed' or 'dynamic'

    Returns:
        matplotlib Figure
    """
    world_proj_clipped = geodata['world_proj_clipped']
    europe_proj        = geodata['europe_proj']
    iceland_shifted    = geodata['iceland_shifted']
    minx, miny, maxx, maxy = geodata['bounds']

    source_color = ENTSOE_COLORS.get(source, '#888888')
    cmap = LinearSegmentedColormap.from_list(source, ['white', source_color])

    if scale == 'dynamic':
        vals = [v for v in values_by_country.values() if v is not None]
        vmax = max(vals) if vals else 1
    else:
        vmax = FIXED_SCALE_MAX.get(source, 100)

    norm = Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patches.append(Rectangle(
        (0, 0.86), 1.0, 0.14,
        transform=fig.transFigure,
        facecolor='#EBEBEB', edgecolor='none', zorder=0
    ))
    plt.subplots_adjust(left=0.01, right=0.89, top=0.84, bottom=0.01)
    ax.set_facecolor('#cce6ff')

    # Context countries: light fill, solid white border
    context_gdf = world_proj_clipped[world_proj_clipped['NAME'].isin(CONTEXT_COUNTRIES)]
    context_gdf = context_gdf[~context_gdf['geometry'].is_empty].copy()
    context_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='white', linewidth=0.5, zorder=1)

    # Gray background for non-ENTSO-E European countries
    europe_proj.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5, zorder=2)

    # ENTSO-E countries colored by value
    for cc in ENTSOE_COUNTRIES:
        country_geom = world_proj_clipped[world_proj_clipped['iso2'] == cc]
        country_geom = country_geom[~country_geom['geometry'].is_empty]
        if country_geom.empty:
            continue
        val = values_by_country.get(cc)
        color = cmap(norm(val)) if val is not None else '#cccccc'
        country_geom.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8, zorder=3)

    # Iceland (relocated)
    if iceland_shifted is not None:
        iceland_gdf = gpd.GeoDataFrame(geometry=[iceland_shifted], crs=world_proj_clipped.crs)
        iceland_gdf.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.8, zorder=4)
        b = iceland_shifted.bounds
        pad = 50000
        ax.add_patch(Rectangle(
            (b[0]-pad, b[1]-pad), b[2]-b[0]+2*pad, b[3]-b[1]+2*pad,
            linewidth=2.0, edgecolor='#555555', facecolor='none',
            linestyle=(0, (6, 3)), zorder=5
        ))
        ax.text((b[0]+b[2])/2, (b[1]+b[3])/2 - 50000, 'IS',
                fontsize=14, fontweight='bold',
                ha='center', va='center', color='black', zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # ENTSO-E country labels
    for cc in ENTSOE_COUNTRIES:
        lx, ly = get_label_pos(geodata, iso2=cc)
        if lx is None:
            continue
        ox, oy = LABEL_OFFSETS.get(cc, (0, 0))
        lx += ox
        ly += oy
        if minx <= lx <= maxx and miny <= ly <= maxy:
            ax.text(lx, ly, DISPLAY_LABEL.get(cc, cc),
                    fontsize=14, fontweight='bold',
                    ha='center', va='center', color='black', zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # Malta label (no polygon in 110m dataset)
    ax.text(MT_X + 80000, MT_Y + 60000, 'MT',
            fontsize=14, fontweight='bold',
            ha='center', va='center', color='black', zorder=7,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # Context country labels
    for name, (label, weight, size, color) in CONTEXT_LABELS.items():
        lx, ly = get_label_pos(geodata, name=name)
        if lx is None:
            continue
        ox, oy = CONTEXT_LABEL_OFFSETS.get(name, (0, 0))
        lx += ox
        ly += oy
        if minx <= lx <= maxx and miny <= ly <= maxy:
            ax.text(lx, ly, label,
                    fontsize=size, fontweight=weight,
                    ha='center', va='center', color=color, zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal')
    ax.axis('off')

    # Titles centered across full banner
    source_display = DISPLAY_NAMES.get(source, source.title())
    fig.text(0.5, 0.965, 'Electricity Generation',
             fontsize=36, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.91, f'{source_display} · Fraction of Total · {date_str}',
             fontsize=28, fontweight='normal', ha='center', va='top')

    # Vertical colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.875, 0.05, 0.03, 0.77])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Fraction of Total (%)', fontsize=22, labelpad=20)
    cbar.ax.tick_params(labelsize=20)

    # Watermark
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    fig.text(0.02, 0.012, "eu-electricity.eu",
             ha='left', va='top', fontsize=12, color='#666', style='italic')
    fig.text(0.88, 0.012, f"Generated: {timestamp}",
             ha='right', va='top', fontsize=12, color='#666', style='italic')

    return fig


# ============================================================
# GOOGLE SHEETS DATA LOADING
# ============================================================

def get_spreadsheet(gc, country_code):
    """Get spreadsheet for a country, using drive_links.json if available."""
    sheet_name = f'{country_code} Electricity Production Data'
    drive_links_file = 'plots/drive_links.json'
    if os.path.exists(drive_links_file):
        try:
            with open(drive_links_file, 'r') as f:
                links = json.load(f)
            if country_code in links and 'data_sheet_id' in links[country_code]:
                return gc.open_by_key(links[country_code]['data_sheet_id'])
        except Exception:
            pass
    return gc.open(sheet_name)


def load_monthly_data_for_country(gc, country_code, source):
    """
    Load monthly GWh data for a specific country and source from Google Sheets.
    Returns dict: {year: {month: gwh}}
    """
    spreadsheet = get_spreadsheet(gc, country_code)
    ws_name = WORKSHEET_NAMES.get(source)
    if not ws_name:
        return {}
    try:
        ws = spreadsheet.worksheet(ws_name)
        time.sleep(2)
        values = ws.get_all_values()
        if len(values) < 2:
            return {}
        import pandas as pd
        df = pd.DataFrame(values[1:], columns=values[0])
        df = df[df['Month'] != 'Total']
        year_cols = [c for c in df.columns if c != 'Month' and c.isdigit()]
        year_data = {}
        for yr_str in year_cols:
            yr = int(yr_str)
            year_data[yr] = {}
            for _, row in df.iterrows():
                try:
                    month_num = list(calendar.month_abbr).index(row['Month'])
                    year_data[yr][month_num] = float(row[yr_str]) if row[yr_str] else 0
                except (ValueError, KeyError):
                    continue
        return year_data
    except Exception as e:
        print(f"  ⚠ Could not load {source} for {country_code}: {e}")
        return {}


def get_yesterday_percentages(gc, source):
    """
    Get yesterday's percentage for each ENTSO-E country for a given source.
    Reads from Summary Table Data worksheet for EU, then individual countries.
    Returns dict {country_iso2: percentage}
    """
    yesterday = datetime.now() - timedelta(days=1)
    month = yesterday.month

    values = {}
    for country_code in ENTSOE_COUNTRIES:
        try:
            source_data = load_monthly_data_for_country(gc, country_code, source)
            total_data = load_monthly_data_for_country(gc, country_code, 'total')

            current_year = datetime.now().year
            source_val = source_data.get(current_year, {}).get(month, None)
            total_val = total_data.get(current_year, {}).get(month, None)

            if source_val is not None and total_val and total_val > 0:
                values[country_code] = round(source_val / total_val * 100, 2)
            else:
                values[country_code] = None
        except Exception as e:
            print(f"  ⚠ {country_code}: {e}")
            values[country_code] = None

    return values


def get_last_month_percentages(gc, source):
    """
    Get last completed month's percentage for each country.
    Returns dict {country_iso2: percentage}
    """
    today = datetime.now()
    last_month = today.month - 1 if today.month > 1 else 12
    year = today.year if today.month > 1 else today.year - 1

    values = {}
    for country_code in ENTSOE_COUNTRIES:
        try:
            source_data = load_monthly_data_for_country(gc, country_code, source)
            total_data = load_monthly_data_for_country(gc, country_code, 'total')

            source_val = source_data.get(year, {}).get(last_month, None)
            total_val = total_data.get(year, {}).get(last_month, None)

            if source_val is not None and total_val and total_val > 0:
                values[country_code] = round(source_val / total_val * 100, 2)
            else:
                values[country_code] = None
        except Exception as e:
            print(f"  ⚠ {country_code}: {e}")
            values[country_code] = None

    return values


def get_annual_percentages(gc, source, year):
    """
    Get full-year percentage for each country for a given year.
    Returns dict {country_iso2: percentage}
    """
    values = {}
    for country_code in ENTSOE_COUNTRIES:
        try:
            source_data = load_monthly_data_for_country(gc, country_code, source)
            total_data = load_monthly_data_for_country(gc, country_code, 'total')

            source_total = sum(source_data.get(year, {}).values())
            total_total = sum(total_data.get(year, {}).values())

            if total_total > 0:
                values[country_code] = round(source_total / total_total * 100, 2)
            else:
                values[country_code] = None
        except Exception as e:
            print(f"  ⚠ {country_code}: {e}")
            values[country_code] = None

    return values


# ============================================================
# GOOGLE DRIVE UPLOAD
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


def get_or_create_folder(service, folder_name, parent_id=None):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    folders = results.get('files', [])
    if folders:
        return folders[0]['id']
    meta = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        meta['parents'] = [parent_id]
    return service.files().create(body=meta, fields='id').execute().get('id')


def upload_map_to_drive(service, file_path, period, year=None):
    """
    Upload map to EU-Electricity-Plots/Maps/{period}/[{year}/]{source}.png
    Returns dict with file_id, view_url, direct_url or None.
    """
    if not GDRIVE_AVAILABLE or service is None:
        return None
    try:
        root_id  = get_or_create_folder(service, 'EU-Electricity-Plots')
        maps_id  = get_or_create_folder(service, 'Maps', root_id)
        period_id = get_or_create_folder(service, period, maps_id)

        if year is not None:
            folder_id = get_or_create_folder(service, str(year), period_id)
        else:
            folder_id = period_id

        filename = os.path.basename(file_path)
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        existing = service.files().list(q=query, spaces='drive', fields='files(id)').execute().get('files', [])

        media = MediaFileUpload(file_path, mimetype='image/png')
        if existing:
            file_id = existing[0]['id']
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            meta = {'name': filename, 'parents': [folder_id]}
            file_id = service.files().create(body=meta, media_body=media, fields='id').execute().get('id')

        # Set public read
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


def save_map_links(period, source, result, year=None):
    """Save map drive links to plots/drive_links.json under Maps section."""
    drive_links_file = 'plots/drive_links.json'
    links = {}
    if os.path.exists(drive_links_file):
        try:
            with open(drive_links_file, 'r') as f:
                links = json.load(f)
        except Exception:
            pass

    if 'Maps' not in links:
        links['Maps'] = {}
    if period not in links['Maps']:
        links['Maps'][period] = {}

    if year is not None:
        if str(year) not in links['Maps'][period]:
            links['Maps'][period][str(year)] = {}
        links['Maps'][period][str(year)][source] = result
    else:
        links['Maps'][period][source] = result

    os.makedirs('plots', exist_ok=True)
    with open(drive_links_file, 'w') as f:
        json.dump(links, f, indent=2)
    print(f"  ✓ drive_links.json updated: Maps/{period}/{source}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate EU electricity generation maps')
    parser.add_argument('--source', default=None,
                        choices=list(ENTSOE_COLORS.keys()),
                        help='Single source (default: all sources)')
    parser.add_argument('--period', default='yesterday',
                        choices=['yesterday', 'last_month', 'annual'],
                        help='Time period (default: yesterday)')
    parser.add_argument('--scale', default='fixed',
                        choices=['fixed', 'dynamic'],
                        help='Color scale (default: fixed)')
    parser.add_argument('--shapefile', default='data/natural_earth/ne_110m_admin_0_countries.shp',
                        help='Path to Natural Earth shapefile')
    args = parser.parse_args()

    if not os.environ.get('GOOGLE_CREDENTIALS_JSON'):
        print("⚠ GOOGLE_CREDENTIALS_JSON not set")
        return

    # Load geodata once
    geodata = load_world_geodata(args.shapefile)

    # Initialize Google Sheets
    creds_dict = json.loads(os.environ.get('GOOGLE_CREDENTIALS_JSON'))
    scope = ['https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
    gc = gspread.authorize(credentials)

    # Initialize Drive
    drive_service = initialize_drive_service()

    # Determine sources to process
    sources = [args.source] if args.source else list(ENTSOE_COLORS.keys())

    # Determine date label and years for annual
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    last_month_num = today.month - 1 if today.month > 1 else 12
    last_month_year = today.year if today.month > 1 else today.year - 1

    if args.period == 'yesterday':
        date_str = yesterday.strftime('%d %B %Y')
        period_folder = 'Yesterday'
    elif args.period == 'last_month':
        date_str = datetime(last_month_year, last_month_num, 1).strftime('%B %Y')
        period_folder = 'LastMonth'
    else:  # annual
        # Process all years from 2015 to current
        years_to_process = list(range(2015, today.year + 1))

    os.makedirs('plots', exist_ok=True)

    print("=" * 60)
    print(f"GENERATING MAPS: {args.period.upper()} | scale={args.scale}")
    print(f"Sources: {', '.join(sources)}")
    print("=" * 60)

    for source in sources:
        print(f"\n--- {DISPLAY_NAMES.get(source, source)} ---")

        if args.period == 'annual':
            for year in years_to_process:
                print(f"  Year {year}...")
                values = get_annual_percentages(gc, source, year)
                date_str = str(year)

                fig = generate_map(geodata, values, source, date_str, scale=args.scale)

                plot_file = f'plots/map_{source}_{year}.png'
                fig.savefig(plot_file, dpi=150, facecolor='white')
                plt.close(fig)
                print(f"  ✓ Saved: {plot_file}")

                if drive_service:
                    result = upload_map_to_drive(drive_service, plot_file, 'Annual', year=year)
                    if result:
                        save_map_links('Annual', source, result, year=year)

        else:
            if args.period == 'yesterday':
                values = get_yesterday_percentages(gc, source)
            else:
                values = get_last_month_percentages(gc, source)

            fig = generate_map(geodata, values, source, date_str, scale=args.scale)

            plot_file = f'plots/map_{source}_{args.period}.png'
            fig.savefig(plot_file, dpi=150, facecolor='white')
            plt.close(fig)
            print(f"  ✓ Saved: {plot_file}")

            if drive_service:
                result = upload_map_to_drive(drive_service, plot_file, period_folder)
                if result:
                    save_map_links(period_folder, source, result)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
