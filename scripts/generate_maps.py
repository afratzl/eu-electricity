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
    'solar':          'Solar Monthly Production',
    'wind':           'Wind Monthly Production',
    'hydro':          'Hydro Monthly Production',
    'biomass':        'Biomass Monthly Production',
    'geothermal':     'Geothermal Monthly Production',
    'gas':            'Gas Monthly Production',
    'coal':           'Coal Monthly Production',
    'nuclear':        'Nuclear Monthly Production',
    'oil':            'Oil Monthly Production',
    'waste':          'Waste Monthly Production',
    'all-renewables': 'All Renewables Monthly Production',
    'total':          'Total Generation Monthly Production',
    # all-non-renewables is derived: Total - All Renewables
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

# Context countries: labeled but not in dashboard
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
    - Clip to European bounding box
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

    # Reproject to EPSG:3035
    world_proj = world.to_crs('EPSG:3035')

    # Clip all geometries to European bounding box
    world_proj_clipped = world_proj.copy()
    world_proj_clipped['geometry'] = world_proj_clipped['geometry'].intersection(EUROPE_CLIP_BOX)

    # Transformer for label coordinates WGS84 -> EPSG:3035
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)

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
        'world':               world,
        'world_proj':          world_proj,
        'world_proj_clipped':  world_proj_clipped,
        'iceland_shifted':     iceland_shifted,
        'transformer':         transformer,
        'bounds':              (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y),
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

    Hatching rule (consistent):
    - Any country WITHOUT data (None) -> white + gray hatch
    - Any country WITH data (including 0.0) -> colored by value

    This applies to: context countries, ENTSO-E NaN countries, Iceland.

    Args:
        geodata: dict from load_world_geodata
        values_by_country: dict {country_iso2: percentage or None}
        source: e.g. 'solar'
        date_str: e.g. '14 May 2026'
        scale: 'fixed' or 'dynamic'

    Returns:
        matplotlib Figure
    """
    world_proj_clipped = geodata['world_proj_clipped']
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

    def plot_hatched(gdf, ax, zorder):
        """White base + gray diagonal stripes = no data"""
        gdf.plot(ax=ax, color='white', edgecolor='#bbbbbb', linewidth=0.6, zorder=zorder)
        gdf.plot(ax=ax, color='none', edgecolor='#999999', linewidth=0.6, hatch='//', zorder=zorder)

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patches.append(Rectangle(
        (0, 0.86), 1.0, 0.14,
        transform=fig.transFigure,
        facecolor='#EBEBEB', edgecolor='none', zorder=0
    ))
    plt.subplots_adjust(left=0.01, right=0.89, top=0.84, bottom=0.01)
    ax.set_facecolor('#cce6ff')

    # Draw ALL non-ENTSO-E European countries as hatched (no data)
    # This includes context countries (RU, BY, TR, UA, Balkans) and
    # tiny countries (Liechtenstein, San Marino, Monaco etc.)
    all_non_entsoe = world_proj_clipped[
        ~world_proj_clipped['iso2'].isin(ENTSOE_COUNTRIES) &
        (world_proj_clipped['CONTINENT'] == 'Europe') |
        world_proj_clipped['NAME'].isin(CONTEXT_COUNTRIES)
    ]
    all_non_entsoe = all_non_entsoe[~all_non_entsoe['NAME'].isin(['Iceland'])].copy()
    all_non_entsoe = all_non_entsoe[~all_non_entsoe['geometry'].is_empty]
    plot_hatched(all_non_entsoe, ax, zorder=1)

    # ENTSO-E countries: colored if data, hatched if None
    for cc in ENTSOE_COUNTRIES:
        country_geom = world_proj_clipped[world_proj_clipped['iso2'] == cc]
        country_geom = country_geom[~country_geom['geometry'].is_empty]
        if country_geom.empty:
            continue
        val = values_by_country.get(cc)
        if val is None:
            plot_hatched(country_geom, ax, zorder=2)
        else:
            country_geom.plot(ax=ax, color=cmap(norm(val)),
                              edgecolor='#bbbbbb', linewidth=0.6, zorder=2)

    # Iceland (relocated): hatched -- no data
    if iceland_shifted is not None:
        iceland_gdf = gpd.GeoDataFrame(geometry=[iceland_shifted], crs=world_proj_clipped.crs)
        plot_hatched(iceland_gdf, ax, zorder=3)
        b = iceland_shifted.bounds
        pad = 50000
        ax.add_patch(Rectangle(
            (b[0]-pad, b[1]-pad), b[2]-b[0]+2*pad, b[3]-b[1]+2*pad,
            linewidth=2.0, edgecolor='#555555', facecolor='none',
            linestyle=(0, (6, 3)), zorder=4
        ))
        ax.text((b[0]+b[2])/2, (b[1]+b[3])/2 - 50000, 'IS',
                fontsize=14, fontweight='bold',
                ha='center', va='center', color='black', zorder=5,
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
                    ha='center', va='center', color='black', zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # Malta label (no polygon in 110m dataset)
    ax.text(MT_X + 80000, MT_Y + 60000, 'MT',
            fontsize=14, fontweight='bold',
            ha='center', va='center', color='black', zorder=6,
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
                    ha='center', va='center', color=color, zorder=6,
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
    Note: all-non-renewables is not stored directly -- use compute_non_renewables().
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


def compute_percentage(source_val, total_val):
    """
    Compute percentage with correct None vs 0 logic:
    - None if total is missing/zero -> hatch (can't compute)
    - 0.0 if source is missing but total available -> white (assume zero generation)
    - percentage otherwise
    """
    if total_val is None or total_val == 0:
        return None
    if source_val is None:
        return 0.0
    return round(source_val / total_val * 100, 2)


def get_values_for_period(gc, source, year, month):
    """
    Get percentage for each ENTSO-E country for a specific year/month.
    Handles all-non-renewables by computing Total - All Renewables.
    Returns dict {country_iso2: percentage or None}
    """
    values = {}
    is_non_renewables = (source == 'all-non-renewables')

    for country_code in ENTSOE_COUNTRIES:
        try:
            total_data = load_monthly_data_for_country(gc, country_code, 'total')
            total_val  = total_data.get(year, {}).get(month, None)

            if is_non_renewables:
                ren_data  = load_monthly_data_for_country(gc, country_code, 'all-renewables')
                ren_val   = ren_data.get(year, {}).get(month, None)
                # Non-renewables = Total - Renewables
                if total_val and total_val > 0 and ren_val is not None:
                    source_val = max(0, total_val - ren_val)
                else:
                    source_val = None
            else:
                source_data = load_monthly_data_for_country(gc, country_code, source)
                source_val  = source_data.get(year, {}).get(month, None)

            values[country_code] = compute_percentage(source_val, total_val)
        except Exception as e:
            print(f"  ⚠ {country_code}: {e}")
            values[country_code] = None

    return values


def get_annual_values(gc, source, year):
    """
    Get full-year percentage for each country for a given year.
    Returns dict {country_iso2: percentage or None}
    """
    values = {}
    is_non_renewables = (source == 'all-non-renewables')

    for country_code in ENTSOE_COUNTRIES:
        try:
            total_data  = load_monthly_data_for_country(gc, country_code, 'total')
            total_total = sum(total_data.get(year, {}).values()) if total_data.get(year) else None

            if is_non_renewables:
                ren_data  = load_monthly_data_for_country(gc, country_code, 'all-renewables')
                ren_total = sum(ren_data.get(year, {}).values()) if ren_data.get(year) else None
                source_total = max(0, total_total - ren_total) if (total_total and ren_total is not None) else None
      
