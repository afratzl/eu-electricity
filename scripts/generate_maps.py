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
from config import ENTSOE_COUNTRIES, ENTSOE_COLORS, DISPLAY_NAMES

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

# ENTSOE_COUNTRIES imported from config.py

# ENTSOE_COLORS imported from config.py

# DISPLAY_NAMES imported from config.py

# Worksheet name mapping (Google Sheets)
# Note: all-non-renewables is derived as Total - All Renewables
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
}

FIXED_SCALE_MAX = {
    'solar':              40,
    'wind':               70,
    'hydro':              80,
    'biomass':            30,
    'geothermal':         5,
    'gas':                70,
    'coal':               60,
    'nuclear':            80,
    'oil':                15,
    'waste':              10,
    'all-renewables':     100,
    'all-non-renewables': 100,
}


# Label position offsets in EPSG:3035 metres
LABEL_OFFSETS = {
    'CY': (-180000, 0),
    'AT': (50000, 0),
    'SE': (-130000, -300000),
    'DE': (50000, 0),
    'HR': (80000, 0),
    'CH': (50000, 0),
    'CZ': (-30000, -30000),
    'ME': (-65000, 0),
    'XK': (65000, 0),
}

CONTEXT_LABEL_OFFSETS = {
    'Ukraine':        (0, -120000),
    'United Kingdom': (0, -150000),
    'Armenia':        (-80000, 0),
    'Azerbaijan':     (50000, 80000),
}

# Malta position (too small for 110m shapefile)
MT_X, MT_Y = 4721805, 1408134 - 60000

# Context countries: labeled but not in dashboard
CONTEXT_COUNTRIES = [
    'Russia', 'Belarus', 'Turkey', 'Ukraine', 'United Kingdom', 'Serbia', 'Albania',
    'Bosnia and Herz.', 'North Macedonia', 'Montenegro', 'Kosovo',
    'Georgia', 'Armenia', 'Azerbaijan'
]

# label, fontweight, fontsize, color
CONTEXT_LABELS = {
    # Suspended from ENTSO-E -- gray, smaller
    'Russia':            ('RU', 'normal', 11, '#888888'),
    'Belarus':           ('BY', 'normal', 11, '#888888'),
    # Non-ENTSO-E neighbors -- bold black
    'Turkey':            ('TR', 'bold',   14, 'black'),
    'Ukraine':           ('UA', 'bold',   14, 'black'),
    'United Kingdom':    ('UK', 'bold',   14, 'black'),
    'Albania':           ('AL', 'bold',   14, 'black'),
    'Armenia':           ('AM', 'bold',   14, 'black'),
    'Azerbaijan':        ('AZ', 'bold',   14, 'black'),
    # RS, BA, ME, MK, XK, GE removed -- now in ENTSOE_COUNTRIES, labeled by ENTSO-E loop
}

# Europe clip box in EPSG:3035 (removes overseas territories)
EUROPE_CLIP_BOX = box(1200000, 900000, 8000000, 5900000)


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
        'world':              world,
        'world_proj':         world_proj,
        'world_proj_clipped': world_proj_clipped,
        'iceland_shifted':    iceland_shifted,
        'transformer':        transformer,
        'bounds':             (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y),
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

    Hatching rule (consistent for all countries):
    - Any country WITHOUT data (None) -> white + gray hatch
    - Any country WITH data (including 0.0) -> colored by value

    Applies to: all non-ENTSO-E countries, ENTSO-E NaN, Iceland.

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
    maxx_extended = 7550000  # Extended east to include Caucasus

    source_color = ENTSOE_COLORS.get(source, '#888888')
    cmap = LinearSegmentedColormap.from_list(source, ['white', source_color])

    if scale == 'dynamic':
        vals = [v for v in values_by_country.values() if v is not None]
        vmax = max(vals) if vals else 1
    else:
        vmax = FIXED_SCALE_MAX.get(source, 100)

    norm = Normalize(vmin=0, vmax=vmax)

    def plot_hatched(gdf, ax, zorder):
        """White base + light sparse diagonal stripes = no data"""
        gdf.plot(ax=ax, color='white', edgecolor='#bbbbbb', linewidth=0.6, zorder=zorder)
        gdf.plot(ax=ax, color='none', edgecolor='#cccccc', linewidth=0.6, hatch='/', zorder=zorder)

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patches.append(Rectangle(
        (0, 0.86), 1.0, 0.14,
        transform=fig.transFigure,
        facecolor='#EBEBEB', edgecolor='none', zorder=0
    ))
    plt.subplots_adjust(left=0.0, right=0.99, top=0.84, bottom=0.11)
    maxy_cropped = maxy + (maxy - miny) * 0.03 * 0.3  # crop top
    ax.set_facecolor('#cce6ff')

    # All non-ENTSO-E countries (except Iceland which is handled separately):
    # hatched -- no data
    non_entsoe = world_proj_clipped[
        (~world_proj_clipped['iso2'].isin(ENTSOE_COUNTRIES)) &
        (~world_proj_clipped['geometry'].is_empty) &
        (world_proj_clipped['NAME'] != 'Iceland') &
        (
            (world_proj_clipped['CONTINENT'] == 'Europe') |
            (world_proj_clipped['NAME'].isin(CONTEXT_COUNTRIES))
        )
    ].copy()
    non_entsoe = non_entsoe[~non_entsoe['geometry'].is_empty]
    if not non_entsoe.empty:
        plot_hatched(non_entsoe, ax, zorder=1)

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
        if minx <= lx <= maxx_extended + 50000 and miny <= ly <= maxy_cropped:
            ax.text(lx, ly, cc,
                    fontsize=14, fontweight='bold',
                    ha='center', va='center', color='black', zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # Malta label (no polygon in 110m dataset)
    ax.text(MT_X + 80000, MT_Y + 60000, 'MT',  # MT_Y already adjusted (-60000)
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
        if minx <= lx <= maxx_extended + 50000 and miny <= ly <= maxy_cropped:
            ax.text(lx, ly, label,
                    fontsize=size, fontweight=weight,
                    ha='center', va='center', color=color, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])

    ax.set_xlim(minx, maxx_extended + 50000)
    ax.set_ylim(miny, maxy_cropped)
    ax.set_aspect('equal')
    ax.axis('off')

    # Titles centered across full banner
    source_display = DISPLAY_NAMES.get(source, source.title())
    fig.text(0.5, 0.965, 'Electricity Generation',
             fontsize=36, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.91, f'{source_display} · Fraction of Total · {date_str}',
             fontsize=28, fontweight='normal', ha='center', va='top')

    # Horizontal colorbar below map
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.07, 0.075, 0.86, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_label('Fraction of Total (%)', fontsize=18, labelpad=12)
    cbar.ax.tick_params(labelsize=18)

    # Watermark
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    fig.text(0.02, 0.028, "eu-electricity.eu",
             ha='left', va='top', fontsize=12, color='#666', style='italic')
    fig.text(0.98, 0.028, f"Generated: {timestamp}",
             ha='right', va='top', fontsize=12, color='#666', style='italic')

    # ── Country legend overlay (Russia space) ──────────────────
    x_range = (maxx_extended + 50000) - minx
    y_range = maxy_cropped - miny
    map_left_f, map_right_f = 0.0, 0.99
    map_bottom_f, map_top_f = 0.11, 0.84

    def data_to_fig(xd, yd):
        ax_x = (xd - minx) / x_range
        ax_y = (yd - miny) / y_range
        return map_left_f + ax_x * (map_right_f - map_left_f), map_bottom_f + ax_y * (map_top_f - map_bottom_f)

    px0, py1 = data_to_fig(5750000, 5600000)
    px1, py0 = data_to_fig(7310000, 3340000)
    pw = px1 - px0
    ph = py1 - py0

    sorted_countries = sorted(
        [(cc, values_by_country.get(cc)) for cc in values_by_country],
        key=lambda x: (x[1] is None, -(x[1] or 0))
    )

    n_leg = len(sorted_countries)
    n_cols_leg = 4
    n_rows_leg = (n_leg + n_cols_leg - 1) // n_cols_leg

    flag_w = 0.019
    flag_h = flag_w * 0.6
    row_h = ph / n_rows_leg * 0.95
    entry_w = flag_w + 0.025
    col_gap = 0.028
    total_w = n_cols_leg * entry_w + (n_cols_leg - 1) * col_gap
    start_x = px0 + (pw - total_w) / 2

    content_top    = py1 - row_h * 0.5 + flag_h / 2 + 0.003
    content_bottom = py1 - (n_rows_leg - 0.5) * row_h - flag_h / 2 - 0.010
    content_left   = start_x - 0.004
    content_right  = start_x + total_w + 0.004

    cw = content_right - content_left
    ch = content_top - content_bottom
    leg_pad = 0.05

    fig.patches.append(Rectangle(
        (content_left - cw * leg_pad, content_bottom - ch * leg_pad),
        cw * (1 + 2 * leg_pad), ch * (1 + 2 * leg_pad),
        transform=fig.transFigure,
        facecolor='#f9f9f9', edgecolor='none', alpha=0.88, zorder=10))

    flags_dir = 'flags'
    for i, (cc, val) in enumerate(sorted_countries):
        row_i = i // n_cols_leg
        col_i = i % n_cols_leg
        fy = py1 - (row_i + 0.5) * row_h
        fx = start_x + col_i * (entry_w + col_gap)

        value_str = f"{val:.0f}%" if val is not None else "—"
        map_color = cmap(norm(val)) if val is not None else '#eeeeee'

        # Try real SVG flag via cairosvg, fall back to colored rectangle
        flag_path = os.path.join(flags_dir, f'{cc}.svg')
        flag_drawn = False
        if os.path.exists(flag_path):
            try:
                import cairosvg, io
                from PIL import Image as PILImage
                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                # Convert SVG to PNG at reasonable resolution
                png_data = cairosvg.svg2png(url=os.path.abspath(flag_path), output_width=60, output_height=40)
                pil_img = PILImage.open(io.BytesIO(png_data)).convert('RGBA')
                img_arr = np.array(pil_img)
                # Compute figure pixel size of flag
                fig_w_px = fig.get_figwidth() * fig.get_dpi()
                fig_h_px = fig.get_figheight() * fig.get_dpi()
                flag_w_px = flag_w * fig_w_px
                # Place flag using AnnotationBbox in figure coords
                imagebox = OffsetImage(img_arr, zoom=flag_w_px / img_arr.shape[1])
                imagebox.image.axes = ax
                # Convert figure coords to display coords
                disp_x = (fx + flag_w / 2) * fig_w_px
                disp_y = (fy) * fig_h_px
                ab = AnnotationBbox(imagebox, (fx + flag_w / 2, fy),
                                    xycoords=fig.transFigure,
                                    frameon=False, zorder=11, pad=0)
                fig.add_artist(ab)
                flag_drawn = True
            except Exception as e:
                pass

        if not flag_drawn:
            fig.patches.append(Rectangle(
                (fx, fy - flag_h / 2), flag_w, flag_h,
                transform=fig.transFigure,
                facecolor=map_color, edgecolor='#888', linewidth=0.4, zorder=11))

        # CC label below flag
        fig.text(fx + flag_w / 2, fy - flag_h / 2 - 0.005, cc,
                 transform=fig.transFigure,
                 fontsize=8, fontweight='bold', va='top', ha='center',
                 color='#333', zorder=12)
        # % to right of flag
        fig.text(fx + flag_w + 0.010, fy - flag_h * 0.15, value_str,
                 transform=fig.transFigure,
                 fontsize=10, fontweight='bold', va='center', ha='left',
                 color='#111', zorder=12)

    return fig


# ============================================================
# GOOGLE SHEETS DATA LOADING
# ============================================================

def get_spreadsheet(gc, country_code):
    """Get spreadsheet for a country, using drive_links.json if available."""
    from gspread.exceptions import APIError
    sheet_name = f'{country_code} Electricity Production Data'
    drive_links_file = 'plots/drive_links.json'
    sheet_id = None
    # Read sheet ID from drive_links.json (local file read only, no API calls)
    if os.path.exists(drive_links_file):
        try:
            with open(drive_links_file, 'r') as f:
                links = json.load(f)
            if country_code in links and 'data_sheet_id' in links[country_code]:
                sheet_id = links[country_code]['data_sheet_id']
        except (json.JSONDecodeError, KeyError, IOError):
            pass
    # API call -- let 429s propagate to api_call_with_retry
    if sheet_id:
        return gc.open_by_key(sheet_id)
    return gc.open(sheet_name)


def parse_worksheet(raw_values):
    """
    Parse raw worksheet values into {year: {month: gwh}} dict.
    """
    import pandas as pd
    if len(raw_values) < 2:
        return {}
    df = pd.DataFrame(raw_values[1:], columns=raw_values[0])
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


def api_call_with_retry(fn, max_retries=5):
    """Execute a Google API call with exponential backoff on 429 quota errors."""
    from gspread.exceptions import APIError
    for attempt in range(max_retries):
        try:
            return fn()
        except APIError as e:
            if '429' in str(e):
                wait = 60 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Max retries exceeded after {max_retries} attempts")


def load_all_data_for_country(gc, country_code):
    """
    Load ALL worksheets for one country in a single pass.
    Returns dict: {source: {year: {month: gwh}}}
    Reads all sheets sequentially with 1s delay between sheets.
    Caller should wait 15s before next country to stay under quota.
    api_call_with_retry handles 429s -- do NOT catch exceptions here
    so retry logic can work properly.
    """
    spreadsheet = api_call_with_retry(lambda: get_spreadsheet(gc, country_code))
    all_data = {}
    for source, ws_name in WORKSHEET_NAMES.items():
        ws  = api_call_with_retry(lambda name=ws_name: spreadsheet.worksheet(name))
        raw = api_call_with_retry(ws.get_all_values)
        all_data[source] = parse_worksheet(raw)
        time.sleep(1)  # 1s between worksheets within same country
    return all_data


def load_summary_table_for_country(gc, country_code):
    """
    Load percentage values from Summary Table Data worksheet.
    Returns dict: {source: {yesterday_pct, last_week_pct}}
    Much faster than loading all monthly sheets.
    """
    spreadsheet = api_call_with_retry(lambda: get_spreadsheet(gc, country_code))
    ws  = api_call_with_retry(lambda: spreadsheet.worksheet('Summary Table Data'))
    raw = api_call_with_retry(ws.get_all_values)
    if len(raw) < 2:
        return {}

    headers = raw[0]
    # Find column indices
    def col(name):
        for i, h in enumerate(headers):
            if name.lower() in h.lower():
                return i
        return None

    yesterday_pct_col = col('Yesterday_%')
    lastweek_pct_col  = col('LastWeek_%')
    ytd_pct_col       = col(f'YTD')
    prev_pct_col      = col(f'_GWh')  # not needed

    # Source name -> lowercase key mapping
    source_map = {
        'All Renewables':     'all-renewables',
        'All Non-Renewables': 'all-non-renewables',
        'Renewables':         'all-renewables',      # fallback
        'Non-Renewables':     'all-non-renewables',  # fallback
        'Solar':   'solar',   'Wind':  'wind',   'Hydro':     'hydro',
        'Biomass': 'biomass', 'Gas':   'gas',    'Coal':      'coal',
        'Nuclear': 'nuclear', 'Oil':   'oil',    'Waste':     'waste',
        'Geothermal': 'geothermal',
    }

    result = {}
    for row in raw[1:]:
        if not row or not row[0]:
            continue
        src_name = row[0].strip()
        src_key  = source_map.get(src_name)
        if not src_key:
            continue
        try:
            y_pct  = float(row[yesterday_pct_col]) if yesterday_pct_col is not None and row[yesterday_pct_col] else None
            lw_pct = float(row[lastweek_pct_col])  if lastweek_pct_col  is not None and row[lastweek_pct_col]  else None
        except (ValueError, IndexError):
            y_pct = lw_pct = None
        result[src_key] = {'yesterday': y_pct, 'last_week': lw_pct}
    return result


def compute_percentage(source_val, total_val):
    """
    Compute percentage with correct None vs 0 logic:
    - None if total is missing/zero -> hatch (can't compute denominator)
    - 0.0 if source is missing but total available -> white (assume zero generation)
    - percentage otherwise
    """
    if total_val is None or total_val == 0:
        return None
    if source_val is None:
        return 0.0
    return round(source_val / total_val * 100, 2)


# Note: get_values_for_period and get_annual_values removed --
# data loading is handled in main() to avoid redundant API calls.


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
    Upload map to EU-Electricity-Plots/Maps/{period}/[{year}/]{filename}
    Returns dict with file_id, view_url, direct_url or None.
    """
    if not GDRIVE_AVAILABLE or service is None:
        return None
    try:
        root_id   = get_or_create_folder(service, 'EU-Electricity-Plots')
        maps_id   = get_or_create_folder(service, 'Maps', root_id)
        period_id = get_or_create_folder(service, period, maps_id)
        folder_id = get_or_create_folder(service, str(year), period_id) if year else period_id

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


def save_map_links(period, source, result, year=None, plot_type='percentage'):
    """
    Save map drive links to plots/drive_links.json under Maps section.
    Structure: Maps -> period -> [year ->] source -> plot_type -> {file_id, direct_url...}
    plot_type: 'percentage' or 'absolute'
    """
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
        if source not in links['Maps'][period][str(year)]:
            links['Maps'][period][str(year)][source] = {}
        links['Maps'][period][str(year)][source][plot_type] = result
    else:
        if source not in links['Maps'][period]:
            links['Maps'][period][source] = {}
        links['Maps'][period][source][plot_type] = result

    os.makedirs('plots', exist_ok=True)
    with open(drive_links_file, 'w') as f:
        json.dump(links, f, indent=2)
    print(f"  ✓ drive_links.json updated: Maps/{period}/{source}/{plot_type}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate EU electricity generation maps')
    parser.add_argument('--source', default=None,
                        choices=list(DISPLAY_NAMES.keys()),
                        help='Single source (default: all sources)')
    parser.add_argument('--period', default='yesterday',
                        choices=['yesterday', 'last_week', 'monthly', 'annual'],
                        help='Time period (default: yesterday)')
    parser.add_argument('--year', type=int, default=None,
                        help='Year for monthly/annual period')
    parser.add_argument('--month', type=int, default=None,
                        help='Month (1-12) for monthly period')
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
    sources = [args.source] if args.source else list(DISPLAY_NAMES.keys())

    # Date info
    today           = datetime.now()
    yesterday       = today - timedelta(days=1)
    last_month_num  = today.month - 1 if today.month > 1 else 12
    last_month_year = today.year if today.month > 1 else today.year - 1

    os.makedirs('plots', exist_ok=True)

    print("=" * 60)
    print(f"GENERATING MAPS: {args.period.upper()} | scale={args.scale}")
    print(f"Sources: {', '.join(sources)}")
    print("=" * 60)

    # Load data -- strategy depends on period
    print("\n📊 Loading data from Google Sheets...")
    all_country_data    = {}  # monthly sheets: {country: {source: {year: {month: gwh}}}}
    summary_table_data  = {}  # summary sheet:  {country: {source: {yesterday, last_week}}}

    if args.period in ('yesterday', 'last_week'):
        # Fast path: read only Summary Table Data worksheet per country
        for i, country_code in enumerate(ENTSOE_COUNTRIES):
            print(f"  Loading {country_code} ({i+1}/{len(ENTSOE_COUNTRIES)})...")
            try:
                summary_table_data[country_code] = load_summary_table_for_country(gc, country_code)
            except Exception as e:
                print(f"  ⚠ Failed to load {country_code}: {e} -- will show as hatched")
                summary_table_data[country_code] = {}
            if i < len(ENTSOE_COUNTRIES) - 1:
                time.sleep(20)
    else:
        # Full path: read monthly production sheets
        for i, country_code in enumerate(ENTSOE_COUNTRIES):
            print(f"  Loading {country_code} ({i+1}/{len(ENTSOE_COUNTRIES)})...")
            try:
                all_country_data[country_code] = load_all_data_for_country(gc, country_code)
            except Exception as e:
                print(f"  ⚠ Failed to load {country_code}: {e} -- will show as hatched")
                all_country_data[country_code] = {}
            if i < len(ENTSOE_COUNTRIES) - 1:
                time.sleep(15)
    print("  ✓ All country data loaded")

    for source in sources:
        print(f"\n--- {DISPLAY_NAMES.get(source, source)} ---")
        is_non_renewables = (source == 'all-non-renewables')

        if args.period == 'yesterday':
            date_str = yesterday.strftime('%d %B %Y')
            values = {}
            for country_code in ENTSOE_COUNTRIES:
                try:
                    pct = summary_table_data.get(country_code, {}).get(source, {}).get('yesterday')
                    values[country_code] = pct
                except Exception as e:
                    print(f"  ⚠ {country_code}: {e}")
                    values[country_code] = None
            fig       = generate_map(geodata, values, source, date_str, scale=args.scale)
            plot_file = f'plots/map_{source}_yesterday.png'
            fig.savefig(plot_file, dpi=150, facecolor='white')
            plt.close(fig)
            print(f"  ✓ Saved: {plot_file}")
            if drive_service:
                result = upload_map_to_drive(drive_service, plot_file, 'Yesterday')
                if result:
                    save_map_links('Yesterday', source, result, plot_type='percentage')

        elif args.period == 'last_week':
            date_str = f"Week ending {yesterday.strftime('%d %B %Y')}"
            values = {}
            for country_code in ENTSOE_COUNTRIES:
                try:
                    pct = summary_table_data.get(country_code, {}).get(source, {}).get('last_week')
                    values[country_code] = pct
                except Exception as e:
                    print(f"  ⚠ {country_code}: {e}")
                    values[country_code] = None
            fig       = generate_map(geodata, values, source, date_str, scale=args.scale)
            plot_file = f'plots/map_{source}_last_week.png'
            fig.savefig(plot_file, dpi=150, facecolor='white')
            plt.close(fig)
            print(f"  ✓ Saved: {plot_file}")
            if drive_service:
                result = upload_map_to_drive(drive_service, plot_file, 'LastWeek')
                if result:
                    save_map_links('LastWeek', source, result, plot_type='percentage')

        elif args.period == 'monthly':
            year  = args.year  or (last_month_year if today.month == 1 else today.year if today.month > 1 else today.year - 1)
            month = args.month or (today.month - 1 if today.month > 1 else 12)
            date_str = datetime(year, month, 1).strftime('%B %Y')
            values = {}
            for country_code in ENTSOE_COUNTRIES:
                try:
                    cd        = all_country_data[country_code]
                    total_val = cd.get('total', {}).get(year, {}).get(month, None)
                    if is_non_renewables:
                        ren_val    = cd.get('all-renewables', {}).get(year, {}).get(month, None)
                        source_val = max(0, total_val - ren_val) if (total_val and ren_val is not None) else None
                    else:
                        source_val = cd.get(source, {}).get(year, {}).get(month, None)
                    values[country_code] = compute_percentage(source_val, total_val)
                except Exception as e:
                    print(f"  ⚠ {country_code}: {e}")
                    values[country_code] = None
            fig       = generate_map(geodata, values, source, date_str, scale=args.scale)
            plot_file = f'plots/map_{source}_{year}_{month:02d}.png'
            fig.savefig(plot_file, dpi=150, facecolor='white')
            plt.close(fig)
            print(f"  ✓ Saved: {plot_file}")
            if drive_service:
                result = upload_map_to_drive(drive_service, plot_file, 'Monthly', year=f"{year}_{month:02d}")
                if result:
                    save_map_links('Monthly', source, result, year=f"{year}_{month:02d}", plot_type='percentage')

        elif args.period == 'annual':
            for year in range(2015, today.year + 1):
                print(f"  Year {year}...")
                values = {}
                for country_code in ENTSOE_COUNTRIES:
                    try:
                        cd = all_country_data[country_code]
                        total_total = sum(cd.get('total', {}).get(year, {}).values()) if cd.get('total', {}).get(year) else None
                        if is_non_renewables:
                            ren_total    = sum(cd.get('all-renewables', {}).get(year, {}).values()) if cd.get('all-renewables', {}).get(year) else None
                            source_total = max(0, total_total - ren_total) if (total_total and ren_total is not None) else None
                        else:
                            source_total = sum(cd.get(source, {}).get(year, {}).values()) if cd.get(source, {}).get(year) else None
                        values[country_code] = compute_percentage(source_total, total_total)
                    except Exception as e:
                        print(f"  ⚠ {country_code}: {e}")
                        values[country_code] = None

                date_str  = str(year)
                fig       = generate_map(geodata, values, source, date_str, scale=args.scale)
                plot_file = f'plots/map_{source}_{year}.png'
                fig.savefig(plot_file, dpi=150, facecolor='white')
                plt.close(fig)
                print(f"  ✓ Saved: {plot_file}")
                if drive_service:
                    result = upload_map_to_drive(drive_service, plot_file, 'Annual', year=year)
                    if result:
                        save_map_links('Annual', source, result, year=year, plot_type='percentage')



    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
