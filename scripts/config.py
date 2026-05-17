# Canonical ENTSO-E source keyword mapping
# Single source of truth for all scripts
# Generated: 2026-05-01

# ============================================================
# COUNTRY LISTS
# ============================================================

# EU member states in ENTSO-E (used for EU aggregate calculations)
EU_COUNTRIES = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
    'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE',
]

# Non-EU countries actively in the dashboard (not included in EU aggregate)
NON_EU_COUNTRIES = [
    'NO', 'CH', 'MD',
    'RS', 'BA', 'ME', 'MK', 'XK',  # Western Balkans
    'GE',                            # Georgia
]

# All countries in the dashboard (EU + non-EU)
ENTSOE_COUNTRIES = EU_COUNTRIES + NON_EU_COUNTRIES

# Countries excluded from dashboard for now:
# GB  -- United Kingdom (Brexit, separate grid data)
# IS  -- Iceland (no ENTSO-E reporting)
# AL  -- Albania (no data)
# AM  -- Armenia (no data)
# AZ  -- Azerbaijan (no data)
# TR  -- Turkey (no data)
# RU  -- Russia (suspended from ENTSO-E 2022)
# BY  -- Belarus (suspended from ENTSO-E 2022)
EXCLUDED_COUNTRIES = [
    'GB',  # United Kingdom (not currently reporting, may change)
    'IS',  # Iceland
    'AL',  # Albania
    'AM',  # Armenia
    'AZ',  # Azerbaijan
    'TR',  # Turkey
    'RU',  # Russia (suspended)
    'BY',  # Belarus (suspended)
]

# ============================================================
# COUNTRY DISPLAY NAMES
# ============================================================

COUNTRY_DISPLAY_NAMES = {
    'EU': 'European Union',
    'AT': 'Austria',
    'BE': 'Belgium',
    'BG': 'Bulgaria',
    'HR': 'Croatia',
    'CY': 'Cyprus',
    'CZ': 'Czechia',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'FI': 'Finland',
    'FR': 'France',
    'DE': 'Germany',
    'GR': 'Greece',
    'HU': 'Hungary',
    'IE': 'Ireland',
    'IT': 'Italy',
    'LV': 'Latvia',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'MT': 'Malta',
    'NL': 'Netherlands',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'SK': 'Slovakia',
    'SI': 'Slovenia',
    'ES': 'Spain',
    'SE': 'Sweden',
    'NO': 'Norway',
    'CH': 'Switzerland',
    'GB': 'United Kingdom',
    'MD': 'Moldova',
    # Western Balkans
    'RS': 'Serbia',
    'BA': 'Bosnia and Herzegovina',
    'ME': 'Montenegro',
    'MK': 'North Macedonia',
    'XK': 'Kosovo',
    'AL': 'Albania',
    # Caucasus
    'GE': 'Georgia',
    'AM': 'Armenia',
    'AZ': 'Azerbaijan',
}

# ============================================================
# SOURCE KEYWORDS
# ============================================================

SOURCE_KEYWORDS = {
    'solar':      ['Solar'],
    'wind':       ['Wind Onshore', 'Wind Offshore'],
    'hydro':      ['Hydro Water Reservoir', 'Hydro Run-of-river and poundage', 'Hydro Pumped Storage'],
    'biomass':    ['Biomass'],
    'geothermal': ['Geothermal'],
    'gas':        ['Fossil Gas', 'Fossil Coal-derived gas'],
    'coal':       ['Fossil Hard coal', 'Fossil Brown coal/Lignite'],
    'nuclear':    ['Nuclear'],
    'oil':        ['Fossil Oil', 'Fossil Oil shale'],
    'waste':      ['Waste'],
}

RENEWABLES = ['solar', 'wind', 'hydro', 'biomass', 'geothermal']
NON_RENEWABLES = ['gas', 'coal', 'nuclear', 'oil', 'waste']

# Small categories: included in aggregates only, not as individual series
SMALL_RENEWABLES = ['Other renewable', 'Marine']
SMALL_NON_RENEWABLES = ['Fossil Peat', 'Other']

# Aggregates derived automatically from individual sources + small categories
SOURCE_KEYWORDS['all-renewables'] = (
    [kw for s in RENEWABLES for kw in SOURCE_KEYWORDS[s]] + SMALL_RENEWABLES
)
SOURCE_KEYWORDS['all-non-renewables'] = (
    [kw for s in NON_RENEWABLES for kw in SOURCE_KEYWORDS[s]] + SMALL_NON_RENEWABLES
)

# ============================================================
# ENERGY SOURCES MAPPING
# Maps display names to ENTSO-E API column keywords
# 'Total Generation' is a special case: sum all columns
# ============================================================

ENERGY_SOURCES = {
    'Solar':            SOURCE_KEYWORDS['solar'],
    'Wind Onshore':     ['Wind Onshore'],
    'Wind Offshore':    ['Wind Offshore'],
    'Wind':             SOURCE_KEYWORDS['wind'],
    'Hydro':            SOURCE_KEYWORDS['hydro'],
    'Biomass':          SOURCE_KEYWORDS['biomass'],
    'Geothermal':       SOURCE_KEYWORDS['geothermal'],
    'Gas':              SOURCE_KEYWORDS['gas'],
    'Coal':             SOURCE_KEYWORDS['coal'],
    'Nuclear':          SOURCE_KEYWORDS['nuclear'],
    'Oil':              SOURCE_KEYWORDS['oil'],
    'Waste':            SOURCE_KEYWORDS['waste'],
    'All Renewables':   SOURCE_KEYWORDS['all-renewables'],
    'Total Generation': 'ALL',
}

# ============================================================
# DISPLAY NAMES
# ============================================================

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

# ============================================================
# COLORS
# ============================================================

ENTSOE_COLORS = {
    # lowercase keys (generate_maps.py, intraday_analysis.py)
    'solar':              '#FFD700',  # Gold
    'wind':               '#228B22',  # Forest Green
    'wind-onshore':       '#2E8B57',  # Sea Green
    'wind-offshore':      '#008B8B',  # Dark Cyan
    'hydro':              '#1E90FF',  # Dodger Blue
    'biomass':            '#9ACD32',  # Yellow Green
    'geothermal':         '#708090',  # Slate Gray
    'gas':                '#FF1493',  # Deep Pink
    'coal':               '#8B008B',  # Dark Magenta
    'nuclear':            '#8B4513',  # Saddle Brown
    'oil':                '#191970',  # Midnight Blue
    'waste':              '#808000',  # Olive
    'all-renewables':     '#00CED1',  # Dark Turquoise
    'all-non-renewables': '#000000',  # Black
    # Title Case aliases (eu_energy_plotting.py)
    'Solar':              '#FFD700',
    'Wind':               '#228B22',
    'Wind Onshore':       '#2E8B57',
    'Wind Offshore':      '#008B8B',
    'Hydro':              '#1E90FF',
    'Biomass':            '#9ACD32',
    'Geothermal':         '#708090',
    'Gas':                '#FF1493',
    'Coal':               '#8B008B',
    'Nuclear':            '#8B4513',
    'Oil':                '#191970',
    'Waste':              '#808000',
    'All Renewables':     '#00CED1',
    'All Non-Renewables': '#000000',
}
