# Canonical ENTSO-E source keyword mapping
# Single source of truth for all scripts
# Generated: 2026-05-01

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
