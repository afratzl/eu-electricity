#!/usr/bin/env python3
"""
Bluesky Bot - Monthly EU Electricity Generation Update
Posts the current-year plot with last completed month's stats.
Identical format to the daily bot, focused on the previous month.
"""

import os
import sys
from datetime import datetime
from atproto import Client, models
import json
import requests
import calendar


def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive"""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    return destination


def get_plot_from_drive():
    """Get current-year percentage plot file ID from drive_links.json and download it"""
    json_path = 'plots/drive_links.json'

    if not os.path.exists(json_path):
        print(f"❌ drive_links.json not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            links = json.load(f)

        if 'EU' not in links:
            print("❌ EU not found in drive_links.json")
            return None

        if 'CurrentYear' not in links['EU']:
            print("❌ CurrentYear not found in EU section")
            return None

        file_id = links['EU']['CurrentYear']['percentage']['file_id']
        if not file_id:
            print("❌ No file_id found for CurrentYear percentage plot")
            return None

        print(f"✓ Found plot file_id: {file_id}")

        plot_path = 'plots/EU_current_year_percentage.png'
        os.makedirs('plots', exist_ok=True)

        print("📥 Downloading plot from Google Drive...")
        download_file_from_google_drive(file_id, plot_path)
        print(f"✓ Downloaded to: {plot_path}")

        return plot_path

    except Exception as e:
        print(f"❌ Error getting plot from Drive: {e}")
        import traceback; traceback.print_exc()
        return None


def get_stats_from_json():
    """
    Read last completed month's percentages from current_year_monthly_stats.json.
    Returns dict with source percentages, or None if unavailable.
    """
    json_path = 'plots/current_year_monthly_stats.json'

    if not os.path.exists(json_path):
        print(f"⚠️  Stats JSON not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'EU' not in data:
            print("⚠️  EU not found in stats JSON")
            return None

        eu = data['EU']
        last_month = str(eu['last_completed_month'])
        month_data = eu['months'].get(last_month)

        if not month_data:
            print(f"⚠️  No data for month {last_month}")
            return None

        # Map to same keys the daily bot uses
        source_map = {
            'Wind':               'wind',
            'Hydro':              'hydro',
            'Solar':              'solar',
            'Nuclear':            'nuclear',
            'Gas':                'gas',
            'Coal':               'coal',
            'All Renewables':     'renewables',
            'All Non-Renewables': 'non_renewables',
        }

        stats = {v: month_data.get(k) for k, v in source_map.items()}
        stats['month_name'] = eu['month_name']
        stats['year']       = eu['year']

        return stats if all(stats[k] is not None for k in source_map.values()) else None

    except Exception as e:
        print(f"⚠️  Error reading stats JSON: {e}")
        return None


def format_percentage(value):
    """Right-align percentage to width 6, matching daily bot exactly"""
    return f"{value:>5.2f}%"


def create_post_text_and_facets():
    """Create post text and facets -- identical layout to daily bot"""
    stats = get_stats_from_json()

    if stats:
        month_name = stats['month_name']   # e.g. "April"
        year       = stats['year']         # e.g. 2026
        date_str   = f"{month_name} {year}"

        wind_pct     = format_percentage(stats['wind'])
        hydro_pct    = format_percentage(stats['hydro'])
        solar_pct    = format_percentage(stats['solar'])
        nuclear_pct  = format_percentage(stats['nuclear'])
        gas_pct      = format_percentage(stats['gas'])
        coal_pct     = format_percentage(stats['coal'])
        ren_pct      = format_percentage(stats['renewables'])
        non_ren_pct  = format_percentage(stats['non_renewables'])

        post_text = f"""EU Electricity Generation - {date_str}

{ren_pct} of EU electricity generation was renewable.

Wind:   {wind_pct}      Nuclear:  {nuclear_pct}
Hydro:  {hydro_pct}     Gas:         {gas_pct}
Solar:    {solar_pct}     Coal:         {coal_pct}

afratzl.github.io/eu-electricity
#EU #Renewables #Electricity #EnergySky #ClimateSky"""

    else:
        # Fallback if stats not available
        current_date = datetime.now()
        last_month   = current_date.month - 1 or 12
        last_year    = current_date.year if current_date.month > 1 else current_date.year - 1
        date_str     = f"{calendar.month_name[last_month]} {last_year}"

        post_text = f"""EU Electricity Generation - {date_str}

Monthly electricity generation breakdown across all EU member states.

Data: ENTSO-E
afratzl.github.io/eu-electricity

#EU #Renewables #Electricity #EnergySky #ClimateSky"""

    # Build facets (clickable link + hashtags) -- identical to daily bot
    facets = []

    link_text  = "afratzl.github.io/eu-electricity"
    link_start = post_text.find(link_text)
    if link_start != -1:
        facets.append(
            models.AppBskyRichtextFacet.Main(
                features=[models.AppBskyRichtextFacet.Link(uri=f"https://{link_text}")],
                index=models.AppBskyRichtextFacet.ByteSlice(
                    byteStart=len(post_text[:link_start].encode('utf-8')),
                    byteEnd=len(post_text[:link_start + len(link_text)].encode('utf-8'))
                )
            )
        )

    for tag in ['#EU', '#Renewables', '#Electricity', '#EnergySky', '#ClimateSky']:
        tag_start = post_text.find(tag)
        if tag_start != -1:
            facets.append(
                models.AppBskyRichtextFacet.Main(
                    features=[models.AppBskyRichtextFacet.Tag(tag=tag[1:])],
                    index=models.AppBskyRichtextFacet.ByteSlice(
                        byteStart=len(post_text[:tag_start].encode('utf-8')),
                        byteEnd=len(post_text[:tag_start + len(tag)].encode('utf-8'))
                    )
                )
            )

    return post_text, facets


def post_to_bluesky():
    """Main function -- mirrors daily bot exactly"""
    print("=" * 60)
    print("BLUESKY BOT - EU ELECTRICITY GENERATION (MONTHLY)")
    print("=" * 60)

    handle   = os.environ.get('BLUESKY_HANDLE')
    password = os.environ.get('BLUESKY_PASSWORD')

    if not handle or not password:
        print("❌ Error: BLUESKY_HANDLE and BLUESKY_PASSWORD must be set")
        sys.exit(1)

    plot_path = get_plot_from_drive()
    if not plot_path:
        print("❌ Error: Could not get current-year plot from Google Drive")
        sys.exit(1)

    post_text, facets = create_post_text_and_facets()
    print(f"\n📝 Post text:\n{post_text}\n")
    print(f"✓ Created {len(facets)} facets")

    try:
        print("🔐 Logging in to Bluesky...")
        client = Client()
        client.login(handle, password)
        print(f"✓ Logged in as {handle}")

        print("📤 Uploading image...")
        with open(plot_path, 'rb') as f:
            img_data = f.read()
        upload_response = client.upload_blob(img_data)

        print("📮 Posting to Bluesky...")
        client.send_post(
            text=post_text,
            facets=facets,
            embed={
                '$type': 'app.bsky.embed.images',
                'images': [{
                    'alt': f'EU electricity generation chart showing monthly breakdown for {datetime.now().year}',
                    'image': upload_response.blob
                }]
            }
        )

        print("✓ Posted successfully!")
        print(f"   Profile: https://bsky.app/profile/{handle}")

    except Exception as e:
        print(f"❌ Error posting to Bluesky: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    post_to_bluesky()
