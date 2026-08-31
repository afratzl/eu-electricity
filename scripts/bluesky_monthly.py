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


def get_monthly_map_from_drive(month_name):
    """Get the month's renewables map file ID from drive_links.json and download it -- mirrors the daily bot's get_map_from_drive"""
    json_path = 'plots/drive_links.json'

    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            links = json.load(f)

        file_id = links.get('Maps', {}).get('Monthly', {}).get(month_name, {}).get('all-renewables', {}).get('percentage', {}).get('file_id')

        if not file_id:
            print(f"⚠️  No renewables map file_id found for {month_name}")
            return None

        print(f"✓ Found map file_id: {file_id}")

        map_path = f'plots/map_all-renewables_{month_name}.png'
        os.makedirs('plots', exist_ok=True)

        print("📥 Downloading map from Google Drive...")
        download_file_from_google_drive(file_id, map_path)
        print(f"✓ Downloaded to: {map_path}")

        return map_path

    except Exception as e:
        print(f"⚠️  Error getting map from Drive: {e}")
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

        # Real, measured advance widths from Inter -- Bluesky's confirmed
        # default font. Same table as the daily bot, mirrored exactly here.
        # See bluesky_bot.py for the full derivation (measured directly
        # from InterVariable.ttf via fonttools, normalized to a 1000-unit
        # scale). Digits are NOT equal width in a proportional font --
        # '1' is noticeably narrower than most other digits -- so a fixed
        # space count drifts out of alignment depending on what values
        # happen to show up each month.
        CHAR_WIDTH = {
            '0': 631, '1': 407, '2': 610, '3': 618, '4': 646,
            '5': 593, '6': 620, '7': 566, '8': 619, '9': 620,
            '.': 288, '%': 982, ':': 288, ' ': 281,
        }
        CHAR_WIDTH.update({
            'i': 242, 'l': 242, 'r': 376,
            's': 528, 'a': 562, 'y': 562, 'c': 571,
            'e': 583, 'n': 591, 'u': 591, 'o': 600, 'd': 612,
            'S': 642, 'C': 731, 'H': 743, 'G': 746, 'N': 753, 'W': 985,
        })
        DEFAULT_WIDTH = 600  # fallback for any other character

        def visual_width(s):
            return sum(CHAR_WIDTH.get(ch, DEFAULT_WIDTH) for ch in s)

        col1_gap_width = 4 * 600  # roughly 4 average-width characters of breathing room
        wind_col1 = f"Wind: {wind_pct}"
        hydro_col1 = f"Hydro: {hydro_pct}"
        solar_col1 = f"Solar: {solar_pct}"
        col1_target_width = max(visual_width(wind_col1), visual_width(hydro_col1), visual_width(solar_col1)) + col1_gap_width

        def pad_to_width(s, target_width):
            while visual_width(s) < target_width:
                s += ' '
            return s

        post_text = f"""EU Electricity Generation - {date_str}

{ren_pct} of EU electricity generation was renewable.

{pad_to_width(wind_col1, col1_target_width)}Nuclear: {nuclear_pct}
{pad_to_width(hydro_col1, col1_target_width)}Gas: {gas_pct}
{pad_to_width(solar_col1, col1_target_width)}Coal: {coal_pct}

eu-electricity.eu
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
eu-electricity.eu

#EU #Renewables #Electricity #EnergySky #ClimateSky"""

    # Build facets (clickable link + hashtags) -- identical to daily bot
    facets = []

    link_text  = "eu-electricity.eu"
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

    # Determine which month's map to fetch -- prefer stats' own month_name
    # (already in the correct 'January'/'August'-style format matching
    # drive_links.json's Maps > Monthly keys), falling back to the same
    # last-completed-month logic used elsewhere in this script if stats
    # aren't available.
    stats_for_map = get_stats_from_json()
    if stats_for_map:
        map_month_name = stats_for_map['month_name']
    else:
        current_date = datetime.now()
        last_month = current_date.month - 1 or 12
        map_month_name = calendar.month_name[last_month]

    # Download renewables map (optional -- post without it if unavailable)
    map_path = get_monthly_map_from_drive(map_month_name)

    post_text, facets = create_post_text_and_facets()
    print(f"\n📝 Post text:\n{post_text}\n")
    print(f"✓ Created {len(facets)} facets")

    try:
        print("🔐 Logging in to Bluesky...")
        client = Client(base_url='https://eurosky.social')
        client.login(handle, password)
        print(f"✓ Logged in as {handle}")

        def upload_with_retry(data, max_retries=3):
            for attempt in range(max_retries):
                try:
                    return client.upload_blob(data)
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  Upload attempt {attempt+1} failed, retrying...")
                        import time
                        time.sleep(5)
                    else:
                        raise

        print("📤 Uploading plot image...")
        with open(plot_path, 'rb') as f:
            img_data = f.read()
        upload_response = upload_with_retry(img_data)

        images = [{
            'alt': f'EU electricity generation chart showing monthly breakdown for {datetime.now().year}',
            'image': upload_response.blob
        }]

        # Upload map image if available
        if map_path:
            print("📤 Uploading renewables map to Bluesky...")
            with open(map_path, 'rb') as f:
                map_data = f.read()
            map_upload = upload_with_retry(map_data)
            images.append({
                'alt': f'Map of EU renewable electricity generation by country for {map_month_name}',
                'image': map_upload.blob
            })
            print("✓ Map image ready")

        print("📮 Posting to Bluesky...")
        client.send_post(
            text=post_text,
            facets=facets,
            embed={
                '$type': 'app.bsky.embed.images',
                'images': images
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
