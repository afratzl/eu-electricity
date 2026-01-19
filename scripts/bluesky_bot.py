#!/usr/bin/env python3
"""
Bluesky Bot for EU Electricity Generation
Downloads plot from Google Drive and posts to Bluesky
With clickable links and hashtags, and exact spacing as specified
"""

import os
import sys
from datetime import datetime, timedelta
from atproto import Client, models
import json
import requests

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive"""
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Save to file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    
    return destination

def get_plot_from_drive():
    """Get yesterday's plot file ID from drive_links.json and download it"""
    json_path = 'plots/drive_links.json'
    
    if not os.path.exists(json_path):
        print(f"❌ drive_links.json not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            links = json.load(f)
        
        # Navigate to EU -> Yesterday -> all_sources -> percentage
        if 'EU' not in links:
            print("❌ EU not found in drive_links.json")
            return None
        
        if 'Yesterday' not in links['EU']:
            print("❌ Yesterday not found in EU section")
            return None
        
        if 'all_sources' not in links['EU']['Yesterday']:
            print("❌ all_sources not found in Yesterday section")
            return None
        
        file_id = links['EU']['Yesterday']['all_sources']['percentage']['file_id']
        
        if not file_id:
            print("❌ No file_id found for yesterday percentage plot")
            return None
        
        print(f"✓ Found plot file_id: {file_id}")
        
        # Download the file
        plot_path = 'plots/EU_yesterday_all_sources_percentage.png'
        os.makedirs('plots', exist_ok=True)
        
        print(f"📥 Downloading plot from Google Drive...")
        download_file_from_google_drive(file_id, plot_path)
        print(f"✓ Downloaded to: {plot_path}")
        
        return plot_path
        
    except Exception as e:
        print(f"❌ Error getting plot from Drive: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_stats_from_json():
    """
    Get yesterday's source percentages from energy_summary_table.json
    This should already be in the repo from the intraday run
    """
    json_path = 'plots/energy_summary_table.json'
    
    if not os.path.exists(json_path):
        print(f"⚠️  JSON not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Navigate to EU -> sources list
        if 'EU' not in data or 'sources' not in data['EU']:
            print("⚠️  EU data not found in JSON")
            return None
        
        sources = data['EU']['sources']
        
        # Extract the 6 main sources AND the aggregates
        stats = {}
        source_map = {
            'Wind': 'wind',
            'Hydro': 'hydro',
            'Solar': 'solar',
            'Nuclear': 'nuclear',
            'Gas': 'gas',
            'Coal': 'coal',
            'All Renewables': 'renewables',
            'All Non-Renewables': 'non_renewables'
        }
        
        for source in sources:
            source_name = source.get('source')
            if source_name in source_map:
                pct = source.get('yesterday', {}).get('percentage')
                if pct is not None:
                    stats[source_map[source_name]] = pct
        
        return stats if len(stats) == 8 else None
    
    except Exception as e:
        print(f"⚠️  Error reading JSON: {e}")
        return None

def format_percentage(value):
    """
    Format percentage with right-alignment using spaces
    Always 2 decimals, right-padded to width 6 (e.g., " 4.16%", "18.30%")
    """
    formatted = f"{value:>5.2f}%"  # Right-align in 5 chars, then add %
    return formatted

def create_post_text_and_facets():
    """Create the post text with yesterday's stats and facets for clickable links/hashtags"""
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%B %d, %Y')
    
    # Get stats from JSON
    stats = get_stats_from_json()
    
    if stats and len(stats) == 8:
        # Format percentages
        wind_pct = format_percentage(stats['wind'])
        hydro_pct = format_percentage(stats['hydro'])
        solar_pct = format_percentage(stats['solar'])
        nuclear_pct = format_percentage(stats['nuclear'])
        gas_pct = format_percentage(stats['gas'])
        coal_pct = format_percentage(stats['coal'])
        ren_pct = format_percentage(stats['renewables'])
        non_ren_pct = format_percentage(stats['non_renewables'])
        
        # EXACT spacing as specified:
        post_text = f"""EU Electricity Generation - {date_str}

{ren_pct} of EU electricity generation was renewable.

Wind:   {wind_pct}        Nuclear:  {nuclear_pct}
Hydro:  {hydro_pct}       Gas:         {gas_pct}
Solar:    {solar_pct}       Coal:         {coal_pct}

afratzl.github.io/eu-electricity
#Energy #EU #Renewables #Electricity"""
    else:
        # Fallback if JSON data not available
        post_text = f"""EU Electricity Generation - {date_str}

Yesterday's electricity generation breakdown across all EU member states.

Data: ENTSO-E
afratzl.github.io/eu-electricity

#Energy #EU #Renewables #Electricity"""
    
    # Create facets for clickable link and hashtags
    facets = []
    
    # Find the website link position
    link_text = "afratzl.github.io/eu-electricity"
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
    
    # Find hashtag positions
    hashtags = ['#Energy', '#EU', '#Renewables', '#Electricity']
    for tag in hashtags:
        tag_start = post_text.find(tag)
        if tag_start != -1:
            facets.append(
                models.AppBskyRichtextFacet.Main(
                    features=[models.AppBskyRichtextFacet.Tag(tag=tag[1:])],  # Remove # from tag
                    index=models.AppBskyRichtextFacet.ByteSlice(
                        byteStart=len(post_text[:tag_start].encode('utf-8')),
                        byteEnd=len(post_text[:tag_start + len(tag)].encode('utf-8'))
                    )
                )
            )
    
    return post_text, facets

def post_to_bluesky():
    """Main function to post to Bluesky"""
    print("=" * 60)
    print("BLUESKY BOT - EU ELECTRICITY GENERATION")
    print("=" * 60)
    
    # Get credentials from environment
    handle = os.environ.get('BLUESKY_HANDLE')
    password = os.environ.get('BLUESKY_PASSWORD')
    
    if not handle or not password:
        print("❌ Error: BLUESKY_HANDLE and BLUESKY_PASSWORD must be set")
        sys.exit(1)
    
    # Download plot from Google Drive
    plot_path = get_plot_from_drive()
    if not plot_path:
        print("❌ Error: Could not get yesterday's plot from Google Drive")
        sys.exit(1)
    
    # Create post text and facets
    post_text, facets = create_post_text_and_facets()
    print(f"\n📝 Post text:\n{post_text}\n")
    print(f"✓ Created {len(facets)} facets (clickable links/hashtags)")
    
    try:
        # Login to Bluesky
        print("🔐 Logging in to Bluesky...")
        client = Client()
        client.login(handle, password)
        print(f"✓ Logged in as {handle}")
        
        # Upload image
        print("📤 Uploading image to Bluesky...")
        with open(plot_path, 'rb') as f:
            img_data = f.read()
        
        upload_response = client.upload_blob(img_data)
        
        # Create post with image and facets
        print("📮 Posting to Bluesky...")
        client.send_post(
            text=post_text,
            facets=facets,
            embed={
                '$type': 'app.bsky.embed.images',
                'images': [{
                    'alt': 'EU electricity generation chart showing Wind, Hydro, Solar, Nuclear, Gas, and Coal percentages for yesterday',
                    'image': upload_response.blob
                }]
            }
        )
        
        print("✓ Posted successfully!")
        print(f"   Profile: https://bsky.app/profile/{handle}")
        
    except Exception as e:
        print(f"❌ Error posting to Bluesky: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    post_to_bluesky()
