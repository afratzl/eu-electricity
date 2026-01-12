#!/usr/bin/env python3
"""
Bluesky Bot for EU Electricity Generation
Posts daily updates with yesterday's percentage plot
Shows 6 main sources in 2-column layout with percentages only
"""

import os
import sys
from datetime import datetime, timedelta
from atproto import Client
import json

def get_yesterday_plot_path():
    """Get the path to yesterday's EU percentage plot"""
    plot_path = 'plots/EU_yesterday_all_sources_percentage.png'
    
    if not os.path.exists(plot_path):
        print(f"❌ Plot not found: {plot_path}")
        return None
    
    return plot_path

def get_stats_from_json():
    """
    Get yesterday's source percentages from energy_summary_table.json
    Returns: dict with source percentages or None if not found
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
        
        # Extract the 6 main sources
        stats = {}
        source_map = {
            'Wind': 'wind',
            'Hydro': 'hydro',
            'Solar': 'solar',
            'Nuclear': 'nuclear',
            'Gas': 'gas',
            'Coal': 'coal'
        }
        
        for source in sources:
            source_name = source.get('source')
            if source_name in source_map:
                pct = source.get('yesterday', {}).get('percentage')
                if pct is not None:
                    stats[source_map[source_name]] = pct
        
        return stats if len(stats) == 6 else None
    
    except Exception as e:
        print(f"⚠️  Error reading JSON: {e}")
        return None

def format_percentage(value):
    """Format percentage: remove decimal if whole number"""
    if value % 1 == 0:
        return f"{int(value)}%"
    else:
        return f"{value:.1f}%"

def create_post_text():
    """Create the post text with yesterday's stats"""
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%B %d, %Y')  # e.g., "January 11, 2026"
    
    # Get stats from JSON
    stats = get_stats_from_json()
    
    if stats and len(stats) == 6:
        # Format the 6 sources in 2 columns using spacing
        wind_pct = format_percentage(stats['wind'])
        hydro_pct = format_percentage(stats['hydro'])
        solar_pct = format_percentage(stats['solar'])
        nuclear_pct = format_percentage(stats['nuclear'])
        gas_pct = format_percentage(stats['gas'])
        coal_pct = format_percentage(stats['coal'])
        
        post_text = f"""EU Electricity Generation - {date_str}

Wind: {wind_pct}       Nuclear: {nuclear_pct}
Hydro: {hydro_pct}      Gas: {gas_pct}
Solar: {solar_pct}       Coal: {coal_pct}

Data: ENTSO-E
afratzl.github.io/eu-electricity

#Electricity #EU #Energy #Renewables"""
    else:
        # Fallback if JSON data not available
        post_text = f"""EU Electricity Generation - {date_str}

Yesterday's electricity generation breakdown across all EU member states.

Data: ENTSO-E
afratzl.github.io/eu-electricity

#Electricity #EU #Energy #Renewables"""
    
    return post_text

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
        print("   Set them as environment variables or GitHub secrets")
        sys.exit(1)
    
    # Get plot path
    plot_path = get_yesterday_plot_path()
    if not plot_path:
        print("❌ Error: Yesterday's plot not found")
        sys.exit(1)
    
    print(f"✓ Found plot: {plot_path}")
    
    # Create post text
    post_text = create_post_text()
    print(f"\n📝 Post text:\n{post_text}\n")
    
    try:
        # Login to Bluesky
        print("🔐 Logging in to Bluesky...")
        client = Client()
        client.login(handle, password)
        print(f"✓ Logged in as {handle}")
        
        # Upload image
        print("📤 Uploading image...")
        with open(plot_path, 'rb') as f:
            img_data = f.read()
        
        upload_response = client.upload_blob(img_data)
        
        # Create post with image
        print("📮 Posting to Bluesky...")
        client.send_post(
            text=post_text,
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
