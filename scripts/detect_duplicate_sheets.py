#!/usr/bin/env python3
"""
Detect duplicate country spreadsheets in Google Drive.

READ-ONLY -- this script only lists and inspects files. It does not modify,
merge, or delete anything. Run it first to see the actual state of your
Drive before running merge_duplicate_sheets.py.

Root cause this is investigating: get_or_create_country_sheet() previously
only looked up a country's sheet via drive_links.json, with no fallback
search of Drive by name. If that JSON entry was ever missing or stale
(corrupted file, a concurrent workflow's write racing with another), the
function would silently create a new same-named spreadsheet instead of
finding the real one -- leaving two files named e.g. "NL Electricity
Production Data" in the same Drive folder, one with years of history, one
with only whatever was being processed at the time. This has since been
fixed at the source (get_or_create_country_sheet now searches Drive by name
before creating), but doesn't undo any duplicates that already exist.

Usage:
    export GOOGLE_CREDENTIALS_JSON='...'
    python scripts/detect_duplicate_sheets.py
"""

import os
import sys
import json
import time


def _open_spreadsheet_with_retry(gc, sheet_id, max_retries=4):
    """
    Open a spreadsheet and read its worksheet list/header row, retrying with
    backoff on 429 (rate limit) errors. Returns (worksheets, error) where
    error is None on success.
    """
    for attempt in range(max_retries):
        try:
            spreadsheet = gc.open_by_key(sheet_id)
            worksheets = spreadsheet.worksheets()
            time.sleep(2)  # pace subsequent reads, matches existing codebase convention
            return worksheets, None
        except Exception as e:
            if '429' in str(e) or 'Quota exceeded' in str(e):
                wait = 30 * (attempt + 1)
                print(f"      ⏳ Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            return None, str(e)
    return None, "Failed after max retries (persistent rate limiting)"


def main():
    google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    if not google_creds_json:
        print("⚠️  ERROR: GOOGLE_CREDENTIALS_JSON environment variable not set!")
        sys.exit(1)

    creds_dict = json.loads(google_creds_json)

    import gspread
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
    gc = gspread.authorize(credentials)

    drive_credentials = Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    drive_service = build('drive', 'v3', credentials=drive_credentials)

    print("=" * 80)
    print("DUPLICATE SHEET DETECTION (read-only)")
    print("=" * 80)

    # Find the root folder
    root_query = "name='EU-Electricity-Plots' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    root_results = drive_service.files().list(q=root_query, spaces='drive', fields='files(id, name)').execute()
    root_folders = root_results.get('files', [])

    if not root_folders:
        print("✗ Could not find 'EU-Electricity-Plots' root folder. Nothing to check.")
        sys.exit(1)

    if len(root_folders) > 1:
        print(f"⚠ Found {len(root_folders)} folders named 'EU-Electricity-Plots' at the root -- "
              f"unusual, but continuing with the first one found.")

    root_folder_id = root_folders[0]['id']
    print(f"✓ Found root folder: EU-Electricity-Plots ({root_folder_id})\n")

    # List country subfolders
    subfolder_query = (f"'{root_folder_id}' in parents and "
                        f"mimeType='application/vnd.google-apps.folder' and trashed=false")
    subfolder_results = drive_service.files().list(
        q=subfolder_query, spaces='drive', fields='files(id, name)', pageSize=1000
    ).execute()
    country_folders = subfolder_results.get('files', [])
    print(f"✓ Found {len(country_folders)} country folders\n")

    duplicates_found = {}

    for folder in sorted(country_folders, key=lambda f: f['name']):
        country_code = folder['name']
        folder_id = folder['id']

        sheet_query = (f"'{folder_id}' in parents and "
                       f"mimeType='application/vnd.google-apps.spreadsheet' and trashed=false")
        sheet_results = drive_service.files().list(
            q=sheet_query, spaces='drive',
            fields='files(id, name, createdTime, modifiedTime)'
        ).execute()
        sheets = sheet_results.get('files', [])

        if len(sheets) <= 1:
            continue

        # Found a duplicate -- gather detail on each candidate
        print(f"⚠ {country_code}: {len(sheets)} spreadsheets found (expected 1)")
        sheets.sort(key=lambda f: f.get('createdTime', ''))

        candidates = []
        for sheet_info in sheets:
            detail = {
                'id': sheet_info['id'],
                'name': sheet_info['name'],
                'created': sheet_info.get('createdTime', '?'),
                'modified': sheet_info.get('modifiedTime', '?'),
                'year_range': None,
                'worksheet_count': None,
            }

            worksheets, error = _open_spreadsheet_with_retry(gc, sheet_info['id'])
            if error:
                detail['error'] = error
            elif worksheets is not None:
                detail['worksheet_count'] = len(worksheets)

                # Use 'Total Generation Monthly Production' as a representative
                # worksheet to read the year range, since every country's data
                # collection always writes this one.
                total_gen_ws = next((ws for ws in worksheets if ws.title == 'Total Generation Monthly Production'), None)
                if total_gen_ws:
                    try:
                        header_row = total_gen_ws.row_values(1)
                        time.sleep(2)
                        years = [int(c) for c in header_row if c.isdigit()]
                        if years:
                            detail['year_range'] = f"{min(years)}-{max(years)}"
                    except Exception as e:
                        detail['error'] = f"Could not read header row: {e}"

            candidates.append(detail)

        for i, c in enumerate(candidates, 1):
            year_info = c['year_range'] or c.get('error', 'unknown')
            print(f"    [{i}] id={c['id']}")
            print(f"        created={c['created']}  modified={c['modified']}")
            print(f"        worksheets={c['worksheet_count']}  year_range={year_info}")

        print()
        duplicates_found[country_code] = candidates

        # Pace between countries too, on top of the per-sheet pacing above --
        # cheap insurance against the same 429s recurring across a long scan.
        time.sleep(3)

    print("=" * 80)
    if duplicates_found:
        print(f"SUMMARY: {len(duplicates_found)} countries with duplicate sheets: "
              f"{', '.join(duplicates_found.keys())}")
        print("\nNothing was changed. Review the above, then use merge_duplicate_sheets.py "
              "(dry-run by default) to merge and clean these up.")

        # Save findings so merge_duplicate_sheets.py doesn't need to re-scan
        with open('duplicate_sheets_report.json', 'w') as f:
            json.dump(duplicates_found, f, indent=2)
        print("\n✓ Saved detailed report to duplicate_sheets_report.json")
    else:
        print("SUMMARY: No duplicates found. Every country folder has exactly one spreadsheet.")
    print("=" * 80)


if __name__ == '__main__':
    main()
