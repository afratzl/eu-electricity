#!/usr/bin/env python3
"""
Merge duplicate country spreadsheets found by detect_duplicate_sheets.py.

SAFETY MODEL:
- Defaults to DRY RUN: prints exactly what would happen, changes nothing.
- Real changes require --apply AND --country (one country at a time --
  deliberately no "apply to all 12" option, so each merge gets reviewed).
- Merge policy is conservative, not "newest/duplicate wins":
    - A year present in a duplicate sheet but MISSING from the primary sheet
      gets copied into the primary. This is the common case (duplicate has
      2026 that the primary might be missing if it hasn't been updated yet).
    - A year present in BOTH sheets is NEVER auto-overwritten, even if the
      values differ. It's reported as a conflict for you to look at and
      resolve by hand. Given this session already found two separate bugs
      that silently wrote wrong data into live sheets (missing NED_API_KEY
      writing zeros, then a pagination bug badly undercounting real
      numbers), this script isn't willing to guess which of two differing
      values is correct.
- "Delete" means moving the non-primary spreadsheet(s) to Google Drive's
  trash (recoverable there for a retention period), never a permanent
  delete. This only happens after a successful merge, under --apply.
- Reads duplicate_sheets_report.json (from detect_duplicate_sheets.py)
  instead of re-scanning Drive, to avoid repeating that rate-limited scan.

Usage:
    export GOOGLE_CREDENTIALS_JSON='...'

    # Dry run for everything in the report (default, safe, no changes):
    python scripts/merge_duplicate_sheets.py

    # Dry run for just one country:
    python scripts/merge_duplicate_sheets.py --country BG

    # Actually apply the merge for one country (required: both flags):
    python scripts/merge_duplicate_sheets.py --country BG --apply
"""

import os
import sys
import json
import time
import argparse

WORKSHEET_SUFFIX = ' Monthly Production'
REQUEST_SLEEP_SECONDS = 3  # matches existing codebase's Sheets API pacing convention
MAX_RETRIES = 4


def _with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying with backoff on 429s."""
    for attempt in range(MAX_RETRIES):
        try:
            result = fn(*args, **kwargs)
            time.sleep(REQUEST_SLEEP_SECONDS)
            return result
        except Exception as e:
            if '429' in str(e) or 'Quota exceeded' in str(e):
                wait = 30 * (attempt + 1)
                print(f"      ⏳ Rate limited, waiting {wait}s before retry ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Failed after max retries (persistent rate limiting)")


def _read_worksheet_table(worksheet):
    """
    Read a worksheet into {year_str: {month_name: value_str}}.
    Assumes the same layout save_all_data_to_google_sheets_with_merge writes:
    header row ['Month', year1, year2, ...], one row per month, a 'Total' row.
    """
    values = _with_retry(worksheet.get_all_values)
    if len(values) < 2:
        return {}, []

    headers = values[0]
    year_cols = [h for h in headers[1:] if h.isdigit()]
    table = {year: {} for year in year_cols}

    for row in values[1:]:
        if not row or row[0] == 'Total':
            continue
        month_name = row[0]
        for i, year in enumerate(headers[1:], start=1):
            if year in year_cols and i < len(row):
                table[year][month_name] = row[i]

    return table, headers


def plan_merge_for_worksheet(primary_ws, duplicate_ws, worksheet_title):
    """
    Compare one worksheet between primary and duplicate.
    Returns a dict describing the plan: years to add, years in conflict,
    and the fully merged table if applied. Read-only -- makes no changes.
    """
    primary_table, primary_headers = _read_worksheet_table(primary_ws)
    dup_table, dup_headers = _read_worksheet_table(duplicate_ws)

    years_to_add = []
    years_in_conflict = []
    years_identical = []

    for year, month_values in dup_table.items():
        if year not in primary_table:
            years_to_add.append(year)
        else:
            # Year exists in both -- compare values
            differs = False
            for month, value in month_values.items():
                primary_value = primary_table.get(year, {}).get(month, '0.00')
                try:
                    if abs(float(value or 0) - float(primary_value or 0)) > 0.01:
                        differs = True
                        break
                except ValueError:
                    if value != primary_value:
                        differs = True
                        break
            if differs:
                years_in_conflict.append(year)
            else:
                years_identical.append(year)

    return {
        'worksheet_title': worksheet_title,
        'years_to_add': sorted(years_to_add),
        'years_in_conflict': sorted(years_in_conflict),
        'years_identical': sorted(years_identical),
        'primary_table': primary_table,
        'dup_table': dup_table,
    }


def apply_merge_for_worksheet(primary_ws, plan):
    """
    Actually write years_to_add from the duplicate into the primary
    worksheet. Never touches years_in_conflict or years_identical -- those
    are left exactly as they are in the primary.
    """
    if not plan['years_to_add']:
        return False

    values = _with_retry(primary_ws.get_all_values)
    headers = values[0]
    month_rows = {row[0]: row for row in values[1:] if row and row[0] != 'Total'}

    for year in plan['years_to_add']:
        headers.append(year)
        for month_name, month_row in month_rows.items():
            value = plan['dup_table'].get(year, {}).get(month_name, '0.00')
            month_row.append(value)

    # Recompute total row
    total_row = ['Total']
    for year in headers[1:]:
        col_idx = headers.index(year)
        total = 0.0
        for month_row in month_rows.values():
            try:
                total += float(month_row[col_idx] or 0)
            except (ValueError, IndexError):
                pass
        total_row.append(f"{total:.2f}")

    final_rows = [headers] + list(month_rows.values()) + [total_row]

    _with_retry(primary_ws.clear)
    _with_retry(primary_ws.update, final_rows)
    return True


def main():
    parser = argparse.ArgumentParser(description='Merge duplicate country spreadsheets')
    parser.add_argument('--country', help='Country code to process (e.g. BG). Omit to dry-run all.')
    parser.add_argument('--apply', action='store_true',
                         help='Actually write changes and trash duplicates. Requires --country. '
                              'Without this flag, always dry-run (no changes).')
    parser.add_argument('--report', default='duplicate_sheets_report.json',
                         help='Path to the report from detect_duplicate_sheets.py')
    args = parser.parse_args()

    if args.apply and not args.country:
        print("⚠️  ERROR: --apply requires --country (one country at a time, by design).")
        sys.exit(1)

    if not os.path.exists(args.report):
        print(f"⚠️  ERROR: {args.report} not found. Run detect_duplicate_sheets.py first.")
        sys.exit(1)

    with open(args.report, 'r') as f:
        duplicates_report = json.load(f)

    countries_to_process = [args.country] if args.country else list(duplicates_report.keys())

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

    drive_service = build('drive', 'v3', credentials=Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/drive']
    ))

    mode_label = "APPLYING CHANGES" if args.apply else "DRY RUN (no changes will be made)"
    print("=" * 80)
    print(f"MERGE DUPLICATE SHEETS -- {mode_label}")
    print("=" * 80)

    for country_code in countries_to_process:
        if country_code not in duplicates_report:
            print(f"⚠ {country_code} not found in {args.report}, skipping.")
            continue

        candidates = duplicates_report[country_code]
        if len(candidates) < 2:
            continue

        # Primary = earliest created. Confirmed as the right rule for every
        # one of the 12 countries checked in this project (oldest always has
        # full history, newer ones are always 2026-only accidental duplicates).
        sorted_candidates = sorted(candidates, key=lambda c: c['created'])
        primary = sorted_candidates[0]
        duplicates = sorted_candidates[1:]

        print(f"\n{'=' * 80}")
        print(f"{country_code}: primary={primary['id']} (created {primary['created']})")
        print(f"  {len(duplicates)} duplicate(s) to merge from")

        primary_spreadsheet = _with_retry(gc.open_by_key, primary['id'])
        primary_worksheets = {ws.title: ws for ws in _with_retry(primary_spreadsheet.worksheets)}

        any_conflicts = False

        for dup in duplicates:
            print(f"\n  --- Duplicate: {dup['id']} (created {dup['created']}) ---")
            dup_spreadsheet = _with_retry(gc.open_by_key, dup['id'])
            dup_worksheets = {ws.title: ws for ws in _with_retry(dup_spreadsheet.worksheets)}

            for worksheet_title, dup_ws in dup_worksheets.items():
                if not worksheet_title.endswith(WORKSHEET_SUFFIX):
                    continue
                if worksheet_title not in primary_worksheets:
                    print(f"    ⚠ '{worksheet_title}' exists in duplicate but not in primary -- "
                          f"skipping (needs manual review, not handled automatically)")
                    continue

                plan = plan_merge_for_worksheet(primary_worksheets[worksheet_title], dup_ws, worksheet_title)

                if plan['years_to_add']:
                    print(f"    {worksheet_title}: would add years {plan['years_to_add']}")
                if plan['years_in_conflict']:
                    any_conflicts = True
                    print(f"    ⚠ {worksheet_title}: CONFLICT on years {plan['years_in_conflict']} "
                          f"(differing values in both sheets -- NOT auto-resolved, review manually)")
                if not plan['years_to_add'] and not plan['years_in_conflict']:
                    print(f"    {worksheet_title}: nothing to add, no conflicts")

                if args.apply:
                    changed = apply_merge_for_worksheet(primary_worksheets[worksheet_title], plan)
                    if changed:
                        print(f"    ✓ Applied: added {plan['years_to_add']} to primary's {worksheet_title}")

        if args.apply:
            if any_conflicts:
                print(f"\n  ⚠ {country_code}: conflicts were found and left untouched. "
                      f"NOT trashing duplicates automatically -- resolve conflicts first, "
                      f"then re-run to trash once you're satisfied everything's merged.")
                continue

            # Update drive_links.json to point at the primary
            drive_links_file = 'plots/drive_links.json'
            links = {}
            if os.path.exists(drive_links_file):
                with open(drive_links_file, 'r') as f:
                    links = json.load(f)
            if country_code not in links:
                links[country_code] = {}
            links[country_code]['data_sheet_id'] = primary['id']
            os.makedirs('plots', exist_ok=True)
            with open(drive_links_file, 'w') as f:
                json.dump(links, f, indent=2)
            print(f"  ✓ Updated drive_links.json to point {country_code} at primary")

            # Trash (not permanently delete) each duplicate
            for dup in duplicates:
                _with_retry(drive_service.files().update(fileId=dup['id'], body={'trashed': True}).execute)
                print(f"  ✓ Moved duplicate {dup['id']} to Drive trash (recoverable, not permanently deleted)")
        else:
            print(f"\n  (dry run -- nothing written, nothing trashed. Re-run with "
                  f"--country {country_code} --apply to actually merge and clean up)")

    print("\n" + "=" * 80)
    print("Done." if args.apply else "Dry run complete. Nothing was changed.")
    print("=" * 80)


if __name__ == '__main__':
    main()
