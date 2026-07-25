#!/usr/bin/env python3
"""
Merge duplicate country spreadsheets found by detect_duplicate_sheets.py.

MERGE POLICY:
- The oldest-created candidate ("primary") is the write target -- it holds
  the deepest history (2015-2026 in every case seen in this project) and
  its non-overlapping years are kept as-is.
- For a year that exists in MORE THAN ONE candidate (in every case seen so
  far, this means the current year, e.g. 2026, since that's the only year
  a fresh duplicate would have): the value comes from whichever candidate
  has the LATEST Drive 'modified' timestamp overall for that country, not
  just "primary" or "the most recent duplicate" by assumption. This matches
  the actual mechanism causing these duplicates: whichever sheet
  drive_links.json pointed at after the split is the one that kept
  receiving live writes, while the other one is a stale snapshot frozen
  as of the split. That "freshest" sheet is not always the same one --
  it's determined per country from the real modification timestamps.
- This resolution is NOT silent: every year that came from a non-primary
  source, and every conflict that got auto-resolved, is printed so you can
  see exactly which candidate's value was used and why.
- "Delete" means moving the non-primary spreadsheet(s) to Google Drive's
  trash (recoverable there for a retention period), never a permanent
  delete. This only happens after a successful merge, under --apply.
- Reads duplicate_sheets_report.json (from detect_duplicate_sheets.py)
  instead of re-scanning Drive, to avoid repeating that rate-limited scan.

Usage:
    export GOOGLE_CREDENTIALS_JSON='...'

    # Dry run for everything in the report (default, safe, no changes):
    python scripts/merge_duplicate_sheets.py

    # Dry run for just one country, with full year-by-year detail:
    python scripts/merge_duplicate_sheets.py --country BG --show-detail

    # Actually apply the merge for one country (required: both flags):
    python scripts/merge_duplicate_sheets.py --country BG --apply
"""

import os
import sys
import json
import time
import argparse

WORKSHEET_SUFFIX = ' Monthly Production'
MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
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
        return {}

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

    return table


def _read_all_worksheets(gc, spreadsheet_id):
    """Returns {worksheet_title: {year: {month: value}}} for one spreadsheet."""
    spreadsheet = _with_retry(gc.open_by_key, spreadsheet_id)
    worksheets = _with_retry(spreadsheet.worksheets)
    result = {}
    for ws in worksheets:
        if ws.title.endswith(WORKSHEET_SUFFIX):
            result[ws.title] = _read_worksheet_table(ws)
    return result, {ws.title: ws for ws in worksheets}


def build_country_merge_plan(gc, country_code, candidates):
    """
    Read every candidate's data and build one merge plan for the whole
    country: for each worksheet, for each year found in any candidate,
    decide which candidate's value to use.

    Returns (plan, primary, freshest, worksheet_objects) where:
    - plan: {worksheet_title: {year: {'value': {...}, 'source_id': ..., 'is_conflict': bool}}}
    - primary: the oldest-created candidate (write target)
    - freshest: the candidate with the latest 'modified' timestamp
    - worksheet_objects: {worksheet_title: gspread Worksheet} for the primary spreadsheet
    """
    sorted_by_created = sorted(candidates, key=lambda c: c['created'])
    primary = sorted_by_created[0]
    freshest = max(candidates, key=lambda c: c['modified'])

    print(f"  Primary (write target, oldest): {primary['id']} created {primary['created']}")
    print(f"  Freshest (wins ties): {freshest['id']} modified {freshest['modified']}"
          + (" -- same as primary" if freshest['id'] == primary['id'] else ""))

    all_candidate_data = {}
    primary_worksheet_objects = {}
    for c in candidates:
        print(f"  Reading {c['id']} ({'primary' if c['id'] == primary['id'] else 'duplicate'})...")
        tables, ws_objects = _read_all_worksheets(gc, c['id'])
        all_candidate_data[c['id']] = tables
        if c['id'] == primary['id']:
            primary_worksheet_objects = ws_objects

    worksheet_titles = set()
    for tables in all_candidate_data.values():
        worksheet_titles.update(tables.keys())

    plan = {}
    for ws_title in sorted(worksheet_titles):
        year_sources = {}  # year -> list of candidate ids that have this year
        for c in candidates:
            table = all_candidate_data[c['id']].get(ws_title, {})
            for year in table.keys():
                year_sources.setdefault(year, []).append(c['id'])

        resolved_years = {}
        for year, source_ids in year_sources.items():
            if len(source_ids) == 1:
                winner_id = source_ids[0]
                is_conflict = False
            else:
                # Multiple candidates have this year -- freshest wins if it's
                # one of them, otherwise fall back to primary if it's one of
                # them, otherwise just pick the first (shouldn't normally happen).
                if freshest['id'] in source_ids:
                    winner_id = freshest['id']
                elif primary['id'] in source_ids:
                    winner_id = primary['id']
                else:
                    winner_id = source_ids[0]
                is_conflict = True

            resolved_years[year] = {
                'value': all_candidate_data[winner_id][ws_title][year],
                'source_id': winner_id,
                'is_conflict': is_conflict,
                'all_sources': source_ids,
            }

        plan[ws_title] = resolved_years

    return plan, primary, freshest, primary_worksheet_objects, all_candidate_data


def print_plan(country_code, plan, primary, freshest):
    for ws_title, resolved_years in plan.items():
        conflicts = {y: v for y, v in resolved_years.items() if v['is_conflict']}
        additions = {y: v for y, v in resolved_years.items() if not v['is_conflict']
                     and v['source_id'] != primary['id']}

        if not conflicts and not additions:
            continue

        print(f"    {ws_title}:")
        for year, info in sorted(additions.items()):
            print(f"      + {year}: adding from {info['source_id']} (not in primary)")
        for year, info in sorted(conflicts.items()):
            winner_label = 'freshest' if info['source_id'] == freshest['id'] else 'primary (fallback)'
            print(f"      ⚠ {year}: CONFLICT across {info['all_sources']} -- "
                  f"using {info['source_id']} ({winner_label})")


def apply_merge(primary_worksheet_objects, plan, primary):
    """Write the resolved plan into the primary spreadsheet's worksheets."""
    for ws_title, resolved_years in plan.items():
        # Only write if there's at least one year that's new or resolved from
        # a non-primary source -- otherwise the worksheet is untouched.
        needs_write = any(info['source_id'] != primary['id'] for info in resolved_years.values())
        if not needs_write:
            continue

        if ws_title not in primary_worksheet_objects:
            print(f"    ⚠ '{ws_title}' has data to merge but doesn't exist in primary -- skipping, needs manual review")
            continue

        worksheet = primary_worksheet_objects[ws_title]
        values = _with_retry(worksheet.get_all_values)
        headers = values[0] if values else ['Month']
        month_rows = {row[0]: row for row in values[1:] if row and row[0] != 'Total'} if len(values) > 1 else {
            m: [m] + ['0.00'] * (len(headers) - 1) for m in MONTH_ORDER
        }

        for year, info in resolved_years.items():
            if info['source_id'] == primary['id']:
                continue  # primary's own value, already correct in the sheet
            if year not in headers:
                headers.append(year)
                for month_row in month_rows.values():
                    month_row.append('0.00')
            col_idx = headers.index(year)
            for month_name, month_row in month_rows.items():
                value = info['value'].get(month_name, '0.00')
                while len(month_row) <= col_idx:
                    month_row.append('0.00')
                month_row[col_idx] = value

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

        final_rows = [headers] + [month_rows[m] for m in MONTH_ORDER if m in month_rows] + [total_row]

        _with_retry(worksheet.clear)
        _with_retry(worksheet.update, final_rows)
        print(f"    ✓ Wrote merged {ws_title}")


def main():
    parser = argparse.ArgumentParser(description='Merge duplicate country spreadsheets')
    parser.add_argument('--country', help='Country code to process (e.g. BG). Omit to dry-run all.')
    parser.add_argument('--apply', action='store_true',
                         help='Actually write changes and trash duplicates. Requires --country. '
                              'Without this flag, always dry-run (no changes).')
    parser.add_argument('--report', default='duplicate_sheets_report.json',
                         help='Path to the report from detect_duplicate_sheets.py')
    parser.add_argument('--show-detail', action='store_true',
                         help='Print month-by-month values for conflicting years. Read-only.')
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

        print(f"\n{'=' * 80}")
        print(f"{country_code}: {len(candidates)} spreadsheets")

        plan, primary, freshest, primary_ws_objects, all_candidate_data = build_country_merge_plan(
            gc, country_code, candidates
        )

        print_plan(country_code, plan, primary, freshest)

        if args.show_detail:
            for ws_title, resolved_years in plan.items():
                conflicts = {y: v for y, v in resolved_years.items() if v['is_conflict']}
                for year, info in conflicts.items():
                    print(f"\n    {ws_title} -- Year {year} month-by-month:")
                    for month in MONTH_ORDER:
                        row = []
                        for source_id in info['all_sources']:
                            val = all_candidate_data[source_id].get(ws_title, {}).get(year, {}).get(month, '-')
                            tag = ' (used)' if source_id == info['source_id'] else ''
                            row.append(f"{source_id[:8]}={val}{tag}")
                        print(f"      {month}: " + "  ".join(row))

        if args.apply:
            apply_merge(primary_ws_objects, plan, primary)

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

            # Trash (not permanently delete) every non-primary candidate
            for c in candidates:
                if c['id'] == primary['id']:
                    continue
                _with_retry(drive_service.files().update(fileId=c['id'], body={'trashed': True}).execute)
                print(f"  ✓ Moved {c['id']} to Drive trash (recoverable, not permanently deleted)")
        else:
            print(f"\n  (dry run -- nothing written, nothing trashed. Re-run with "
                  f"--country {country_code} --apply to actually merge and clean up)")

    print("\n" + "=" * 80)
    print("Done." if args.apply else "Dry run complete. Nothing was changed.")
    print("=" * 80)


if __name__ == '__main__':
    main()
