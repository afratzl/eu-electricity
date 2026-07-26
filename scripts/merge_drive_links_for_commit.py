#!/usr/bin/env python3
"""
Merge this job's locally-modified plots/drive_links.json with whatever is
currently on origin/main, before committing.

WHY THIS EXISTS:
Workflows in this repo used to either (a) back up this job's own copy of
drive_links.json, hard-reset to origin/main, then blindly paste the local
copy back on top (update_intraday.yml's old approach), or (b) do a plain
`git pull --rebase --autostash` (update_historical.yml's approach). Both
can silently lose data: (a) never actually merges, so whichever job's
commit step runs last wins with its OWN stale full-file copy, discarding
any newer sections another concurrent job already pushed. (b) risks git
producing textual conflict markers inside the JSON file itself if both
sides touched it (this is exactly what corrupted plots/last_update.html
and, separately, wiped every country's Monthly/Trends sections down to
almost nothing).

drive_links.json is written by MULTIPLE workflows, each legitimately
responsible for different sections:
    - update_intraday.yml computes per-country 'Intraday' and 'Yesterday',
      plus 'Maps' sub-keys 'Yesterday' and 'LastWeek'.
    - update_historical.yml (via eu_energy_plotting.py) computes per-country
      'Monthly' and 'Trends'.
    - generate_maps_monthly.yml computes 'Maps' sub-keys 'Monthly' and
      'Annual'.
No single job should ever blindly overwrite the whole file -- each should
only write the sections it actually owns, and defer to whatever's on
origin/main for everything else, since another job may have updated those
sections more recently than this job's own checkout.

Usage (from repo root, after `git fetch origin main` but BEFORE any
`git reset`):
    python scripts/merge_drive_links_for_commit.py \
        --owned-country-keys Intraday,Yesterday \
        --owned-maps-keys Yesterday,LastWeek

    # update_historical.yml's equivalent call:
    python scripts/merge_drive_links_for_commit.py \
        --owned-country-keys Monthly,Trends

    # generate_maps_monthly.yml's equivalent call:
    python scripts/merge_drive_links_for_commit.py \
        --owned-maps-keys Monthly,Annual
"""

import json
import subprocess
import sys
import argparse

DRIVE_LINKS_PATH = 'plots/drive_links.json'


def get_remote_version():
    """
    Fetch plots/drive_links.json as it currently exists on origin/main,
    without touching the working directory (git show reads from the git
    object database, not the checkout).
    """
    try:
        result = subprocess.run(
            ['git', 'show', f'origin/main:{DRIVE_LINKS_PATH}'],
            capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError:
        # File doesn't exist on origin/main yet (e.g. first-ever run) --
        # nothing to merge against.
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️  WARNING: origin/main's {DRIVE_LINKS_PATH} is not valid JSON ({e}). "
              f"Treating as empty rather than merging against corrupted data.")
        return {}


def merge(local, remote, owned_country_keys, owned_maps_keys):
    """
    local: this job's own freshly-computed drive_links.json content
    remote: whatever is currently on origin/main
    owned_country_keys: set of per-country keys (e.g. {'Intraday', 'Yesterday'})
        that THIS job legitimately computes -- taken from local, everything
        else per country comes from remote.
    owned_maps_keys: set of keys inside the top-level 'Maps' dict that THIS
        job legitimately computes (e.g. {'Yesterday', 'LastWeek'}) -- same
        logic, one level deeper.
    Returns the merged dict.
    """
    merged = {}
    all_country_codes = (set(local.keys()) | set(remote.keys())) - {'Maps'}

    for country_code in all_country_codes:
        local_entry = local.get(country_code, {})
        remote_entry = remote.get(country_code, {})

        merged_entry = dict(remote_entry)  # start from remote (owns everything by default)
        for key in owned_country_keys:
            if key in local_entry:
                merged_entry[key] = local_entry[key]

        merged[country_code] = merged_entry

    # Top-level Maps: merge at the sub-key level, same ownership logic.
    local_maps = local.get('Maps', {})
    remote_maps = remote.get('Maps', {})
    merged_maps = dict(remote_maps)
    for key in owned_maps_keys:
        if key in local_maps:
            merged_maps[key] = local_maps[key]
    if merged_maps:
        merged['Maps'] = merged_maps

    return merged


def main():
    parser = argparse.ArgumentParser(description='Merge drive_links.json against origin/main before committing')
    parser.add_argument('--owned-country-keys', default='',
                         help='Comma-separated per-country keys this job computes (e.g. Intraday,Yesterday)')
    parser.add_argument('--owned-maps-keys', default='',
                         help='Comma-separated keys inside the top-level Maps dict this job computes (e.g. Yesterday,LastWeek)')
    args = parser.parse_args()

    owned_country_keys = set(k.strip() for k in args.owned_country_keys.split(',') if k.strip())
    owned_maps_keys = set(k.strip() for k in args.owned_maps_keys.split(',') if k.strip())

    if not owned_country_keys and not owned_maps_keys:
        print("⚠️  WARNING: no --owned-country-keys or --owned-maps-keys given -- this job "
              "won't write anything to drive_links.json (fully deferring to origin/main). "
              "If that's not intended, pass the keys this job actually computes.")

    try:
        with open(DRIVE_LINKS_PATH, 'r') as f:
            local = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  {DRIVE_LINKS_PATH} doesn't exist locally -- nothing to merge, exiting cleanly.")
        sys.exit(0)
    except json.JSONDecodeError as e:
        print(f"✗ ERROR: local {DRIVE_LINKS_PATH} is not valid JSON ({e}). "
              f"Refusing to merge against corrupted local data -- fix this manually first.")
        sys.exit(1)

    remote = get_remote_version()

    if not remote:
        print("No usable remote version found -- keeping local version as-is (nothing to merge).")
        sys.exit(0)

    merged = merge(local, remote, owned_country_keys, owned_maps_keys)

    # Report what changed, for visibility in the workflow log
    for country_code in sorted(merged.keys()):
        if country_code == 'Maps':
            continue
        remote_keys = set(remote.get(country_code, {}).keys())
        merged_keys = set(merged[country_code].keys())
        kept_from_remote = (merged_keys & remote_keys) - owned_country_keys
        if kept_from_remote:
            print(f"  {country_code}: kept {sorted(kept_from_remote)} from origin/main (not this job's to own)")

    if 'Maps' in merged:
        remote_maps_keys = set(remote.get('Maps', {}).keys())
        kept_maps_from_remote = (set(merged['Maps'].keys()) & remote_maps_keys) - owned_maps_keys
        if kept_maps_from_remote:
            print(f"  Maps: kept {sorted(kept_maps_from_remote)} from origin/main (not this job's to own)")

    with open(DRIVE_LINKS_PATH, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"✓ Merged {DRIVE_LINKS_PATH}: {len(merged)} top-level keys "
          f"(this job's {sorted(owned_country_keys)} + Maps.{sorted(owned_maps_keys)} "
          f"+ origin/main's everything else)")


if __name__ == '__main__':
    main()
