#!/usr/bin/env python3
"""
Fix JSON files with git conflict markers.
Handles both energy_summary_table.json and drive_links.json.
Keeps the second version (after =======) as it's typically newer.
"""
import json
import re
import sys
import os


def remove_conflict_markers(content):
    """
    Remove git conflict markers from content.
    Keeps the second block (after =======) as it's typically the newer version.
    Handles any branch name variant.
    """
    # General pattern: <<<<<<< anything ... ======= ... >>>>>>> anything
    pattern = r'<<<<<<< [^\n]*\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]*'
    
    def keep_second(match):
        return match.group(2)
    
    fixed = re.sub(pattern, keep_second, content, flags=re.DOTALL)
    
    # Also remove any remaining stray markers just in case
    fixed = re.sub(r'^<<<<<<< .*$', '', fixed, flags=re.MULTILINE)
    fixed = re.sub(r'^=======$', '', fixed, flags=re.MULTILINE)
    fixed = re.sub(r'^>>>>>>> .*$', '', fixed, flags=re.MULTILINE)
    
    return fixed


def fix_json_file(filepath):
    if not os.path.exists(filepath):
        print(f"  ⚠ File not found: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if '<<<<<<< ' not in content:
        print(f"  ✓ No conflicts in {filepath}")
        return True

    print(f"  ⚠ Found conflict markers in {filepath}, fixing...")
    fixed = remove_conflict_markers(content)

    try:
        json.loads(fixed)
        print(f"  ✓ Fixed JSON is valid")
    except json.JSONDecodeError as e:
        print(f"  ✗ Fixed JSON still invalid: {e}")
        return False

    with open(filepath, 'w') as f:
        f.write(fixed)

    print(f"  ✓ Saved fixed {filepath}")
    return True


if __name__ == '__main__':
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        'plots/energy_summary_table.json',
        'plots/drive_links.json'
    ]

    all_ok = True
    for filepath in files:
        ok = fix_json_file(filepath)
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)
