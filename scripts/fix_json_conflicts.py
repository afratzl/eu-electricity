#!/usr/bin/env python3
"""
Fix broken summary_table.json with git conflict markers
Removes conflict markers and keeps the most recent timestamp
"""

import json
import re
import sys

def fix_json_conflicts(filepath='plots/energy_summary_table.json'):
    """
    Fix JSON file with git conflict markers
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if file has conflict markers
        if '<<<<<<< ' not in content:
            print("✓ No conflict markers found. JSON is clean.")
            return True
        
        print("⚠ Found conflict markers. Attempting to fix...")
        
        # Remove conflict markers and keep the newer version (after =======)
        # Pattern: <<<<<<< ... ======= KEEP_THIS >>>>>>> ...
        pattern = r'<<<<<<< Updated upstream\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> Stashed changes'
        
        def keep_newer(match):
            # Usually the "Stashed changes" (second part) is newer
            return match.group(2)
        
        fixed_content = re.sub(pattern, keep_newer, content, flags=re.DOTALL)
        
        # Validate the fixed JSON
        try:
            json.loads(fixed_content)
            print("✓ Fixed JSON is valid!")
        except json.JSONDecodeError as e:
            print(f"✗ Fixed JSON is still invalid: {e}")
            return False
        
        # Write the fixed content
        with open(filepath, 'w') as f:
            f.write(fixed_content)
        
        print(f"✓ Successfully fixed {filepath}")
        return True
        
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return False
    except Exception as e:
        print(f"✗ Error fixing JSON: {e}")
        return False

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'plots/energy_summary_table.json'
    success = fix_json_conflicts(filepath)
    sys.exit(0 if success else 1)
