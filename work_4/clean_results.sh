#!/usr/bin/env bash
# Clean default work_4 output locations independently of the working directory.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 - "$script_dir" "$@" <<'PY'
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

root = Path(sys.argv.pop(1))
parser = argparse.ArgumentParser(description='Delete results for one method in the default work_4 directories. Requires Python 3.')
parser.add_argument('method', choices=['fvm', 'pinn', 'pinn-de'])
parser.add_argument('--dry-run', action='store_true', help='List targets without deleting them')
args = parser.parse_args()

def read_metadata(path):
    if path.is_symlink():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

def method(data):
    optimizer = str(data.get('optimizer', '')).lower()
    if data.get('de') or data.get('de_generations') or 'de/rand' in optimizer:
        return 'pinn-de'
    if 'adam' in optimizer:
        return 'pinn'
    return None

targets = set()
stamp = r'[0-9]+(?:\.[0-9]+)?'
if args.method == 'fvm':
    patterns = {
        'fvm_sim': rf'(?:solution_{stamp}\.npz|metadata_{stamp}\.json|solution\.npz|metadata\.json)',
        'plots': rf'velocity_(?:heatmaps(?:_{stamp})?\.png|evolution(?:_{stamp})?\.gif)',
    }
    for folder, pattern in patterns.items():
        directory = root / folder
        if directory.is_symlink():
            parser.error(f'Refusing symlink output directory: {directory}')
        if directory.is_dir():
            targets.update(p for p in directory.iterdir() if p.is_file() and re.fullmatch(pattern, p.name))
else:
    directory = root / 'pinn_sim'
    if directory.is_symlink():
        parser.error(f'Refusing symlink output directory: {directory}')
    if directory.is_dir():
        for path in directory.iterdir():
            if path.is_symlink():
                continue
            if path.is_dir() and re.fullmatch(stamp, path.name):
                kind = method(read_metadata(path / 'metadata.json'))
                if kind == args.method:
                    targets.add(path)
                elif kind is None:
                    print(f'Skipping unclassified run: {path}', file=sys.stderr)
            elif args.method == 'pinn' and path.is_file() and re.fullmatch(
                rf'pinn_(?:model_{stamp}\.pt|losses_{stamp}\.npz|metadata_{stamp}\.json)', path.name
            ):
                # These flat filenames are produced by the Adam notebook.
                targets.add(path)

for path in sorted(targets):
    print(f'{"Would remove" if args.dry_run else "Removing"}: {path}', flush=True)
    if not args.dry_run:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
print(f'{len(targets)} target(s) {"selected" if args.dry_run else "removed"}.')
PY
