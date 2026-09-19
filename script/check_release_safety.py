"""Scan tracked files and explicitly listed new release files; never print secrets.

No .git history, local credential files, or research archives are exported/scanned
by default. Binary image/PDF payloads require separate review. Parquet string
columns are inspected when pandas/pyarrow are installed.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXTRA_DIRS = ['superchem', 'tests', 'configs', 'demo', '.github']
EXTRA_FILES = ['eval/cli_utils.py', 'eval/eval_revision_api.py', 'requirements-offline.txt',
               'script/build_release_manifests.py', 'script/anonymize_baseline_ids.py',
               'script/check_release_safety.py', 'docs/release_protocol.md',
               'docs/release_security_review.md', 'docs/baseline_pseudonymization.json']
EXTRA_FILES += ['script/install_opsin.py', 'requirements-mol.txt', 'docs/mol_compare.md']
PATTERNS = {
    'credential_token': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{30,}|ghp_[A-Za-z0-9]{25,})'),
    'private_absolute_path': re.compile(r'/(?:home|hdd01|data/projects)/[A-Za-z0-9_.-]+'),
    'email_address_review': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}


def selected_files():
    result = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')) - {''}
    result.update(EXTRA_FILES)
    for directory in EXTRA_DIRS:
        result.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts)
    return sorted(name for name in result if (ROOT/name).is_file())


def inspect_text(text, name):
    return [{'path': name, 'rule': rule, 'count': len(pattern.findall(text))}
            for rule, pattern in PATTERNS.items() if pattern.search(text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optional report containing paths/counts only')
    args = parser.parse_args()
    findings = []; skipped = []; checked = 0
    for name in selected_files():
        path = ROOT/name
        if path.name == 'config.yaml' or path.name.startswith('.env'):
            findings.append({'path': name, 'rule': 'private_configuration_in_release', 'count': 1})
            continue
        if path.suffix == '.parquet':
            import pandas as pd
            frame = pd.read_parquet(path)
            # Bytes/image blobs are excluded; JSON serialization handles nested strings.
            def strings(value):
                if isinstance(value, str): return value
                if isinstance(value, dict): return '\n'.join(strings(k)+'\n'+strings(v) for k,v in value.items())
                if isinstance(value, (list, tuple)): return '\n'.join(map(strings, value))
                return ''
            for column in frame:
                findings.extend(inspect_text('\n'.join(strings(v) for v in frame[column]), name+':'+column))
            checked += 1
        else:
            try: text = path.read_text(encoding='utf-8')
            except UnicodeError:
                skipped.append(name); continue
            findings.extend(inspect_text(text, name)); checked += 1
        if path.suffix == '.csv' and 'baseline' in name:
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    if 'user_id' in row and not re.fullmatch(r'participant_\d{3}', row['user_id']):
                        findings.append({'path': name, 'rule': 'unmapped_participant_id', 'count': 1}); break
    report = {'scope': 'tracked existing files plus explicit new release code/demo; no local secrets or old archives',
              'files_checked': checked, 'binary_payloads_not_inspected': skipped, 'findings': findings,
              'limitation': 'Heuristic scan, not proof of anonymity; review images, participant consent, and historical Git separately.'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(findings))


if __name__ == '__main__': main()
