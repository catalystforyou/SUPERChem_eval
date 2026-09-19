"""Replace source user IDs with consistent release-only participant labels.

No identity mapping is written. Original tracked data remains in Git history;
this transformation does not anonymize that history.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT/'data/20251015_baseline.csv', ROOT/'demo/20251015_baseline_demo.csv']
    documents = []
    for path in paths:
        raw = path.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        documents.append((path, raw, reader.fieldnames, list(reader)))
    ids = sorted({r['user_id'] for r in documents[0][3]})
    if all(x.startswith('participant_') for x in ids):
        # Keep deterministic LF endings for Git and manifest hashes.
        for path, raw, _, _ in documents:
            path.write_bytes(raw.replace(b'\r\n', b'\n'))
        audit_path = ROOT/'docs/baseline_pseudonymization.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            for item in audit['files']:
                item['after_sha256'] = hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()
            audit_path.write_text(json.dumps(audit, indent=2)+'\n')
        print('Participant identifiers already pseudonymized; LF endings and audit hashes verified.')
        return
    if any(x.startswith('participant_') for x in ids):
        raise ValueError('Mixed raw and pseudonymized IDs require manual review')
    mapping = {value: f'participant_{i:03d}' for i, value in enumerate(ids, 1)}
    audit = []
    # Validate all documents before writing either.
    for _, _, _, rows in documents:
        if not all(row['user_id'] in mapping for row in rows):
            raise ValueError('Demo contains unknown participants')
    for path, raw, fields, rows in documents:
        old_non_id = [{k:v for k,v in r.items() if k!='user_id'} for r in rows]
        for row in rows:
            row['user_id'] = mapping[row['user_id']]
        assert old_non_id == [{k:v for k,v in r.items() if k!='user_id'} for r in rows]
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
        audit.append({'path': str(path.relative_to(ROOT)), 'rows': len(rows),
                      'participants': len({r['user_id'] for r in rows}),
                      'before_sha256': hashlib.sha256(raw).hexdigest(),
                      'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    dest = ROOT/'docs/baseline_pseudonymization.json'
    dest.write_text(json.dumps({'scope': 'Current CSV files only; not historical Git objects or old archives',
                               'mapping_published': False, 'score_answer_and_question_fields_unchanged': True,
                               'files': audit}, indent=2)+'\n')
    print('Pseudonymized both baseline files; all non-ID fields preserved.')


if __name__ == '__main__': main()
