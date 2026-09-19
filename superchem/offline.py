"""Hash-verified offline manifest evaluation. No API configuration is loaded."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, variance
from .metrics import FORMULA_VERSION, graph_metrics, checked_graph


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_index(path):
    result = {}
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            uuid = row.get('uuid')
            if not isinstance(uuid, str) or not uuid or uuid in result:
                raise ValueError(f'{path}:{line_number}: missing or duplicate uuid')
            result[uuid] = row
    return result


def verified_file(root, entry):
    if not isinstance(entry, dict) or not entry.get('path') or not entry.get('sha256'):
        raise ValueError('File entry requires path and sha256')
    path = root / entry['path']
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Hash mismatch: {path}')
    return path


def validate_manifest(path, *, require_matches=False):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('formula_version') != FORMULA_VERSION:
        raise ValueError('Unsupported manifest or formula version')
    ids = manifest['question_ids']
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('question_ids must be nonempty and unique')
    root = path.parent
    for artifact in manifest.get('artifacts', []):
        verified_file(root, artifact)
    gt = read_index(verified_file(root, manifest['ground_truth']))
    if set(gt) != set(ids):
        raise ValueError('GT/question UUID mismatch')
    for row in gt.values():
        checked_graph(row['ground_truth_graph'])
    keys = set()
    loaded = []
    for run in manifest['runs']:
        key = (run['config_id'], run['replicate'])
        if key in keys or run.get('synthetic_resample') is not False or run.get('independent_generation') is not True:
            raise ValueError(f'Duplicate or non-independent run: {key}')
        keys.add(key)
        answers = read_index(verified_file(root, run['answers']))
        if set(answers) != set(ids):
            raise ValueError(f'Answer/question UUID mismatch: {key}')
        for row in answers.values():
            if row.get('score') not in (0, 1):
                raise ValueError(f'Nonbinary score: {key}/{row["uuid"]}')
            if (row.get('status') is not True or not str(row.get('llm_output') or '').strip()) and row['score'] != 0:
                raise ValueError('Invalid source answer must have score zero')
        matches = None
        if run.get('matches'):
            matches = read_index(verified_file(root, run['matches']))
            if set(matches) != set(ids):
                raise ValueError(f'Match/question UUID mismatch: {key}')
            for uuid, record in matches.items():
                if not isinstance(record.get('status'), bool):
                    raise ValueError(f'Match needs explicit boolean status: {uuid}')
                if record['status']:
                    graph = record.get('parsed')
                    if not isinstance(graph, dict) or not graph.get('nodes'):
                        raise ValueError(f'Successful match has no graph: {uuid}')
                    graph_metrics(gt[uuid]['ground_truth_graph'], graph)
        elif require_matches:
            raise ValueError(f'Final match file not supplied: {key}')
        loaded.append((run, answers, matches))
    if not loaded:
        raise ValueError('Manifest contains no runs')
    return manifest, gt, loaded


def evaluate(path):
    manifest, gt, runs = validate_manifest(path, require_matches=True)
    rows = []
    empty = {'nodes': [], 'edges': [], 'matches': []}
    for run, answers, matches in runs:
        for uuid in manifest['question_ids']:
            answer, match = answers[uuid], matches[uuid]
            source_valid = answer.get('status') is True and bool(str(answer.get('llm_output') or '').strip())
            match_valid = match.get('status') is True
            graph = match.get('parsed') if source_valid and match_valid else empty
            if source_valid and match_valid and (not isinstance(graph, dict) or not graph.get('nodes')):
                raise ValueError(f'Successful match has no graph: {uuid}')
            rows.append({'config_id': run['config_id'], 'replicate': run['replicate'], 'uuid': uuid,
                         'accuracy': answer['score'], 'dag_valid': source_valid and match_valid,
                         **graph_metrics(gt[uuid]['ground_truth_graph'], graph)})
    summaries = []
    for run, _, _ in runs:
        group = [r for r in rows if (r['config_id'], r['replicate']) == (run['config_id'], run['replicate'])]
        record = {'config_id': run['config_id'], 'replicate': run['replicate'], 'n': len(group),
                  'valid_dags': sum(r['dag_valid'] for r in group)}
        for name in ['accuracy', 'rpf', 'node_only', 'logic_penalty', 'branching_factor', 'dangling_count']:
            values = [r[name] for r in group]
            record[name + '_mean'] = mean(values)
            record[name + '_sample_variance'] = variance(values) if len(values) > 1 else None
        summaries.append(record)
    return rows, summaries


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
