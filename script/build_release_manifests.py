"""Build hashed demo and candidate-input manifests from verified local artifacts.

This is a maintainer-only offline command, not required by reviewers.
It never declares the manuscript data final or invents missing match results.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superchem.offline import sha256, read_index, evaluate, validate_manifest
from superchem.metrics import FORMULA_VERSION


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def entry(path, manifest):
    return {'path': os.path.relpath(path, manifest.parent), 'sha256': sha256(path)}


def demo():
    folder = ROOT / 'demo'
    manifest = folder / 'manifest.json'
    answer_name = '20251014164938_questions_release_en_false__gemini-2_5-pro_high__1_0_1.jsonl'
    answers = read_index(folder / answer_name)
    full_answers_path = ROOT / 'data' / answer_name
    full_gt_path = ROOT / 'DAG_eval/data/ground_truth_graphs_detail.jsonl'
    matches_path = ROOT / 'DAG_eval/cleaned/match_results_false__gemini-2_5-pro_high__v5_merged.jsonl'
    full_answers, full_gt, matches = map(read_index, [full_answers_path, full_gt_path, matches_path])
    gt = read_index(folder / 'ground_truth_graphs_detail.jsonl')
    selected = []
    for uuid, answer in answers.items():
        if answer['llm_output'] != full_answers[uuid]['llm_output'] or answer['score'] != matches[uuid]['score']:
            raise ValueError(f'Demo answer source mismatch: {uuid}')
        if gt[uuid]['ground_truth_graph'] != full_gt[uuid]['ground_truth_graph']:
            raise ValueError(f'Demo GT source mismatch: {uuid}')
        selected.append({'uuid': uuid, 'status': True, 'parsed': {k: matches[uuid][k] for k in ['nodes', 'edges', 'matches']}})
    match_file = folder / 'precomputed_matches.jsonl'
    match_file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected), encoding='utf-8')
    dump(manifest, {'schema_version': 1, 'formula_version': FORMULA_VERSION, 'state': 'historical_demo',
                    'description': 'Historical Gemini 2.5 Pro text sample; not final manuscript results. Extractor/judge provenance is not inferred from target model name.',
                    'question_ids': list(answers), 'ground_truth': entry(folder / 'ground_truth_graphs_detail.jsonl', manifest),
                    'artifacts': [entry(folder / name, manifest) for name in ['questions_demo.parquet', 'dataset_split_map.json', '20251015_baseline_demo.csv']],
                    'runs': [{'config_id': 'gemini-2_5-pro_high__text', 'replicate': 1, 'independent_generation': True,
                              'synthetic_resample': False, 'answers': entry(folder / answer_name, manifest),
                              'matches': entry(match_file, manifest)}]})
    dump(folder / 'dag_provenance.json', {'historical_sources': [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in [full_answers_path, full_gt_path, matches_path]],
                                       'selection': 'Existing 10 demo UUIDs; answer text and GT equality verified against historical sources; no API calls'})
    _, summaries = evaluate(manifest)
    dump(folder / 'expected_metrics.json', summaries)
    print(f'Demo: {manifest}')


def candidate():
    packages = [ROOT / name for name in [
        'DAG_full_repeats_ds_gt_v4_handoff_20260909',
        'DAG_o3_real_rep03_ds_gt_v4_addon_20260909',
        'DAG_gpt5_low_medium_replacement_ds_gt_20260910',
        'DAG_gpt5_low_medium_rep01_ds_gt_20260910',
    ]]
    manifest = ROOT / 'configs/release_inputs.candidate.json'
    selected = {}
    replacements = []
    gt_hash = None
    for package in packages:
        gt = package / 'data/gt/deepseek-v4-pro.jsonl'
        digest = sha256(gt)
        if gt_hash is not None and digest != gt_hash:
            raise ValueError('Packages have different GT hashes')
        gt_hash = digest
        for config in json.loads((package / 'data/configs.json').read_text())['configs']:
            if config['synthetic_resample'] or not config['independent_generation']:
                raise ValueError('Synthetic batch cannot enter candidate release')
            key = (config['base_config_id'], config['replicate_index'])
            answer = package / config['answer_file']
            if sha256(answer) != config['sha256']:
                raise ValueError(f'Package answer hash mismatch: {answer}')
            if key in selected:
                if config['base_config_id'] not in {f'gpt-5_{e}__{m}' for e in ['low', 'medium'] for m in ['text', 'multimodal']}:
                    raise ValueError(f'Unexpected replacement: {key}')
                replacements.append({'config_id': key[0], 'replicate': key[1], 'old': selected[key]['answers'], 'new': entry(answer, manifest)})
            selected[key] = {'config_id': key[0], 'replicate': key[1], 'independent_generation': True,
                             'synthetic_resample': False, 'answers': entry(answer, manifest), 'matches': None}
    gt_file = packages[0] / 'data/gt/deepseek-v4-pro.jsonl'
    if len(selected) != 75 or len({k[0] for k in selected}) != 25 or len(replacements) != 12:
        raise ValueError('Expected 25 configs, 75 batches, 12 GPT-5 replacements')
    dump(manifest, {'schema_version': 1, 'formula_version': FORMULA_VERSION, 'state': 'candidate_inputs_only',
                    'description': 'Resolved local input lineage only. Not the final manuscript freeze; final returned DS-GT match outputs still need reconciliation. Kimi/Intern inclusion is not inferred.',
                    'question_ids': sorted(read_index(gt_file)), 'ground_truth': entry(gt_file, manifest),
                    'runs': [selected[k] for k in sorted(selected)], 'replacements': replacements})
    validate_manifest(manifest)
    print(f'Candidate inputs only: {manifest} (75 batches; matches deliberately pending)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    if not (args.demo or args.candidate):
        parser.error('Choose --demo and/or --candidate')
    if args.demo:
        demo()
    if args.candidate:
        candidate()
