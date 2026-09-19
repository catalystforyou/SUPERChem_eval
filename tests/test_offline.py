"""Run from repository root: python -m unittest discover -s tests -v."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import argparse
from eval.cli_utils import parse_bool

from superchem.metrics import graph_metrics
from superchem.offline import read_index, evaluate, validate_manifest, sha256

ROOT = Path(__file__).resolve().parents[1]


def graph(nodes, edges=(), matches=()):
    return {'nodes': [{'id': n, 'points': w} for n, w in nodes],
            'edges': [{'from': a, 'to': b} for a, b in edges],
            'matches': [{'h_id': h, 'r_id': r} for h, r in matches]}


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = graph([('R1', 1), ('R2', 3)], [('R1', 'R2')])
        self.answer = graph([('H1', 1), ('H2', 1)], [('H1', 'H2')], [('H1', 'R1'), ('H2', 'R2')])

    def test_perfect(self):
        m = graph_metrics(self.gt, self.answer)
        self.assertEqual((m['rpf'], m['node_only'], m['branching_factor'], m['dangling_count']), (1, 1, 0, 0))

    def test_wrong_direction_penalizes_path_not_node(self):
        self.answer['edges'] = [{'from': 'H2', 'to': 'H1'}]
        m = graph_metrics(self.gt, self.answer)
        self.assertEqual(m['node_only'], 1)
        self.assertEqual(m['rpf'], .25)

    def test_indirect_path_counts(self):
        a = graph([('H1', 1), ('H2', 1), ('H3', 1)], [('H1', 'H3'), ('H3', 'H2')], [('H1', 'R1'), ('H2', 'R2')])
        self.assertEqual(graph_metrics(self.gt, a)['rpf'], 1)

    def test_partial_parents(self):
        gt = graph([('R1', 1), ('R2', 1), ('R3', 2)], [('R1', 'R3'), ('R2', 'R3')])
        a = graph([('H1', 1), ('H2', 1), ('H3', 1)], [('H1', 'H3')], [('H1', 'R1'), ('H2', 'R2'), ('H3', 'R3')])
        self.assertEqual(graph_metrics(gt, a)['rpf'], .75)

    def test_empty_answer(self):
        m = graph_metrics(self.gt, graph([]))
        self.assertEqual((m['rpf'], m['node_only'], m['dangling_count']), (0, 0, 0))

    def test_branching_and_dangling(self):
        a = graph([('H1', 1), ('H2', 1), ('H3', 1)], [('H1', 'H2'), ('H1', 'H3')])
        m = graph_metrics(self.gt, a)
        self.assertAlmostEqual(m['branching_factor'], 1/3)
        self.assertEqual(m['dangling_count'], 1)

    def test_duplicate_matches_cannot_inflate(self):
        self.answer['matches'] *= 3
        self.assertEqual(graph_metrics(self.gt, self.answer)['rpf'], 1)

    def test_explicit_unmatched_node(self):
        self.answer['matches'].append({'h_id': 'H1', 'r_id': None})
        self.assertEqual(graph_metrics(self.gt, self.answer)['rpf'], 1)

    def test_invalid_graphs_rejected(self):
        cases = []
        a = copy.deepcopy(self.answer); a['edges'].append({'from': 'H2', 'to': 'H1'}); cases.append(a)
        a = copy.deepcopy(self.answer); a['nodes'].append({'id': 'H1'}); cases.append(a)
        a = copy.deepcopy(self.answer); a['edges'].append({'from': 'H1', 'to': 'ghost'}); cases.append(a)
        a = copy.deepcopy(self.answer); a['matches'].append({'h_id': 'H1', 'r_id': 'ghost'}); cases.append(a)
        for a in cases:
            with self.subTest(a=a), self.assertRaises(ValueError):
                graph_metrics(self.gt, a)

    def test_invalid_weight(self):
        self.gt['nodes'][0]['points'] = float('nan')
        with self.assertRaises(ValueError): graph_metrics(self.gt, self.answer)


class ManifestTests(unittest.TestCase):
    def test_boolean_cli(self):
        for value in ['False', 'false', '0', '', False]:
            self.assertIs(parse_bool(value), False)
        for value in ['True', 'true', '1', True]:
            self.assertIs(parse_bool(value), True)
        with self.assertRaises(argparse.ArgumentTypeError): parse_bool('maybe')

    def test_frozen_demo(self):
        rows, summary = evaluate(ROOT / 'demo/manifest.json')
        self.assertEqual(len(rows), 10)
        expected = json.loads((ROOT / 'demo/expected_metrics.json').read_text())
        self.assertEqual(summary, expected)
        self.assertEqual(summary[0]['accuracy_mean'], .5)

    def test_duplicate_uuid(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'test.jsonl'
            p.write_text('{"uuid":"a"}\n{"uuid":"a"}\n')
            with self.assertRaises(ValueError): read_index(p)

    def test_hash_missing_match_and_synthetic_rejected(self):
        source = ROOT / 'demo/manifest.json'
        manifest = json.loads(source.read_text())
        for item in [manifest['ground_truth'], manifest['runs'][0]['answers'], manifest['runs'][0]['matches'], *manifest.get('artifacts', [])]:
            item['path'] = str(source.parent / item['path'])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'manifest.json'
            bad = copy.deepcopy(manifest); bad['ground_truth']['sha256'] = '0'*64
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): validate_manifest(p)
            bad = copy.deepcopy(manifest); bad['runs'][0]['matches'] = None
            p.write_text(json.dumps(bad))
            validate_manifest(p)
            with self.assertRaises(ValueError): evaluate(p)
            bad = copy.deepcopy(manifest); bad['runs'][0]['synthetic_resample'] = True
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): validate_manifest(p)

    def test_explicit_failed_dag_keeps_denominator(self):
        source = ROOT / 'demo/manifest.json'
        manifest = json.loads(source.read_text())
        for item in [manifest['ground_truth'], manifest['runs'][0]['answers'], manifest['runs'][0]['matches'], *manifest.get('artifacts', [])]:
            item['path'] = str(source.parent / item['path'])
        matches = list(read_index(manifest['runs'][0]['matches']['path']).values())
        matches[0]['status'] = False
        with tempfile.TemporaryDirectory() as directory:
            m = Path(directory) / 'matches.jsonl'
            m.write_text(''.join(json.dumps(row)+'\n' for row in matches))
            manifest['runs'][0]['matches'] = {'path': str(m), 'sha256': sha256(m)}
            p = Path(directory) / 'manifest.json'; p.write_text(json.dumps(manifest))
            rows, summaries = evaluate(p)
            self.assertEqual((len(rows), summaries[0]['valid_dags']), (10, 9))
            self.assertEqual(rows[0]['rpf'], 0)


if __name__ == '__main__': unittest.main()
