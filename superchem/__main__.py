"""python -m superchem: portable offline validation and metric computation."""
import argparse
import json
import sys
from pathlib import Path
from .offline import validate_manifest, evaluate, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate', help='Verify frozen data hashes and UUID coverage')
    validate.add_argument('--manifest', required=True, type=Path)
    validate.add_argument('--require-matches', action='store_true')
    metrics = sub.add_parser('metrics', help='Compute metrics without API access')
    metrics.add_argument('--manifest', required=True, type=Path)
    metrics.add_argument('--output', required=True, type=Path, help='New output directory; refuses overwrite')
    molecules = sub.add_parser('mol-compare', help='OPSIN/RDKit local or ChemDraw HTTP comparison')
    molecules.add_argument('--backend', choices=['opsin','chemdraw'], default=None)
    molecules.add_argument('--config', type=Path, help='YAML containing mol_compare configuration')
    molecules.add_argument('--opsin-jar', type=Path)
    molecules.add_argument('--pairs', type=Path, help='JSON array or object containing pairs')
    molecules.add_argument('--mol1')
    molecules.add_argument('--mol2')
    molecules.add_argument('--mol1-format', choices=['auto','smiles','iupac'], default='auto')
    molecules.add_argument('--mol2-format', choices=['auto','smiles','iupac'], default='auto')
    args = parser.parse_args()
    try:
        if args.command == 'validate':
            m, _, runs = validate_manifest(args.manifest, require_matches=args.require_matches)
            print(json.dumps({'state': m.get('state'), 'runs': len(runs), 'questions': len(m['question_ids']),
                              'runs_with_matches': sum(r[2] is not None for r in runs)}, indent=2))
        elif args.command == 'mol-compare':
            from .mol_compare import batch_compare, resolve_config
            config = {'backend':'opsin'}
            if args.config:
                import yaml
                config = dict((yaml.safe_load(args.config.read_text()) or {}).get('mol_compare', {}))
            if args.backend: config['backend'] = args.backend
            if args.opsin_jar: config['jar_path'] = str(args.opsin_jar.resolve())
            config = resolve_config(config, args.config.resolve().parent if args.config else None)
            if args.pairs:
                if args.mol1 is not None or args.mol2 is not None:
                    raise ValueError('Use --pairs OR --mol1/--mol2')
                pairs = json.loads(args.pairs.read_text())
                if isinstance(pairs, dict): pairs = pairs['pairs']
            else:
                if args.mol1 is None or args.mol2 is None: raise ValueError('Supply --pairs or both molecules')
                pairs = [{'pair_id':'pair1','mol1':args.mol1,'mol2':args.mol2,
                          'mol1_format':args.mol1_format,'mol2_format':args.mol2_format}]
            result = batch_compare(pairs, config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get('failed'): return 2
        else:
            if args.output.exists():
                raise ValueError('Output directory exists; choose a new directory')
            rows, summaries = evaluate(args.manifest)
            args.output.mkdir(parents=True)
            write_csv(args.output / 'per_item.csv', rows)
            write_csv(args.output / 'summary.csv', summaries)
            print(json.dumps(summaries, indent=2))
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
