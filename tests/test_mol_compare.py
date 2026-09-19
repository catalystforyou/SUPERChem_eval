"""Contract tests always run; real OPSIN tests require SUPERCHEM_RUN_OPSIN_TESTS=1."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from superchem import mol_compare as mc


class BackendContractTests(unittest.TestCase):
    def test_legacy_default_is_chemdraw(self):
        self.assertEqual(mc.backend_name({}), 'chemdraw')

    def test_no_silent_backend_fallback(self):
        with self.assertRaises(ValueError): mc.backend_name({'backend':'unknown'})
        with self.assertRaises(ValueError): mc.validate_config({'backend':'chemdraw'})

    def test_legacy_http_contract(self):
        pair={'pair_id':'p','mol1':'ethanol','mol2':'CCO'}
        response={'total':1,'success':1,'failed':0,'results':[dict(pair,exact_match=True,tanimoto=1.,warning=None,error=None)]}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
            result=mc.batch_compare([pair],{'url':'https://example.invalid/compare','api_key':'test-placeholder'})
        self.assertTrue(result['results'][0]['exact_match'])
        self.assertEqual(result['backend'],'chemdraw')
        self.assertEqual(json.loads(call.call_args.args[0].data), {'pairs':[pair]})

    def test_relative_jar_is_relative_to_config(self):
        resolved=mc.resolve_config({'backend':'opsin','jar_path':'../opsin.jar'},Path('/tmp/config'))
        self.assertEqual(resolved['jar_path'],'/tmp/opsin.jar')

    def test_output_cannot_mix_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.jsonl'
            config={'backend':'chemdraw','url':'https://example.invalid','api_key':'placeholder'}
            mc.ensure_output_backend(path,config)
            mc.ensure_output_backend(path,config)
            with patch.object(mc,'provenance',return_value={'backend':'opsin'}):
                with self.assertRaises(ValueError):mc.ensure_output_backend(path,{'backend':'opsin'})

    def test_unmarked_old_output_not_resumed_as_opsin(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.jsonl';path.write_text('{}\n')
            with patch.object(mc,'provenance',return_value={'backend':'opsin'}):
                with self.assertRaises(ValueError):mc.ensure_output_backend(path,{'backend':'opsin'})


@unittest.skipUnless(os.environ.get('SUPERCHEM_RUN_OPSIN_TESTS')=='1', 'optional Java/OPSIN/RDKit integration')
class OpsinIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config={'backend':'opsin','timeout':30}
        mc.validate_config(cls.config)

    def compare(self,a,b,**extra):
        result=mc.batch_compare([{'pair_id':'p','mol1':a,'mol2':b,**extra}],self.config)
        self.assertEqual(result['total'],1)
        return result['results'][0]

    def test_name_smiles(self):
        r=self.compare('ethanol','CCO')
        self.assertTrue(r['exact_match']);self.assertEqual(r['input_type1'],'opsin_name')
        self.assertEqual(r['input_type2'],'smiles')

    def test_names_and_retained_name(self):
        self.assertTrue(self.compare('ethanoic acid','acetic acid')['exact_match'])

    def test_different_molecules(self):
        self.assertFalse(self.compare('ethanol','methanol')['exact_match'])

    def test_aromatic_kekule(self):
        self.assertTrue(self.compare('c1ccccc1','C1=CC=CC=C1')['exact_match'])

    def test_order_and_atom_maps_do_not_change_identity(self):
        self.assertTrue(self.compare('[CH3:1][OH:2]','OC')['exact_match'])

    def test_charge_salts_and_isotopes_are_preserved(self):
        for a,b in [('CC(=O)O','CC(=O)[O-]'),('CCN','CCN.Cl'),('[13CH4]','C')]:
            with self.subTest(a=a,b=b):self.assertFalse(self.compare(a,b)['exact_match'])

    def test_enantiomers_and_unspecified_stereo_differ(self):
        self.assertFalse(self.compare('N[C@@H](C)C(=O)O','N[C@H](C)C(=O)O')['exact_match'])
        self.assertFalse(self.compare('N[C@@H](C)C(=O)O','NC(C)C(=O)O')['exact_match'])

    def test_name_stereochemistry(self):
        self.assertTrue(self.compare('(2S)-2-aminopropanoic acid','N[C@@H](C)C(=O)O')['exact_match'])

    def test_tautomers_not_collapsed(self):
        self.assertFalse(self.compare('CC=O','C=CO')['exact_match'])

    def test_salt_fragment_order(self):
        self.assertTrue(self.compare('[Na+].[Cl-]','[Cl-].[Na+]')['exact_match'])

    def test_invalid_name_is_unknown_not_false(self):
        r=self.compare('not-a-real-molecule-xyz987','CCO')
        self.assertIsNone(r['exact_match']);self.assertIsNone(r['tanimoto']);self.assertTrue(r['error'])

    def test_smiles_trailing_name_not_silently_accepted(self):
        r=self.compare('CCO unrelated description','CCO',mol1_format='smiles')
        self.assertIsNone(r['exact_match'])

    def test_explicit_format_override(self):
        r=self.compare('CCO','ethanol',mol1_format='smiles',mol2_format='iupac')
        self.assertTrue(r['exact_match'])
        self.assertIsNone(self.compare('ethanol','CCO',mol1_format='smiles')['exact_match'])

    def test_wildcard_rejected(self):
        self.assertIsNone(self.compare('*CC','CCC')['exact_match'])

    def test_warning_never_becomes_equivalence(self):
        runner=mc.get_runner(self.config)
        with patch.object(runner,'parse',return_value={'warning-name':{'status':'WARNING','smiles':'CCO','message':'ambiguous'}}):
            self.assertIsNone(self.compare('warning-name','CCO')['exact_match'])

    def test_real_optical_rotation_warning_is_unresolved(self):
        r=self.compare('(+)-lactic acid','CC(O)C(=O)O')
        self.assertIsNone(r['exact_match'])
        self.assertIn('WARNING',r['error'])

    def test_batch_mixed_status(self):
        pairs=[{'pair_id':'same','mol1':'benzene','mol2':'c1ccccc1'},
               {'pair_id':'different','mol1':'CCO','mol2':'CO'},
               {'pair_id':'unknown','mol1':'badchemical987','mol2':'CO'}]
        r=mc.batch_compare(pairs,self.config)
        self.assertEqual((r['total'],r['success'],r['failed']),(3,2,1))
        self.assertEqual([x['pair_id'] for x in r['results']],['same','different','unknown'])
        self.assertEqual(r['versions']['opsin'],'2.9.0')

    def test_timeout_is_explicit(self):
        runner=mc.get_runner(self.config)
        with patch('subprocess.run',side_effect=subprocess.TimeoutExpired('java',0.01)):
            with self.assertRaisesRegex(RuntimeError,'timed out'):
                runner.parse(['uncached-timeout-name987'],0.01)

    def test_dag_tool_path_without_chemdraw_credentials(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DAG_eval/src'))
        import match_dag
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                fn=SimpleNamespace(name='batch_compare_molecules',arguments=json.dumps({'pairs':[{'pair_id':'p','mol1':'ethanol','mol2':'CCO'}]}))
                msg=SimpleNamespace(content='',tool_calls=[SimpleNamespace(id='tool1',function=fn)])
            else:msg=SimpleNamespace(content='{"nodes":[],"edges":[],"matches":[]}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)],usage=None)
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch('urllib.request.urlopen',side_effect=AssertionError('Local OPSIN must not use HTTP')):
            result=match_dag.call_llm_with_tools(client,'mock-judge',[{'role':'user','content':'test'}],0.2,30,'high',self.config)
        self.assertEqual(len(calls),2)
        self.assertTrue(result['tool_calls'][0]['response']['result']['results'][0]['exact_match'])
        self.assertEqual(result['tool_calls'][0]['request']['backend'],'opsin')


if __name__=='__main__':unittest.main()
