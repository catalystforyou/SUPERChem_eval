"""Switchable ChemDraw HTTP / local OPSIN + RDKit molecular comparison.

OPSIN recognizes supported systematic and retained names, not IUPAC compliance.
Unknown inputs remain exact_match=null. No silent fallback or network in OPSIN.
"""
import base64
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PINNED_JAR = ROOT/'tools/opsin/opsin-cli-2.9.0-jar-with-dependencies.jar'
_LOCK = threading.RLock()
_JAVA_SLOTS = threading.BoundedSemaphore(4)


def backend_name(config):
    value = config.get('backend', 'chemdraw')
    if value not in ('chemdraw', 'opsin'):
        raise ValueError('mol_compare.backend must be chemdraw or opsin')
    return value


def backend_endpoint(config):
    return 'local://opsin-rdkit' if backend_name(config) == 'opsin' else config.get('url', '')


def _rdkit():
    try:
        from rdkit import Chem, rdBase, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise RuntimeError('OPSIN backend requires RDKit: pip install -r requirements-mol.txt') from exc
    return Chem, rdBase, DataStructs, rdFingerprintGenerator


def resolve_config(config, base_directory=None):
    config = dict(config)
    if backend_name(config) == 'opsin':
        value = config.get('jar_path') or os.environ.get('OPSIN_JAR') or str(PINNED_JAR)
        path = Path(value).expanduser()
        if not path.is_absolute(): path = Path(base_directory or Path.cwd())/path
        config['jar_path'] = str(path.resolve())
    return config


@lru_cache(maxsize=8)
def _runner(jar, java, jar_mtime, class_mtime, manifest_mtime):
    return OpsinRunner(Path(jar), java)


def get_runner(config):
    jar = Path(resolve_config(config)['jar_path'])
    clazz = jar.parent/'classes/OpsinBridge.class'
    metadata = jar.parent/'manifest.json'
    if not all(p.is_file() for p in [jar, clazz, metadata]):
        raise RuntimeError('OPSIN installation incomplete; run python script/install_opsin.py')
    with _LOCK:
        return _runner(str(jar), config.get('java', 'java'), jar.stat().st_mtime_ns,
                       clazz.stat().st_mtime_ns, metadata.stat().st_mtime_ns)


def validate_config(config):
    if backend_name(config) == 'chemdraw':
        if not config.get('url') or not config.get('api_key'):
            raise ValueError('ChemDraw backend requires mol_compare.url and api_key')
    else:
        _rdkit(); get_runner(config)
    if float(config.get('timeout', 30)) <= 0:
        raise ValueError('mol_compare.timeout must be positive')


def provenance(config):
    if backend_name(config) == 'chemdraw':
        return {'backend':'chemdraw','endpoint':backend_endpoint(config)}
    runner=get_runner(config)
    return {'backend':'opsin','opsin_version':runner.metadata['version'],
            'jar_sha256':runner.metadata['jar_sha256'],
            'bridge_source_sha256':runner.metadata['bridge_source_sha256'],
            'rdkit_version':_rdkit()[1].rdkitVersion,'comparison_policy':'strict-isomeric-smiles-v1'}


def ensure_output_backend(output, config):
    """Prevent a resumed file mixing backends/versions. Old unmarked files are ChemDraw."""
    output=Path(output)
    sidecar=Path(str(output)+'.mol_compare.json')
    expected=provenance(config)
    if sidecar.exists():
        if json.loads(sidecar.read_text())!=expected:
            raise ValueError('Output was created with a different molecular backend/version; choose a new output path')
        return
    if output.exists() and output.stat().st_size and expected['backend']!='chemdraw':
        raise ValueError('Existing output has no backend provenance; use a new output path for OPSIN')
    sidecar.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=sidecar.parent,delete=False) as handle:
        json.dump(expected,handle,indent=2);handle.write('\n');tmp=Path(handle.name)
    tmp.replace(sidecar)


class OpsinRunner:
    def __init__(self, jar, java):
        if not shutil.which(java): raise RuntimeError('java not found; install a Java runtime 8+')
        self.jar, self.java = jar, java
        self.metadata = json.loads((jar.parent/'manifest.json').read_text())
        bridge = Path(__file__).parent/'resources/OpsinBridge.java'
        for path, key in [(jar, 'jar_sha256'), (jar.parent/'classes/OpsinBridge.class', 'bridge_class_sha256'), (bridge, 'bridge_source_sha256')]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata[key]:
                raise RuntimeError('OPSIN installation/source hash mismatch; rerun installer')
        self.cache = OrderedDict(); self.lock = threading.RLock()

    def parse(self, names, timeout):
        names = list(dict.fromkeys(names))
        with self.lock:
            found = {n:self.cache[n] for n in names if n in self.cache}
        missing = [n for n in names if n not in found]
        if missing:
            payload = '\n'.join(base64.b64encode(n.encode()).decode() for n in missing)+'\n'
            cmd = [self.java, '-Xmx256m', '-cp', str(self.jar)+os.pathsep+str(self.jar.parent/'classes'), 'OpsinBridge']
            try:
                with _JAVA_SLOTS:
                    result = subprocess.run(cmd, input=payload, capture_output=True, text=True,
                                            encoding='utf-8', timeout=timeout, check=True)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError('OPSIN name parsing timed out') from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError('OPSIN Java process failed; check Java and installation') from exc
            lines = result.stdout.splitlines()
            if len(lines) != len(missing): raise RuntimeError('OPSIN returned an unexpected number of results')
            for name, line in zip(missing, lines):
                parts = line.split('\t')
                if len(parts) != 3 or parts[0] not in ('SUCCESS','WARNING','FAILURE'):
                    raise RuntimeError('Invalid OPSIN bridge response')
                found[name] = {'status':parts[0], 'smiles':base64.b64decode(parts[1]).decode(),
                               'message':base64.b64decode(parts[2]).decode()}
            with self.lock:
                for name in missing:
                    self.cache[name] = found[name]
                    self.cache.move_to_end(name)
                while len(self.cache) > 4096: self.cache.popitem(last=False)
        return found


def _smiles(value):
    Chem, rdBase, _, _ = _rdkit()
    params = Chem.SmilesParserParams(); params.parseName = False; params.allowCXSMILES = False
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(value, params)
    if mol is None or not mol.GetNumAtoms(): return None
    return mol


def _canonical(mol):
    Chem, _, _, _ = _rdkit()
    if any(a.GetAtomicNum() == 0 or a.HasQuery() for a in mol.GetAtoms()):
        raise ValueError('Wildcard/query structures are not fully specified molecules')
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    return mol, Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def batch_compare(pairs, config):
    config = resolve_config(config)
    backend = backend_name(config)
    if not isinstance(pairs, list) or len(pairs) > 100:
        raise ValueError('pairs must be a list containing at most 100 pairs')
    if backend == 'chemdraw':
        validate_config(config)
        payload = json.dumps({'pairs':pairs}).encode()
        request = urllib.request.Request(config['url'], data=payload,
                    headers={'accept':'application/json','Content-Type':'application/json','Authorization':'Bearer '+config['api_key']})
        with urllib.request.urlopen(request, timeout=float(config.get('timeout',15))) as response:
            result = json.load(response)
        return {**result, 'backend':'chemdraw'}
    validate_config(config)
    runner = get_runner(config)
    Chem, rdBase, DataStructs, generators = _rdkit()
    pending = []; normalized = {}; errors = {}
    for pair in pairs:
        if not isinstance(pair, dict) or not all(k in pair for k in ['pair_id','mol1','mol2']):
            raise ValueError('Each pair needs pair_id, mol1 and mol2')
        if not all(isinstance(pair.get(f+'_format','auto'),str) for f in ['mol1','mol2']):
            raise ValueError('Molecule format hints must be strings')
        for field in ['mol1','mol2']:
            value = pair[field]; fmt = pair.get(field+'_format','auto')
            key = (value, fmt) if isinstance(value,str) else (repr(value),fmt)
            if fmt not in ('auto','smiles','iupac') or not isinstance(value,str) or not value.strip() or len(value)>4096 or '\n' in value or '\r' in value:
                errors[key] = 'Invalid format or empty/oversized/multiline molecule'; continue
            text = value.strip()
            mol = _smiles(text) if fmt != 'iupac' else None
            if mol is not None:
                try:normalized[key]=(*_canonical(mol),'smiles','SUCCESS',None)
                except ValueError as exc:errors[key]=str(exc)
            elif fmt == 'smiles':errors[key]='Invalid SMILES'
            else:pending.append((key,text))
    converted = runner.parse([text for _,text in pending],float(config.get('timeout',30))) if pending else {}
    for key,text in pending:
        parsed = converted[text]
        if parsed['status'] != 'SUCCESS':
            errors[key] = 'OPSIN '+parsed['status']+': '+parsed['message']; continue
        mol = _smiles(parsed['smiles'])
        if mol is None:errors[key]='RDKit could not read OPSIN SMILES'; continue
        try:normalized[key]=(*_canonical(mol),'opsin_name',parsed['status'],None)
        except ValueError as exc:errors[key]=str(exc)
    fingerprint = generators.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    results = []
    for pair in pairs:
        item = {k:pair[k] for k in ['pair_id','mol1','mol2']}
        item.update(exact_match=None,tanimoto=None,warning=None,error=None,backend='opsin',comparison_status='unresolved')
        keys=[(pair[f] if isinstance(pair[f],str) else repr(pair[f]),pair.get(f+'_format','auto')) for f in ['mol1','mol2']]
        failure=[errors[k] for k in keys if k in errors]
        if failure:item['error']='; '.join(failure)
        else:
            left,right=[normalized[k] for k in keys]
            item.update(exact_match=left[1]==right[1],tanimoto=float(DataStructs.TanimotoSimilarity(fingerprint.GetFingerprint(left[0]),fingerprint.GetFingerprint(right[0]))),
                        comparison_status='compared',canonical_smiles1=left[1],canonical_smiles2=right[1],input_type1=left[2],input_type2=right[2])
        results.append(item)
    success=sum(r['comparison_status']=='compared' for r in results)
    return {'total':len(results),'success':success,'failed':len(results)-success,'results':results,'backend':'opsin',
            'versions':{'opsin':runner.metadata['version'],'opsin_jar_sha256':runner.metadata['jar_sha256'],'rdkit':rdBase.rdkitVersion},
            'policy':'strict canonical isomeric SMILES; preserve stereo/isotopes/charges/fragments; no tautomer/salt standardization; OPSIN warnings unresolved'}
