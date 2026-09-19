# Molecular comparison: ChemDraw or open-source OPSIN

The maintained matcher `DAG_eval/src/match_dag.py` now supports two explicit
backends. The default remains `chemdraw` for compatibility. `opsin` uses local
OPSIN name parsing and RDKit structure comparison; no ChemDraw installation,
tool-service API key, online name service or LLM is needed for the comparison.
The surrounding DAG judge still requires its normal LLM API configuration.

## Install the open-source backend

From the repository root, install Java JDK 8 or newer (`java` and `javac` on PATH),
then:

```bash
pip install -r requirements-offline.txt -r requirements-mol.txt
python script/install_opsin.py
python -m superchem mol-compare --backend opsin --mol1 ethanol --mol2 CCO
```

Expected: `exact_match: true`, `canonical_smiles1/2: CCO`, backend `opsin`.
The installer downloads the official [OPSIN 2.9.0 release](https://github.com/dan2097/opsin/releases/tag/2.9.0),
verified as latest stable on 2026-09-19, checks its published SHA256, and compiles
the included status-aware Java bridge. The tested JAR digest is
`c2e29326c281f87b59a05d934d8589adac6e9d17b95b984931b3e739111b360f`.

The default is pinned for reproducibility. `--version latest` queries GitHub's
latest release and verifies its asset digest; it can change results and should
be recorded as a new software version. Downloads respect environment proxies;
use `--no-proxy` only if a configured proxy is broken. For offline installation,
download the official JAR elsewhere and use `--jar /path/to/the.jar` (JDK still
required). JARs/classes are installed under ignored `tools/opsin/` and are not
committed. OPSIN is MIT licensed; its bundled dependencies retain their own
licenses. RDKit is a separately licensed BSD open-source dependency.

Local verification: Java runtime 21, javac 8, OPSIN 2.9.0, RDKit 2026.03.6,
Python 3.13.12 on Linux. Other supported platforms require their own validation.

## Switch the DAG matcher

Copy `DAG_eval/src/config.example.yaml` to `DAG_eval/src/config.yaml` and configure
the judge model. Set the molecular section to:

```yaml
mol_compare:
  backend: opsin
  jar_path: ../../tools/opsin/opsin-cli-2.9.0-jar-with-dependencies.jar
  java: java
  timeout: 30
```

Relative `jar_path` values resolve against the YAML file's directory, not the
working directory. `OPSIN_JAR` is supported when `jar_path` is omitted. For
ChemDraw, use:

```yaml
mol_compare:
  backend: chemdraw
  url: https://your-service.example/batch_mol_compare
  api_key: YOUR_TOOL_API_KEY
  timeout: 30
```

You can override the YAML choice without editing it:

```bash
python DAG_eval/src/match_dag.py \
  --config DAG_eval/src/config.yaml \
  --questions demo/questions_demo.parquet \
  --answers demo/20251014164938_questions_release_en_false__gemini-2_5-pro_high__1_0_1.jsonl \
  --ground-truth demo/ground_truth_graphs_detail.jsonl \
  --prompt DAG_eval/prompts/match_prompt_v5.md \
  --output outputs/opsin_demo_match.jsonl \
  --model YOUR_JUDGE_MODEL --language en --limit 2 --workers 1 \
  --mol-compare-backend opsin
```

The command above calls the judge API. It is distinct from the free local
`python -m superchem mol-compare` example. Add `--opsin-jar /path/to/the.jar` to
override the JAR (relative CLI paths resolve against the working directory).
Switch to `--mol-compare-backend chemdraw` to use the HTTP service.
The legacy shell pipeline also follows the `mol_compare.backend` YAML setting.

Use a **new output path** when switching backend or software versions. The matcher
writes `<output>.mol_compare.json` and refuses resumption with a different backend,
JAR/bridge hash or RDKit version. Historical outputs without a sidecar are treated
as the legacy ChemDraw workflow and cannot be resumed as OPSIN. Archived handoff
packages retain their frozen code; they are not silently upgraded by this change.

## Name detection and equivalence policy

1. In `auto` mode, attempt strict SMILES parsing first. Trailing molecule names
   and CXSMILES extensions are disabled, preventing partial acceptance of text.
2. If not valid SMILES, ask OPSIN to interpret it as a chemical name and produce
   SMILES. This is a practical recognition test, **not proof of valid/preferred
   IUPAC nomenclature**: OPSIN also accepts some retained/trivial names.
3. Parse that SMILES with RDKit and compare canonical isomeric SMILES. Ordinary
   explicit hydrogens and atom-map labels are normalized, while stereo, isotopes,
   formal charges and disconnected salt/mixture components are retained.

Enantiomers, stereospecified versus unspecified structures, differing protonation,
tautomers, or omitted counterions are not automatically made equivalent. Aromatic
and valid Kekulé encodings and disconnected-fragment order are normalized by RDKit.
Wildcard/query structures are unresolved, not compared as fully defined molecules.
Coordination compounds, organometallics, polymers and unsupported names may fail;
there is no claim of universal coverage or exact ChemDraw equivalence.

For ambiguous strings, the pair may specify `mol1_format` / `mol2_format` as
`smiles`, `iupac`, or `auto`. Example:

```bash
python -m superchem mol-compare \
  --mol1 'ethanoic acid' --mol1-format iupac \
  --mol2 'CC(=O)O' --mol2-format smiles
```

For batches, pass `--pairs pairs.json`, a JSON array (or object with `pairs`) of
objects containing `pair_id`, `mol1`, `mol2` and optional format hints. The CLI
also accepts `--config` and `--backend`. Maximum batch size is 100 pairs, maximum
input string length 4096 characters. Name conversions are cached in process;
Java subprocesses are capped at four simultaneously, independently of judge
concurrency. Timeout applies once a parsing subprocess starts.

## Output and failure semantics

The existing `total`, `success`, `failed`, `results` and per-pair `pair_id`,
`mol1`, `mol2`, `exact_match`, `tanimoto`, `warning`, `error` fields are retained.
OPSIN adds `comparison_status`, canonical SMILES, recognized input types and
backend/version metadata. `success` counts comparable pairs, including different
molecules; it is not the number of equal pairs.

- `exact_match=true`: equivalent under the stated canonicalization policy.
- `exact_match=false`: both parsed successfully but the structures differ.
- `exact_match=null`: invalid input, unrecognized name, OPSIN WARNING/FAILURE,
  or unsupported structure. This is **unknown**, never evidence of a mismatch.

Warnings (including ambiguous names/ignored stereochemistry) are conservative:
they remain unresolved. Java/runtime errors raise an explicit tool error. No
automatic fallback to ChemDraw or external services occurs. The judge's tool
description explicitly warns against treating unresolved results as mismatch.
The standalone CLI returns exit code 2 if any pair is unresolved, while retaining
the full JSON result. `tanimoto` is a diagnostic Morgan(radius=2, 2048-bit,
chirality-aware) fingerprint similarity, not the identity criterion or a promise
of parity with ChemDraw fingerprints.

## Tests

```bash
python -m unittest discover -s tests -v
SUPERCHEM_RUN_OPSIN_TESTS=1 python -m unittest discover -s tests -v
```

The optional command exercises the real downloaded JAR/RDKit and a simulated
judge-tool interaction, without any paid LLM requests. Tests cover mixed formats,
invalid names, aromatic encodings, salts, isotope/charge/stereo preservation,
tautomers, timeouts, unknown results and backend-resumption isolation. ChemDraw
HTTP compatibility is checked with a mocked response, not a live proprietary
service. This implementation does not retroactively change existing paper scores.
The simulated matcher integration imports the normal judge runner dependencies;
install `openai PyYAML loguru tqdm` as well if only the two minimal requirements
files were installed. The dedicated CI job installs these explicitly. Local
tests passed; the newly configured remote CI job has not yet been executed.

## References

- [OPSIN source, API and limitations](https://github.com/dan2097/opsin)
- [OPSIN 2.9.0 release](https://github.com/dan2097/opsin/releases/tag/2.9.0)
- [RDKit SMILES and molecule handling](https://www.rdkit.org/docs/GettingStartedInPython.html)
- Lowe et al., *Chemical Name to Structure: OPSIN, an Open Source Solution*,
  J. Chem. Inf. Model. 2011, 51, 739–753, DOI: 10.1021/ci100384d.
