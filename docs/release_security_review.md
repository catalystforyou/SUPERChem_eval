# Release code, privacy and reproducibility review — 2026-09-18

## Scope

Current tracked files plus new offline code, tests, demo, manifests and explicitly
selected release documentation were checked. Local API configs were used only
for in-memory exact credential comparison, not copied or printed. The research
workspace and old tar/zip deliveries are not a sanitized release artifact.

## Changes

- Replaced numeric source `user_id` values with `participant_###` pseudonyms in
  the full human CSV and demo CSV. A one-to-one mapping is consistent across both.
  No identity mapping is published. All 870 full and 13 demo records retain their
  exact non-ID fields; grouping and scores were compared with Git HEAD.
- Fixed invalid YAML in `eval/config.yaml.sample` (a bare trailing placeholder).
  Both API configuration templates now parse successfully.
- Added hashes for demo questions, split metadata and pseudonymized human data.
- Added an API-independent release scanner, which reports paths/counts only.
- Added minimal offline dependencies and corrected README environment claims,
  runnable clone command, DAG working directory and external service requirement.
- Preserved both direct-script and module-import forms of the boolean helper.

## Verification

- 17 unit/integration tests passed, including metrics, hashes, failed-output
  denominator, participant format, boolean flags and CSV overwrite protection.
- A separate Python 3.13.12 environment was created and installed from local
  cache using `requirements-offline.txt`: pandas 3.0.5, pyarrow 24.0.0,
  networkx 3.6.1. All 17 tests, demo and metrics export passed in a clean source
  copy with no API configuration, archived experiment directories or full data.
- Existing environment also passed (pandas 3.0.3, same pyarrow/networkx).
- Local README links checked successfully. Both YAML templates parsed.
- Current-file pattern scan found no credential tokens, private absolute paths,
  email patterns or private-key headers in the checked text/Parquet strings.
- Exact matching of six distinct locally configured credential strings against
  selected release files found no matches. Values are not recorded in this report.
- Previous 500-question formula parity check remains valid; formulas unchanged.

## History and limitations

36 reachable Git commits / 99 unique non-PDF/PNG/Parquet blobs were scanned for
credential-token patterns and private paths. No token-pattern hit was found.
Old `DAG_eval/view/dag_viewer.py` in commit `22aefb230ee3` still contains two
private-path occurrences. Historical human CSV versions retain source identifiers.
The current pseudonymization does not remove them from Git history or old archives.
No history rewrite, commit or push was performed.

An export of current files without `.git/` avoids shipping history. Publishing the
repository history itself needs a separate decision and audit; do not claim that
the history is anonymized. Pseudonyms retain participant linkability and are not
a guarantee against reidentification from other records.

PDF/image contents and embedded image bytes have not been exhaustively reviewed;
the scanner lists skipped binary files. It scans Parquet string fields, not image
metadata/OCR. Git-history binary data and old research archives are out of scope.
No live API generation or matcher service was invoked in this review. CI jobs for
other Python versions are configured, not claimed as remotely executed.

The candidate final-input manifest still lacks reconciled final match files and
is deliberately not a final manuscript result freeze.
