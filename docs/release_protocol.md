# Release cleanup: offline protocol and remaining work

## Implemented in this pass

- `superchem/metrics.py`: API-independent original weighted RPF/node-only/BF/DC.
- `python -m superchem validate|metrics`: hashed manifests, UUID coverage, strict
  DAG validation, per-question CSV, per-run means and sample variances (ddof=1).
- `demo/run_demo.py`: historical 10-question offline ACC + DAG reproduction.
- `tests/`: directionality, indirect paths, partial parents, null matches,
  duplicates, cycles, failures, data hashes and demo regression.
- `configs/release_inputs.candidate.json`: unique candidate inputs for 75 answer
  batches; no synthetic resamples, 12 explicit GPT-5 replacements.
- `eval/cli_utils.py`: fixes `--multimodal False` interpreting as true; the
  historical empty-string text-only flag remains supported.

## Exact metric semantics

For reference node r of weight w(r), node-only grants its points when at least one
answer node matches it. RPF multiplies those points by the fraction of direct GT
parents whose matched answer nodes can reach a matched answer node for r.
Matched GT roots receive factor 1. Both sums divide by total GT weight.
Reachability includes paths of one or more edges, not a zero-length self path.
Duplicate matches do not add credit. Explicit `r_id: null` means unmatched.

- `logic_penalty = node_only - rpf`.
- `BF = sum(max(out_degree - 1, 0)) / number_of_answer_nodes`.
- `DC = max(number_of_zero_out_degree_answer_nodes - 1, 0)`.
- Scores are proportions, not percentages. Empty answer DAGs have zero metrics.
- For known source/judge failures, keep the question in the denominator with zero
  DAG credit. Nonempty malformed successful DAGs cause an error, not a silent repair.
- Descriptive per-question sample variance is **not** a confidence interval.
  Question-level paired bootstrap/statistical release scripts remain to be integrated.

The new implementation was compared with the frozen handoff formula on all 500
questions in `DAG_gt_sensitivity_full500_20260908`, agreeing to 1e-12 for all metrics.
Archive scripts are preserved for provenance rather than edited in place.

## Data versions and manuscript alignment

The demo uses its original historical GT and matched answers. It is not evidence
for final leaderboard claims. The candidate input manifest instead uses the frozen
DeepSeek V4 Pro GT from the latest prepared handoff packages. Neither is silently
promoted to the manuscript's final choice.

Current candidate scope: 25 configurations × 3 runs; GPT-5 Low/Medium use new
rep01/02/03, o3 uses genuinely generated rep03. Failed requests remain recorded.
The final returned match files and whether Kimi/Intern enter the final paper still
need reconciliation. CLI scoring refuses candidate manifests without matches.

## Next implementation stages

1. Reconcile final server returns, check answer/GT hashes, populate matches and
   publish a separate final manifest with figure/table mappings.
2. Consolidate API generation/matching entry points without changing the prompts
   or semantic-vs-structural recovery protocol. Move credential loading to request
   time; preserve provider/request-model/returned-model and all attempt logs.
3. Consolidate answer parsing as a versioned, audited operation; apply consistently
   to all final batches. Do not rewrite raw generations or select answers by score.
4. Port overall/subdiscipline statistics, judge/expert agreement, alternative-path
   audit and component comparisons to manifest-driven scripts. Use only analyses
   actually retained in the revised paper; HI treatment follows the final manuscript.
5. Molecular comparison now supports explicit ChemDraw/OPSIN backends; see
   [mol_compare.md](mol_compare.md). Real local OPSIN/RDKit tests and mocked judge
   integration are provided. Scientific cross-backend agreement on the final
   benchmark remains a separate analysis; unsupported structures are unresolved.
6. Separate minimal/runtime/optional dependencies and record a tested lockfile,
   actual platform/runtime/cost measurements, release allowlist and data downloads.

## Git and release policy

Initial branch: main; existing edits to eval.py/eval.sh/eval_ckpt.py/eval_cot.py and
deletion of prompt_en_eval.txt were preserved. No commit, reset or history rewrite
was performed. Historical delivery archives were not modified.
The root MIT license remains. New source, tests and demo artifacts must be added
explicitly when a commit is requested; do not `git add .` across the research workspace.
The new GitHub Actions configuration tests offline code on Python 3.10/3.12;
local checks used the existing environment, not a claim those CI jobs already ran.
