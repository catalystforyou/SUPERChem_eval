# DAG Re-matching Prompt

**Role:** You are an expert Logic Analyst and Chemistry Evaluation Specialist tasked with correcting problematic DAG node matches based on validation feedback.

## Context

The original DAG matching task identified several issues:
- **Problematic matches**: Incorrect matches that violate semantic equivalence rules
- **Missing matches**: Valid matches that were not identified
- **False null matches**: Nodes incorrectly marked as null that should have been matched
- **Edge issues**: Incorrect DAG structure (missing edges, extra edges, or wrong edge directions)

Your task is to **re-evaluate and correct ONLY the problematic nodes and edges** while keeping all other matches and edges unchanged.

---

## Input Data

1. **[CHEMISTRY_QUESTION]**: `{question}`
2. **[OPTIONS]**: `{options}` (if any)
3. **[GROUND_TRUTH_ANALYSIS]**: `{ground_truth_analysis}`
4. **[GROUND_TRUTH_GRAPH]**: `{ground_truth_graph}`
5. **[MODEL_ANSWER]**: `{model_answer}`
6. **[EXTRACTED_MODEL_DAG]**: `{extracted_dag}`
7. **[ORIGINAL_MATCHES]**: `{original_matches}`
8. **[VALIDATION_ISSUES]**: `{validation_issues}`

---

## Validation Issues Summary

### Problematic Matches
{problematic_matches_detail}

### Missing Matches
{missing_matches_detail}

### False Null Matches
{false_null_matches_detail}

### Edge Issues
{edge_issues_detail}

---

## Your Task

Re-evaluate **ONLY** the nodes mentioned in the validation issues above. For each problematic node:

1. Review the validation feedback carefully
2. Re-examine the semantic equivalence between hypothesis node (H) and reference node (R)
3. Determine the correct match according to the strict matching rules below
4. Update the match assignment

**IMPORTANT**: 
- Only modify matches for nodes mentioned in validation issues
- Keep all other matches exactly as they were in the original matching
- Follow the same strict matching rules as defined in match_prompt_v7.md

---

## Core Matching Rules (Reference)

### Rule 1: Semantic Equivalence
A match is valid **ONLY if** both nodes express the **same logical reasoning step** or **same chemical fact**:
- Paraphrasing is allowed
- Core semantic content must be identical
- "Similar" or "related" is NOT enough

### Rule 2: Final Answer Nodes (CRITICAL)
- Both must be final answer nodes
- Option letter MUST match EXACTLY
- "Answer is C or D" vs "Answer is C" → INVALID (ambiguous)
- "Answer is D" vs "Answer is C" → INVALID (wrong option)

### Rule 3: Molecular Structure Nodes (CRITICAL)
- Both must contain molecular identifiers (SMILES, IUPAC names, formulas)
- Must represent the SAME molecule
- Vague vs specific → INVALID match
- Different oxidation states/ligands/aggregation → INVALID

### Rule 4: Numerical Values
- Specific numerical values must match (within reasonable tolerance for continuous values)
- Different values → INVALID match
- Generic statement vs quantitative → INVALID match
- Discrete counts (atoms, rings, bonds) must match exactly

### Rule 5: Option Evaluation Nodes
- Both must evaluate the same option
- Both must have the same judgment (correct/incorrect)
- Different options → INVALID match
- Contradictory judgments → INVALID match

### Rule 5b: Option Nodes Must Point to Final Answer (CRITICAL)
- Any node that mentions or evaluates an option (A/B/C/D, "option", "choice", etc.) MUST have an edge pointing to the final answer node that states the model's selected option
- If the model's final answer is option C, all option-related nodes must point to the final answer node for option C
- Missing edge or pointing to a different option's final answer → STRUCTURE ERROR

### Rule 6: Type Compatibility
- Nodes must have compatible types
- FINAL_ANSWER can only match FINAL_ANSWER
- OPTION_EVALUATION can only match OPTION_EVALUATION
- STRUCTURE_ID can only match STRUCTURE_ID
- NUMERIC_VALUE can only match NUMERIC_VALUE
- Cross-type matches are generally invalid

### Rule 7: No Partial Credit
- A node that is "close but wrong" should be NULL match
- Do not overclaim matches

---

## Output Format

Return **strict JSON only**, no markdown code fences, no commentary.

**Schema:**
```json
{{
  "corrected_matches": [
    {{ "h_id": "H1", "r_id": "R1", "correction_reason": "Brief explanation of why this was corrected" }},
    {{ "h_id": "H2", "r_id": null, "correction_reason": "Brief explanation of why this was corrected" }},
    ...
  ],
  "unchanged_matches": [
    {{ "h_id": "H3", "r_id": "R3" }},
    {{ "h_id": "H4", "r_id": null }},
    ...
  ],
  "corrected_edges": [
    {{ "from": "H1", "to": "H2", "action": "add|remove", "correction_reason": "Brief explanation" }},
    ...
  ],
  "unchanged_edges": [
    {{ "from": "H3", "to": "H4" }},
    ...
  ],
  "summary": {{
    "total_corrected": <number>,
    "problematic_resolved": <number>,
    "missing_added": <number>,
    "false_null_fixed": <number>,
    "edges_corrected": <number>,
    "overall_assessment": "Brief assessment of the re-matching quality"
  }}
}}
```

**Output Requirements:**
1. ✅ `corrected_matches` contains ONLY nodes mentioned in validation issues
2. ✅ `unchanged_matches` contains all other nodes with their original matches
3. ✅ Each corrected match includes a brief `correction_reason`
4. ✅ Use `null` (not `"null"`, not empty string) for no match
5. ✅ Valid JSON syntax (proper quotes, commas, no trailing commas)
6. ❌ Do NOT wrap JSON in markdown code fences
7. ❌ Do NOT add any commentary before or after JSON

---

## Key Principles

1. **Conservative Correction**: Only change what validation identified as problematic
2. **Evidence-Based**: Base corrections on validation feedback and strict rules
3. **Consistency**: Maintain consistency with unchanged matches
4. **Strictness**: When in doubt, prefer NULL over questionable matches
5. **Completeness**: Ensure all validation issues are addressed

---

## Decision Framework

For each problematic node, ask:
1. Does the validation feedback identify a clear rule violation?
2. Is there a valid alternative match (for missing/false_null cases)?
3. Should this match be changed to null (for problematic matches)?
4. Does the correction align with semantic equivalence principles?

If uncertain, prefer the more conservative option (NULL over match).
