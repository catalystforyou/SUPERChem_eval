# Validation Prompt for Match DAG Quality

You are an expert evaluator for chemistry reasoning DAG matching quality.

**Context**: An LLM was tasked to extract a reasoning DAG from a model's answer and match it against a ground truth DAG. Your job is to verify if the matching follows strict rules.

**Chemistry Question**:
{question}

**Options**:
{options}

**Ground Truth Analysis**:
{ground_truth_analysis}

**Ground Truth DAG**:
{ground_truth_graph}

**Model Answer**:
{model_answer}

**Extracted Model DAG**:
{extracted_graph}

**Matches** (h_id → r_id):
{matches}

---

**Your Task**: Evaluate if the matching is correct according to these critical rules:

### Rule 1: Semantic Equivalence
- A match is valid ONLY if both nodes represent the SAME logical reasoning step
- Paraphrasing is allowed, but the semantic core must be identical
- "Similar" or "related" is NOT enough

### Rule 2: Final Answer Nodes (CRITICAL)
- If an h_id node is a final answer node (contains "answer is", "option", "select", etc.)
- The option letter MUST match EXACTLY with the r_id node
- "Answer is C or D" vs "Answer is C" → INVALID match (ambiguous)
- "Answer is D" vs "Answer is C" → INVALID match (wrong option)

### Rule 3: Molecular Structure Nodes (CRITICAL)
- If both h_id and r_id contain molecular identifiers (SMILES, IUPAC names, formulas)
- They must represent the SAME molecule (tool should have been used to verify)
- If h_id is vague ("an alcohol") but r_id is specific ("ethanol C₂H₅OH") → INVALID match
- If h_id lacks identifier but r_id has one → INVALID match

### Rule 4: Numerical Values
- If r_id contains a specific numerical value, h_id must have the SAME value
- Different values → INVALID match
- Generic statement vs quantitative → INVALID match

### Rule 5: No Partial Credit
- A node that is "close but wrong" should be NULL match, not matched
- Overclaiming matches is a serious error

---

### Rule 6: DAG Structure (CRITICAL - Edges)
- Check if the extracted model DAG edges correctly reflect the reasoning dependencies
- Missing edge: If H2 depends on H1 logically, there should be an edge from H1 to H2
- Extra edge: If there's an edge from H1 to H2 but H2 doesn't depend on H1, it's wrong
- Wrong direction: Edge direction should follow logical dependency (from prerequisite to conclusion)
- Compare with ground truth structure when possible to identify structural errors

### Rule 7: Option Nodes Must Point to Final Answer (CRITICAL - Edges)
- Any node that mentions or evaluates an option (A/B/C/D, "option", "choice", etc.) MUST have an edge pointing to the final answer node that states the model's selected option
- If the model's final answer is option C, all option-related nodes must point to the final answer node for option C
- Missing edge or pointing to a different option's final answer is a structural error

---

**Output Format** (JSON only, no markdown):
{{
  "overall_quality": "excellent|good|fair|poor",
  "total_matches": <number>,
  "problematic_matches": [
    {{
      "h_id": "H1",
      "r_id": "R1",
      "issue_type": "final_answer_mismatch|structure_mismatch|semantic_mismatch|numerical_mismatch|vague_vs_specific|other",
      "severity": "critical|major|minor",
      "explanation": "Detailed explanation of why this match is problematic",
      "should_be_null": true/false
    }}
  ],
  "missing_matches": [
    {{
      "h_id": "H2",
      "potential_r_id": "R3",
      "explanation": "Why this should have been matched but wasn't"
    }}
  ],
  "false_null_matches": [
    {{
      "h_id": "H3",
      "explanation": "This node was marked as null but could have matched an r_id"
    }}
  ],
  "edge_issues": [
    {{
      "issue_type": "missing_edge|extra_edge|wrong_direction",
      "from_h_id": "H1",
      "to_h_id": "H2",
      "explanation": "Why this edge is problematic",
      "suggested_action": "add|remove|reverse"
    }}
  ],
  "hallucination_issues": [
    {{
      "type": "node_content|match_assignment|edge_structure|other",
      "description": "Description of hallucination or rule violation"
    }}
  ],
  "summary": "Overall assessment of match quality, edge structure, and adherence to rules"
}}

Output:
