"""Original weighted RPF and topology formulas with explicit input validation."""
from collections import defaultdict
import math
import networkx as nx

FORMULA_VERSION = 'original-rpf-v1'


def checked_graph(graph, *, allow_empty=False):
    if not isinstance(graph, dict):
        raise ValueError('Graph must be an object')
    nodes, edges = graph.get('nodes'), graph.get('edges')
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError('Graph requires nodes and edges lists')
    if not nodes and not allow_empty:
        raise ValueError('Reference graph must not be empty')
    ids = []
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get('id'), str) or not node['id']:
            raise ValueError('Node requires a nonempty string id')
        ids.append(node['id'])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate node id')
    pairs = []
    for edge in edges:
        if not isinstance(edge, dict) or edge.get('from') not in ids or edge.get('to') not in ids:
            raise ValueError('Edge refers to an unknown node')
        pairs.append((edge['from'], edge['to']))
    g = nx.DiGraph()
    g.add_nodes_from(ids)
    g.add_edges_from(pairs)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError('Graph contains a cycle')
    return g


def graph_metrics(gt, answer):
    """Use the released formula; edges are tested via directed reachability.

    An empty answer DAG is valid for explicit failures and receives zero.
    Malformed nonempty graphs raise, rather than silently changing scores.
    Multiple H nodes per R node are allowed; repeated matches cannot add points.
    """
    reference = checked_graph(gt)
    predicted = checked_graph(answer, allow_empty=True)
    weights = {}
    for node in gt['nodes']:
        weight = float(node.get('points', 1))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('GT points must be finite and nonnegative')
        weights[node['id']] = weight
    matches = answer.get('matches', [])
    if not isinstance(matches, list):
        raise ValueError('matches must be a list')
    mapping = defaultdict(set)
    for match in matches:
        if not isinstance(match, dict) or 'r_id' not in match or match.get('h_id') not in predicted:
            raise ValueError('Match refers to an unknown GT or answer node')
        if match['r_id'] is None:
            continue  # Explicitly unmatched answer node in the historical format.
        if match['r_id'] not in reference:
            raise ValueError('Match refers to an unknown GT node')
        mapping[match['r_id']].add(match['h_id'])
    closure = nx.transitive_closure_dag(predicted)
    maximum = sum(weights.values())
    node_total = full_total = 0.0
    for rid, points in weights.items():
        if not mapping[rid]:
            continue
        node_total += points
        parents = list(reference.predecessors(rid))
        logic = (sum(any(closure.has_edge(hp, hc) for hp in mapping[parent]
                         for hc in mapping[rid]) for parent in parents) / len(parents)) if parents else 1.0
        full_total += points * logic
    node_only = node_total / maximum if maximum else 0.0
    rpf = full_total / maximum if maximum else 0.0
    degrees = [d for _, d in predicted.out_degree()]
    return {
        'rpf': rpf, 'node_only': node_only, 'logic_penalty': node_only - rpf,
        'gt_nodes': len(reference), 'gt_edges': reference.number_of_edges(),
        'llm_nodes': len(predicted), 'llm_edges': predicted.number_of_edges(),
        'branching_factor': sum(max(0, d - 1) for d in degrees) / len(predicted) if predicted else 0.0,
        'dangling_count': max(0, sum(d == 0 for d in degrees) - 1),
    }
