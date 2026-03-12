#!/usr/bin/env python3
"""
Validate extracted DAGs before analysis to avoid runtime errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, Iterable, List, Set, Tuple


NODE_ID_RE = re.compile(r"^H\d+$")


def _is_node_id(value: Any) -> bool:
    return isinstance(value, str) and NODE_ID_RE.match(value) is not None


def _int_id(value: str) -> int:
    return int(value[1:])


def validate_graph(graph: Dict[str, Any], require_matches: bool = True) -> List[str]:
    errors: List[str] = []

    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0:
        errors.append("nodes must be a non-empty list")
        nodes = []

    node_ids: Set[str] = set()
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{idx}] must be an object")
            continue
        nid = node.get("id")
        if not _is_node_id(nid):
            errors.append(f"nodes[{idx}].id must match H<integer>, got {nid!r}")
            continue
        if nid in node_ids:
            errors.append(f"duplicate node id {nid!r}")
            continue
        node_ids.add(nid)

    edges = graph.get("edges")
    if not isinstance(edges, list):
        errors.append("edges must be a list")
        edges = []

    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edges[{idx}] must be an object")
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if not _is_node_id(src):
            errors.append(f"edges[{idx}].from must be a node id, got {src!r}")
        if not _is_node_id(dst):
            errors.append(f"edges[{idx}].to must be a node id, got {dst!r}")
        if _is_node_id(src) and src not in node_ids:
            errors.append(f"edges[{idx}].from refers to missing node {src!r}")
        if _is_node_id(dst) and dst not in node_ids:
            errors.append(f"edges[{idx}].to refers to missing node {dst!r}")
        if src == dst and _is_node_id(src):
            errors.append(f"edges[{idx}] has self-loop on {src!r}")

    if node_ids:
        out_degree = {nid: 0 for nid in node_ids}
        for edge in edges:
            src = edge.get("from")
            if _is_node_id(src) and src in out_degree:
                out_degree[src] += 1
        zero_out = [nid for nid, d in out_degree.items() if d == 0]
        if len(zero_out) == 0:
            errors.append("no zero-out-degree node found")
        else:
            # Ensure IDs are sortable by int suffix (H1, H2...)
            try:
                _ = sorted(zero_out, key=_int_id)
            except Exception as exc:
                errors.append(f"zero-out-degree node ids not sortable: {exc}")

    if require_matches:
        matches = graph.get("matches")
        if not isinstance(matches, list):
            errors.append("matches must be a list")
        else:
            for idx, match in enumerate(matches):
                if not isinstance(match, dict):
                    errors.append(f"matches[{idx}] must be an object")
                    continue
                if "h_id" not in match:
                    errors.append(f"matches[{idx}] missing h_id")
                if "r_id" not in match:
                    errors.append(f"matches[{idx}] missing r_id")

    # Optional DAG cycle check
    if node_ids and edges:
        if _has_cycle(node_ids, edges):
            errors.append("graph contains a cycle (not a DAG)")

    return errors


def _has_cycle(node_ids: Set[str], edges: Iterable[Dict[str, Any]]) -> bool:
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if _is_node_id(src) and _is_node_id(dst) and src in adj:
            adj[src].append(dst)

    visiting: Set[str] = set()
    visited: Set[str] = set()

    def dfs(nid: str) -> bool:
        if nid in visiting:
            return True
        if nid in visited:
            return False
        visiting.add(nid)
        for nxt in adj.get(nid, []):
            if dfs(nxt):
                return True
        visiting.remove(nid)
        visited.add(nid)
        return False

    for nid in node_ids:
        if dfs(nid):
            return True
    return False


def validate_jsonl(path: str, max_errors: int | None = None) -> Tuple[int, List[Dict[str, Any]]]:
    total_errors = 0
    error_records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as exc:
                total_errors += 1
                error_records.append({
                    "line": line_num,
                    "uuid": None,
                    "errors": [f"invalid json: {exc}"],
                })
                if max_errors and total_errors >= max_errors:
                    break
                continue

            uuid = data.get("uuid")
            graph = {
                "nodes": data.get("nodes"),
                "edges": data.get("edges"),
                "matches": data.get("matches"),
            }
            errors = validate_graph(graph, require_matches=True)
            if errors:
                total_errors += len(errors)
                error_records.append({
                    "line": line_num,
                    "uuid": uuid,
                    "errors": errors,
                })
                if max_errors and total_errors >= max_errors:
                    break

    return total_errors, error_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DAG JSONL for analysis safety.")
    parser.add_argument("--input", required=True, help="Path to JSONL containing DAGs")
    parser.add_argument("--max-errors", type=int, default=None, help="Stop after this many errors")
    parser.add_argument("--output", type=str, default=None, help="Write error report JSONL to file")
    args = parser.parse_args()

    total_errors, error_records = validate_jsonl(args.input, args.max_errors)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for record in error_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Checked: {args.input}")
    print(f"Errors: {total_errors}")
    print(f"Invalid records: {len(error_records)}")

    if error_records:
        preview = error_records[:3]
        print("Sample errors:")
        for rec in preview:
            print(json.dumps(rec, ensure_ascii=False))

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
