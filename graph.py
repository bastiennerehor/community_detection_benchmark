import igraph as ig

from utils import compute_minus_log_p


def build_edge_dict(selected_links, edge_weight="log-raw-P"):
    selectors = {
        "abs-raw-E": lambda e: abs(e["raw-E"]), # oder "abs-LS"
        "1-raw-P": lambda e: 1-e["raw-P"],
        "abs-probit-E": lambda e: abs(e["probit-E"]), #oder "abs-probit-LS"
        "abs-std-E": lambda e: abs(e["std-E"]), #oder "abs-std-LS"
        "-log-raw-P": lambda e: compute_minus_log_p(e["raw-P"]),
        "-logp_e_abs_raw": lambda e: compute_minus_log_p(e["raw-P"])*abs(e["raw-E"]),
        "-logp_abs_probit-E": lambda e: compute_minus_log_p(e["raw-P"])*abs(e["probit-E"]),
        "-logp_abs_std-E": lambda e: compute_minus_log_p(e["raw-P"])*abs(e["std-E"]),
    }

    if edge_weight not in selectors:
        raise ValueError(f"Invalid edge_weight '{edge_weight}'")

    select = selectors[edge_weight]
    edge_map = {}

    for _, row in selected_links.iterrows():
        s = row["label1"]
        t = row["label2"]

        if s is None or t is None:
            continue

        key = tuple(sorted((str(s), str(t))))

        if key in edge_map:
            raise ValueError(f"Duplicate edge detected for pair {key}")

        try:
            value = select(row)
        except KeyError as e:
            raise ValueError(f"Missing required field for mode '{edge_weight}': {e}")

        edge_map[key] = value

    return edge_map

def build_weighted_graph(selected_links, used_node_ids, edge_weight="log-raw-P"):
    sorted_nodes = sorted(map(str, used_node_ids))
    node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}

    edge_map = build_edge_dict(selected_links, edge_weight=edge_weight)

    edges = []
    weights = []

    for (s, t), w in edge_map.items():
        if s not in node_to_idx or t not in node_to_idx:
            continue
        edges.append((node_to_idx[s], node_to_idx[t]))
        weights.append(w)

    graph = ig.Graph(n=len(sorted_nodes), edges=edges, directed=False)
    graph.vs["name"] = sorted_nodes
    graph.es["weight"] = weights

    return graph
