import json
import importlib
from math import ceil
import os
import time
import timeit
from datetime import datetime
from collections import defaultdict
from pathlib import Path


from modina import probit_rescaling
import pandas as pd
import numpy as np
from rdflib import logger
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
import asyncio
import leidenalg
import graph_tool.all as gt
from modina.context_net_inference import compute_context_scores
from modina.edge_filtering import filter_single, std_rescaling
import igraph as ig
from environs import Env

env = Env()

env.read_env()

types = ["protein", "metabolite", "phenotype", "variant"]  # "disorders", "genes"
layers_to_source_table = {
    "proteomics": "cohort_protein",
    "metabolomics": "cohort_metabolite",
    "phenomics": "cohort_phenotype",
    "variants": "cohort_variant"
}

DEFAULT_LEIDEN_RESOLUTIONS = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
SUPPORTED_COMMUNITY_METHODS = ('louvain', 'leiden', 'recursive_leiden', 'agglomerative', 'infomap', 'hierarchical_infomap', 'hsbm')
SUPPORTED_WEIGHT_OPTIONS = ('abs-raw-E','log-raw-P', '-logp_e_abs_raw', 'abs-probit-E', 'abs-std-E', '-logp_abs_probit-E', '-logp_abs_std-E')
BENCHMARK_DEFAULT_DENSITY = 0.1
BENCHMARK_DEFAULT_EDGE_WEIGHT = '-logp_e_abs_raw'


def resolution_to_key(value):
    """Normalize resolution keys so 1.0 and 3.0 keep one decimal."""
    text = f"{float(value):.6f}".rstrip('0').rstrip('.')
    if '.' not in text:
        text = f"{text}.0"
    return text


def parse_resolution_values(request):
    """
    Parse comma-separated Leiden resolutions from `resolutions` query parameter.
    Falls back to the default slider-friendly resolution set.
    """
    raw = request.GET.get('resolutions', None)
    if raw in ['', 'null', None]:
        return DEFAULT_LEIDEN_RESOLUTIONS

    try:
        values = [float(part.strip()) for part in str(raw).split(',') if part.strip()]
    except ValueError:
        raise ValueError("resolutions must be a comma-separated list of numbers.")

    if not values:
        raise ValueError('resolutions must contain at least one value.', status=405)

    for value in values:
        if value <= 0:
            raise ValueError('all resolution values must be > 0.', status=405)

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for value in values:
        key = resolution_to_key(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped

def compute_minus_log_p(p_value):

    if p_value is None:
        return 0.0

    if p_value <= 0:
        # A p-value of exactly 0 is the strongest possible signal, not the
        # weakest. Clamp to the smallest representable positive float so
        # -log10(p_value) stays a large finite number instead of dropping
        # to 0 (or blowing up to inf).
        p_value = np.finfo(float).tiny

    return -np.log10(p_value)

def build_edge_dict(selected_links, edge_weight="log-raw-P"):
    selectors = {
        "abs-raw-E": lambda e: abs(e["raw-E"]), # oder "abs-LS"
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

    # using pandas for aggregation (fast and readable)
# TODO: do faster graph building??
# df = pd.DataFrame(selected_links)  # must have 'source','target', optional 'weight'
# df['source'] = df['source'].astype(str)
# df['target'] = df['target'].astype(str)
# # normalize undirected pair
# df[['u','v']] = pd.DataFrame(
#     np.sort(df[['source','target']].values, axis=1),
#     index=df.index
# )
# # choose aggregation policy: sum / max / existence
# df['weight'] = df.get('weight').fillna(1.0)
# agg = df.groupby(['u','v'], sort=False, as_index=False)['weight'].sum()

# triples = list(agg.itertuples(index=False, name=None))  # (u,v,weight)
# g = ig.Graph.TupleList(triples, directed=False, weights=True, vertex_name_attr='name')


def communities_from_mapping(node_to_community):
    grouped = defaultdict(set)
    for node_id, community_id in node_to_community.items():
        grouped[community_id].add(node_id)
    return [members for _, members in sorted(grouped.items(), key=lambda item: item[0]) if members]


def compute_modularity(graph, node_to_community):
    if graph.vcount() == 0:
        return 0.0

    try:
        membership = [node_to_community.get(vertex['name'], idx) for idx, vertex in enumerate(graph.vs)]
        if graph.ecount() == 0:
            return 0.0
        return float(graph.modularity(membership, weights=graph.es['weight']))
    except Exception:
        return 0.0


def compute_conductance(graph, node_to_community):
    if graph.vcount() == 0 or graph.ecount() == 0 or not node_to_community:
        return 0.0

    try:
        membership = [node_to_community.get(vertex['name'], idx) for idx, vertex in enumerate(graph.vs)]
        if not membership:
            return 0.0

        weights = graph.es['weight'] if 'weight' in graph.es.attributes() else None
        strengths = graph.strength(weights=weights)
        total_strength = float(sum(strengths))
        if total_strength <= 0:
            return 0.0

        community_vertices = defaultdict(list)
        for vertex_index, community_id in enumerate(membership):
            community_vertices[community_id].append(vertex_index)

        boundary_weights = defaultdict(float)
        for edge in graph.es:
            source = int(edge.source)
            target = int(edge.target)
            if membership[source] == membership[target]:
                continue
            weight = float(edge['weight']) if 'weight' in edge.attributes() else 1.0
            boundary_weights[membership[source]] += weight
            boundary_weights[membership[target]] += weight

        conductances = []
        for community_id, vertices in community_vertices.items():
            community_volume = float(sum(strengths[index] for index in vertices))
            other_volume = total_strength - community_volume
            denominator = min(community_volume, other_volume)
            if denominator <= 0:
                continue
            conductances.append(boundary_weights.get(community_id, 0.0) / denominator)

        if not conductances:
            return 0.0

        return float(sum(conductances) / len(conductances))
    except Exception:
        return 0.0


def _assign_membership(graph, membership):
    return {graph.vs[idx]['name']: community_id for idx, community_id in enumerate(membership)}


def _membership_vector(graph, node_to_community):
    return [node_to_community.get(vertex['name'], idx) for idx, vertex in enumerate(graph.vs)]


def _community_family_name(method_name):
    method_name = (method_name or '').strip().lower()
    if method_name.startswith('hsbm_level_'):
        return 'hsbm'
    if method_name.startswith('agglomerative_cut_'):
        return 'agglomerative'
    if method_name.startswith('recursive_leiden_level_'):
        return 'recursive_leiden'
    if method_name.startswith('hierarchical_infomap_level_'):
        return 'hierarchical_infomap'
    return method_name


def _select_best_rows_for_comparison(benchmark_df):
    if benchmark_df.empty:
        return pd.DataFrame(columns=benchmark_df.columns)

    temp = benchmark_df.copy()
    temp['family'] = temp['method'].map(_community_family_name)

    selected_rows = []
    for _, group in temp.groupby('family', sort=False):
        best_idx = group['modularity'].astype(float).idxmax()
        selected_rows.append(temp.loc[best_idx])

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _build_pairwise_comparison_table(graph, benchmark_df, benchmark_assignments):
    # for hierarchical methods take the best modularity level to represent the family in the pairwise comparison table
    selected_rows = _select_best_rows_for_comparison(benchmark_df)
    if selected_rows.empty:
        return pd.DataFrame(columns=['comparison'])

    selected_entries = []
    for _, row in selected_rows.iterrows():
        selected_entries.append({
            'method': row['method'],
            'family': _community_family_name(row['method']),
            'display_name': row['method'] if _community_family_name(row['method']) == row['method'] else f"{_community_family_name(row['method'])} (best modularity)",
            'modularity': float(row['modularity']) if row['modularity'] is not None else 0.0,
            'community_count': int(row['community_count']) if row['community_count'] is not None else 0,
        })

    labels = [entry['display_name'] for entry in selected_entries]
    memberships = {
        entry['method']: _membership_vector(graph, benchmark_assignments[entry['method']])
        for entry in selected_entries
    }

    rows = []
    for left in selected_entries:
        left_label = left['display_name']
        left_membership = memberships[left['method']]
        for metric_name, scorer in (
            ('RI', rand_score),
            ('ARI', adjusted_rand_score),
            ('NMI', normalized_mutual_info_score),
        ):
            row = {'comparison': f'{metric_name} {left_label}'}
            for right in selected_entries:
                right_label = right['display_name']
                right_membership = memberships[right['method']]
                row[right_label] = round(float(scorer(left_membership, right_membership)), 4)
            rows.append(row)

    comparison_df = pd.DataFrame(rows)
    ordered_columns = ['comparison'] + labels
    comparison_df = comparison_df.loc[:, ordered_columns]
    return comparison_df


def _singleton_membership(graph):
    return list(range(graph.vcount()))

def _run_leiden_clustering(graph, resolution=1.0, seed=42):
    """Run Leiden on an igraph graph."""
    if graph.vcount() == 0:
        return {}, 'none'
    
    if graph.ecount() == 0:
        return _assign_membership(graph, _singleton_membership(graph)), 'leiden'

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es['weight'] if graph.ecount() else None,
        resolution_parameter=resolution,
        seed=seed,
    )

    return _assign_membership(graph, partition.membership), 'leiden'

def _run_louvain_clustering(graph):
    if graph.vcount() == 0:
        return {}, 'none'

    if graph.ecount() == 0:
        return _assign_membership(graph, _singleton_membership(graph)), 'louvain'

    communities = graph.community_multilevel(weights=graph.es['weight'])
    return _assign_membership(graph, communities.membership), 'louvain'

def _run_recursive_leiden_clustering(graph, resolution=1.0, seed=42, max_depth=4, min_cluster_size=4):
    """Return the deepest recursive Leiden split together with all level summaries."""
    if graph.vcount() == 0:
        return {}, 'none', []

    level_summaries = _summarize_recursive_leiden_levels(
        graph,
        resolution=resolution,
        seed=seed,
        max_depth=max_depth,
        min_cluster_size=min_cluster_size,
    )
    if not level_summaries:
        return _assign_membership(graph, _singleton_membership(graph)), 'recursive_leiden', []

    return level_summaries[-1]['node_to_community'], 'recursive_leiden', level_summaries

def _summarize_recursive_leiden_levels(graph, resolution=1.0, seed=42, max_depth=4, min_cluster_size=4):
    """Recursively split the graph with Leiden and report each level summary."""
    if graph.vcount() == 0:
        return []

    level_summaries = []

    current_clusters = [list(graph.vs['name'])]
    total_levels = 0

    for depth in range(max_depth):
        next_clusters = []
        node_to_community = {}
        split_occurred = False

        for cluster_names in current_clusters:
            indices = [graph.vs.find(name=node_id).index for node_id in cluster_names]
            subgraph = graph.subgraph(indices)

            if subgraph.vcount() <= min_cluster_size or subgraph.ecount() == 0:
                for node_id in cluster_names:
                    node_to_community[node_id] = len(next_clusters)
                next_clusters.append(cluster_names)
                continue

            partition = leidenalg.find_partition(
                subgraph,
                leidenalg.RBConfigurationVertexPartition,
                weights=subgraph.es['weight'] if subgraph.ecount() else None,
                resolution_parameter=resolution,
                seed=seed,
            )
            membership = list(partition.membership)

            unique_communities = sorted(set(membership))
            if len(unique_communities) <= 1:
                for node_id in cluster_names:
                    node_to_community[node_id] = len(next_clusters)
                next_clusters.append(cluster_names)
                continue

            split_occurred = True
            for community_id in unique_communities:
                community_names = [subgraph.vs[idx]['name'] for idx, comm in enumerate(membership) if comm == community_id]
                for node_id in community_names:
                    node_to_community[node_id] = len(next_clusters)
                next_clusters.append(community_names)

        if not node_to_community:
            break

        size_map = {}
        for comm in node_to_community.values():
            size_map[comm] = size_map.get(comm, 0) + 1

        level_summaries.append({
            'level': depth,
            'node_to_community': node_to_community,
            'community_count': len(size_map),
            'modularity': round(compute_modularity(graph, node_to_community), 6),
            'communities': sorted(size_map.values(), reverse=True),
            'hierarchy_label': f'level {depth + 1}',
            'hierarchy_depth': max_depth,
        })

        total_levels = depth + 1
        current_clusters = next_clusters
        if not split_occurred:
            break

    for level_summary in level_summaries:
        level_summary['hierarchy_depth'] = total_levels
    return level_summaries

def _run_infomap_clustering(graph, trials=10):
    """Run flat Infomap via python-igraph's community_infomap (if available)."""
    if graph.vcount() == 0:
        return {}, 'none'

    if graph.ecount() == 0:
        return _assign_membership(graph, _singleton_membership(graph)), 'infomap'

    communities = graph.community_infomap(edge_weights=graph.es['weight'] if graph.ecount() else None, trials=trials)
    return _assign_membership(graph, communities.membership), 'infomap'


def _summarize_infomap_levels(graph, seed=42, trials=10):
    """Run the official Infomap bindings and return per-level hierarchy summaries."""
    if graph.vcount() == 0:
        return []

    try:
        from infomap import Infomap
    except ImportError as exc:
        raise ImportError(
            'The `infomap` package is required for hierarchical Infomap. Install it in the active environment.'
        ) from exc

    infomap_instance = Infomap(
        seed=seed,
        silent=True,
        no_file_output=True,
        num_trials=trials,
        two_level=False,
    )

    for edge in graph.es:
        source_id = int(edge.source)
        target_id = int(edge.target)
        weight = float(edge['weight']) if 'weight' in edge.attributes() else 1.0
        infomap_instance.addLink(source_id, target_id, weight)

    infomap_instance.run()

    multilevel_modules = infomap_instance.get_multilevel_modules()
    if not multilevel_modules:
        return []

    max_depth = max(len(path) for path in multilevel_modules.values())
    level_summaries = []
    for level_index in range(max_depth):
        membership = []
        for vertex in graph.vs:
            path = multilevel_modules.get(vertex.index, ())
            if not path:
                membership.append(0)
                continue
            prefix = path[: level_index + 1] if level_index < len(path) else path
            membership.append(tuple(prefix))

        # Make tuple prefixes hashable community labels in a compact integer space.
        prefix_to_id = {}
        compact_membership = []
        for prefix in membership:
            if prefix not in prefix_to_id:
                prefix_to_id[prefix] = len(prefix_to_id)
            compact_membership.append(prefix_to_id[prefix])

        node_to_community = _assign_membership(graph, compact_membership)

        size_map = {}
        for community_id in compact_membership:
            size_map[community_id] = size_map.get(community_id, 0) + 1

        level_summaries.append({
            'level': level_index,
            'node_to_community': node_to_community,
            'community_count': len(size_map),
            'modularity': round(compute_modularity(graph, node_to_community), 6),
            'communities': sorted(size_map.values(), reverse=True),
            'hierarchy_label': f'level {level_index + 1} of {max_depth}',
            'hierarchy_depth': max_depth,
        })

    return level_summaries


def _run_hierarchical_infomap_clustering(graph, seed=42, trials=10):
    """Run the official Infomap hierarchy and return the deepest-level assignment with all level summaries."""
    if graph.vcount() == 0:
        return {}, 'none', []

    level_summaries = _summarize_infomap_levels(graph, seed=seed, trials=trials)
    if not level_summaries:
        return _assign_membership(graph, _singleton_membership(graph)), 'hierarchical_infomap', []

    return level_summaries[-1]['node_to_community'], 'hierarchical_infomap', level_summaries

def _run_hsbm_clustering_with_levels(graph, seed=42):
    """Run a real hierarchical stochastic block model using graph-tool."""
    if graph.vcount() == 0:
        return {}, 'none', []

    edges = np.array([(e.source, e.target) for e in graph.es], dtype="int32")
    weights = np.array([e["weight"] if "weight" in e.attributes() else 1.0 for e in graph.es], dtype="float64")

    gt_graph = gt.Graph(directed=False)
    gt_graph.add_edge_list(edges)
    gt_graph.ep.weight = gt_graph.new_edge_property("double")
    gt_graph.ep.weight.a = weights  # FAST bulk assignment
    gt_graph.vp.name = gt_graph.new_vertex_property('string')

    for vertex in graph.vs:
        gt_graph.vp.name[vertex.index] = vertex["name"]

    state = gt.minimize_nested_blockmodel_dl(gt_graph, state_args={'deg_corr': True})
    blocks = state.get_bs()[0]  # lowest level keeps the existing flat clustering behaviour.
    membership = [int(blocks[v]) for v in gt_graph.vertices()]
    hierarchy_levels = _summarize_hsbm_levels(graph, gt_graph, state)
    return _assign_membership(graph, membership), 'hsbm', hierarchy_levels

def _summarize_hsbm_levels(graph, gt_graph, state):
    """Return per-level community counts and modularity for the nested HSBM tree."""
    level_summaries = []
    for level_index, blocks in enumerate(state.get_bs()):
        projected_state = state.project_level(level_index)
        membership = [int(block_id) for block_id in projected_state.get_blocks()]
        node_to_community = {
            graph.vs[idx]['name']: membership[idx]
            for idx in range(len(membership))
        }

        size_map = {}
        for comm in membership:
            size_map[comm] = size_map.get(comm, 0) + 1

        level_summaries.append({
            'level': level_index,
            'node_to_community': node_to_community,
            'community_count': len(set(membership)) if membership else 0,
            'modularity': round(compute_modularity(graph, node_to_community), 6),
            'communities': sorted(size_map.values(), reverse=True),
        })

        # Once the projection collapses to a single community, coarser levels are
        # redundant for benchmarking because they will remain identical.
        if level_summaries[-1]['community_count'] <= 1:
            break

    return level_summaries

def _run_agglomerative_clustering(graph):
    """Return the finest meaningful agglomerative cut together with all level summaries."""
    if graph.vcount() == 0:
        return {}, 'none', []

    level_summaries = _summarize_agglomerative_levels(graph)
    if not level_summaries:
        return _assign_membership(graph, _singleton_membership(graph)), 'agglomerative', []

    return level_summaries[-1]['node_to_community'], 'agglomerative', level_summaries

def _summarize_agglomerative_levels(graph):
    """Return the meaningful coarser-side cuts for the agglomerative merge tree."""
    if graph.vcount() == 0:
        return []

    if graph.ecount() == 0:
        membership = _singleton_membership(graph)
        return [{
            'level': 0,
            'node_to_community': _assign_membership(graph, membership),
            'community_count': len(set(membership)),
            'modularity': 0.0,
            'communities': [1] * graph.vcount(),
            'hierarchy_label': 'cut 1 communities',
            'hierarchy_depth': 1,
            'cut_count': 1,
        }]

    dendrogram = graph.community_fastgreedy(weights=graph.es['weight'])
    optimal_count = int(dendrogram.optimal_count)

    # For agglomerative clustering, cuts above the modularity-optimal partition
    # quickly turn into singleton-heavy refinements. Benchmark only the coarser
    # side up to the optimal cut so the rows stay meaningful.
    cut_counts = list(range(1, optimal_count + 1))

    level_summaries = []
    total_levels = len(cut_counts)

    for level_index, cluster_count in enumerate(cut_counts):
        min_communities = graph.vcount() - len(dendrogram.merges)
        if cluster_count < min_communities:
            logger.info(f"Skipping cut {cluster_count} because it produces fewer than {min_communities} communities. Inspect: Graph is connected: {graph.is_connected()}. Connected components: {len(graph.connected_components())}.")
            continue
        clustering = dendrogram.as_clustering(n=cluster_count)
        membership = list(clustering.membership)
        node_to_community = _assign_membership(graph, membership)

        size_map = {}
        for comm in membership:
            size_map[comm] = size_map.get(comm, 0) + 1

        level_summaries.append({
            'level': level_index,
            'node_to_community': node_to_community,
            'community_count': len(set(membership)) if membership else 0,
            'modularity': round(compute_modularity(graph, node_to_community), 6),
            'communities': sorted(size_map.values(), reverse=True),
            'hierarchy_label': f'cut {cluster_count} communities',
            'hierarchy_depth': total_levels,
            'cut_count': cluster_count,
        })

    return level_summaries


def run_community_clustering(graph, method='leiden', resolution=1.0, seed=42):
    method = (method or 'leiden').strip().lower()

    if method == 'leiden':
        return _run_leiden_clustering(graph, resolution=resolution, seed=seed)
    if method == 'louvain':
        return _run_louvain_clustering(graph)
    if method == 'infomap':
        return _run_infomap_clustering(graph)

    raise ValueError(f"Unsupported community detection method '{method}'. Choose from: {', '.join(SUPPORTED_COMMUNITY_METHODS)}.")

def benchmark_community_detection(selected_links, used_node_ids, methods=None, resolution=1.0, edge_weight=None, seed=42):
    graph = build_weighted_graph(selected_links, used_node_ids, edge_weight)
    methods = methods or SUPPORTED_COMMUNITY_METHODS
    node_count = graph.vcount()
    edge_count = graph.ecount()

    benchmark_rows = []
    benchmark_assignments = {}
    for method in methods:
        start = time.perf_counter()
        if method == 'hsbm':
            node_to_community, algorithm_name, hierarchy_levels = _run_hsbm_clustering_with_levels(
                graph,
                seed=seed,
            )
            runtime = time.perf_counter() - start

            total_levels = len(hierarchy_levels)
            for level_summary in hierarchy_levels:
                level_number = level_summary['level'] + 1
                row = {
                    'method': f'hsbm_level_{level_number}_of_{total_levels}',
                    'algorithm': algorithm_name,
                    'hierarchy_level': level_number,
                    'hierarchy_depth': total_levels,
                    'hierarchy_label': f'level {level_number} of {total_levels}',
                    'runtime_seconds': round(runtime, 6),
                    'modularity': level_summary['modularity'],
                    'conductance': round(compute_conductance(graph, level_summary['node_to_community']), 6),
                    'community_count': level_summary['community_count'],
                    'node_count': node_count,
                    'edge_count': edge_count,
                    # JSON-encode the community id/size list so it fits cleanly into CSV
                    'communities': json.dumps(level_summary['communities']),
                }
                benchmark_rows.append(row)
                benchmark_assignments[row['method']] = level_summary['node_to_community']
                print(
                    f"{row['method']}: runtime={row['runtime_seconds']:.6f}s modularity={row['modularity']:.6f}"
                )

            continue
        if method == 'agglomerative':
            node_to_community, algorithm_name, hierarchy_levels = _run_agglomerative_clustering(graph)
            runtime = time.perf_counter() - start

            total_levels = len(hierarchy_levels)
            for level_summary in hierarchy_levels:
                level_number = level_summary['level'] + 1
                row = {
                    'method': f'agglomerative_cut_{level_summary["cut_count"]}',
                    'algorithm': algorithm_name,
                    'hierarchy_level': level_number,
                    'hierarchy_depth': total_levels,
                    'hierarchy_label': level_summary['hierarchy_label'],
                    'runtime_seconds': round(runtime, 6),
                    'modularity': level_summary['modularity'],
                    'conductance': round(compute_conductance(graph, level_summary['node_to_community']), 6),
                    'community_count': level_summary['community_count'],
                    'node_count': node_count,
                    'edge_count': edge_count,
                    'communities': json.dumps(level_summary['communities']),
                }
                benchmark_rows.append(row)
                benchmark_assignments[row['method']] = level_summary['node_to_community']
                print(
                    f"{row['method']}: runtime={row['runtime_seconds']:.6f}s modularity={row['modularity']:.6f}"
                )

            continue
        if method == 'hierarchical_infomap':
            node_to_community, algorithm_name, hierarchy_levels = _run_hierarchical_infomap_clustering(graph, seed=seed)
            runtime = time.perf_counter() - start

            total_levels = len(hierarchy_levels)
            for level_summary in hierarchy_levels:
                level_number = level_summary['level'] + 1
                row = {
                    'method': f'hierarchical_infomap_level_{level_number}_of_{total_levels}',
                    'algorithm': algorithm_name,
                    'hierarchy_level': level_number,
                    'hierarchy_depth': total_levels,
                    'hierarchy_label': level_summary['hierarchy_label'],
                    'runtime_seconds': round(runtime, 6),
                    'modularity': level_summary['modularity'],
                    'conductance': round(compute_conductance(graph, level_summary['node_to_community']), 6),
                    'community_count': level_summary['community_count'],
                    'node_count': node_count,
                    'edge_count': edge_count,
                    'communities': json.dumps(level_summary['communities']),
                }
                benchmark_rows.append(row)
                benchmark_assignments[row['method']] = level_summary['node_to_community']
                print(
                    f"{row['method']}: runtime={row['runtime_seconds']:.6f}s modularity={row['modularity']:.6f}"
                )

            continue
        if method == 'recursive_leiden':
            node_to_community, algorithm_name, hierarchy_levels = _run_recursive_leiden_clustering(
                graph,
                resolution=resolution,
                seed=seed,
            )
            runtime = time.perf_counter() - start

            total_levels = len(hierarchy_levels)
            for level_summary in hierarchy_levels:
                level_number = level_summary['level'] + 1
                row = {
                    'method': f'recursive_leiden_level_{level_number}_of_{total_levels}',
                    'algorithm': algorithm_name,
                    'hierarchy_level': level_number,
                    'hierarchy_depth': total_levels,
                    'hierarchy_label': level_summary['hierarchy_label'],
                    'runtime_seconds': round(runtime, 6),
                    'modularity': level_summary['modularity'],
                    'conductance': round(compute_conductance(graph, level_summary['node_to_community']), 6),
                    'community_count': level_summary['community_count'],
                    'node_count': node_count,
                    'edge_count': edge_count,
                    'communities': json.dumps(level_summary['communities']),
                }
                benchmark_rows.append(row)
                benchmark_assignments[row['method']] = level_summary['node_to_community']
                print(
                    f"{row['method']}: runtime={row['runtime_seconds']:.6f}s modularity={row['modularity']:.6f}"
                )

            continue
        else:
            node_to_community, algorithm_name = run_community_clustering(
                graph=graph,
                method=method,
                resolution=resolution,
                seed=seed,
            )
            runtime = time.perf_counter() - start
        modularity = compute_modularity(graph, node_to_community)
        community_count = len(set(node_to_community.values())) if node_to_community else 0

        # Build a compact representation: ordered community sizes (largest first)
        if node_to_community:
            size_map = {}
            for comm in node_to_community.values():
                size_map[comm] = size_map.get(comm, 0) + 1
            communities_with_size = sorted(size_map.values(), reverse=True)
        else:
            communities_with_size = []

        row = {
            'method': method,
            'algorithm': algorithm_name,
            'hierarchy_level': None,
            'hierarchy_depth': None,
            'hierarchy_label': None,
            'runtime_seconds': round(runtime, 6),
            'modularity': round(modularity, 6),
            'conductance': round(compute_conductance(graph, node_to_community), 6),
            'community_count': community_count,
            'node_count': node_count,
            'edge_count': edge_count,
            # JSON-encode the community id/size list so it fits cleanly into CSV
            'communities': json.dumps(communities_with_size),
        }
        benchmark_rows.append(row)
        benchmark_assignments[row['method']] = node_to_community
        print(
            f"{method}: runtime={row['runtime_seconds']:.6f}s modularity={row['modularity']:.6f}"
        )

    return pd.DataFrame(benchmark_rows), benchmark_assignments

def run_benchmark_from_whole_network(
    density=BENCHMARK_DEFAULT_DENSITY,
    edge_weight=BENCHMARK_DEFAULT_EDGE_WEIGHT,
    methods=None,
    resolution=1.0,
    seed=42,
    output_dir='./benchmarking/runs',
):
    # calculate association network
    network_data = pd.read_csv(env("PHENOTYPE_PATH"), sep=",", index_col=env("PATIENT_ID_COLUMN"), low_memory=False)
    if not os.path.exists(env("CALCULATED_EDGES_PATH")+"/"+env("OBSERVATION_SOURCE")+"_scores.csv"):
        print(env("CALCULATED_EDGES_PATH")+env("OBSERVATION_SOURCE")+"_scores.csv")
        logger.info("No pre-calculated edges found, computing context scores from whole network data...")
        network_meta_data = pd.read_csv(env("PHENOTYPE_META_PATH"), sep=",", low_memory=False)
        assoc_scores = compute_context_scores(context_data=network_data, meta_file=network_meta_data, test_type="parametric",
                                        correction= 'bh', num_workers=env.int("NUMBER_OF_WORKERS"),
                            path=env("CALCULATED_EDGES_PATH"), nan_value=env.int("NAN_VALUE"),
                            name=env("OBSERVATION_SOURCE"))
    else:
        logger.info("Pre-calculated edges found, loading from file...")
        assoc_scores = pd.read_csv(env("CALCULATED_EDGES_PATH")+"/"+env("OBSERVATION_SOURCE")+"_scores.csv", sep=",", low_memory=False)

    # filter the network with modina based on density (node degree would also be a possible filtering method for the future)
    filter_method="density"
    filter_param=density

    filtered_assoc_scores, filtered_network_data = filter_single(scores=assoc_scores,
        context=network_data,
        filter_method=filter_method,
        filter_param=filter_param,
        filter_metric= "raw-P",
        path=None)

    # add rescaled Effect size values
    #TODO which to include in weights and how
    filtered_assoc_scores = std_rescaling(scores1=filtered_assoc_scores, scores2=None, metric='std-E')
    filtered_assoc_scores = probit_rescaling(scores1=filtered_assoc_scores, scores2=None, metric='probit-E')

    print(filtered_assoc_scores.head())

    # For logging
    selected_nodes = set(filtered_assoc_scores["label1"]).union(set(filtered_assoc_scores["label2"]))
    selected_node_count = len(selected_nodes)
    selected_edge_count = len(filtered_assoc_scores)
    used_network_node_count = selected_node_count
    used_network_edge_count = selected_edge_count
    possible_edge_count = selected_node_count * (selected_node_count - 1) / 2 if selected_node_count > 1 else 0
    network_density = (selected_edge_count / possible_edge_count) if possible_edge_count else 0.0
    logger.info(
        f"Benchmark network size: {selected_node_count} nodes, {selected_edge_count} edges "
        f"(density: {network_density:.6f})"
    )

    benchmark_df, benchmark_assignments = benchmark_community_detection(
        selected_links=filtered_assoc_scores,
        used_node_ids=selected_nodes,
        methods=methods or SUPPORTED_COMMUNITY_METHODS,
        resolution=resolution,
        edge_weight=edge_weight,
        seed=seed,
    )

    # Only result plotting from here-on

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = Path(output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'benchmarking_cm_detection_{timestamp}.csv'
    output_path = run_output_dir / file_name
    comparison_file_name = f'benchmarking_cm_detection_pairwise_{timestamp}.csv'
    comparison_output_path = run_output_dir / comparison_file_name

    # Shared metadata is written once above the table to keep method rows compact.
    metadata_rows = [
        ('selected_node_count', selected_node_count),
        ('selected_edge_count', selected_edge_count),
        ('used_network_node_count', used_network_node_count),
        ('used_network_edge_count', used_network_edge_count),
        ('network_density', round(network_density, 6)),
        ('edge_weight', edge_weight),
    ]

    table_columns = [
        'method',
        'algorithm',
        'hierarchy_level',
        'hierarchy_depth',
        'hierarchy_label',
        'runtime_seconds',
        'modularity',
        'conductance',
        'community_count',
        'communities',
    ]
    method_table_df = benchmark_df.loc[:, table_columns].copy()

    pairwise_table_df = _build_pairwise_comparison_table(
        graph=build_weighted_graph(filtered_assoc_scores, selected_nodes, edge_weight),
        benchmark_df=benchmark_df,
        benchmark_assignments=benchmark_assignments,
    )

    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write('metric,value\n')
        for key, value in metadata_rows:
            output_file.write(f'{key},{"" if value is None else value}\n')
        output_file.write('\n')
        method_table_df.to_csv(output_file, index=False)

    pairwise_table_df.to_csv(comparison_output_path, index=False)

    try:
        from benchmarking.benchmark_plotting import generate_plots

        benchmark_root_dir = Path(output_dir).resolve().parent
        plots_output_dir = benchmark_root_dir / 'plots' / timestamp

        heatmap_path, metrics_path, runtime_path = generate_plots(
            benchmark_csv=Path(output_path),
            pairwise_csv=Path(comparison_output_path),
            output_dir=plots_output_dir,
        )
        print(f"Pairwise heatmap written to: {heatmap_path}")
        print(f"Modularity/conductance plot written to: {metrics_path}")
        print(f"Runtime plot written to: {runtime_path}")
    except Exception as exc:
        print(f'Plot generation skipped: {exc}')

    print(f"Benchmark results written to: {output_path}")
    print(f"Pairwise comparison results written to: {comparison_output_path}")
    print('Shared benchmark metadata:')
    for key, value in metadata_rows:
        print(f'  {key}: {value}')
    return method_table_df, output_path, pairwise_table_df, comparison_output_path

def summarize_categories(df, cols):
    result = {}

    for col in cols:
        if col not in df.columns:
            result[col] = "Column not found"
            continue

        value_counts = df[col].value_counts(dropna=False)

        result[col] = {
            "n_unique": df[col].nunique(dropna=False),
            "categories": value_counts.index.tolist(),
            "counts": value_counts.to_dict()
        }

    return result

if __name__ == '__main__':
    # # Allows running this module directly for local benchmarking without using the API endpoint.
    # os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dyhealthnet_project.settings')
    # import django

    # django.setup()

    run_benchmark_from_whole_network(
        edge_weight=BENCHMARK_DEFAULT_EDGE_WEIGHT,
        methods=SUPPORTED_COMMUNITY_METHODS,
        resolution=1.0,
        seed=42,
        output_dir='./benchmarking/runs',
    )
