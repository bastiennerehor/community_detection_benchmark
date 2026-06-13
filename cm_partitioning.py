import json

import numpy as np
import leidenalg
import graph_tool.all as gt
from rdflib import logger

from utils import compute_modularity, compute_conductance


SUPPORTED_COMMUNITY_METHODS = ('louvain', 'leiden', 'recursive_leiden', 'agglomerative', 'infomap', 'hierarchical_infomap', 'hsbm')


def _assign_membership(graph, membership):
    return {graph.vs[idx]['name']: community_id for idx, community_id in enumerate(membership)}


def _membership_vector(graph, node_to_community):
    return [node_to_community.get(vertex['name'], idx) for idx, vertex in enumerate(graph.vs)]


def _singleton_membership(graph):
    return list(range(graph.vcount()))


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


HIERARCHICAL_METHOD_NAMING = {
    'hsbm': lambda level_summary, total_levels: f'hsbm_level_{level_summary["level"] + 1}_of_{total_levels}',
    'agglomerative': lambda level_summary, total_levels: f'agglomerative_cut_{level_summary["cut_count"]}',
    'hierarchical_infomap': lambda level_summary, total_levels: f'hierarchical_infomap_level_{level_summary["level"] + 1}_of_{total_levels}',
    'recursive_leiden': lambda level_summary, total_levels: f'recursive_leiden_level_{level_summary["level"] + 1}_of_{total_levels}',
}


def _append_hierarchy_rows(benchmark_rows, benchmark_assignments, graph, node_count, edge_count,
                            algorithm_name, hierarchy_levels, runtime, method_name_fn):
    """Append one benchmark row per level of a hierarchical method's results."""
    total_levels = len(hierarchy_levels)
    for level_summary in hierarchy_levels:
        level_number = level_summary['level'] + 1
        row = {
            'method': method_name_fn(level_summary, total_levels),
            'algorithm': algorithm_name,
            'hierarchy_level': level_number,
            'hierarchy_depth': level_summary.get('hierarchy_depth', total_levels),
            'hierarchy_label': level_summary.get('hierarchy_label', f'level {level_number} of {total_levels}'),
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


def run_community_clustering(graph, method='leiden', resolution=1.0, seed=42):
    method = (method or 'leiden').strip().lower()

    if method == 'leiden':
        return _run_leiden_clustering(graph, resolution=resolution, seed=seed)
    if method == 'louvain':
        return _run_louvain_clustering(graph)
    if method == 'infomap':
        return _run_infomap_clustering(graph)

    raise ValueError(f"Unsupported community detection method '{method}'. Choose from: {', '.join(SUPPORTED_COMMUNITY_METHODS)}.")
