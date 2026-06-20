import pandas as pd

from cm_partitioning import _community_family_name, _run_leiden_clustering
from utils import resolution_to_key, DEFAULT_LEIDEN_RESOLUTIONS


def _run_leiden_partitions_by_resolution(graph, resolutions, seed=42):
    """Run Leiden once per resolution and return {resolution_key: node_to_community}."""
    partitions_by_resolution = {}
    for resolution in resolutions:
        node_to_community, _ = _run_leiden_clustering(graph, resolution=resolution, seed=seed)
        partitions_by_resolution[resolution_to_key(resolution)] = node_to_community
    return partitions_by_resolution


def _collect_family_partitions(benchmark_df, benchmark_assignments, family):
    """Return {level_key: node_to_community} for a hierarchical method family, ordered by level."""
    family_rows = benchmark_df[benchmark_df['method'].map(_community_family_name) == family]
    family_rows = family_rows.dropna(subset=['hierarchy_level']).sort_values('hierarchy_level')
    return {
        resolution_to_key(row['hierarchy_level']): benchmark_assignments[row['method']]
        for _, row in family_rows.iterrows()
    }


def _build_cm_partition_links(filtered_assoc_scores):
    """Build the shared 'links' list reused across all per-algorithm partition files."""
    links = []
    for _, row in filtered_assoc_scores.iterrows():
        source = str(row['label1'])
        target = str(row['label2'])
        test_type = row.get('test_type')
        if pd.isna(test_type):
            test_type = None
        links.append({
            'id': f'{source}__{target}',
            'source': source,
            'target': target,
            'edge_type': test_type,
            'final_e_value': float(row['raw-E']),
            'test_type': test_type,
            'final_e_value_rescaled': float(row['probit-E']),
        })
    return links


def _build_cm_partitions_payload(graph, links, candidate_link_count, algorithm, partitions_by_variant, seed):
    """Build a {meta, points, links} payload for one algorithm's partition(s)."""
    points = []
    for vertex in graph.vs:
        node_id = vertex['name']
        point = {
            'id': node_id,
            'label': node_id,
            'type': None,
            'source_table': None,
        }
        for variant_key, node_to_community in partitions_by_variant.items():
            point[f'community_r{variant_key}'] = node_to_community.get(node_id)
        points.append(point)

    community_counts_by_resolution = {
        variant_key: len(set(node_to_community.values()))
        for variant_key, node_to_community in partitions_by_variant.items()
    }

    return {
        'meta': {
            'point_count': len(points),
            'link_count': len(links),
            'candidate_link_count': candidate_link_count,
            'community_counts_by_resolution': community_counts_by_resolution,
            'resolutions': list(partitions_by_variant.keys()),
            'algorithm': algorithm,
            'seed': seed,
            'limit': None,
            'threshold': None,
            'per_node_limit': None,
        },
        'points': points,
        'links': links,
    }


def build_cm_partitions_for_methods(graph, filtered_assoc_scores, candidate_link_count, methods, benchmark_df, benchmark_assignments, seed):
    """Build one {meta, points, links} payload per community detection method.

    - 'leiden' sweeps DEFAULT_LEIDEN_RESOLUTIONS, keyed by resolution (e.g. 'community_r0.2').
    - Flat methods ('louvain', 'infomap') get a single partition keyed 'community_r1.0'.
    - Hierarchical families get one partition per level, keyed by decimal level
      (e.g. 'community_r1.0', 'community_r2.0', ...).
    """
    links = _build_cm_partition_links(filtered_assoc_scores)
    payloads = {}

    for method in methods:
        if method == 'leiden':
            partitions_by_variant = _run_leiden_partitions_by_resolution(graph, DEFAULT_LEIDEN_RESOLUTIONS, seed=seed)
        elif method in ('louvain', 'infomap'):
            if method not in benchmark_assignments:
                continue
            partitions_by_variant = {'1.0': benchmark_assignments[method]}
        else:
            partitions_by_variant = _collect_family_partitions(benchmark_df, benchmark_assignments, method)
            if not partitions_by_variant:
                continue

        payloads[method] = _build_cm_partitions_payload(
            graph=graph,
            links=links,
            candidate_link_count=candidate_link_count,
            algorithm=method,
            partitions_by_variant=partitions_by_variant,
            seed=seed,
        )

    return payloads
