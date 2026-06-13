import json
import os
import time
from datetime import datetime
from pathlib import Path


from modina import probit_rescaling
from utils import compute_modularity, compute_conductance
from graph import build_weighted_graph
from cm_partitioning import (
    SUPPORTED_COMMUNITY_METHODS,
    HIERARCHICAL_METHOD_NAMING,
    _append_hierarchy_rows,
    _run_hsbm_clustering_with_levels,
    _run_agglomerative_clustering,
    _run_hierarchical_infomap_clustering,
    _run_recursive_leiden_clustering,
    run_community_clustering,
)
from cm_export import build_cm_partitions_for_methods
from benchmarking.benchmark_plotting import build_pairwise_comparison_table
import pandas as pd
from rdflib import logger
from modina.context_net_inference import compute_context_scores
from modina.edge_filtering import filter_single, std_rescaling
from environs import Env

env = Env()

env.read_env()

SUPPORTED_WEIGHT_OPTIONS = ('abs-raw-E','log-raw-P', '-logp_e_abs_raw', 'abs-probit-E', 'abs-std-E', '-logp_abs_probit-E', '-logp_abs_std-E')
BENCHMARK_DEFAULT_DENSITY = 0.1
BENCHMARK_DEFAULT_EDGE_WEIGHT = '1-raw-P'
TEST_MODE = 'nonparametric'  # 'nonparametric' or 'parametric'

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
            node_to_community, algorithm_name, hierarchy_levels = _run_hsbm_clustering_with_levels(graph, seed=seed)
        elif method == 'agglomerative':
            node_to_community, algorithm_name, hierarchy_levels = _run_agglomerative_clustering(graph)
        elif method == 'hierarchical_infomap':
            node_to_community, algorithm_name, hierarchy_levels = _run_hierarchical_infomap_clustering(graph, seed=seed)
        elif method == 'recursive_leiden':
            node_to_community, algorithm_name, hierarchy_levels = _run_recursive_leiden_clustering(
                graph,
                resolution=resolution,
                seed=seed,
            )
        else:
            node_to_community, algorithm_name = run_community_clustering(
                graph=graph,
                method=method,
                resolution=resolution,
                seed=seed,
            )
            hierarchy_levels = None

        runtime = time.perf_counter() - start

        if hierarchy_levels is not None:
            _append_hierarchy_rows(
                benchmark_rows=benchmark_rows,
                benchmark_assignments=benchmark_assignments,
                graph=graph,
                node_count=node_count,
                edge_count=edge_count,
                algorithm_name=algorithm_name,
                hierarchy_levels=hierarchy_levels,
                runtime=runtime,
                method_name_fn=HIERARCHICAL_METHOD_NAMING[method],
            )
            continue

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
    test_mode=TEST_MODE,
    methods=None,
    resolution=1.0,
    seed=42,
    output_dir='./benchmarking/runs',
):
    # calculate association network
    network_data = pd.read_csv(env("PHENOTYPE_PATH"), sep=",", index_col=env("PATIENT_ID_COLUMN"), low_memory=False)
    if not os.path.exists(env("CALCULATED_EDGES_PATH")+"/"+env("OBSERVATION_SOURCE")+"_"+test_mode+"_scores.csv"):
        print(env("CALCULATED_EDGES_PATH")+env("OBSERVATION_SOURCE")+"_"+test_mode+"_scores.csv")
        logger.info("No pre-calculated edges found, computing context scores from whole network data...")
        network_meta_data = pd.read_csv(env("PHENOTYPE_META_PATH"), sep=",", low_memory=False)
        assoc_scores = compute_context_scores(context_data=network_data, meta_file=network_meta_data, test_type=test_mode,
                                        correction= 'bh', num_workers=env.int("NUMBER_OF_WORKERS"),
                            path=env("CALCULATED_EDGES_PATH"), nan_value=env.int("NAN_VALUE"),
                            name=env("OBSERVATION_SOURCE")+"_"+test_mode)
    else:
        logger.info("Pre-calculated edges found, loading from file...")
        assoc_scores = pd.read_csv(env("CALCULATED_EDGES_PATH")+"/"+env("OBSERVATION_SOURCE")+"_"+test_mode+"_scores.csv", sep=",", low_memory=False)
        
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
        ('test_mode', test_mode),
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

    pairwise_table_df = build_pairwise_comparison_table(
        graph=build_weighted_graph(filtered_assoc_scores, selected_nodes, edge_weight),
        benchmark_df=benchmark_df,
        benchmark_assignments=benchmark_assignments,
    )

    cm_partitions_payloads = build_cm_partitions_for_methods(
        graph=build_weighted_graph(filtered_assoc_scores, selected_nodes, edge_weight),
        filtered_assoc_scores=filtered_assoc_scores,
        candidate_link_count=len(assoc_scores),
        methods=methods or SUPPORTED_COMMUNITY_METHODS,
        benchmark_df=benchmark_df,
        benchmark_assignments=benchmark_assignments,
        seed=seed,
    )
    partitions_dir = Path(output_dir).resolve().parent / 'cm_partitions' / timestamp
    partitions_dir.mkdir(parents=True, exist_ok=True)
    partitions_output_paths = {}
    for method, payload in cm_partitions_payloads.items():
        partitions_output_path = partitions_dir / f'benchmarking_cm_detection_partitions_{method}.json'
        with open(partitions_output_path, 'w', encoding='utf-8') as partitions_file:
            json.dump(payload, partitions_file)
        partitions_output_paths[method] = partitions_output_path

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
    print(f"CM partitions written to: {partitions_dir}")
    for method, partitions_output_path in partitions_output_paths.items():
        print(f"  {method}: {partitions_output_path}")
    print('Shared benchmark metadata:')
    for key, value in metadata_rows:
        print(f'  {key}: {value}')
    return method_table_df, output_path, pairwise_table_df, comparison_output_path, partitions_output_paths

if __name__ == '__main__':
    run_benchmark_from_whole_network(
        edge_weight=BENCHMARK_DEFAULT_EDGE_WEIGHT,
        methods=SUPPORTED_COMMUNITY_METHODS,
        test_mode=TEST_MODE,
        resolution=1.0,
        seed=42,
        output_dir='./benchmarking/runs',
    )
