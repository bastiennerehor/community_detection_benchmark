from collections import defaultdict

import numpy as np


DEFAULT_LEIDEN_RESOLUTIONS = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]

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