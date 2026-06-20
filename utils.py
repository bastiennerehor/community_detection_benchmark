import os
from collections import defaultdict

import numpy as np
import pandas as pd
from rdflib import logger


DEFAULT_LEIDEN_RESOLUTIONS = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]


def _parse_list_env(env, name):
    """Parse a comma-separated env var into a list of stripped, non-empty strings."""
    raw = env(name, default="")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _infer_separator(path):
    """Infer the CSV separator from a file's extension (.tsv -> tab, else comma)."""
    return "\t" if path.lower().endswith(".tsv") else ","


def _resolve_path(path, root):
    """Join a relative path with `root` (if set); absolute paths are returned unchanged."""
    if root and not os.path.isabs(path):
        return os.path.join(root, path)
    return path


# Types accepted by modina's _separate_types().
VALID_VARIABLE_TYPES = {"ordinal", "nominal", "binary", "continuous"}

# Common alternative spellings, mapped to the canonical modina type they represent.
TYPE_ALIASES = {
    "boolean": "binary",
    "categorical": "nominal",
    "float": "continuous",
    "integer": "continuous",
}

ALL_VALID_TYPES = VALID_VARIABLE_TYPES | set(TYPE_ALIASES)

# Dtype passed to read_csv per modina variable type. Low-cardinality types are read
# as 'category' (1 byte/code instead of a full int64/object per cell); continuous
# stays float64 so reading doesn't change downstream numeric precision.
TYPE_TO_DTYPE = {
    "ordinal": "category",
    "nominal": "category",
    "binary": "category",
    "continuous": "float64",
}


def _apply_type_column(meta, type_column, meta_path):
    """
    Rename `type_column` to 'type' if it is an existing column in `meta`.
    Otherwise, treat `type_column` as a literal type value (e.g. 'continuous') and
    assign it to every row, so a single fixed type can be used for an entire
    data source whose meta file has no type column.
    """
    if type_column in meta.columns:
        return meta.rename(columns={type_column: "type"})

    if type_column.lower() not in ALL_VALID_TYPES:
        logger.warning(
            f"{meta_path}: '{type_column}' is neither a column in this meta file nor one of "
            f"the recognized types {sorted(ALL_VALID_TYPES)}."
        )

    meta = meta.copy()
    meta["type"] = type_column
    return meta


def _normalize_meta_types(meta, meta_path):
    """
    Lowercase the 'type' column and map alternative spellings (TYPE_ALIASES, e.g.
    'boolean' -> 'binary', 'categorical' -> 'nominal', 'float'/'integer' -> 'continuous')
    to the canonical types modina expects. Rows whose type still isn't one of
    VALID_VARIABLE_TYPES are dropped, since modina cannot assign them to a test and
    the matching data column never needs to be read.
    """
    meta = meta.copy()
    normalized = meta["type"].astype(str).str.lower().map(lambda t: TYPE_ALIASES.get(t, t))
    meta["type"] = normalized

    invalid_mask = ~normalized.isin(VALID_VARIABLE_TYPES)
    if invalid_mask.any():
        dropped = meta.loc[invalid_mask, ["label", "type"]]
        logger.info(
            f"{meta_path}: dropping {len(dropped)} variable(s) with unrecognized type "
            f"(expected one of {sorted(VALID_VARIABLE_TYPES)} or aliases {sorted(TYPE_ALIASES)}): "
            f"{list(dropped.itertuples(index=False, name=None))}"
        )
        meta = meta[~invalid_mask].reset_index(drop=True)

    return meta


def _sample_ids(ids, limit=20):
    """Sorted ids for a log message, capped so a huge mismatch can't blow up the log."""
    values = sorted(ids)
    if len(values) <= limit:
        return values
    return values[:limit] + [f"...and {len(values) - limit} more"]


def _load_typed_data_source(data_path, meta_path, label_column, type_column, patient_id_column):
    """
    Load one data/meta pair, restricting the (potentially huge) data file to the
    columns modina can actually use. The meta file is read first since it's tiny, so
    the relevant columns and their dtypes are known before the data file is touched:
    variables with an unrecognized type and any per-column dtype inference are
    skipped entirely via `usecols`/`dtype` instead of being parsed and then dropped.

    Raises if `data_path` contains columns with no matching label in `meta_path`,
    since modina cannot assign those a type and would error out later anyway. Meta
    rows with no matching column in `data_path` (e.g. variables not simulated/
    collected) are dropped, logging how many.
    """
    meta_sep = _infer_separator(meta_path)
    meta = pd.read_csv(meta_path, sep=meta_sep, low_memory=False)
    meta = meta.rename(columns={label_column: "label"})
    meta = _apply_type_column(meta, type_column, meta_path)[["label", "type"]]
    all_meta_labels = set(meta["label"])

    meta = _normalize_meta_types(meta, meta_path)

    data_sep = _infer_separator(data_path)
    header_columns = pd.read_csv(data_path, sep=data_sep, nrows=0).columns.drop(patient_id_column)

    missing_in_meta = set(header_columns) - all_meta_labels
    if missing_in_meta:
        raise ValueError(
            f"{data_path}: {len(missing_in_meta)} column(s) have no matching label in "
            f"{meta_path}: {sorted(missing_in_meta)}"
        )

    missing_in_data = set(meta["label"]) - set(header_columns)
    if missing_in_data:
        logger.info(
            f"{meta_path}: dropping {len(missing_in_data)} label(s) with no matching "
            f"column in {data_path}: {sorted(missing_in_data)}"
        )
        meta = meta[meta["label"].isin(header_columns)].reset_index(drop=True)

    usecols = [patient_id_column] + meta["label"].tolist()
    dtype = {label: TYPE_TO_DTYPE[type_] for label, type_ in zip(meta["label"], meta["type"])}

    data = pd.read_csv(
        data_path,
        sep=data_sep,
        index_col=patient_id_column,
        usecols=usecols,
        dtype=dtype,
    )

    return data, meta


def combine_data(env):
    """
    Load and combine any number of data sources (e.g. phenotypes, proteins,
    metabolites, genomic variants), each with their own meta data file, label
    column and type column.

    Sources are configured via the comma-separated env vars DATA_PATHS,
    DATA_META_PATHS, DATA_LABEL_COLUMNS and DATA_TYPE_COLUMNS, which must all have
    the same number of entries (one per source) and at least one entry. If
    DATA_ROOT is set, it is prepended to any relative entry in DATA_PATHS and
    DATA_META_PATHS, so only file names need to be given there.

    Patients are combined via an inner join on the patient id index, so any patient
    missing from one of the data sources is dropped from the combined result. Such
    drops are logged. For every data/meta pair, the meta file's 'label' column is
    checked against the data file's column names and mismatches are logged.

    :return: tuple (combined_data, meta_file) ready for modina's compute_context_scores.
    """
    patient_id_column = env("PATIENT_ID_COLUMN")
    data_root = env("DATA_ROOT", default=None)

    data_paths = _parse_list_env(env, "DATA_PATHS")
    meta_paths = _parse_list_env(env, "DATA_META_PATHS")
    label_columns = _parse_list_env(env, "DATA_LABEL_COLUMNS")
    type_columns = _parse_list_env(env, "DATA_TYPE_COLUMNS")

    data_paths = [_resolve_path(path, data_root) for path in data_paths]
    meta_paths = [_resolve_path(path, data_root) for path in meta_paths]

    if not (len(data_paths) == len(meta_paths) == len(label_columns) == len(type_columns)):
        raise ValueError(
            "DATA_PATHS, DATA_META_PATHS, DATA_LABEL_COLUMNS and DATA_TYPE_COLUMNS "
            "must all have the same number of comma-separated entries."
        )

    if not data_paths:
        raise ValueError(
            "No data source configured. Set DATA_PATHS, DATA_META_PATHS, "
            "DATA_LABEL_COLUMNS and DATA_TYPE_COLUMNS."
        )

    combined_data = None
    meta_file = None

    for data_path, meta_path, label_column, type_column in zip(data_paths, meta_paths, label_columns, type_columns):
        extra_data, extra_meta = _load_typed_data_source(
            data_path, meta_path, label_column, type_column, patient_id_column,
        )

        if combined_data is None:
            combined_data = extra_data
            meta_file = extra_meta
            logger.info(
                f"Loaded base data from {data_path}: "
                f"{combined_data.shape[0]} patients, {combined_data.shape[1]} variables."
            )
            continue

        dropped_existing = combined_data.index.difference(extra_data.index)
        dropped_incoming = extra_data.index.difference(combined_data.index)
        if len(dropped_existing):
            logger.info(
                f"{data_path}: dropping {len(dropped_existing)} patient(s) not present "
                f"in this file: {_sample_ids(dropped_existing)}"
            )
        if len(dropped_incoming):
            logger.info(
                f"{data_path}: dropping {len(dropped_incoming)} patient(s) from this file "
                f"not present in previously loaded data: {_sample_ids(dropped_incoming)}"
            )

        combined_data = combined_data.join(extra_data, how="inner")
        meta_file = pd.concat([meta_file, extra_meta], ignore_index=True)
        logger.info(
            f"Combined with {data_path}: "
            f"{combined_data.shape[0]} patients, {combined_data.shape[1]} variables remaining."
        )

    return combined_data, meta_file

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