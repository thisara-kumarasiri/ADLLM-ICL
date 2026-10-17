from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline


def load_xgboost_model(
    model_path: str | Path,
) -> Any:
    """Load a saved XGBoost model or sklearn pipeline."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {model_path}"
        )

    return joblib.load(model_path)


def get_xgboost_estimator(
    model: Any,
) -> Any:
    """
    Return the final XGBoost estimator.

    Supports:
    - Directly saved XGBClassifier
    - sklearn Pipeline ending in XGBClassifier
    """
    if isinstance(model, Pipeline):
        return model.steps[-1][1]

    return model


def transform_features(
    model: Any,
    data: pd.DataFrame,
) -> Any:
    """
    Apply the same preprocessing used during model training.

    For a pipeline, all steps except the final XGBoost model are applied.
    """
    if isinstance(model, Pipeline):
        preprocessing_pipeline = model[:-1]
        return preprocessing_pipeline.transform(data)

    return data


def _to_2d_array(data: Any) -> np.ndarray:
    """Convert transformed input into a dense two-dimensional array."""
    if sparse.issparse(data):
        array = data.toarray()
    elif isinstance(data, pd.DataFrame):
        array = data.to_numpy()
    else:
        array = np.asarray(data)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError(
            f"Expected two-dimensional feature data, received shape {array.shape}."
        )

    return array


def _build_node_lookup(tree: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map every node ID in one dumped XGBoost tree to its node dictionary."""
    lookup: dict[int, dict[str, Any]] = {}

    def visit(node: dict[str, Any]) -> None:
        node_id = int(node["nodeid"])
        lookup[node_id] = node

        for child in node.get("children", []):
            visit(child)

    visit(tree)
    return lookup


def _resolve_feature_index(
    split_name: str,
    booster_feature_names: list[str] | None,
    input_feature_names: list[str] | None,
) -> int:
    """
    Resolve an XGBoost split name to the corresponding transformed column index.

    XGBoost commonly stores split features as f0, f1, ... when explicit
    feature names were not supplied.
    """
    if split_name.startswith("f") and split_name[1:].isdigit():
        return int(split_name[1:])

    if booster_feature_names and split_name in booster_feature_names:
        return booster_feature_names.index(split_name)

    if input_feature_names and split_name in input_feature_names:
        return input_feature_names.index(split_name)

    raise ValueError(
        f"Could not map XGBoost split feature {split_name!r} "
        "to a transformed input column."
    )


def _follow_tree_path(
    row: np.ndarray,
    node_lookup: dict[int, dict[str, Any]],
    booster_feature_names: list[str] | None,
    input_feature_names: list[str] | None,
) -> tuple[tuple[int, int], ...]:
    """
    Return the ordered branching decisions followed in one tree.

    Every decision is represented as:
        (current_node_id, selected_child_node_id)

    Including the selected child is essential. Two samples can visit the same
    root node but choose opposite branches; those root decisions must not count
    as shared.
    """
    current_node_id = 0
    path: list[tuple[int, int]] = []

    while True:
        node = node_lookup[current_node_id]

        if "leaf" in node:
            return tuple(path)

        split_name = str(node["split"])
        feature_index = _resolve_feature_index(
            split_name=split_name,
            booster_feature_names=booster_feature_names,
            input_feature_names=input_feature_names,
        )

        if feature_index >= row.shape[0]:
            raise IndexError(
                f"Tree requests feature index {feature_index}, but the "
                f"transformed row contains only {row.shape[0]} features."
            )

        value = row[feature_index]

        if pd.isna(value):
            next_node_id = int(node["missing"])
            path.append((current_node_id, next_node_id))
            current_node_id = next_node_id
            continue

        split_condition = node["split_condition"]

        # This implementation is intended for numerical threshold splits.
        # One-hot encoded categorical variables are numerical and are supported.
        if isinstance(split_condition, list):
            raise NotImplementedError(
                "Native XGBoost categorical split detected. "
                "Use numerical/one-hot encoded features, or extend the "
                "categorical traversal rule for this model."
            )

        if float(value) < float(split_condition):
            next_node_id = int(node["yes"])
        else:
            next_node_id = int(node["no"])

        path.append((current_node_id, next_node_id))
        current_node_id = next_node_id


def get_decision_paths(
    model: Any,
    data: pd.DataFrame,
) -> list[list[tuple[tuple[int, int], ...]]]:
    """
    Extract the root-to-leaf internal-node path for every sample and tree.

    Returns:
        paths[sample_index][tree_index] -> tuple of internal node IDs
    """
    estimator = get_xgboost_estimator(model)
    booster = estimator.get_booster()

    transformed_data = transform_features(
        model=model,
        data=data,
    )
    feature_array = _to_2d_array(transformed_data)

    input_feature_names = (
        list(transformed_data.columns)
        if isinstance(transformed_data, pd.DataFrame)
        else None
    )
    booster_feature_names = (
        list(booster.feature_names)
        if booster.feature_names is not None
        else None
    )

    dumped_trees = [
        json.loads(tree_json)
        for tree_json in booster.get_dump(
            dump_format="json"
        )
    ]
    tree_lookups = [
        _build_node_lookup(tree)
        for tree in dumped_trees
    ]

    all_paths: list[list[tuple[tuple[int, int], ...]]] = []

    for row in feature_array:
        sample_paths = [
            _follow_tree_path(
                row=row,
                node_lookup=node_lookup,
                booster_feature_names=booster_feature_names,
                input_feature_names=input_feature_names,
            )
            for node_lookup in tree_lookups
        ]
        all_paths.append(sample_paths)

    return all_paths


def calculate_tree_path_similarity(
    first_path: tuple[tuple[int, int], ...],
    second_path: tuple[tuple[int, int], ...],
) -> float:
    """
    Compare two root-to-leaf paths from the same tree.

    Shared path length is the common prefix before the paths diverge.

    Similarity:
        shared_prefix_length / max(path_length_1, path_length_2)

    Therefore:
        1.0 = identical branching path
        0.0 = diverge at the root decision
    """
    first_path = tuple(first_path)
    second_path = tuple(second_path)

    maximum_length = max(
        len(first_path),
        len(second_path),
    )

    # A tree containing only one leaf has no decisions. Every sample is
    # structurally identical in that tree.
    if maximum_length == 0:
        return 1.0

    shared_prefix_length = 0

    for first_node, second_node in zip(
        first_path,
        second_path,
    ):
        if first_node != second_node:
            break

        shared_prefix_length += 1

    return shared_prefix_length / maximum_length


def calculate_path_similarity(
    test_paths: list[tuple[tuple[int, int], ...]],
    train_paths: list[list[tuple[tuple[int, int], ...]]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare one test subject with every training subject.

    For each train-test pair:
      1. Calculate path similarity separately in every tree.
      2. Average the per-tree similarities.

    Returns:
        average_similarities:
            One similarity value per training subject.

        matching_tree_counts:
            Number of trees with completely identical paths.
    """
    number_of_trees = len(test_paths)

    if number_of_trees == 0:
        raise ValueError("No XGBoost trees were found.")

    similarities: list[float] = []
    matching_tree_counts: list[int] = []

    for candidate_paths in train_paths:
        if len(candidate_paths) != number_of_trees:
            raise ValueError(
                "Train and test subjects contain different numbers of trees."
            )

        tree_similarities = [
            calculate_tree_path_similarity(
                first_path=test_path,
                second_path=candidate_path,
            )
            for test_path, candidate_path in zip(
                test_paths,
                candidate_paths,
            )
        ]

        similarities.append(
            float(np.mean(tree_similarities))
        )
        matching_tree_counts.append(
            int(
                sum(
                    test_path == candidate_path
                    for test_path, candidate_path in zip(
                        test_paths,
                        candidate_paths,
                    )
                )
            )
        )

    return (
        np.asarray(similarities, dtype=float),
        np.asarray(matching_tree_counts, dtype=int),
    )


def calculate_path_distance(
    test_paths: list[tuple[tuple[int, int], ...]],
    train_paths: list[list[tuple[tuple[int, int], ...]]],
) -> np.ndarray:
    """
    Convert average path similarity into distance.

    Distance:
        0.0 = identical paths in every tree
        1.0 = paths diverge at the root in every tree
    """
    similarities, _ = calculate_path_similarity(
        test_paths=test_paths,
        train_paths=train_paths,
    )

    return 1.0 - similarities