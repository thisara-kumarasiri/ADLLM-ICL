from __future__ import annotations

import json
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ...proposed_method import feature_weights as weighted_core
from ...proposed_method import xgboost_paths as path_core


class DecisionStep(NamedTuple):
    """One XGBoost split decision made by a subject."""

    node_id: int
    child_node_id: int
    feature: str


# PATH EXTRACTION

def _follow_feature_path(
    row: np.ndarray,
    node_lookup: dict[int, dict[str, Any]],
    booster_feature_names: list[str] | None,
    transformed_feature_names: list[str],
) -> tuple[DecisionStep, ...]:
    """
    Follow one subject through one XGBoost tree.

    Returns the consecutive split decisions made by the subject.
    """

    current_node_id = 0
    path: list[DecisionStep] = []

    while True:

        node = node_lookup[current_node_id]

        # Stop when a leaf is reached
        if "leaf" in node:
            return tuple(path)

        split_name = str(node["split"])

        feature_index = path_core._resolve_feature_index(
            split_name=split_name,
            booster_feature_names=booster_feature_names,
            input_feature_names=transformed_feature_names,
        )

        if feature_index >= row.shape[0]:
            raise IndexError(
                f"Tree requests feature index {feature_index}, "
                f"but transformed row contains only "
                f"{row.shape[0]} features."
            )

        value = row[feature_index]

        # Follow XGBoost branch
        if pd.isna(value):

            next_node_id = int(
                node["missing"]
            )

        else:

            split_condition = node["split_condition"]

            if isinstance(split_condition, list):
                raise NotImplementedError(
                    "Native XGBoost categorical splits are "
                    "not supported. Use numerical or "
                    "one-hot encoded features."
                )

            next_node_id = int(
                node["yes"]
                if float(value) < float(split_condition)
                else node["no"]
            )

        # Clean feature name
        clean_feature = weighted_core.clean_feature_name(
            transformed_feature_names[feature_index]
        )

        path.append(
            DecisionStep(
                node_id=current_node_id,
                child_node_id=next_node_id,
                feature=clean_feature,
            )
        )

        current_node_id = next_node_id


def get_feature_decision_paths(
    model: Any,
    data: pd.DataFrame,
) -> list[list[tuple[DecisionStep, ...]]]:
    """
    Extract every subject's path through every XGBoost tree.

    Returns
    -------
    paths[sample_index][tree_index]
        -> ordered DecisionStep tuple
    """

    estimator = path_core.get_xgboost_estimator(
        model
    )

    booster = estimator.get_booster()

    transformed_data = path_core.transform_features(
        model,
        data,
    )

    feature_array = path_core._to_2d_array(
        transformed_data
    )

    transformed_feature_names = (
        weighted_core.get_model_feature_names(
            model=model,
            estimator=estimator,
            fallback_feature_names=list(data.columns),
            expected_feature_count=feature_array.shape[1],
        )
    )

    booster_feature_names = (
        list(booster.feature_names)
        if booster.feature_names is not None
        else None
    )

    # Load every XGBoost tree
    dumped_trees = [
        json.loads(tree_json)
        for tree_json in booster.get_dump(
            dump_format="json"
        )
    ]

    tree_lookups = [
        path_core._build_node_lookup(tree)
        for tree in dumped_trees
    ]

    # Follow every subject through every tree
    return [
        [
            _follow_feature_path(
                row=row,
                node_lookup=node_lookup,
                booster_feature_names=booster_feature_names,
                transformed_feature_names=transformed_feature_names,
            )
            for node_lookup in tree_lookups
        ]
        for row in feature_array
    ]


# XGBOOST TREE IMPORTANCE USING LOCAL GAIN

def get_tree_gain_weights(
    model: Any,
) -> pd.DataFrame:
    """
    Calculate an importance weight for every XGBoost tree.

    For each tree:

        Tree_Total_Gain_t
            =
        sum of the local split gains of all split nodes
        inside tree t

    Then normalise:

        Tree_Weight_t
            =
        Tree_Total_Gain_t
        ------------------------------
        sum(Tree_Total_Gain_all_trees)

    Therefore, trees containing more total XGBoost split gain
    contribute more strongly to the final candidate distance.

    Important
    ---------
    Leaf rows are excluded because the Gain value associated
    with a leaf is a leaf score rather than split gain.

    Returns
    -------
    pd.DataFrame

    Columns:
        Tree
        Tree_Total_Gain
        Tree_Weight
    """

    estimator = path_core.get_xgboost_estimator(
        model
    )

    booster = estimator.get_booster()

    # Get XGBoost tree structure as dataframe
    trees_df = booster.trees_to_dataframe()

    number_of_trees = len(
        booster.get_dump()
    )

    if number_of_trees == 0:
        raise ValueError(
            "No XGBoost trees were found."
        )

    # Keep only split nodes
    split_df = trees_df[
        trees_df["Feature"] != "Leaf"
    ].copy()

    # Sum LOCAL split gain inside each tree (GAIN)
    tree_gain = (
        split_df
        .groupby("Tree")["Gain"]
        .sum()
        .reindex(
            range(number_of_trees),
            fill_value=0.0,
        )
        .astype(float)
    )


    # Safety: gain should normally be non-negative
    tree_gain = tree_gain.clip(
        lower=0.0
    )

    # Normalise across all trees
    total_gain = float(
        tree_gain.sum()
    )

    if total_gain <= 0.0:

        # Safety fallback:
        # all trees receive equal importance
        tree_weights = np.full(
            number_of_trees,
            1.0 / number_of_trees,
            dtype=float,
        )

    else:

        tree_weights = (
            tree_gain.to_numpy(
                dtype=float
            )
            / total_gain
        )

    result = pd.DataFrame(
        {
            "Tree": np.arange(
                number_of_trees
            ),
            "Tree_Total_Gain": (
                tree_gain.to_numpy(
                    dtype=float
                )
            ),
            "Tree_Weight": (
                tree_weights
            ),
        }
    )

    return result


# PER-TREE SHARED-PATH DISTANCE

def calculate_tree_distance(
    test_path: tuple[DecisionStep, ...],
    candidate_path: tuple[DecisionStep, ...],
    test_scaled_row: pd.Series,
    candidate_scaled_row: pd.Series,
    feature_weights: dict[str, float],
) -> tuple[float, int, int]:
    """
    Calculate LR coefficient-weighted shared-path Euclidean
    distance for ONE XGBoost tree.

    Shared-path definition
    ----------------------
    Starting from the root:

    - Test and candidate must be at the same node.
    - They must take the same child branch.
    - Stop immediately at the first divergence.

    For all eligible shared-path features:

        D_t =
        sqrt(
            sum(
                w_f * (test_f - candidate_f)^2
            )
            /
            sum(w_f)
        )

    where:

        w_f = LR feature importance weight

    Smaller distance means greater similarity.

    Returns
    -------
    tree_distance
    shared_node_count
    eligible_test_node_count
    """

    # Build case-insensitive feature lookup
    canonical_features = {
        str(feature).strip().lower(): feature
        for feature in feature_weights
    }

    def resolve_feature(
        step: DecisionStep,
    ) -> str | None:

        return canonical_features.get(
            str(step.feature)
            .strip()
            .lower()
        )

    # 1. Count eligible nodes on complete test path
    eligible_test_node_count = 0

    for step in test_path:

        feature = resolve_feature(
            step
        )

        if feature is None:
            continue

        weight = float(
            feature_weights[feature]
        )

        if weight > 0.0:
            eligible_test_node_count += 1

    # 2. Consecutive shared path
    weighted_squared_difference = 0.0

    shared_weight_sum = 0.0

    shared_node_count = 0

    for test_step, candidate_step in zip(
        test_path,
        candidate_path,
    ):

        # Stop at first path divergence
        if (
            test_step.node_id
            != candidate_step.node_id
            or
            test_step.child_node_id
            != candidate_step.child_node_id
        ):
            break

        # Identify feature
        feature = resolve_feature(
            test_step
        )

        if feature is None:
            continue

        weight = float(
            feature_weights[feature]
        )

        if weight <= 0.0:
            continue

        # Standardised feature difference
        difference = (
            float(
                test_scaled_row[feature]
            )
            -
            float(
                candidate_scaled_row[feature]
            )
        )

        # LR-weighted squared difference
        weighted_squared_difference += (
            weight
            * (difference ** 2)
        )

        shared_weight_sum += weight

        shared_node_count += 1

    # 3. No eligible shared features
    if shared_weight_sum <= 0.0:

        return (
            float("nan"),
            0,
            int(
                eligible_test_node_count
            ),
        )

    # 4. LR-weighted Euclidean distance
    weighted_euclidean_distance = float(
        np.sqrt(
            weighted_squared_difference
            /
            shared_weight_sum
        )
    )

    return (
        float(
            weighted_euclidean_distance
        ),
        int(
            shared_node_count
        ),
        int(
            eligible_test_node_count
        ),
    )


# ENSEMBLE DISTANCE WITH TREE-LEVEL XGBOOST GAIN

def calculate_ensemble_similarity(
    test_paths: list[tuple[DecisionStep, ...]],
    candidate_paths: list[tuple[DecisionStep, ...]],
    test_scaled_row: pd.Series,
    candidate_scaled_row: pd.Series,
    feature_weights: dict[str, float],
    tree_weights: list[float] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """
    Calculate the final test-candidate distance.

    Stage 1
    -------
    For every XGBoost tree, calculate the LR-weighted
    shared-path Euclidean distance:

        D_t

    Stage 2
    -------
    Weight each tree according to its TOTAL LOCAL
    XGBoost split gain:

        G_t

    Final distance:

                     sum(G_t * D_t)
        D_final = -------------------------
                       sum(G_t)

    Only trees that produce a valid shared-path distance
    are included in the final weighted average.

    If tree_weights=None, equal tree weighting is used.
    """

    # Validation
    if not test_paths:
        raise ValueError(
            "No XGBoost trees were found."
        )

    if len(test_paths) != len(
        candidate_paths
    ):
        raise ValueError(
            "Train and test subjects contain "
            "different numbers of trees."
        )

    number_of_trees = len(
        test_paths
    )

    # 1. Prepare tree weights
    if tree_weights is None:

        # Original behaviour:
        # all trees have equal influence
        tree_weights_array = np.ones(
            number_of_trees,
            dtype=float,
        )

    else:

        tree_weights_array = np.asarray(
            tree_weights,
            dtype=float,
        )

        if len(
            tree_weights_array
        ) != number_of_trees:

            raise ValueError(
                "Number of tree weights does not match "
                "number of XGBoost trees.\n"
                f"Trees: {number_of_trees}\n"
                f"Weights: "
                f"{len(tree_weights_array)}"
            )

    # 2. Calculate every tree's distance
    tree_distances: list[float] = []

    valid_tree_weights: list[float] = []

    total_shared_nodes = 0

    total_eligible_test_nodes = 0

    trees_with_shared_nodes = 0

    identical_path_trees = 0

    valid_tree_count = 0

    for tree_index, (
        test_path,
        candidate_path,
    ) in enumerate(
        zip(
            test_paths,
            candidate_paths,
        )
    ):

        distance, shared_nodes, eligible_nodes = (
            calculate_tree_distance(
                test_path=test_path,
                candidate_path=candidate_path,
                test_scaled_row=test_scaled_row,
                candidate_scaled_row=candidate_scaled_row,
                feature_weights=feature_weights,
            )
        )

        tree_weight = float(
            tree_weights_array[
                tree_index
            ]
        )

        # Include only trees producing a valid distance
        if not np.isnan(
            distance
        ):

            tree_distances.append(
                float(
                    distance
                )
            )

            valid_tree_weights.append(
                tree_weight
            )

            valid_tree_count += 1

        # Diagnostics
        total_shared_nodes += (
            shared_nodes
        )

        total_eligible_test_nodes += (
            eligible_nodes
        )

        trees_with_shared_nodes += int(
            shared_nodes > 0
        )

        identical_path_trees += int(
            test_path
            == candidate_path
        )

    # 3. Gain-weighted mean distance
    if not tree_distances:

        final_distance = float(
            "inf"
        )

        total_valid_tree_weight = 0.0

    else:

        distances_array = np.asarray(
            tree_distances,
            dtype=float,
        )

        weights_array = np.asarray(
            valid_tree_weights,
            dtype=float,
        )

        total_valid_tree_weight = float(
            weights_array.sum()
        )

        # Safety fallback
        if total_valid_tree_weight <= 0.0:

            final_distance = float(
                np.mean(
                    distances_array
                )
            )

        else:

            final_distance = float(
                np.average(
                    distances_array,
                    weights=weights_array,
                )
            )

    # 4. Return metrics
    return {
        "Distance":
            final_distance,

        "Shared_LR_Weighted_Nodes":
            total_shared_nodes,

        "Eligible_Test_Path_Nodes":
            total_eligible_test_nodes,

        "Trees_With_Shared_LR_Nodes":
            trees_with_shared_nodes,

        "Identical_Path_Trees":
            identical_path_trees,

        "Valid_Trees":
            valid_tree_count,

        "Valid_Tree_Gain_Weight":
            total_valid_tree_weight,

        "Total_Trees":
            number_of_trees,
    }