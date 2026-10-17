from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ...proposed_method import path_gain_distance as core
from ...proposed_method import feature_weights as weighted_core
from ...proposed_method import xgboost_paths as path_core


class LRWeightedXGBoostPathGainSelector:
    """
    Select balanced examples using:

    1. XGBoost shared decision paths.
    2. LR coefficients for feature weighting within each tree.
    3. XGBoost local split gain for weighting trees across
       the complete ensemble.

    For every test-training pair:

        STEP 1
        Follow both subjects through every XGBoost tree.

        STEP 2
        Keep consecutive nodes where both subjects take
        exactly the same branch.

        STEP 3
        Calculate LR coefficient-weighted Euclidean distance
        over the shared features inside each tree.

        STEP 4
        Calculate each XGBoost tree's importance as:

            Tree_Total_Gain =
                sum(local split gains in the whole tree)

        STEP 5
        Calculate the final distance as:

            sum(Tree_Weight_t * Tree_Distance_t)
            -------------------------------------
                  sum(valid Tree_Weight_t)

    Smaller final distance means that the candidate is more
    similar to the test subject.
    """

    def __init__(
        self,
        logistic_regression_model_path: str | Path,
        xgboost_model_path: str | Path,
        subject_column: str = "Subject",
        group_column: str = "Group",
        age_column: str = "Age",
        sex_column: str = "Sex",
        ad_label: str = "AD",
        cn_label: str = "CN",
        examples_per_class: int = 2,
        lr_feature_columns: list[str] | None = None,
        xgboost_feature_columns: list[str] | None = None,
        exclude_same_subject: bool = True,
    ) -> None:

        if examples_per_class < 1:
            raise ValueError(
                "examples_per_class must be at least 1."
            )

        self.logistic_regression_model_path = Path(
            logistic_regression_model_path
        )

        self.xgboost_model_path = Path(
            xgboost_model_path
        )

        self.subject_column = subject_column

        self.group_column = group_column

        self.age_column = age_column

        self.sex_column = sex_column

        self.ad_label = ad_label

        self.cn_label = cn_label

        self.examples_per_class = (
            examples_per_class
        )

        self.lr_feature_columns = (
            lr_feature_columns
        )

        self.xgboost_feature_columns = (
            xgboost_feature_columns
        )

        self.exclude_same_subject = (
            exclude_same_subject
        )

        # Models
        self.logistic_model: Any | None = None

        self.xgboost_model: Any | None = None

        # LR feature weights
        self.lr_weights_df: (
            pd.DataFrame | None
        ) = None

        self.lr_feature_weights: (
            dict[str, float]
        ) = {}

        # XGBoost tree-level gain weights
        self.xgboost_tree_gain_df: (
            pd.DataFrame | None
        ) = None

        self.xgboost_tree_weights: (
            list[float]
        ) = []

        # Scaling
        self.scaler = None

        # Results
        self.selected_examples_long: (
            pd.DataFrame | None
        ) = None

        self.selected_examples_wide: (
            pd.DataFrame | None
        ) = None

        self.all_candidate_distances: (
            pd.DataFrame | None
        ) = None

    # DATA LOADING
    @staticmethod
    def _load_dataframe(
        data: str | Path | pd.DataFrame,
        name: str,
    ) -> pd.DataFrame:

        if isinstance(
            data,
            pd.DataFrame,
        ):
            return (
                data.copy()
                .reset_index(
                    drop=True
                )
            )

        path = Path(
            data
        )

        if not path.exists():

            raise FileNotFoundError(
                f"{name} file not found: "
                f"{path}"
            )

        return (
            pd.read_csv(
                path
            )
            .reset_index(
                drop=True
            )
        )

    # VALIDATION
    def _validate_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:

        missing_train = {
            self.subject_column,
            self.group_column,
        } - set(
            train_df.columns
        )

        missing_test = {
            self.subject_column
        } - set(
            test_df.columns
        )

        if missing_train:

            raise ValueError(
                "Missing training columns: "
                + ", ".join(
                    sorted(
                        missing_train
                    )
                )
            )

        if missing_test:

            raise ValueError(
                "Missing test columns: "
                + ", ".join(
                    sorted(
                        missing_test
                    )
                )
            )

    # LOAD MODELS + LR WEIGHTS + TREE GAIN WEIGHTS
    def _load_models_and_weights(
        self,
        train_df: pd.DataFrame,
    ) -> None:

        # Logistic Regression
        self.logistic_model = (
            weighted_core.load_model(
                self.logistic_regression_model_path
            )
        )

        # XGBoost
        self.xgboost_model = (
            path_core.load_xgboost_model(
                self.xgboost_model_path
            )
        )

        # LR FEATURE WEIGHTS
        weights_df = (
            weighted_core.extract_logistic_weights(
                model=self.logistic_model,
                fallback_feature_names=(
                    self.lr_feature_columns
                ),
                exclude_features={
                    self.subject_column,
                    self.group_column,
                    self.age_column,
                    self.sex_column,
                },
            )
        )

        weights_df = weights_df[
            weights_df["Feature"].isin(
                train_df.columns
            )
        ].copy()

        if weights_df.empty:

            raise ValueError(
                "No Logistic Regression features matched "
                "the training CSV."
            )

        total_weight = float(
            weights_df[
                "Absolute_Coefficient"
            ].sum()
        )

        if total_weight <= 0.0:

            raise ValueError(
                "All matched LR coefficients are zero."
            )

         
        # Normalise LR weights
         

        weights_df[
            "Normalized_Weight"
        ] = (
            weights_df[
                "Absolute_Coefficient"
            ]
            /
            total_weight
        )

        weights_df = (
            weights_df
            .sort_values(
                "Normalized_Weight",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        self.lr_weights_df = (
            weights_df
        )

        self.lr_feature_columns = (
            weights_df[
                "Feature"
            ].tolist()
        )

        self.lr_feature_weights = dict(
            zip(
                weights_df[
                    "Feature"
                ],
                weights_df[
                    "Normalized_Weight"
                ],
            )
        )

        # XGBOOST TREE-LEVEL GAIN WEIGHTS
        self.xgboost_tree_gain_df = (
            core.get_tree_gain_weights(
                self.xgboost_model
            )
        )

        self.xgboost_tree_weights = (
            self.xgboost_tree_gain_df[
                "Tree_Weight"
            ]
            .astype(float)
            .tolist()
        )

    # XGBOOST INPUT FEATURES
    def _get_xgboost_features(
        self,
        train_df: pd.DataFrame,
    ) -> list[str]:

        if (
            self.xgboost_feature_columns
            is not None
        ):

            features = list(
                self.xgboost_feature_columns
            )

        elif hasattr(
            self.xgboost_model,
            "feature_names_in_",
        ):

            features = list(
                self.xgboost_model
                .feature_names_in_
            )

        else:

            features = [
                column
                for column in train_df.columns
                if column
                not in {
                    self.subject_column,
                    self.group_column,
                }
            ]

        missing = [
            feature
            for feature in features
            if feature
            not in train_df.columns
        ]

        if missing:

            raise ValueError(
                "XGBoost features missing from "
                "training data:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in missing
                )
            )

        return features

    # STANDARDISE LR FEATURES
    def _prepare_scaled_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:

        (
            train_scaled,
            test_scaled,
            self.scaler,
        ) = (
            weighted_core.prepare_scaled_data(
                train_df=train_df,
                test_df=test_df,
                feature_columns=(
                    self.lr_feature_columns
                ),
            )
        )

        return (
            train_scaled.reset_index(
                drop=True
            ),
            test_scaled.reset_index(
                drop=True
            ),
        )

    # CALCULATE TEST-CANDIDATE DISTANCES
    def _calculate_subject_matches(
        self,
        test_index: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_scaled_df: pd.DataFrame,
        test_scaled_df: pd.DataFrame,
        train_paths: list[
            list[
                tuple[
                    core.DecisionStep,
                    ...
                ]
            ]
        ],
        test_paths: list[
            list[
                tuple[
                    core.DecisionStep,
                    ...
                ]
            ]
        ],
    ) -> pd.DataFrame:

        test_subject = (
            test_df
            .iloc[test_index][
                self.subject_column
            ]
        )

        rows: list[
            dict[str, Any]
        ] = []

        for candidate_index in range(
            len(train_df)
        ):

            candidate_subject = (
                train_df
                .iloc[candidate_index][
                    self.subject_column
                ]
            )

            # Exclude same subject if requested
            if (
                self.exclude_same_subject
                and
                str(candidate_subject)
                == str(test_subject)
            ):
                continue

             
            # Calculate:
            # LR shared-path Euclidean within each tree
            # +
            # XGBoost local-gain weighting across trees
            metrics = (
                core.calculate_ensemble_similarity(
                    test_paths=(
                        test_paths[
                            test_index
                        ]
                    ),
                    candidate_paths=(
                        train_paths[
                            candidate_index
                        ]
                    ),
                    test_scaled_row=(
                        test_scaled_df
                        .iloc[test_index]
                    ),
                    candidate_scaled_row=(
                        train_scaled_df
                        .iloc[
                            candidate_index
                        ]
                    ),
                    feature_weights=(
                        self.lr_feature_weights
                    ),
                    tree_weights=(
                        self.xgboost_tree_weights
                    ),
                )
            )

            rows.append(
                {
                    "Test_Subject":
                        test_subject,

                    "Candidate_Subject":
                        candidate_subject,

                    "Candidate_Group":
                        train_df
                        .iloc[candidate_index][
                            self.group_column
                        ],

                    **metrics,

                    "Train_Row_Index":
                        train_df.index[
                            candidate_index
                        ],
                }
            )

        candidates_df = (
            pd.DataFrame(
                rows
            )
        )

        # Smaller distance = more similar
        return (
            candidates_df
            .sort_values(
                by=[
                    "Distance",
                    "Shared_LR_Weighted_Nodes",
                    "Candidate_Subject",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    # SELECT AD / CN EXAMPLES
    def _select_balanced_examples(
        self,
        candidates_df: pd.DataFrame,
    ) -> pd.DataFrame:

        selected_frames: list[
            pd.DataFrame
        ] = []

        class_slots = {

            self.ad_label: [
                f"AD{index + 1}"
                for index in range(
                    self.examples_per_class
                )
            ],

            self.cn_label: [
                f"CN{index + 1}"
                for index in range(
                    self.examples_per_class
                )
            ],
        }

        for class_label in (
            self.ad_label,
            self.cn_label,
        ):

            candidates = (
                candidates_df[
                    candidates_df[
                        "Candidate_Group"
                    ]
                    == class_label
                ]
                .head(
                    self.examples_per_class
                )
                .copy()
            )

            if (
                len(candidates)
                <
                self.examples_per_class
            ):

                raise ValueError(
                    f"Not enough {class_label} "
                    f"candidates. Required: "
                    f"{self.examples_per_class}, "
                    f"available: "
                    f"{len(candidates)}"
                )

            candidates[
                "Shot_Label"
            ] = (
                class_slots[
                    class_label
                ]
            )

            selected_frames.append(
                candidates
            )

        selected = pd.concat(
            selected_frames,
            ignore_index=True,
        )

        shot_order = (
            class_slots[
                self.ad_label
            ]
            +
            class_slots[
                self.cn_label
            ]
        )

        selected[
            "Shot_Label"
        ] = pd.Categorical(
            selected[
                "Shot_Label"
            ],
            categories=shot_order,
            ordered=True,
        )

        return (
            selected
            .rename(
                columns={
                    "Candidate_Subject":
                        "Example_Subject"
                }
            )
            .sort_values(
                "Shot_Label"
            )
            .reset_index(
                drop=True
            )
        )

    # MAIN FIT + SELECT
    def fit_select(
        self,
        train_data: (
            str
            | Path
            | pd.DataFrame
        ),
        test_data: (
            str
            | Path
            | pd.DataFrame
        ),
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:

         
        # Load data
         

        train_df = (
            self._load_dataframe(
                train_data,
                "Training data",
            )
        )

        test_df = (
            self._load_dataframe(
                test_data,
                "Test data",
            )
        )

         
        # Validate
         

        self._validate_data(
            train_df,
            test_df,
        )

         
        # Load models
        #
        # LR weights
        #
        # XGBoost tree gain weights
         

        self._load_models_and_weights(
            train_df
        )

         
        # Determine XGBoost feature columns
         

        self.xgboost_feature_columns = (
            self._get_xgboost_features(
                train_df
            )
        )

        # Make sure test data has them
        missing_test_features = [
            feature
            for feature
            in self.xgboost_feature_columns
            if feature
            not in test_df.columns
        ]

        if missing_test_features:

            raise ValueError(
                "XGBoost features missing from "
                "test data:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature
                    in missing_test_features
                )
            )

        # Standardise LR features
        (
            train_scaled,
            test_scaled,
        ) = (
            self._prepare_scaled_data(
                train_df,
                test_df,
            )
        )

        # Extract XGBoost paths
        train_paths = (
            core.get_feature_decision_paths(
                self.xgboost_model,
                train_df[
                    self.xgboost_feature_columns
                ],
            )
        )

        test_paths = (
            core.get_feature_decision_paths(
                self.xgboost_model,
                test_df[
                    self.xgboost_feature_columns
                ],
            )
        )

        # Validate tree count
        number_of_path_trees = (
            len(train_paths[0])
            if train_paths
            else 0
        )

        number_of_gain_weights = (
            len(
                self.xgboost_tree_weights
            )
        )

        if (
            number_of_path_trees
            != number_of_gain_weights
        ):

            raise ValueError(
                "Tree path count does not match "
                "XGBoost tree gain count.\n"
                f"Path trees: "
                f"{number_of_path_trees}\n"
                f"Gain weights: "
                f"{number_of_gain_weights}"
            )

        # Validate feature-name overlap
        path_features = {
            str(
                step.feature
            )
            .strip()
            .lower()

            for subject_paths
            in (
                train_paths
                +
                test_paths
            )

            for tree_path
            in subject_paths

            for step
            in tree_path
        }

        lr_features = {
            str(
                feature
            )
            .strip()
            .lower()

            for feature
            in self.lr_feature_weights
        }

        if not path_features.intersection(
            lr_features
        ):

            raise ValueError(
                "No XGBoost split features matched "
                "the retained Logistic Regression "
                "features. Check preprocessing "
                "feature names."
            )

        # Calculate candidate distances
        all_candidates: list[
            pd.DataFrame
        ] = []

        all_selected: list[
            pd.DataFrame
        ] = []

        for test_index in range(
            len(test_df)
        ):

            candidates = (
                self._calculate_subject_matches(
                    test_index=test_index,
                    train_df=train_df,
                    test_df=test_df,
                    train_scaled_df=(
                        train_scaled
                    ),
                    test_scaled_df=(
                        test_scaled
                    ),
                    train_paths=(
                        train_paths
                    ),
                    test_paths=(
                        test_paths
                    ),
                )
            )

            all_candidates.append(
                candidates
            )

            all_selected.append(
                self._select_balanced_examples(
                    candidates
                )
            )

        # Store all results
        self.all_candidate_distances = (
            pd.concat(
                all_candidates,
                ignore_index=True,
            )
        )

        self.selected_examples_long = (
            pd.concat(
                all_selected,
                ignore_index=True,
            )
        )

        self.selected_examples_wide = (
            self.selected_examples_long
            .pivot(
                index="Test_Subject",
                columns="Shot_Label",
                values="Example_Subject",
            )
            .reset_index()
        )

        self.selected_examples_wide.columns.name = (
            None
        )

        return (
            self.selected_examples_long,
            self.selected_examples_wide,
            self.all_candidate_distances,
        )

    # optional diagnostic exports
    def save_results(
        self,
        output_directory: str | Path,
    ) -> dict[str, Path]:

        if (
            self.selected_examples_long
            is None
            or
            self.selected_examples_wide
            is None
            or
            self.all_candidate_distances
            is None
            or
            self.lr_weights_df
            is None
            or
            self.xgboost_tree_gain_df
            is None
        ):

            raise RuntimeError(
                "Run fit_select() before "
                "save_results()."
            )

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths = {

            "selected_examples_long":
                output_directory
                / "selected_examples_long.csv",

            "selected_examples_wide":
                output_directory
                / "selected_examples_wide.csv",

            "all_candidate_distances":
                output_directory
                / "all_candidate_distances.csv",

            "logistic_feature_weights":
                output_directory
                / "logistic_feature_weights.csv",

            "xgboost_tree_gain_weights":
                output_directory
                / "xgboost_tree_gain_weights.csv",

            "selection_metadata":
                output_directory
                / "selection_metadata.json",
        }

        # Save selections
        self.selected_examples_long.to_csv(
            paths[
                "selected_examples_long"
            ],
            index=False,
        )

        self.selected_examples_wide.to_csv(
            paths[
                "selected_examples_wide"
            ],
            index=False,
        )

        # Save every candidate distance
        self.all_candidate_distances.to_csv(
            paths[
                "all_candidate_distances"
            ],
            index=False,
        )

        # Save LR feature weights
        self.lr_weights_df.to_csv(
            paths[
                "logistic_feature_weights"
            ],
            index=False,
        )

        # Save XGBoost tree gain weights
        self.xgboost_tree_gain_df.to_csv(
            paths[
                "xgboost_tree_gain_weights"
            ],
            index=False,
        )

        # Metadata
        metadata = {

            "selection_method":
                (
                    "lr_coefficient_weighted_shared_path_"
                    "with_xgboost_tree_gain_weighting"
                ),

            "logistic_regression_model_path":
                str(
                    self.logistic_regression_model_path
                ),

            "xgboost_model_path":
                str(
                    self.xgboost_model_path
                ),

            "within_tree_distance":
                (
                    "LR coefficient weighted Euclidean "
                    "distance over consecutive shared "
                    "XGBoost path features"
                ),

            "within_tree_feature_weight":
                (
                    "absolute Logistic Regression "
                    "coefficient"
                ),

            "tree_importance":
                (
                    "sum of local XGBoost split gains "
                    "over all split nodes in each tree"
                ),

            "tree_weight_normalisation":
                (
                    "tree_total_gain / "
                    "sum_total_gain_all_trees"
                ),

            "ensemble_distance":
                (
                    "gain-weighted mean of valid "
                    "per-tree shared-path distances"
                ),

            "invalid_tree_handling":
                (
                    "trees without an eligible "
                    "shared-path distance are excluded "
                    "and valid tree weights are "
                    "renormalized implicitly by "
                    "weighted average"
                ),

            "uses_alpha":
                False,

            "uses_xgboost_gain":
                True,

            "xgboost_gain_scope":
                "whole_tree",

            "number_of_lr_features":
                len(
                    self.lr_feature_columns
                ),

            "number_of_xgboost_input_features":
                len(
                    self.xgboost_feature_columns
                ),

            "number_of_xgboost_trees":
                len(
                    self.xgboost_tree_weights
                ),

            "examples_per_class":
                self.examples_per_class,

            "ad_label":
                self.ad_label,

            "cn_label":
                self.cn_label,

            "exclude_same_subject":
                self.exclude_same_subject,
        }

        with paths[
            "selection_metadata"
        ].open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        return paths