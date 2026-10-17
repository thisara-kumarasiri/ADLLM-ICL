from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ModelType = Literal["logistic", "xgboost"]


def load_model(
    model_path: str | Path,
) -> Any:
    """
    Load a saved estimator or sklearn Pipeline.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def get_final_estimator(model: Any) -> Any:
    """
    Return the final estimator.

    Supports:
    - A directly saved estimator
    - An sklearn Pipeline whose final step is the estimator
    """
    if isinstance(model, Pipeline):
        return model.steps[-1][1]

    return model


# Backward-compatible alias for the existing logistic selector.
# def get_logistic_estimator(model: Any) -> Any:
#     """
#     Return the final estimator.

#     Kept for compatibility with code that previously imported
#     get_logistic_estimator().
#     """
#     return get_final_estimator(model)


def clean_feature_name(
    transformed_name: str,
) -> str:
    """
    Remove sklearn transformer prefixes.

    Examples
    --------
    numeric__Left Hippocampus Volume
        -> Left Hippocampus Volume

    preprocessing__numeric__Left Hippocampus Volume
        -> Left Hippocampus Volume
    """
    transformed_name = str(transformed_name)

    if "__" in transformed_name:
        return transformed_name.split("__")[-1]

    return transformed_name


def get_model_feature_names(
    model: Any,
    estimator: Any | None = None,
    fallback_feature_names: list[str] | None = None,
    expected_feature_count: int | None = None,
) -> list[str]:
    """
    Get feature names corresponding to the features seen by the final
    estimator.

    This works for logistic regression and XGBoost. For an sklearn
    Pipeline, the names are taken from the preprocessing output.

    Parameters
    ----------
    model:
        Saved estimator or sklearn Pipeline.

    estimator:
        Final estimator. It is inferred when omitted.

    fallback_feature_names:
        Names to use when the saved model does not expose feature names.

    expected_feature_count:
        Number of estimator input features. This is used to validate
        candidate feature-name sources.
    """
    if estimator is None:
        estimator = get_final_estimator(model)

    if isinstance(model, Pipeline):
        preprocessing_pipeline = model[:-1]

        if hasattr(preprocessing_pipeline,
            "get_feature_names_out",
        ):
            try:
                names = list(
                    preprocessing_pipeline.get_feature_names_out()
                )

                if (
                    expected_feature_count is None
                    or len(names) == expected_feature_count
                ):
                    return names

            except (AttributeError, TypeError, ValueError):
                pass

    if hasattr(estimator, "feature_names_in_"):
        names = list(estimator.feature_names_in_)

        if (
            expected_feature_count is None
            or len(names) == expected_feature_count
        ):
            return names

    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)

        if (
            expected_feature_count is None
            or len(names) == expected_feature_count
        ):
            return names

    # XGBoost's native Booster may retain feature names.
    if hasattr(estimator, "get_booster"):
        booster = estimator.get_booster()

        if booster.feature_names is not None:
            names = list(booster.feature_names)

            if (
                expected_feature_count is None
                or len(names) == expected_feature_count
            ):
                return names

    if fallback_feature_names is not None:
        if (
            expected_feature_count is None
            or len(fallback_feature_names)
            == expected_feature_count
        ):
            return list(fallback_feature_names)

    count_message = (
        ""
        if expected_feature_count is None
        else f"\nExpected features: {expected_feature_count}"
    )

    raise ValueError(
        "Could not determine the transformed feature names "
        "corresponding to the model inputs."
        f"{count_message}"
    )


def _apply_feature_exclusions(
    weights_df: pd.DataFrame,
    exclude_features: set[str] | None,
) -> pd.DataFrame:
    """
    Remove excluded features using a case-insensitive comparison.
    """
    if not exclude_features:
        return weights_df.copy()

    excluded_lower = {
        str(feature).strip().lower()
        for feature in exclude_features
    }

    return weights_df[
        ~weights_df["Feature"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(excluded_lower)
    ].copy()


def extract_logistic_weights(
    model: Any,
    fallback_feature_names: list[str] | None = None,
    exclude_features: set[str] | None = None,
) -> pd.DataFrame:
    """
    Convert logistic-regression coefficients into normalized,
    non-negative feature weights.
    """
    estimator = get_final_estimator(model)

    if not hasattr(estimator, "coef_"):
        raise TypeError(
            "The saved object does not contain logistic-regression "
            "coefficients."
        )

    coefficients = np.asarray(
        estimator.coef_,
        dtype=float,
    )

    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)

    feature_count = coefficients.shape[1]

    transformed_feature_names = get_model_feature_names(
        model=model,
        estimator=estimator,
        fallback_feature_names=fallback_feature_names,
        expected_feature_count=feature_count,
    )

    if coefficients.shape[0] == 1:
        signed_coefficients = coefficients[0]
        absolute_coefficients = np.abs(coefficients[0])
    else:
        signed_coefficients = coefficients.mean(axis=0)
        absolute_coefficients = np.abs(
            coefficients
        ).mean(axis=0)

    weights_df = pd.DataFrame(
        {
            "Feature": [
                clean_feature_name(name)
                for name in transformed_feature_names
            ],
            "Transformed_Feature": transformed_feature_names,
            "Coefficient": signed_coefficients,
            "Absolute_Coefficient": absolute_coefficients,
        }
    )

    weights_df = _apply_feature_exclusions(
        weights_df=weights_df,
        exclude_features=exclude_features,
    )

    total_weight = weights_df[
        "Absolute_Coefficient"
    ].sum()

    if total_weight <= 0:
        raise ValueError(
            "All retained logistic-regression coefficients are zero."
        )

    weights_df["Normalized_Weight"] = (
        weights_df["Absolute_Coefficient"]
        / total_weight
    )

    return weights_df[
        [
            "Feature",
            "Transformed_Feature",
            "Coefficient",
            "Absolute_Coefficient",
            "Normalized_Weight",
        ]
    ].sort_values(
        by="Normalized_Weight",
        ascending=False,
    ).reset_index(drop=True)


def extract_xgboost_gain_weights(
    model: Any,
    fallback_feature_names: list[str] | None = None,
    exclude_features: set[str] | None = None,
) -> pd.DataFrame:
    """
    Extract XGBoost gain importance and convert it into normalized
    non-negative feature weights.

    Features unused in any split receive a gain of zero. They are
    retained in the returned table but contribute no distance weight.
    """
    estimator = get_final_estimator(model)

    if not hasattr(estimator, "get_booster"):
        raise TypeError(
            "The saved object does not appear to be an XGBoost "
            "estimator because get_booster() is unavailable."
        )

    booster = estimator.get_booster()

    feature_count = getattr(
        estimator,
        "n_features_in_",
        None,
    )

    if feature_count is None:
        feature_count = booster.num_features()

    transformed_feature_names = get_model_feature_names(
        model=model,
        estimator=estimator,
        fallback_feature_names=fallback_feature_names,
        expected_feature_count=feature_count,
    )

    raw_gain = booster.get_score(
        importance_type="gain"
    )

    mapped_gain: dict[str, float] = {
        name: 0.0
        for name in transformed_feature_names
    }

    for raw_name, gain_value in raw_gain.items():
        if raw_name in mapped_gain:
            mapped_gain[raw_name] = float(gain_value)
            continue

        # Native XGBoost names may be f0, f1, f2, ...
        if (
            raw_name.startswith("f")
            and raw_name[1:].isdigit()
        ):
            feature_index = int(raw_name[1:])

            if feature_index < len(
                transformed_feature_names
            ):
                mapped_name = transformed_feature_names[
                    feature_index
                ]
                mapped_gain[mapped_name] = float(
                    gain_value
                )

    weights_df = pd.DataFrame(
        {
            "Feature": [
                clean_feature_name(name)
                for name in transformed_feature_names
            ],
            "Transformed_Feature": transformed_feature_names,
            "Gain": [
                mapped_gain[name]
                for name in transformed_feature_names
            ],
        }
    )

    weights_df = _apply_feature_exclusions(
        weights_df=weights_df,
        exclude_features=exclude_features,
    )

    total_gain = weights_df["Gain"].sum()

    if total_gain <= 0:
        raise ValueError(
            "All retained XGBoost gain values are zero."
        )

    weights_df["Normalized_Weight"] = (
        weights_df["Gain"] / total_gain
    )

    return weights_df[
        [
            "Feature",
            "Transformed_Feature",
            "Gain",
            "Normalized_Weight",
        ]
    ].sort_values(
        by="Normalized_Weight",
        ascending=False,
    ).reset_index(drop=True)


def extract_model_weights(
    model: Any,
    model_type: ModelType,
    fallback_feature_names: list[str] | None = None,
    exclude_features: set[str] | None = None,
) -> pd.DataFrame:
    """
    Dispatch to the correct feature-weight extraction method.

    Parameters
    ----------
    model_type:
        Either "logistic" or "xgboost".
    """
    normalized_model_type = model_type.strip().lower()

    if normalized_model_type == "logistic":
        return extract_logistic_weights(
            model=model,
            fallback_feature_names=fallback_feature_names,
            exclude_features=exclude_features,
        )

    if normalized_model_type == "xgboost":
        return extract_xgboost_gain_weights(
            model=model,
            fallback_feature_names=fallback_feature_names,
            exclude_features=exclude_features,
        )

    raise ValueError(
        "Unsupported model_type "
        f"{model_type!r}. Use 'logistic' or 'xgboost'."
    )


def prepare_scaled_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardise features before calculating distances.

    The scaler and missing-value medians are fitted using only the
    training data.
    """
    missing_train = [
        column
        for column in feature_columns
        if column not in train_df.columns
    ]

    missing_test = [
        column
        for column in feature_columns
        if column not in test_df.columns
    ]

    if missing_train:
        raise ValueError(
            "Features missing from training data:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_train
            )
        )

    if missing_test:
        raise ValueError(
            "Features missing from test data:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_test
            )
        )

    train_features = (
        train_df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    test_features = (
        test_df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    training_medians = train_features.median()

    train_features = train_features.fillna(
        training_medians
    )
    test_features = test_features.fillna(
        training_medians
    )

    if train_features.isna().any().any():
        all_missing_columns = train_features.columns[
            train_features.isna().any()
        ].tolist()

        raise ValueError(
            "Some training features contain only missing values "
            "and cannot be imputed:\n"
            + "\n".join(
                f"  - {column}"
                for column in all_missing_columns
            )
        )

    scaler = StandardScaler()

    train_scaled_values = scaler.fit_transform(
        train_features
    )
    test_scaled_values = scaler.transform(
        test_features
    )

    train_scaled_df = pd.DataFrame(
        train_scaled_values,
        columns=feature_columns,
        index=train_df.index,
    )

    test_scaled_df = pd.DataFrame(
        test_scaled_values,
        columns=feature_columns,
        index=test_df.index,
    )

    return (
        train_scaled_df,
        test_scaled_df,
        scaler,
    )


def calculate_weighted_distance(
    test_row: pd.Series,
    candidate_row: pd.Series,
    feature_columns: list[str],
    feature_weights: dict[str, float],
) -> float:
    """
    Calculate weighted Euclidean distance.

    The weights may come from logistic coefficients, XGBoost gain,
    SHAP importance, or another non-negative weighting method.
    """
    missing_weights = [
        feature
        for feature in feature_columns
        if feature not in feature_weights
    ]

    if missing_weights:
        raise ValueError(
            "Weights are missing for these features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_weights
            )
        )

    test_values = test_row[
        feature_columns
    ].to_numpy(dtype=float)

    candidate_values = candidate_row[
        feature_columns
    ].to_numpy(dtype=float)

    weights = np.array(
        [
            feature_weights[feature]
            for feature in feature_columns
        ],
        dtype=float,
    )

    if np.any(weights < 0):
        raise ValueError(
            "Feature weights must be non-negative."
        )

    total_weight = weights.sum()

    if total_weight <= 0:
        raise ValueError(
            "The feature weights must sum to a positive value."
        )

    # Protect against callers passing weights that were not yet normalized.
    weights = weights / total_weight

    squared_difference = (
        test_values - candidate_values
    ) ** 2

    return float(
        np.sqrt(
            np.sum(weights * squared_difference)
        )
    )


def calculate_all_candidate_distances(
    test_subject: str | int,
    test_scaled_df: pd.DataFrame,
    train_scaled_df: pd.DataFrame,
    train_original_df: pd.DataFrame,
    feature_columns: list[str],
    feature_weights: dict[str, float],
    subject_column: str = "Subject",
    group_column: str = "Group",
    test_original_df: pd.DataFrame | None = None,
    exclude_same_subject: bool = True,
) -> pd.DataFrame:
    """
    Calculate distances from one test subject to all training subjects.

    The preferred approach is to provide test_original_df. Its index is
    then used to locate the corresponding row in test_scaled_df.

    For backward compatibility, the function also supports a scaled
    DataFrame that already contains the subject column.
    """
    if test_original_df is not None:
        if subject_column not in test_original_df.columns:
            raise ValueError(
                f"{subject_column!r} is missing from test_original_df."
            )

        test_mask = (
            test_original_df[subject_column]
            .astype(str)
            == str(test_subject)
        )

        matching_indices = test_original_df.index[
            test_mask
        ].tolist()

        if not matching_indices:
            raise ValueError(
                f"Test subject {test_subject!r} was not found "
                "in test_original_df."
            )

        test_index = matching_indices[0]

        if test_index not in test_scaled_df.index:
            raise ValueError(
                f"Index {test_index!r} for test subject "
                f"{test_subject!r} is missing from test_scaled_df."
            )

        test_row = test_scaled_df.loc[test_index]

    elif subject_column in test_scaled_df.columns:
        test_matches = test_scaled_df[
            test_scaled_df[subject_column]
            .astype(str)
            == str(test_subject)
        ]

        if test_matches.empty:
            raise ValueError(
                f"Test subject {test_subject!r} was not found "
                "in test_scaled_df."
            )

        test_row = test_matches.iloc[0]

    else:
        raise ValueError(
            "The scaled test data normally contains only feature "
            "columns. Pass test_original_df so the test subject can "
            "be mapped to the correct scaled row."
        )

    required_train_columns = {
        subject_column,
        group_column,
    }

    missing_original_columns = (
        required_train_columns
        - set(train_original_df.columns)
    )

    if missing_original_columns:
        raise ValueError(
            "Columns missing from train_original_df: "
            f"{sorted(missing_original_columns)}"
        )

    results: list[dict[str, object]] = []

    for index, candidate_row in (
        train_scaled_df.iterrows()
    ):
        if index not in train_original_df.index:
            raise ValueError(
                f"Training index {index!r} is missing from "
                "train_original_df."
            )

        candidate_subject = train_original_df.loc[
            index,
            subject_column,
        ]

        if (
            exclude_same_subject
            and str(candidate_subject)
            == str(test_subject)
        ):
            continue

        distance = calculate_weighted_distance(
            test_row=test_row,
            candidate_row=candidate_row,
            feature_columns=feature_columns,
            feature_weights=feature_weights,
        )

        results.append(
            {
                "Test_Subject": test_subject,
                "Candidate_Subject": candidate_subject,
                "Candidate_Group": train_original_df.loc[
                    index,
                    group_column,
                ],
                "Distance": distance,
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "Test_Subject",
                "Candidate_Subject",
                "Candidate_Group",
                "Distance",
            ]
        )

    return pd.DataFrame(results).sort_values(
        by="Distance",
        ascending=True,
    ).reset_index(drop=True)