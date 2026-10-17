from __future__ import annotations

import hashlib
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import Settings as cfg

SUPPORTED_METHODS = {
    "random",
    "cosine",
    "euclidean",
    "manhattan",
    "mahalanobis",
}

def validate_docs(
    train_docs,
    test_docs,
):
    """
    Validate the common MRI subject representation.

    Expected subject structure:

        {
            "name": "002_S_XXXX",
            "gold": "AD" or "CN",
            "features": np.array([...]),
            "feature_dict": {...}
        }
    """

    if not train_docs:

        raise ValueError(
            "train_docs is empty."
        )

    if not test_docs:

        raise ValueError(
            "test_docs is empty."
        )

    # Validate training subjects
    for doc in train_docs:

        for key in [
            "name",
            "gold",
            "features",
        ]:

            if key not in doc:

                raise ValueError(
                    f"Training subject is missing "
                    f"'{key}': {doc}"
                )

        label = str(
            doc["gold"]
        ).strip().upper()

        if label not in cfg.VALID_LABELS:

            raise ValueError(
                f"Unexpected label "
                f"'{label}' for "
                f"{doc['name']}."
            )

    # Validate test subjects
    for doc in test_docs:

        for key in [
            "name",
            "features",
        ]:

            if key not in doc:

                raise ValueError(
                    f"Test subject is missing "
                    f"'{key}': {doc}"
                )
    # Check feature-vector lengths
    n_features = len(
        train_docs[0]["features"]
    )

    for doc in train_docs:

        if (
            len(doc["features"])
            != n_features
        ):

            raise ValueError(
                "Training feature-vector "
                "lengths are inconsistent."
            )

    for doc in test_docs:

        if (
            len(doc["features"])
            != n_features
        ):

            raise ValueError(
                "Training and test feature-vector "
                "lengths do not match."
            )


# PREPARE FEATURES

def prepare_scaled_features(
    train_docs,
    test_docs,
):
    """
    Standardise MRI features.

    Scaling parameters are learned ONLY from the
    training subjects.

                    x - training mean
            z = -------------------------
                training standard deviation
    """

    X_train = np.vstack([
        np.asarray(
            doc["features"],
            dtype=float,
        )
        for doc in train_docs
    ])

    X_test = np.vstack([
        np.asarray(
            doc["features"],
            dtype=float,
        )
        for doc in test_docs
    ])

    # Missing / invalid values
    if not np.all(
        np.isfinite(X_train)
    ):

        raise ValueError(
            "Training MRI features contain "
            "NaN or infinite values."
        )

    if not np.all(
        np.isfinite(X_test)
    ):

        raise ValueError(
            "Test MRI features contain "
            "NaN or infinite values."
        )

    # Standardisation
    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    return (
        X_train_scaled,
        X_test_scaled,
    )


# MAHALANOBIS MATRIX

def prepare_mahalanobis_matrix(
    X_train_scaled,
):
    """
    Prepare inverse covariance matrix for
    Mahalanobis distance.

    Covariance is calculated using training data only.

    pinv() is used instead of inv() because MRI features
    can be highly correlated and the covariance matrix
    may be singular.
    """

    covariance_matrix = np.cov(
        X_train_scaled,
        rowvar=False,
    )

    covariance_matrix = np.atleast_2d(
        covariance_matrix
    )

    # Small regularisation for numerical stability
    covariance_matrix = (
        covariance_matrix
        +
        cfg.MAHALANOBIS_REGULARIZATION
        *
        np.eye(
            covariance_matrix.shape[0]
        )
    )

    precision_matrix = np.linalg.pinv(
        covariance_matrix
    )

    return precision_matrix


# CALCULATE DISTANCES

def calculate_distances(
    method,
    test_vector,
    X_train_scaled,
    mahalanobis_vi=None,
):
    """
    Calculate the distance from one test subject
    to every training subject.

    Smaller value = more similar.

    For cosine we use cosine DISTANCE:

        cosine_distance = 1 - cosine_similarity
    """

    test_vector = (
        np.asarray(
            test_vector,
            dtype=float,
        )
        .reshape(
            1,
            -1,
        )
    )

    # EUCLIDEAN

    if method == "euclidean":

        distances = cdist(
            test_vector,
            X_train_scaled,
            metric="euclidean",
        )[0]

    # MANHATTAN

    elif method == "manhattan":

        # scipy calls Manhattan distance "cityblock"

        distances = cdist(
            test_vector,
            X_train_scaled,
            metric="cityblock",
        )[0]

    # COSINE

    elif method == "cosine":

        distances = cdist(
            test_vector,
            X_train_scaled,
            metric="cosine",
        )[0]

    # MAHALANOBIS

    elif method == "mahalanobis":

        if mahalanobis_vi is None:

            raise ValueError(
                "Mahalanobis inverse covariance "
                "matrix was not provided."
            )

        distances = cdist(
            test_vector,
            X_train_scaled,
            metric="mahalanobis",
            VI=mahalanobis_vi,
        )[0]

    else:

        raise ValueError(
            f"Unsupported distance method: "
            f"{method}"
        )

    distances = np.asarray(
        distances,
        dtype=float,
    )

    # Invalid distances should never rank highly
    distances[
        ~np.isfinite(
            distances
        )
    ] = np.inf

    return distances


# SELECT CLOSEST AD + CN

def select_distance_examples(
    train_docs,
    distances,
    test_name,
    number_per_class,
):
    """
    Rank all training subjects from closest to furthest,
    then independently select:

        number_per_class AD
        number_per_class CN
    """

    ranked_indices = np.argsort(
        distances
    )

    selected_ad = []
    selected_cn = []

    for train_idx in ranked_indices:

        train_doc = (
            train_docs[
                train_idx
            ]
        )

        train_name = str(
            train_doc["name"]
        )

        label = str(
            train_doc["gold"]
        ).strip().upper()

        # Do not select the test subject itself if datasets
        # happen to contain overlapping subject IDs.
        if (
            train_name
            == str(test_name)
        ):

            continue

        # AD
        if (
            label == "AD"
            and
            len(selected_ad)
            < number_per_class
        ):

            selected_ad.append(
                train_name
            )

        # CN
        elif (
            label == "CN"
            and
            len(selected_cn)
            < number_per_class
        ):

            selected_cn.append(
                train_name
            )

        # Finished
        if (
            len(selected_ad)
            == number_per_class
            and
            len(selected_cn)
            == number_per_class
        ):

            break

    if (
        len(selected_ad)
        != number_per_class
    ):

        raise ValueError(
            f"Could not find enough AD "
            f"examples for {test_name}."
        )

    if (
        len(selected_cn)
        != number_per_class
    ):

        raise ValueError(
            f"Could not find enough CN "
            f"examples for {test_name}."
        )

    return (
        selected_ad,
        selected_cn,
    )


# REPRODUCIBLE RANDOM SEED

def get_subject_random_seed(
    test_name,
):
    """
    Create a deterministic random seed for one
    test subject.

    This means random selections are reproducible
    but can differ between test subjects.
    """

    seed_string = (
        f"{cfg.RANDOM_SELECTION_SEED}:"
        f"{test_name}"
    )

    digest = hashlib.sha256(
        seed_string.encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


# RANDOM SELECTION

def select_random_examples(
    train_docs,
    test_name,
    number_per_class,
):
    """
    Random balanced baseline.

    Select:

        number_per_class AD
        number_per_class CN

    without replacement.
    """

    ad_indices = []

    cn_indices = []

    for index, doc in enumerate(
        train_docs
    ):

        subject_name = str(
            doc["name"]
        )

        # Exclude identical subject if train/test overlap
        if (
            subject_name
            == str(test_name)
        ):

            continue

        label = str(
            doc["gold"]
        ).strip().upper()

        if label == "AD":

            ad_indices.append(
                index
            )

        elif label == "CN":

            cn_indices.append(
                index
            )

    if (
        len(ad_indices)
        < number_per_class
    ):

        raise ValueError(
            "Not enough AD training "
            "subjects for random selection."
        )

    if (
        len(cn_indices)
        < number_per_class
    ):

        raise ValueError(
            "Not enough CN training "
            "subjects for random selection."
        )

    # Subject-specific deterministic generator
    rng = np.random.default_rng(
        get_subject_random_seed(
            test_name
        )
    )

    selected_ad_indices = (
        rng.choice(
            ad_indices,
            size=number_per_class,
            replace=False,
        )
    )

    selected_cn_indices = (
        rng.choice(
            cn_indices,
            size=number_per_class,
            replace=False,
        )
    )

    selected_ad = [
        str(
            train_docs[index]["name"]
        )
        for index
        in selected_ad_indices
    ]

    selected_cn = [
        str(
            train_docs[index]["name"]
        )
        for index
        in selected_cn_indices
    ]

    return (
        selected_ad,
        selected_cn,
    )


# MAIN EXAMPLE-SELECTION FUNCTION

def prepare_target_examples(
    train_docs,
    test_docs,
    method,
    n_shot=4,
):
    """
    Select examples using one of the standard baseline
    methods.

    Supported methods:

        random
        cosine
        euclidean
        manhattan
        mahalanobis


    Returns
    -------
    batches : list of dict

    Example:

        {
            "name": "002_S_0413",
            "gold": "CN",

            "selected_ad": [
                "AD_subject_1",
                "AD_subject_2"
            ],

            "selected_cn": [
                "CN_subject_1",
                "CN_subject_2"
            ],

            "target_docs": [
                ...
            ]
        }

    This is the same common structure returned by
    Delta-KNN and ByCS.
    """

    method = str(
        method
    ).strip().lower()

    # VALIDATE CONFIGURATION

    if (
        method
        not in SUPPORTED_METHODS
    ):

        raise ValueError(
            f"Unsupported baseline method "
            f"'{method}'. "
            f"Available methods: "
            f"{sorted(SUPPORTED_METHODS)}"
        )

    if n_shot <= 0:

        raise ValueError(
            "n_shot must be greater than 0."
        )

    if n_shot % 2 != 0:

        raise ValueError(
            "n_shot must be even because "
            "examples are balanced between "
            "AD and CN."
        )

    number_per_class = (
        n_shot // 2
    )

    # VALIDATE DATA

    validate_docs(
        train_docs,
        test_docs,
    )

    # FEATURE PREPARATION
    # RANDOM does not need MRI distances.

    X_train_scaled = None
    X_test_scaled = None

    mahalanobis_vi = None

    if method != "random":

        (
            X_train_scaled,
            X_test_scaled,
        ) = prepare_scaled_features(
            train_docs,
            test_docs,
        )

        # Mahalanobis matrix calculated once for all subjects
        if (
            method
            == "mahalanobis"
        ):

            mahalanobis_vi = (
                prepare_mahalanobis_matrix(
                    X_train_scaled
                )
            )

    # SELECT EXAMPLES FOR EVERY TEST SUBJECT

    batches = []

    for test_idx, test_doc in enumerate(
        test_docs
    ):

        test_name = str(
            test_doc["name"]
        )

        print(
            f"\n{method.upper()} "
            f"{test_idx + 1}/"
            f"{len(test_docs)}: "
            f"{test_name}"
        )

        # RANDOM

        if method == "random":

            (
                selected_ad,
                selected_cn,
            ) = select_random_examples(
                train_docs=
                    train_docs,

                test_name=
                    test_name,

                number_per_class=
                    number_per_class,
            )

        # DISTANCE METHODS

        else:

            test_vector = (
                X_test_scaled[
                    test_idx
                ]
            )

            distances = (
                calculate_distances(
                    method=
                        method,

                    test_vector=
                        test_vector,

                    X_train_scaled=
                        X_train_scaled,

                    mahalanobis_vi=
                        mahalanobis_vi,
                )
            )

            (
                selected_ad,
                selected_cn,
            ) = select_distance_examples(
                train_docs=
                    train_docs,

                distances=
                    distances,

                test_name=
                    test_name,

                number_per_class=
                    number_per_class,
            )

        # PRINT SELECTION

        print(
            f"Selected AD: "
            f"{selected_ad}"
        )

        print(
            f"Selected CN: "
            f"{selected_cn}"
        )

        # COMMON RETURN FORMAT

        batches.append({

            "name":
                test_name,

            "gold":
                test_doc.get(
                    "gold"
                ),

            "selected_ad":
                selected_ad,

            "selected_cn":
                selected_cn,

            "target_docs":
                (
                    selected_ad
                    +
                    selected_cn
                ),
        })

    return batches