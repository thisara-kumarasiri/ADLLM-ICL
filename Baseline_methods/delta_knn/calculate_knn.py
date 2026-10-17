import os
import numpy as np
from sklearn.preprocessing import StandardScaler


def cosine_similarity(a, b):
   return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def get_dk_nearest_examples(dataset_test, dataset_train, test_doc, train_docs, dk=13):
#     # Note: embedding model should be the model you used to obtain the embedding for each document,
#     # make sure that you alredy obtained the `doc_name.npy` file for each doc under the 
#     # corresponding embedding model folder in `data/embeddings/{dataset}/{embedding_model}/`
#     # dk is a hyperparameter for the number of nearest neighbors to select, can be tuned based on validation set performance.
    
#     embedding_model = "your_embedding_model_name" # e.g., "text-embedding-3-small"
#     train_set_save_path = f'data/embeddings/{dataset_train}/{embedding_model}'
#     test_set_save_path = f'data/embeddings/{dataset_test}/{embedding_model}'
    
#     doc_ij_sim = []

#     doci_np = np.load(os.path.join(test_set_save_path, f"{test_doc}.npy"))
#     for train_doc in train_docs:
#         docj_np = np.load(os.path.join(train_set_save_path, f"{train_doc}.npy"))
#         doc_ij_sim.append([train_doc, cosine_similarity(doci_np, docj_np)])
#     doc_ij_sim = sorted(doc_ij_sim, key=lambda x: x[1], reverse=True)

#     return [doc_ij[0] for doc_ij in doc_ij_sim[:dk]], [doc_ij[1] for doc_ij in doc_ij_sim[:dk]]

def get_dk_nearest_examples(
    train_docs,
    test_doc,
    dk=13
):
    """
    Find the dk training subjects most similar to one test subject
    using standardized brain-region features.

    Parameters
    ----------
    train_docs : list of dict
        220 training subjects.

    test_doc : dict
        One test subject.

    dk : int
        Number of nearest training representatives.

    Returns
    -------
    nearest_subjects : list
        Subject IDs.

    nearest_similarities : list
        Cosine similarities.
    """

    X_train = np.vstack([
        doc["features"]
        for doc in train_docs
    ])

    X_test = np.asarray(
        test_doc["features"],
        dtype=float
    ).reshape(1, -1)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )[0]

    similarities = []

    for train_vector in X_train_scaled:

        similarity = cosine_similarity(
            X_test_scaled,
            train_vector
        )

        similarities.append(
            similarity
        )

    similarities = np.asarray(
        similarities
    )

    sorted_indices = np.argsort(
        similarities
    )[::-1]

    nearest_indices = sorted_indices[:dk]

    nearest_subjects = [
        train_docs[i]["name"]
        for i in nearest_indices
    ]

    nearest_similarities = [
        float(similarities[i])
        for i in nearest_indices
    ]

    return (
        nearest_subjects,
        nearest_similarities
    )