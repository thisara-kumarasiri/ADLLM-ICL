For each test subject and candidate training subject:

1. Standardize MRI features using training-set statistics.

2. Follow both subjects through every trained XGBoost tree.

3. Starting from each tree root, retain consecutive split features for which both subjects take the same branch.

4. Compute an LR-weighted Euclidean distance using only those shared-path features.

5. Weight each tree by the sum of its local XGBoost split gains.

6. Calculate the gain-weighted mean distance across valid trees.

7. Rank training subjects by increasing distance.

8. Select the two closest AD and two closest CN subjects.