import unittest

import numpy as np

from validation import holdout_evaluate, nested_cv_evaluate


class NestedValidationTests(unittest.TestCase):
    def test_outer_test_rows_never_reach_weight_fitting(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(0.5, 8.0, size=(20, 3))
        y = 3.0 * X[:, 0] + 1.5 * X[:, 1] + 2.0
        seen_training_sizes = []

        def weights(X_train, _y_train):
            seen_training_sizes.append(len(X_train))
            return np.ones(X_train.shape[1]) / X_train.shape[1]

        result = nested_cv_evaluate(
            X,
            y,
            weights,
            weighted=True,
            param_grid={"C": [1], "gamma": [0.1], "epsilon": [0.05]},
            outer_splits=5,
            inner_splits=3,
            random_state=11,
        )

        self.assertEqual(len(result["predictions"]), len(y))
        self.assertTrue(np.isfinite(result["predictions"]).all())
        self.assertTrue(seen_training_sizes)
        self.assertLess(max(seen_training_sizes), len(y))

    def test_nested_metrics_are_bounded(self):
        X = np.arange(1, 49, dtype=float).reshape(16, 3)
        y = np.linspace(2.0, 30.0, 16)

        result = nested_cv_evaluate(
            X,
            y,
            lambda X_train, _y_train: np.ones(X_train.shape[1]),
            weighted=False,
            param_grid={"C": [0.5, 1], "gamma": [0.1], "epsilon": [0.1]},
            outer_splits=4,
            inner_splits=3,
            random_state=5,
        )

        self.assertGreaterEqual(result["mmre"], 0.0)
        self.assertGreaterEqual(result["pred25"], 0.0)
        self.assertLessEqual(result["pred25"], 1.0)
        self.assertEqual(len(result["fold_details"]), 4)

    def test_holdout_selection_is_confined_to_training_partition(self):
        X_train = np.arange(1, 37, dtype=float).reshape(12, 3)
        y_train = np.linspace(3.0, 24.0, 12)
        X_test = np.arange(101, 113, dtype=float).reshape(4, 3)
        y_test = np.linspace(30.0, 42.0, 4)
        seen_training_sizes = []

        def weights(X_subset, _y_subset):
            seen_training_sizes.append(len(X_subset))
            self.assertLess(float(np.max(X_subset)), 100.0)
            return np.ones(X_subset.shape[1])

        result = holdout_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            weights,
            weighted=True,
            param_grid={"C": [1], "gamma": [0.1], "epsilon": [0.1]},
            inner_splits=3,
            random_state=9,
        )

        self.assertTrue(seen_training_sizes)
        self.assertEqual(max(seen_training_sizes), len(X_train))
        self.assertEqual(len(result["predictions"]), len(y_test))


if __name__ == "__main__":
    unittest.main()
