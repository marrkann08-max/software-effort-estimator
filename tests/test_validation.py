import unittest

import numpy as np

from validation import nested_cv_evaluate


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


if __name__ == "__main__":
    unittest.main()
