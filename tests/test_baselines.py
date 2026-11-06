import unittest

import numpy as np

import baselines


def project(kloc, mode, multipliers=1.0):
    row = np.ones(17)
    row[0], row[16] = kloc, mode
    row[1:16] = multipliers ** (1 / 15)   # the 15 multipliers multiply to `multipliers`
    return row


class CocomoTests(unittest.TestCase):
    def test_published_equation_matches_boehm_by_hand(self):
        X = np.array([project(10, 1), project(10, 3, multipliers=1.5)])
        expected = [3.2 * 10 ** 1.05, 2.8 * 10 ** 1.20 * 1.5]
        np.testing.assert_allclose(baselines.cocomo_published()(X), expected, rtol=1e-9)

    def test_calibration_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        X = np.array([project(k, m, e) for k, m, e in zip(rng.uniform(2, 300, 30), np.tile([1, 2, 3], 10),
                                                            rng.uniform(.5, 2, 30))])
        a = np.array([2.0, 2.5, 3.0])[X[:, 16].astype(int) - 1]
        y = a * X[:, 0] ** 1.1 * X[:, 1:16].prod(axis=1)
        np.testing.assert_allclose(baselines.cocomo_calibrated(X, y)(X), y, rtol=1e-8)


class MeasureTests(unittest.TestCase):
    def test_random_guessing_mae_is_mean_pairwise_difference(self):
        y = np.array([1.0, 2.0, 4.0])
        # pairs: |1-2|=1, |1-4|=3, |2-4|=2 -> mean over ordered pairs = 2
        self.assertAlmostEqual(baselines.random_guess_mae(y), 2.0)

    def test_paired_comparison_sign_and_interval(self):
        y = np.linspace(10, 100, 40)
        better, worse = y + 1.0, y + 5.0
        r = baselines.paired(y, better, worse)
        self.assertAlmostEqual(r["mae_difference"], -4.0)
        self.assertLess(r["ci95"][1], 0)
        self.assertEqual(r["projects_a_better"], 40)

    def test_folds_never_see_their_test_rows(self):
        X = np.array([project(k, 1) for k in range(1, 21)], dtype=float)
        y = np.arange(1.0, 21.0)
        seen = []

        def fit(X_train, y_train):
            seen.append(set(y_train))
            return lambda X_test: np.zeros(len(X_test))

        baselines.cross_validated(fit, X, y)
        self.assertEqual(len(seen), 10)
        self.assertTrue(all(len(s) == 18 for s in seen))


if __name__ == "__main__":
    unittest.main()
