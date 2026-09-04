import unittest

from validation import prediction_interval


class PredictionIntervalTests(unittest.TestCase):
    Q10 = -0.432
    Q90 = 0.568

    def assert_interval_is_algebraically_consistent(self, prediction):
        lower, upper = prediction_interval(prediction, self.Q10, self.Q90)
        self.assertAlmostEqual(prediction / lower - 1.0, self.Q90, places=10)
        self.assertAlmostEqual(prediction / upper - 1.0, self.Q10, places=10)
        self.assertLess(lower, prediction)
        self.assertGreater(upper, prediction)
        return lower, upper

    def test_125_person_month_example(self):
        lower, upper = self.assert_interval_is_algebraically_consistent(125.0)
        self.assertAlmostEqual(lower, 79.72, places=2)
        self.assertAlmostEqual(upper, 220.07, places=2)

    def test_50_person_month_example(self):
        lower, upper = self.assert_interval_is_algebraically_consistent(50.0)
        self.assertAlmostEqual(lower, 31.89, places=2)
        self.assertAlmostEqual(upper, 88.03, places=2)

    def test_200_person_month_example(self):
        lower, upper = self.assert_interval_is_algebraically_consistent(200.0)
        self.assertAlmostEqual(lower, 127.55, places=2)
        self.assertAlmostEqual(upper, 352.11, places=2)


if __name__ == "__main__":
    unittest.main()
