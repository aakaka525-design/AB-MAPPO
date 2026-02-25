import unittest

from generate_figures import _sort_series_labels_by_k


class TestGenerateFiguresSorting(unittest.TestCase):
    def test_numeric_k_order(self):
        labels = ["K=60", "K=100", "K=80"]
        self.assertEqual(_sort_series_labels_by_k(labels), ["K=60", "K=80", "K=100"])

    def test_non_k_label_fallback(self):
        labels = ["foo", "K=10", "bar"]
        self.assertEqual(_sort_series_labels_by_k(labels), sorted(labels))


if __name__ == "__main__":
    unittest.main()
