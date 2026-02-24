"""Unit tests for retrieval metrics."""

import math

import pytest

from harness.metrics.retrieval import (
    compute_all_retrieval_metrics,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect_retrieval(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_no_overlap(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b", "c"], k=3) == 0.0

    def test_partial_overlap(self):
        assert precision_at_k(["a", "x", "b"], ["a", "b", "c"], k=3) == pytest.approx(2 / 3)

    def test_k_larger_than_retrieved(self):
        assert precision_at_k(["a", "b"], ["a", "b", "c"], k=5) == pytest.approx(2 / 5)

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_k_one(self):
        assert precision_at_k(["a", "b"], ["a"], k=1) == 1.0
        assert precision_at_k(["b", "a"], ["a"], k=1) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_no_overlap(self):
        assert recall_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial(self):
        assert recall_at_k(["a", "x"], ["a", "b"], k=2) == pytest.approx(0.5)

    def test_empty_gold(self):
        assert recall_at_k(["a", "b"], [], k=3) == 0.0

    def test_k_zero(self):
        assert recall_at_k(["a"], ["a"], k=0) == 0.0


class TestMRR:
    def test_first_is_relevant(self):
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_is_relevant(self):
        assert mrr(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)

    def test_no_relevant(self):
        assert mrr(["x", "y", "z"], ["a"]) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], ["a"]) == 0.0


class TestNDCGAtK:
    def test_perfect(self):
        result = ndcg_at_k(["a", "b"], ["a", "b"], k=2)
        assert result == pytest.approx(1.0)

    def test_no_relevant(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_empty_gold(self):
        assert ndcg_at_k(["a", "b"], [], k=2) == 0.0

    def test_single_relevant_at_pos2(self):
        # Relevant item at position 2 → DCG = 1/log2(3), IDCG = 1/log2(2)
        result = ndcg_at_k(["x", "a"], ["a"], k=2)
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert result == pytest.approx(expected)


class TestComputeAll:
    def test_returns_all_keys(self):
        result = compute_all_retrieval_metrics(
            ["a", "b", "c", "d", "e"],
            ["a", "c"],
            k_values=[1, 3, 5],
        )
        assert "precision@1" in result
        assert "recall@3" in result
        assert "ndcg@5" in result
        assert "mrr" in result

    def test_values_in_range(self):
        result = compute_all_retrieval_metrics(
            ["a", "x", "b", "y", "z"],
            ["a", "b", "c"],
            k_values=[1, 3, 5, 10],
        )
        for key, value in result.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"
