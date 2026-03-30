"""Unit tests for retrieval metrics."""

import math

import pytest

from harness.metrics.retrieval import (
    compute_all_retrieval_metrics,
    f1_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    success_at_k,
)


class TestPrecisionAtK:
    def test_perfect_retrieval(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_no_overlap(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b", "c"], k=3) == 0.0

    def test_partial_overlap(self):
        assert precision_at_k(["a", "x", "b"], ["a", "b", "c"], k=3) == pytest.approx(2 / 3)

    def test_k_larger_than_retrieved(self):
        assert precision_at_k(["a", "b"], ["a", "b", "c"], k=5) == pytest.approx(1.0)

    def test_k_larger_than_retrieved_partial(self):
        assert precision_at_k(["a", "x"], ["a", "b", "c"], k=5) == pytest.approx(0.5)

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], k=5) == 0.0

    def test_k_one(self):
        assert precision_at_k(["a", "b"], ["a"], k=1) == 1.0
        assert precision_at_k(["b", "a"], ["a"], k=1) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_perfect_recall_when_k_less_than_gold_size(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b", "c", "d", "e"], k=3) == 1.0

    def test_no_overlap(self):
        assert recall_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial(self):
        assert recall_at_k(["a", "x"], ["a", "b"], k=2) == pytest.approx(0.5)

    def test_partial_when_k_less_than_gold_size(self):
        assert recall_at_k(["a", "x", "b"], ["a", "b", "c", "d"], k=3) == pytest.approx(2 / 3)

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
        assert "success@5" in result
        assert "f1@3" in result
        assert "mrr" in result

    def test_values_in_range(self):
        result = compute_all_retrieval_metrics(
            ["a", "x", "b", "y", "z"],
            ["a", "b", "c"],
            k_values=[1, 3, 5, 10],
        )
        for key, value in result.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"

    def test_precision_not_penalized_for_short_result_list(self):
        result = compute_all_retrieval_metrics(
            ["a", "b"],
            ["a", "b", "c"],
            k_values=[1, 3, 5],
        )
        assert result["precision@1"] == 1.0
        assert result["precision@3"] == 1.0
        assert result["precision@5"] == 1.0

    def test_recall_can_reach_one_when_k_less_than_gold_size(self):
        result = compute_all_retrieval_metrics(
            ["a", "b", "c"],
            ["a", "b", "c", "d", "e"],
            k_values=[1, 3, 5],
        )
        assert result["recall@1"] == 1.0
        assert result["recall@3"] == 1.0
        assert result["recall@5"] == 0.6


class TestSuccessAtK:
    def test_all_found(self):
        assert success_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_not_all_found(self):
        assert success_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.0

    def test_none_found(self):
        assert success_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_single_gold_found(self):
        assert success_at_k(["a", "b", "c"], ["a"], k=3) == 1.0

    def test_single_gold_not_found(self):
        assert success_at_k(["x", "y", "z"], ["a"], k=3) == 0.0

    def test_k_zero(self):
        assert success_at_k(["a"], ["a"], k=0) == 0.0

    def test_empty_gold(self):
        assert success_at_k(["a", "b"], [], k=3) == 1.0

    def test_gold_at_boundary(self):
        # Gold item exactly at position K
        assert success_at_k(["x", "y", "a"], ["a"], k=3) == 1.0
        # Gold item just beyond K
        assert success_at_k(["x", "y", "z", "a"], ["a"], k=3) == 0.0


class TestF1AtK:
    def test_perfect(self):
        # All retrieved are relevant, all gold found
        assert f1_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert f1_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial(self):
        # P@2 = 1/2, R@2 = 1/2 → F1 = 0.5
        assert f1_at_k(["a", "x"], ["a", "b"], k=2) == pytest.approx(0.5)

    def test_high_recall_low_precision(self):
        # P@5 = 2/5, R@5 = 2/2=1.0 → F1 = 2*(0.4*1.0)/(0.4+1.0) = 4/7
        result = f1_at_k(["a", "x", "b", "y", "z"], ["a", "b"], k=5)
        assert result == pytest.approx(4 / 7)

    def test_k_zero(self):
        assert f1_at_k(["a"], ["a"], k=0) == 0.0

