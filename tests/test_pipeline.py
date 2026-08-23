"""Tests for the alignment-speech pipeline.

These tests cover the faults that corrupt a result without raising an error:
cache key collisions, missing-score handling, and the statistics behind the
construct-validity claims.
"""

import json
import math
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from alignment_speech import analysis as an
from alignment_speech.corpus import Document
from alignment_speech.scoring import (
    Judgment, LegacyCache, ScoreCache, probe_id, PROBE_TEMPLATE,
)
from alignment_speech.traits import DEFAULT_TAXONOMY, ALL_TRAITS
from alignment_speech.validation import cohens_kappa, evaluate_against_gold, stratified_sample


def test_taxonomy_is_wellformed():
    DEFAULT_TAXONOMY.validate()
    assert len(ALL_TRAITS) == 48
    assert len(set(ALL_TRAITS)) == 48


def test_probe_id_changes_with_template_and_model():
    base = probe_id(PROBE_TEMPLATE, "gpt-4.1-nano")
    assert probe_id(PROBE_TEMPLATE + " ", "gpt-4.1-nano") != base
    assert probe_id(PROBE_TEMPLATE, "gpt-4.1-mini") != base


def test_cache_roundtrip_and_probe_isolation():
    """A cache entry must not be visible to a different probe.

    The legacy layout keys on md5(post_id + trait). That key collides across
    prompt wordings. An edit to the prompt then returns the old prompt's
    answers.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cache_a = ScoreCache(tmp, probe="AAA", model="m")
        cache_b = ScoreCache(tmp, probe="BBB", model="m")
        judgment = Judgment("doc1", "desire for self improvement", 0.9, 1, "m", "AAA")
        cache_a.put(judgment)

        assert cache_a.get("doc1", "desire for self improvement").label == 1
        assert cache_b.get("doc1", "desire for self improvement") is None
        assert cache_a.scan() == {"doc1"}
        assert cache_b.scan() == set()


def test_legacy_cache_reads_zero_scores():
    """A cached 0 must not be mistaken for a cache miss."""
    with tempfile.TemporaryDirectory() as tmp:
        key = LegacyCache._key("doc1", "conscientiousness")
        with open(os.path.join(tmp, f"{key}.json"), "w") as handle:
            json.dump({"post_id": "doc1", "trait": "conscientiousness", "score": 0}, handle)

        legacy = LegacyCache(tmp)
        hit = legacy.get("doc1", "conscientiousness")
        assert hit is not None and hit.label == 0
        assert hit.p_yes is None  # legacy entries carry a label and no probability
        assert legacy.scan() == {"doc1"}


def _frame(rows, traits):
    docs = [Document(id=r["id"], text=r.get("text", "t"), corpus="test",
                     author_id=r.get("author")) for r in rows]
    judgments = [
        Judgment(r["id"], trait, None, r[trait], "m", "p")
        for r in rows for trait in traits if r.get(trait) is not None
    ]
    return an.build_frame(docs, judgments, traits)


def test_partial_scores_are_not_summed_as_zero():
    """A document missing a trait must be excluded from trait-count stats."""
    traits = ["a", "b"]
    frame = _frame(
        [{"id": "d1", "a": 1, "b": 1}, {"id": "d2", "a": 1, "b": None}], traits
    )
    assert frame.loc[frame.doc_id == "d1", "complete"].item()
    assert not frame.loc[frame.doc_id == "d2", "complete"].item()

    summary = an.traits_per_document(frame)
    assert summary["n_documents"] == 1  # the complete document is the only one counted


def test_prevalence_ignores_unscored_documents():
    traits = ["a"]
    frame = _frame([{"id": "d1", "a": 1}, {"id": "d2", "a": 0}, {"id": "d3", "a": None}], traits)
    prev = an.prevalence(frame, traits)
    row = prev.iloc[0]
    assert row["n_scored"] == 2 and row["n_positive"] == 1
    assert row["prevalence"] == pytest.approx(0.5)


def test_wilson_interval_stays_in_bounds_for_rare_events():
    low, high = an.wilson_interval(4, 1015)
    assert 0 < low < high < 1
    low0, high0 = an.wilson_interval(0, 100)
    assert low0 == 0.0 and high0 > 0


def test_max_phi_bounds_correlation_for_skewed_marginals():
    """Two traits with very different base rates cannot correlate at 1.0."""
    assert an.max_phi(0.3, 0.3) == pytest.approx(1.0)
    assert an.max_phi(0.542, 0.064) < 0.3
    assert math.isnan(an.max_phi(0.0, 0.5))


def test_attenuation_distinguishes_artefact_from_disagreement():
    """Perfect agreement must yield phi_ratio 1.0 even with skewed marginals."""
    traits = ["x", "y"]
    # y fires where x fires and nowhere else. This is the maximum agreement
    # two traits with these base rates can reach.
    rows = [{"id": f"d{i}", "x": 1, "y": 1} for i in range(10)]
    rows += [{"id": f"e{i}", "x": 0, "y": 0} for i in range(90)]
    frame = _frame(rows, traits)
    report = an.attenuation_report(frame, pairs=[("x", "y")])
    assert report.iloc[0]["phi_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert report.iloc[0]["jaccard"] == pytest.approx(1.0)


def test_trait_concentration_detects_author_clustering():
    traits = ["t"]
    # Two authors. One is always positive. One is never positive. The global
    # base rate matches a random split. The concentration by author is total.
    rows = [{"id": f"a{i}", "author": "A", "t": 1} for i in range(5)]
    rows += [{"id": f"b{i}", "author": "B", "t": 0} for i in range(5)]
    frame = _frame(rows, traits)
    result = an.trait_concentration(frame, traits, min_posts=3)
    assert result.iloc[0]["concentration"] > 1.5


def test_compare_corpora_flags_disjoint_intervals():
    traits = ["t"]
    agent = _frame([{"id": f"a{i}", "t": 1} for i in range(100)], traits)
    baseline = _frame([{"id": f"b{i}", "t": 0} for i in range(100)], traits)
    comparison = an.compare_corpora(agent, baseline, traits)
    assert bool(comparison.iloc[0]["ci_disjoint"])


def test_cohens_kappa_endpoints():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0


def test_stratified_sample_balances_rare_traits():
    """A trait firing on 1% of documents must still get positives sampled."""
    traits = ["common", "rare"]
    rows = [
        {"id": f"d{i}", "common": 1 if i < 500 else 0, "rare": 1 if i < 10 else 0}
        for i in range(1000)
    ]
    frame = _frame(rows, traits)
    sample = stratified_sample(frame, traits, per_trait_positive=8, per_trait_negative=8, seed=1)
    rare = sample[sample["trait"] == "rare"]
    assert (rare["judge_label"] == 1).sum() == 8
    assert "judge_label" in sample.columns


def test_export_is_blind_by_default():
    from alignment_speech.validation import export_for_annotation

    traits = ["t"]
    frame = _frame([{"id": f"d{i}", "t": i % 2} for i in range(10)], traits)
    sample = stratified_sample(frame, traits, per_trait_positive=2, per_trait_negative=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ann.csv")
        export_for_annotation(sample, path)
        header = open(path).readline()
        assert "judge_label" not in header  # a visible probe label anchors annotators


def test_evaluate_against_gold_scores_perfect_and_inverted():
    traits = ["t"]
    frame = _frame([{"id": "d1", "t": 1}, {"id": "d2", "t": 0}], traits)
    perfect = pd.DataFrame(
        [{"doc_id": "d1", "trait": "t", "human_label": 1},
         {"doc_id": "d2", "trait": "t", "human_label": 0}]
    )
    assert evaluate_against_gold(perfect, frame, traits).iloc[0]["f1"] == pytest.approx(1.0)

    inverted = pd.DataFrame(
        [{"doc_id": "d1", "trait": "t", "human_label": 0},
         {"doc_id": "d2", "trait": "t", "human_label": 1}]
    )
    assert evaluate_against_gold(inverted, frame, traits).iloc[0]["tp"] == 0
