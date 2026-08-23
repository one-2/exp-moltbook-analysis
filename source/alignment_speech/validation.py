"""Validation of the probe against reference labels.

Published classifiers on this material report validation. Zhu et al.
(2504.10286) benchmark Perspective API above F1 0.85 on three labelled
datasets. 2602.02625 hand-checks 400 posts across 4 harm classes. Neither
dataset covers a construct such as "desire to escape sandbox". This package
therefore builds its own reference set. `data/gold/` holds it and section 2
of `source/alignment_relevant_speech.ipynb` records how it is produced.

Workflow:
    1. `stratified_sample` picks documents to annotate. It oversamples
       positives so a rare trait gets evaluated.
    2. Annotators fill in the exported CSV.
    3. `agreement` reports Cohen's kappa between annotators. Below kappa 0.6
       the label definition is the fault, not the probe.
    4. `evaluate_against_gold` reports per-trait precision, recall and F1.
    5. `select_thresholds` picks per-trait cut points on P(yes).

**Not collected:** human labels. The reference labels in `data/gold/` are
model-produced.
"""

from __future__ import annotations

import csv
import math
import random

import numpy as np
import pandas as pd


def stratified_sample(
    frame: pd.DataFrame,
    traits,
    *,
    per_trait_positive: int = 15,
    per_trait_negative: int = 15,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample documents for annotation, balanced per trait.

    Random sampling spends the budget on negatives. A trait that fires on
    0.4% of posts draws near-zero positives in a feasible random sample. Its
    precision is then not measurable.
    """
    rng = random.Random(seed)
    rows = []
    for trait in traits:
        scored = frame[frame[trait].notna()]
        positives = scored[scored[trait] == 1]
        negatives = scored[scored[trait] == 0]

        pos_ids = list(positives["doc_id"])
        neg_ids = list(negatives["doc_id"])
        rng.shuffle(pos_ids)
        rng.shuffle(neg_ids)

        for doc_id in pos_ids[:per_trait_positive]:
            rows.append({"doc_id": doc_id, "trait": trait, "judge_label": 1})
        for doc_id in neg_ids[:per_trait_negative]:
            rows.append({"doc_id": doc_id, "trait": trait, "judge_label": 0})

    sample = pd.DataFrame(rows)
    if sample.empty:
        return sample
    texts = frame.set_index("doc_id")["text"].to_dict()
    sample["text"] = sample["doc_id"].map(texts)
    # Shuffle the rows. An annotator would otherwise infer the probe label
    # from the row order. `export_for_annotation` also drops that column.
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def export_for_annotation(sample: pd.DataFrame, path: str, *, blind: bool = True) -> int:
    """Write the annotation sheet. Return the row count.

    `blind=True` withholds the probe label. A visible probe label anchors the
    annotator. The resulting F1 then measures agreement with the probe, not
    accuracy.
    """
    columns = ["doc_id", "trait", "text", "human_label", "notes"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns if blind else columns + ["judge_label"])
        writer.writeheader()
        for _, row in sample.iterrows():
            record = {
                "doc_id": row["doc_id"],
                "trait": row["trait"],
                "text": row["text"],
                "human_label": "",
                "notes": "",
            }
            if not blind:
                record["judge_label"] = row["judge_label"]
            writer.writerow(record)
    return len(sample)


def cohens_kappa(labels_a, labels_b) -> float:
    """Cohen's kappa for two annotators on binary labels."""
    a = np.asarray(labels_a, dtype=float)
    b = np.asarray(labels_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size == 0:
        return float("nan")

    observed = float((a == b).mean())
    pa1, pb1 = a.mean(), b.mean()
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def agreement(annotations: pd.DataFrame, *, annotator_column: str = "annotator") -> pd.DataFrame:
    """Pairwise inter-annotator kappa per trait.

    `annotations` needs long-form rows: doc_id, trait, annotator,
    human_label.
    """
    records = []
    for trait, group in annotations.groupby("trait"):
        wide = group.pivot_table(
            index="doc_id", columns=annotator_column, values="human_label", aggfunc="first"
        )
        annotators = list(wide.columns)
        for i, first in enumerate(annotators):
            for second in annotators[i + 1 :]:
                records.append(
                    {
                        "trait": trait,
                        "annotator_a": first,
                        "annotator_b": second,
                        "n": int(wide[[first, second]].dropna().shape[0]),
                        "kappa": cohens_kappa(wide[first], wide[second]),
                    }
                )
    return pd.DataFrame(records)


def evaluate_against_gold(
    gold: pd.DataFrame, frame: pd.DataFrame, traits=None
) -> pd.DataFrame:
    """Per-trait precision, recall and F1 of the probe against reference labels.

    `gold` needs the columns doc_id, trait, human_label.

    A trait below F1 0.6 carries no interpretive claim. Report it as measured
    and unreliable. Do not drop it: selective reporting on a validation pass
    is itself a bias.

    **Measured:** 0 of 10 dispositions reach F1 0.60 (n=1,015 posts, median
    F1 0.27, range 0.10 to 0.33).
    """
    lookup = frame.set_index("doc_id")
    records = []
    traits = traits if traits is not None else sorted(gold["trait"].unique())

    for trait in traits:
        subset = gold[gold["trait"] == trait]
        if subset.empty or trait not in lookup.columns:
            continue
        pairs = [
            (row["human_label"], lookup.at[row["doc_id"], trait])
            for _, row in subset.iterrows()
            if row["doc_id"] in lookup.index
        ]
        pairs = [(h, j) for h, j in pairs if not (pd.isna(h) or pd.isna(j))]
        if not pairs:
            continue

        tp = sum(1 for h, j in pairs if h == 1 and j == 1)
        fp = sum(1 for h, j in pairs if h == 0 and j == 1)
        fn = sum(1 for h, j in pairs if h == 1 and j == 0)
        tn = sum(1 for h, j in pairs if h == 0 and j == 0)

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and not math.isnan(precision) and not math.isnan(recall)
            else float("nan")
        )
        records.append(
            {
                "trait": trait,
                "n": len(pairs),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": (tp + tn) / len(pairs),
                "kappa": cohens_kappa([h for h, _ in pairs], [j for _, j in pairs]),
            }
        )
    return pd.DataFrame(records).sort_values("f1").reset_index(drop=True)


def select_thresholds(
    gold: pd.DataFrame, continuous: pd.DataFrame, *, grid: int = 101
) -> pd.DataFrame:
    """Pick a per-trait threshold on P(yes) that maximises F1 against reference labels.

    A single global cut point of 0.5 assumes one calibration for every
    disposition. **Not tested:** whether that assumption holds across
    constructs as different as "conscientiousness" and "desire to escape
    sandbox". Per-trait thresholds remove the assumption.

    **Not run:** this function needs `p_yes`, and the January 2026 cache
    holds binary labels only.
    """
    lookup = continuous.set_index("doc_id")
    records = []
    for trait, subset in gold.groupby("trait"):
        if trait not in lookup.columns:
            continue
        pairs = [
            (row["human_label"], lookup.at[row["doc_id"], trait])
            for _, row in subset.iterrows()
            if row["doc_id"] in lookup.index
        ]
        pairs = [(h, p) for h, p in pairs if not (pd.isna(h) or pd.isna(p))]
        if not pairs:
            continue

        best = {"trait": trait, "threshold": 0.5, "f1": float("nan"), "n": len(pairs)}
        for step in range(grid):
            cut = step / (grid - 1)
            tp = sum(1 for h, p in pairs if h == 1 and p >= cut)
            fp = sum(1 for h, p in pairs if h == 0 and p >= cut)
            fn = sum(1 for h, p in pairs if h == 1 and p < cut)
            if not tp:
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            if math.isnan(best["f1"]) or f1 > best["f1"]:
                best = {"trait": trait, "threshold": cut, "f1": f1, "n": len(pairs)}
        records.append(best)
    return pd.DataFrame(records)
