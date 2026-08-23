"""Analysis over scored documents.

This module reports prevalence, co-occurrence, construct-validity checks,
author-level consistency, and baseline comparison. Every function takes a
wide dataframe: one row per document, one column per trait. `build_frame`
produces it.

These functions describe the labels they are given. They describe the
platform only when the labels are valid. **Measured:** the probe reaches
median F1 0.27 against opus5 reference labels (range 0.10 to 0.33, n=10
dispositions, 1,015 posts). `source/alignment_relevant_speech.ipynb` gives every number.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .traits import FAMILIES, NEAR_SYNONYM_PAIRS, SOURCE_GROUP


META_COLUMNS = [
    "doc_id", "corpus", "author_id", "author_name",
    "created_at", "community", "upvotes", "downvotes", "comment_count", "text",
]


def build_frame(documents, judgments, traits) -> pd.DataFrame:
    """Pivot judgments into one row per document, one column per trait.

    This function reads `label`, which is binary. `build_frame_continuous`
    reads `p_yes`. Correlation work reads `p_yes` where it exists.
    """
    traits = list(traits)
    by_doc: dict[str, dict[str, int | None]] = {}
    for judgment in judgments:
        by_doc.setdefault(judgment.doc_id, {})[judgment.trait] = judgment.label

    rows = []
    for doc in documents:
        scores = by_doc.get(doc.id, {})
        row = {
            "doc_id": doc.id,
            "corpus": doc.corpus,
            "author_id": doc.author_id,
            "author_name": doc.author_name,
            "created_at": doc.created_at,
            "community": doc.community,
            "upvotes": doc.upvotes,
            "downvotes": doc.downvotes,
            "comment_count": doc.comment_count,
            "text": doc.text,
        }
        for trait in traits:
            row[trait] = scores.get(trait)
        rows.append(row)

    frame = pd.DataFrame(rows)
    # A document missing any trait is excluded from trait-count arithmetic.
    # A missing value is not a zero. `complete` marks the usable rows.
    frame["n_scored"] = frame[traits].notna().sum(axis=1)
    frame["complete"] = frame["n_scored"] == len(traits)
    frame["num_traits"] = frame[traits].sum(axis=1, skipna=True)
    return frame


def build_frame_continuous(documents, judgments, traits) -> pd.DataFrame:
    traits = list(traits)
    by_doc: dict[str, dict[str, float | None]] = {}
    for judgment in judgments:
        by_doc.setdefault(judgment.doc_id, {})[judgment.trait] = judgment.p_yes

    rows = []
    for doc in documents:
        scores = by_doc.get(doc.id, {})
        row = {"doc_id": doc.id, "corpus": doc.corpus}
        for trait in traits:
            row[trait] = scores.get(trait)
        rows.append(row)
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval on a proportion.

    The normal approximation returns a negative lower bound near 0%. Several
    dispositions sit there. The rarest fires on 0.4% of posts (4 of 1,015).
    """
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def prevalence(frame: pd.DataFrame, traits) -> pd.DataFrame:
    """Per-trait prevalence with 95% Wilson confidence intervals."""
    records = []
    for trait in traits:
        column = frame[trait].dropna()
        total = len(column)
        hits = int(column.sum())
        low, high = wilson_interval(hits, total)
        records.append(
            {
                "trait": trait,
                "family": next((f for f, m in FAMILIES.items() if trait in m), None),
                "group": SOURCE_GROUP.get(trait),
                "n_scored": total,
                "n_positive": hits,
                "prevalence": hits / total if total else float("nan"),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(records).sort_values("prevalence", ascending=False).reset_index(drop=True)


def traits_per_document(frame: pd.DataFrame) -> dict:
    complete = frame[frame["complete"]]
    counts = complete["num_traits"].to_numpy()
    if counts.size == 0:
        return {}
    return {
        "n_documents": int(counts.size),
        "mean": float(np.mean(counts)),
        "sem": float(np.std(counts, ddof=1) / math.sqrt(counts.size)),
        "median": float(np.median(counts)),
        "std": float(np.std(counts, ddof=1)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "pct_zero": float((counts == 0).mean()),
    }


def phi_matrix(frame: pd.DataFrame, traits) -> pd.DataFrame:
    """Pairwise phi coefficients between binary traits.

    Phi is Pearson's r on two binary variables. The two base rates bound its
    ceiling. Two traits with different base rates cannot reach phi 1.0. This
    holds however well they agree. `max_phi` computes the ceiling and
    `attenuation_report` divides phi by it.
    """
    return frame[list(traits)].astype(float).corr()


def max_phi(p1: float, p2: float) -> float:
    """Maximum phi that two marginal rates allow. NaN when either rate is 0 or 1."""
    if not (0 < p1 < 1 and 0 < p2 < 1):
        return float("nan")
    lo, hi = min(p1, p2), max(p1, p2)
    return math.sqrt((lo * (1 - hi)) / (hi * (1 - lo)))


def attenuation_report(frame: pd.DataFrame, pairs=NEAR_SYNONYM_PAIRS) -> pd.DataFrame:
    """Phi against the ceiling the base rates allow, for near-synonym pairs.

    `phi_ratio` is phi divided by that ceiling. It separates two cases. A
    ratio near 1.0 means the two traits agree as strongly as their base rates
    permit, and a low raw phi is an artefact of binarising skewed variables.
    A low ratio means the probe applies the two labels to different
    documents.

    **Measured:** the coordination pair reaches ratio 0.45 under probe labels
    and 1.00 under opus5 reference labels (n=1,015).
    """
    records = []
    for a, b in pairs:
        if a not in frame.columns or b not in frame.columns:
            continue
        sub = frame[[a, b]].dropna()
        if sub.empty:
            continue
        p1, p2 = sub[a].mean(), sub[b].mean()
        observed = sub[a].astype(float).corr(sub[b].astype(float))
        ceiling = max_phi(p1, p2)
        both = int(((sub[a] == 1) & (sub[b] == 1)).sum())
        either = int(((sub[a] == 1) | (sub[b] == 1)).sum())
        records.append(
            {
                "trait_a": a,
                "trait_b": b,
                "prevalence_a": p1,
                "prevalence_b": p2,
                "phi": observed,
                "max_phi": ceiling,
                "phi_ratio": observed / ceiling if ceiling and not math.isnan(ceiling) else float("nan"),
                "jaccard": both / either if either else float("nan"),
                "n": len(sub),
            }
        )
    return pd.DataFrame(records)


def check_family_coherence(frame: pd.DataFrame, traits) -> pd.DataFrame:
    """Within-family against between-family mean correlation, per family.

    `separation` is the difference. A probe that measures its labels returns
    a positive separation. A family at or below zero does not name a
    construct the probe detects.

    **Measured:** 3 of 9 families separate at or below +0.023 under probe
    labels. `self_model` separates at -0.000 (n=1,015 posts).
    """
    corr = phi_matrix(frame, traits)
    records = []
    for family, members in FAMILIES.items():
        members = [m for m in members if m in corr.columns]
        if len(members) < 2:
            continue
        within = []
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                value = corr.loc[a, b]
                if not math.isnan(value):
                    within.append(value)
        others = [t for t in corr.columns if t not in members]
        between = corr.loc[members, others].to_numpy().flatten()
        between = between[~np.isnan(between)]
        records.append(
            {
                "family": family,
                "n_traits": len(members),
                "mean_within": float(np.mean(within)) if within else float("nan"),
                "mean_between": float(np.mean(between)) if between.size else float("nan"),
                "separation": (float(np.mean(within)) - float(np.mean(between)))
                if within and between.size
                else float("nan"),
            }
        )
    return pd.DataFrame(records).sort_values("separation", ascending=False).reset_index(drop=True)


def author_profiles(frame: pd.DataFrame, traits, *, min_posts: int = 3) -> pd.DataFrame:
    """Per-author disposition rates, for authors with at least `min_posts` posts.

    A disposition that belongs to an agent concentrates in a subset of
    authors. A disposition that belongs to the platform's discourse spreads
    evenly. `trait_concentration` tests which holds.
    """
    complete = frame[frame["complete"] & frame["author_id"].notna()]
    grouped = complete.groupby("author_id")
    records = []
    for author_id, group in grouped:
        if len(group) < min_posts:
            continue
        record = {
            "author_id": author_id,
            "author_name": group["author_name"].iloc[0],
            "n_posts": len(group),
            "mean_traits": float(group["num_traits"].mean()),
            "std_traits": float(group["num_traits"].std(ddof=1)) if len(group) > 1 else 0.0,
        }
        for trait in traits:
            record[trait] = float(group[trait].mean())
        records.append(record)
    return pd.DataFrame(records)


def trait_concentration(frame: pd.DataFrame, traits, *, min_posts: int = 3) -> pd.DataFrame:
    """Observed against expected variance in per-author trait rates.

    The expected variance is the variance of independent draws at the global
    base rate. A ratio above 1 means the trait clusters by author. A ratio
    near 1 means it does not, and the trait is a feature of the conversation
    rather than of the agent.

    **Not checked:** author clustering on this corpus. Only 19 authors have 3
    or more posts in the scored sample of 1,015.
    """
    complete = frame[frame["complete"] & frame["author_id"].notna()]
    counts = complete.groupby("author_id").size()
    eligible = counts[counts >= min_posts].index
    subset = complete[complete["author_id"].isin(eligible)]
    if subset.empty:
        return pd.DataFrame()

    records = []
    for trait in traits:
        base_rate = subset[trait].mean()
        per_author = subset.groupby("author_id")[trait].agg(["mean", "size"])
        if per_author.empty or base_rate in (0.0, 1.0):
            continue
        observed = per_author["mean"].var(ddof=1)
        expected = float(np.mean(base_rate * (1 - base_rate) / per_author["size"]))
        records.append(
            {
                "trait": trait,
                "base_rate": float(base_rate),
                "n_authors": int(len(per_author)),
                "observed_var": float(observed),
                "expected_var": expected,
                "concentration": float(observed / expected) if expected else float("nan"),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("concentration", ascending=False)
        .reset_index(drop=True)
    )


def compare_corpora(frame_a: pd.DataFrame, frame_b: pd.DataFrame, traits,
                    *, name_a: str = "agent", name_b: str = "baseline") -> pd.DataFrame:
    """Prevalence ratio between two corpora scored under the same probe.

    A prevalence figure needs a comparison rate. `ci_disjoint` marks the
    traits whose 95% Wilson intervals do not overlap.

    **Not run:** this comparison. No human-authored corpus is scored.
    """
    a = prevalence(frame_a, traits).set_index("trait")
    b = prevalence(frame_b, traits).set_index("trait")
    records = []
    for trait in traits:
        if trait not in a.index or trait not in b.index:
            continue
        pa, pb = a.loc[trait, "prevalence"], b.loc[trait, "prevalence"]
        records.append(
            {
                "trait": trait,
                f"prevalence_{name_a}": pa,
                f"prevalence_{name_b}": pb,
                "ratio": (pa / pb) if pb else float("inf"),
                "difference": pa - pb,
                f"ci_low_{name_a}": a.loc[trait, "ci_low"],
                f"ci_high_{name_a}": a.loc[trait, "ci_high"],
                f"ci_low_{name_b}": b.loc[trait, "ci_low"],
                f"ci_high_{name_b}": b.loc[trait, "ci_high"],
                "ci_disjoint": (a.loc[trait, "ci_low"] > b.loc[trait, "ci_high"])
                or (b.loc[trait, "ci_low"] > a.loc[trait, "ci_high"]),
            }
        )
    return pd.DataFrame(records).sort_values("ratio", ascending=False).reset_index(drop=True)
