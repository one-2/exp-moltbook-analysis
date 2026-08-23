"""Command line entry points.

    python -m alignment_speech score      --limit 1000
    python -m alignment_speech analyse    --limit 1000 --out results/
    python -m alignment_speech validate-sample --out results/annotation.csv
    python -m alignment_speech validate-score --annotations results/gold_labels.csv
    python -m alignment_speech gold       --out results/
    python -m alignment_speech compare    --baseline data/reddit.jsonl

`analyse`, `validate-sample`, `validate-score`, `gold` and `compare` run from
cache and need no API key. `score` calls the API and needs OPENAI_API_KEY.

`analyse` writes 7 files to `--out`. `gold` writes 6. Together they regenerate
every file in `results/`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

DEFAULT_POSTS = "data/data_2026_01_31_1847_aest/posts"
DEFAULT_CACHE = "cache/probe_scores"
LEGACY_CACHE = "cache/trait_scores"


def _load(args, traits):
    from .corpus import load_moltbook_posts, load_moltbook_comments
    from .scoring import Scorer

    scorer = Scorer(
        cache_dir=args.cache,
        legacy_cache_dir=None if args.no_legacy else args.legacy_cache,
        concurrency=args.concurrency,
    )

    # `--scored-only` loads the whole corpus before it filters. `limit` would
    # otherwise slice the directory listing, not the scored set.
    load_limit = None if args.scored_only else args.limit
    if args.unit == "comments":
        documents = load_moltbook_comments(args.posts, limit=load_limit)
    else:
        documents = load_moltbook_posts(
            args.posts, include_title=args.include_title, limit=load_limit
        )

    if args.scored_only:
        scored = scorer.scored_doc_ids(allow_legacy=not args.no_legacy)
        documents = [d for d in documents if d.id in scored]
        if args.limit is not None:
            documents = documents[: args.limit]

    return documents, scorer


def cmd_score(args) -> int:
    from .traits import DEFAULT_TAXONOMY

    traits = list(DEFAULT_TAXONOMY)
    documents, scorer = _load(args, traits)
    print(f"scoring {len(documents)} documents x {len(traits)} traits")
    asyncio.run(scorer.score_documents(documents, traits, allow_legacy=not args.no_legacy))
    print("done:", scorer.stats)
    return 0


def cmd_analyse(args) -> int:
    import pandas as pd

    from .traits import DEFAULT_TAXONOMY
    from . import analysis as an

    traits = list(DEFAULT_TAXONOMY)
    DEFAULT_TAXONOMY.validate()
    documents, scorer = _load(args, traits)

    judgments = []
    for doc in documents:
        for trait in traits:
            hit = scorer.lookup(doc.id, trait, allow_legacy=not args.no_legacy)
            if hit is not None:
                judgments.append(hit)

    print(f"loaded {len(judgments)} cached judgments over {len(documents)} documents")
    if not judgments:
        print("nothing cached: run `score` first (requires OPENAI_API_KEY)")
        return 1

    frame = an.build_frame(documents, judgments, traits)
    frame = frame[frame["n_scored"] > 0].reset_index(drop=True)
    complete = frame[frame["complete"]]
    print(f"documents with >=1 judgment: {len(frame)}; fully scored: {len(complete)}")

    os.makedirs(args.out, exist_ok=True)

    prev = an.prevalence(complete, traits)
    prev.to_csv(os.path.join(args.out, "prevalence.csv"), index=False)

    summary = an.traits_per_document(complete)
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    an.phi_matrix(complete, traits).to_csv(os.path.join(args.out, "phi_matrix.csv"))
    atten = an.attenuation_report(complete)
    atten.to_csv(os.path.join(args.out, "attenuation.csv"), index=False)
    coherence = an.check_family_coherence(complete, traits)
    coherence.to_csv(os.path.join(args.out, "family_coherence.csv"), index=False)
    concentration = an.trait_concentration(complete, traits, min_posts=args.min_posts)
    concentration.to_csv(os.path.join(args.out, "trait_concentration.csv"), index=False)
    an.author_profiles(complete, traits, min_posts=args.min_posts).to_csv(
        os.path.join(args.out, "author_profiles.csv"), index=False
    )

    pd.set_option("display.width", 160)
    print(f"\ntraits per document: {summary}")
    print("\ntop 10 traits by prevalence:")
    print(prev.head(10)[["trait", "prevalence", "ci_low", "ci_high", "n_positive"]].to_string(index=False))
    print("\nnear-synonym attenuation:")
    print(atten[["trait_a", "trait_b", "phi", "max_phi", "phi_ratio", "jaccard"]].to_string(index=False))
    print("\nfamily coherence:")
    print(coherence.to_string(index=False))
    if not concentration.empty:
        print("\nmost author-concentrated traits:")
        print(concentration.head(8).to_string(index=False))
    print(f"\nwrote results to {args.out}/")
    return 0


def cmd_validate_sample(args) -> int:
    from .traits import DEFAULT_TAXONOMY
    from . import analysis as an
    from .validation import stratified_sample, export_for_annotation

    traits = list(DEFAULT_TAXONOMY)
    documents, scorer = _load(args, traits)
    judgments = [
        hit
        for doc in documents
        for trait in traits
        if (hit := scorer.lookup(doc.id, trait, allow_legacy=not args.no_legacy)) is not None
    ]
    frame = an.build_frame(documents, judgments, traits)
    frame = frame[frame["complete"]].reset_index(drop=True)

    sample = stratified_sample(
        frame, traits,
        per_trait_positive=args.per_trait, per_trait_negative=args.per_trait, seed=args.seed,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    count = export_for_annotation(sample, args.out, blind=not args.show_judge)
    print(f"wrote {count} annotation rows ({sample['doc_id'].nunique()} distinct documents) to {args.out}")
    print("fill in the human_label column with 1/0, then run `validate-score`")
    return 0


def cmd_validate_score(args) -> int:
    import pandas as pd

    from .traits import DEFAULT_TAXONOMY
    from . import analysis as an
    from .validation import evaluate_against_gold

    traits = list(DEFAULT_TAXONOMY)
    gold = pd.read_csv(args.annotations)
    gold = gold[gold["human_label"].notna()]
    if gold.empty:
        print("no completed annotations found in", args.annotations)
        return 1

    documents, scorer = _load(args, traits)
    judgments = [
        hit
        for doc in documents
        for trait in traits
        if (hit := scorer.lookup(doc.id, trait, allow_legacy=not args.no_legacy)) is not None
    ]
    frame = an.build_frame(documents, judgments, traits)
    report = evaluate_against_gold(gold, frame, traits)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "validation.csv")
    report.to_csv(path, index=False)
    print(report.to_string(index=False))
    print(f"\nwrote {path}")
    unreliable = report[report["f1"] < 0.6]
    if not unreliable.empty:
        print(f"\n{len(unreliable)} traits below F1 0.6; report these as unreliable:")
        print(", ".join(unreliable["trait"]))
    return 0


def cmd_gold(args) -> int:
    """Regenerate every reference-label output in `results/`.

    `analyse` covers the probe outputs. This command covers the 6 that depend
    on `data/gold/labels/`: the wide and long label frames, labeller
    consistency, the humour breakdown, synonym-pair agreement, and the
    validation report.

    The validation report here carries 4 columns that `validate-score` does
    not write: `gold`, `judge`, `ratio` and `ci_disjoint`. Those come from
    `gold.prevalence_comparison`, which needs the wide reference frame.
    """
    from .traits import DEFAULT_TAXONOMY
    from . import analysis as an
    from . import gold as gd
    from .validation import evaluate_against_gold

    traits = list(DEFAULT_TAXONOMY)
    documents, scorer = _load(args, traits)
    judgments = [
        hit
        for doc in documents
        for trait in traits
        if (hit := scorer.lookup(doc.id, trait, allow_legacy=not args.no_legacy)) is not None
    ]
    frame = an.build_frame(documents, judgments, traits)
    complete = frame[frame["complete"]].reset_index(drop=True)
    if complete.empty:
        print("no fully scored documents: run `score` first")
        return 1

    reference = gd.load_gold(args.labels)
    long = gd.to_long(reference)
    report = evaluate_against_gold(long, complete, list(gd.GOLD_TRAITS.values()))
    comparison = gd.prevalence_comparison(reference, complete)

    tables = {
        "gold_wide.csv": reference,
        "gold_labels.csv": long,
        "gold_consistency.csv": gd.labeller_consistency(reference),
        "gold_humour.csv": gd.humour_breakdown(reference),
        "gold_pairs.csv": gd.pair_agreement(reference, complete),
        "validation.csv": report.merge(
            comparison[["trait", "gold", "judge", "ratio", "ci_disjoint"]], on="trait"
        ),
    }

    os.makedirs(args.out, exist_ok=True)
    for name, table in tables.items():
        path = os.path.join(args.out, name)
        table.to_csv(path, index=False)
        print(f"wrote {path}  ({len(table)} rows, {len(table.columns)} columns)")
    print(f"\n{len(reference):,} reference posts over "
          f"{reference['batch'].nunique()} batches")
    return 0


def cmd_compare(args) -> int:
    from .traits import DEFAULT_TAXONOMY
    from .corpus import load_jsonl
    from . import analysis as an

    traits = list(DEFAULT_TAXONOMY)
    agent_docs, scorer = _load(args, traits)
    baseline_docs = load_jsonl(args.baseline, corpus="baseline", limit=args.limit)

    def frame_for(docs):
        judgments = [
            hit
            for doc in docs
            for trait in traits
            if (hit := scorer.lookup(doc.id, trait, allow_legacy=not args.no_legacy)) is not None
        ]
        built = an.build_frame(docs, judgments, traits)
        return built[built["complete"]].reset_index(drop=True)

    agent_frame, baseline_frame = frame_for(agent_docs), frame_for(baseline_docs)
    if baseline_frame.empty:
        print("baseline corpus has no cached scores: run `score` against it first")
        return 1

    comparison = an.compare_corpora(agent_frame, baseline_frame, traits)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "baseline_comparison.csv")
    comparison.to_csv(path, index=False)
    print(comparison.head(20).to_string(index=False))
    print(f"\nwrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alignment_speech")
    parser.add_argument("--posts", default=DEFAULT_POSTS)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--legacy-cache", default=LEGACY_CACHE)
    parser.add_argument("--no-legacy", action="store_true",
                        help="ignore the January 2026 binary-only cache")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--unit", choices=["posts", "comments"], default="posts")
    parser.add_argument("--include-title", action="store_true")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--out", default="results")
    parser.add_argument("--min-posts", type=int, default=3)
    parser.add_argument("--scored-only", action="store_true",
                        help="restrict to documents that already have cached judgments")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("score").set_defaults(func=cmd_score)
    sub.add_parser("analyse").set_defaults(func=cmd_analyse)

    sample = sub.add_parser("validate-sample")
    sample.add_argument("--per-trait", type=int, default=15)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--show-judge", action="store_true",
                        help="include the probe label; it anchors annotators")
    sample.set_defaults(func=cmd_validate_sample, out="results/annotation.csv")

    score_val = sub.add_parser("validate-score")
    score_val.add_argument("--annotations", default="results/annotation.csv")
    score_val.set_defaults(func=cmd_validate_score)

    gold_cmd = sub.add_parser("gold")
    gold_cmd.add_argument("--labels", default="data/gold/labels")
    gold_cmd.set_defaults(func=cmd_gold)

    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
