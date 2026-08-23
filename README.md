# Moltbook Analysis

This repository holds two analyses of [Moltbook](https://www.moltbook.com/), a
Reddit-style social platform whose posters are AI agents rather than people.
Both read the same scrape, taken 31 January 2026.

| analysis | notebook | question |
|---|---|---|
| Alignment-relevant speech | [`source/alignment_relevant_speech.ipynb`](source/alignment_relevant_speech.ipynb) | Does a `gpt-4.1-nano` probe measure the dispositions its labels name? |
| February exploratory analysis | [`february-analysis/`](february-analysis/) | How do agents behave on Moltbook? |

Each notebook computes every number and chart it reports. No notebook reads a
stored result file.

## Layout

| path | content |
|---|---|
| `source/alignment_speech/` | the probe, the taxonomy, and the analysis library |
| `source/alignment_relevant_speech.ipynb` | method, label provenance, and results |
| `february-analysis/` | the February 2026 exploratory analysis and its README |
| `data/data_2026_01_31_1847_aest/` | the scrape |
| `data/opus5_reference/` | opus5 reference labels and the rubric given to the opus5 labellers |
| `cache/trait_scores/` | `gpt-4.1-nano` judgments from the January 2026 run |
| `results/` | CSV output of the command line interface; every file is regenerable |
| `tests/` | 15 tests over the cache keys and the statistics |

## Naming

`opus5_reference` names the labels 24 `claude-opus-5` subagents produced against
`data/opus5_reference/RUBRIC.md`. The name states which model produced them.

**Not measured:** the model that serves each subagent turn. The session reports
`claude-opus-5` as the last served model. An individual turn may use a different
model if the runtime falls back. The name asserts the session model, not the
per-turn model.

**Not collected:** human labels. These are not a human gold standard.

## Data

| item | count |
|---|---|
| Post files | 16,844 |
| Posts with non-empty content | 16,377 |
| Distinct post authors | 6,599 |
| Comments embedded in post payloads | 143,955 |
| Cache entries | 51,430 |
| Posts with cache entries | 1,041 |
| Posts scored against all 48 dispositions | 1,015 |
| Posts with opus5 reference labels | 1,015 |

**Measured:** the cache holds 54 distinct trait strings. 48 are in the current
taxonomy. 6 are traits the taxonomy later excluded, covering 1,462 entries. Of
the 1,041 posts in the cache, 26 have empty content and the loader skips them,
which leaves 1,015.

**Not scored:** 15,362 of 16,377 posts, and all 143,955 comments.

The scrape comes from Newman and Rimey (2026). The disposition names come from
Perez et al. (2022).

## Running

Analysis needs no API key. It reads the committed cache.

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q

# Recommended: read the notebook. It computes everything it reports.
jupyter nbconvert --to notebook --execute --inplace \
    source/alignment_relevant_speech.ipynb

# Command line interface. These two rebuild every file in results/.
PYTHONPATH=source python3 -m alignment_speech --scored-only --out results analyse
PYTHONPATH=source python3 -m alignment_speech --scored-only --out results opus5-reference
```

`analyse` writes 7 files. `opus5-reference` writes 6. **Measured:** deleting `results/` and
running both rebuilds all 13 files. Every value matches the committed copy to
within 5.6e-15, and every text column is identical.

Scoring new documents calls the OpenAI API and needs `OPENAI_API_KEY`.

```bash
PYTHONPATH=source python3 -m alignment_speech --limit 5000 score
```

## Status

Section 5 of the alignment-speech notebook reports that the probe does not
measure the dispositions its labels name: 0 of 10 dispositions reach F1 0.60
(n=1,015 posts, median F1 0.27, range 0.10 to 0.33). The notebook holds the
detail, the charts, and the reference-label provenance.

Li (2026) finds that Moltbook's viral behaviour is substantially human-driven.
**Not estimated:** contamination by human authors. Any reading of these results
as emergent agent behaviour needs that estimate first.

## Writing style

Write every document in this repository in Simplified Technical English.

| rule | requirement |
|---|---|
| Sentence length | short |
| Ideas per sentence | one |
| Voice | active |
| Tense | present |
| Terms | use the same word for the same thing every time |
| Content | give facts and numbers, not justifications |
| Numbers | state every number with its n and its spread |
| Claims | label what you measured, observed, inferred, and assumed |
| Gaps | name what you did not check |
| Parallel items | use a table for three or more |
| Prohibited | metaphor, praise, filler, stacked hedges |
| Unknowns | say "I do not know", then name the test that would settle it |
| First sentence | answer the question |

This applies to every README, the notebook prose, `data/opus5_reference/RUBRIC.md`, every
docstring and comment, commit subjects, and pull request bodies.

## Licence

MIT. See [LICENSE](LICENSE).

## Acknowledgements

This work was undertaken as part of the
[Sydney AI Safety Fellowship 2026](https://sasf26.com/).

Follow the broader problem this speaks to, safety in very-large systems of AI,
at [Gigascale Labs](https://www.gigascale-labs.org).

## References

Li, N. (2026). *The Moltbook illusion: Separating human influence from emergent
behavior in AI agent societies*. arXiv:2602.07432

Newman, E., & Rimey, K. (2026). *Moltbook Data*. GitHub.
https://github.com/ExtraE113/moltbook_data

Perez, E., Ringer, S., Lukošiūtė, K., Nguyen, K., Chen, E., Heiner, S., Pettit,
C., Olsson, C., Kundu, S., Kadavath, S., Jones, A., Chen, A., Mann, B., Israel,
B., Seethor, B., McKinnon, C., Olah, C., Yan, D., Amodei, D., . . . Kaplan, J.
(2022). *Discovering language model behaviors with model-written evaluations*.
arXiv:2212.09251
