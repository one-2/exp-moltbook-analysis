"""Measure alignment-relevant speech in a multi-agent system.

This package reads text that AI agents write to each other. It labels each
text against 48 dispositions from Anthropic's model-generated evals (Perez
et al., 2022). The dispositions are a rubric applied to text the agent
already wrote. They are not prompts put to the agent.

Prior work on agent-only platforms measures content harm. Zhu et al.
(2504.10286) and Coppolillo et al. (2601.01090) score Chirper.ai with
moderation taxonomies. The February 2026 Moltbook papers score Moltbook with
topic and harm taxonomies. Those instruments score dispositional speech near
zero. This probe measures that speech.

**Measured:** the probe against reference labels on 1,015 posts and 10 of
the 48 dispositions. Median F1 is 0.27 (range 0.10 to 0.33, n=10). 0 of 10
dispositions reach F1 0.60. `source/alignment_relevant_speech.ipynb`
gives every number.

**Not checked:** 38 of 48 dispositions, 15,362 of 16,377 posts, 143,955
comments, and a human baseline corpus.
"""

from .traits import DEFAULT_TAXONOMY, ALL_TRAITS, FAMILIES, Taxonomy
from .corpus import Document, load_moltbook_posts, load_moltbook_comments, load_jsonl
from .scoring import Scorer, Judgment, PROBE_TEMPLATE, JUDGE_MODEL
from .gold import GOLD_TRAITS, load_gold

__all__ = [
    "DEFAULT_TAXONOMY", "ALL_TRAITS", "FAMILIES", "Taxonomy",
    "Document", "load_moltbook_posts", "load_moltbook_comments", "load_jsonl",
    "Scorer", "Judgment", "PROBE_TEMPLATE", "JUDGE_MODEL",
    "GOLD_TRAITS", "load_gold",
]

__version__ = "0.1.0"
