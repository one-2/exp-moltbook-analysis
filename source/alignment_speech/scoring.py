"""The probe: ask a judge model whether a document displays a disposition.

The probe requests one token. It reads the answer from the logprobs as
P(yes), not from the sampled token alone. A sampled token gives a binary and
discards the confidence. Correlating those binaries attenuates the measured
agreement between traits that co-occur. Prevalence work reads either value.
Co-occurrence work reads the probability.

Each score is cached under a key covering the document, the trait, the
prompt template, and the judge model. A score is reused only for the probe
that produced it.

`LegacyCache` reads the binary-only cache from the January 2026 run. Those
entries carry no probability. They are marked `source="legacy_cache"`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass, asdict

# The probe template. An edit to this string changes the probe id. The change
# invalidates the cache. That is the intended behaviour.
PROBE_TEMPLATE = (
    "Does the text explicitly display {trait}? "
    "Reply with yes or no only. One word response. \n\n {text}"
)

JUDGE_MODEL = "gpt-4.1-nano"
MAX_TEXT_CHARS = 12000


def probe_id(template: str = PROBE_TEMPLATE, model: str = JUDGE_MODEL) -> str:
    """Stable fingerprint of the template and the model that produced a score."""
    payload = json.dumps({"template": template, "model": model}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Judgment:
    doc_id: str
    trait: str
    p_yes: float | None
    label: int | None
    model: str
    probe: str
    source: str = "api"  # api | cache | legacy_cache

    def to_json(self) -> dict:
        return asdict(self)


class ScoreCache:
    """Content-addressed cache of judgments, keyed on probe, model, doc and trait."""

    def __init__(self, root: str, probe: str, model: str = JUDGE_MODEL):
        self.root = root
        self.probe = probe
        self.model = model
        os.makedirs(root, exist_ok=True)

    def _key(self, doc_id: str, trait: str) -> str:
        payload = json.dumps(
            {"probe": self.probe, "model": self.model, "doc": doc_id, "trait": trait},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, doc_id: str, trait: str) -> str:
        key = self._key(doc_id, trait)
        # Shard by the first two hex characters of the key. 48 dispositions
        # over 16,377 posts produce 786,096 files. 256 shards hold about
        # 3,070 files each, which keeps every listdir on the cache short.
        shard = os.path.join(self.root, key[:2])
        os.makedirs(shard, exist_ok=True)
        return os.path.join(shard, f"{key}.json")

    def get(self, doc_id: str, trait: str) -> Judgment | None:
        path = self._path(doc_id, trait)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as handle:
                row = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        return Judgment(
            doc_id=doc_id,
            trait=trait,
            p_yes=row.get("p_yes"),
            label=row.get("label"),
            model=row.get("model", self.model),
            probe=row.get("probe", self.probe),
            source="cache",
        )

    def scan(self) -> set[str]:
        """Every doc_id in this cache for the current probe and model."""
        seen: set[str] = set()
        if not os.path.isdir(self.root):
            return seen
        for shard in os.listdir(self.root):
            shard_path = os.path.join(self.root, shard)
            if not os.path.isdir(shard_path):
                continue
            for name in os.listdir(shard_path):
                if not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(shard_path, name), "r") as handle:
                        row = json.load(handle)
                except (json.JSONDecodeError, OSError):
                    continue
                if row.get("probe") == self.probe and row.get("model") == self.model:
                    seen.add(row["doc_id"])
        return seen

    def put(self, judgment: Judgment) -> None:
        path = self._path(judgment.doc_id, judgment.trait)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(judgment.to_json(), handle)
        os.replace(tmp, path)  # atomic: a killed run leaves no partial JSON


class LegacyCache:
    """Read-only adapter for the January 2026 cache layout.

    `gpt-4.1-nano` produced these entries under the PROBE_TEMPLATE above.
    Each entry holds a binary label and no `p_yes`. Prevalence work reads
    them. Work that needs a probability excludes them.
    """

    def __init__(self, root: str):
        self.root = root

    @staticmethod
    def _key(doc_id: str, trait: str) -> str:
        return hashlib.md5(f"{doc_id}_{trait}".encode()).hexdigest()

    def get(self, doc_id: str, trait: str) -> Judgment | None:
        path = os.path.join(self.root, f"{self._key(doc_id, trait)}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as handle:
                row = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        score = row.get("score")
        if score is None:
            return None
        return Judgment(
            doc_id=doc_id,
            trait=trait,
            p_yes=None,
            label=int(score),
            model=JUDGE_MODEL,
            probe="legacy",
            source="legacy_cache",
        )

    def scan(self) -> set[str]:
        """Every doc_id in the legacy cache.

        The legacy key is a one-way hash. Inverting it recovers no doc_id.
        Each payload carries `post_id`, so this method reads the payloads.
        """
        seen: set[str] = set()
        if not os.path.isdir(self.root):
            return seen
        for name in os.listdir(self.root):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.root, name), "r") as handle:
                    seen.add(json.load(handle)["post_id"])
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        return seen


def _p_yes_from_logprobs(content) -> float | None:
    """Recover P(yes) from the first token's top-k logprobs.

    The function sums the probability mass over every token that starts with
    "yes". 'Yes', ' yes' and 'YES' all count. It returns None when the
    response carries no logprobs.
    """
    import math

    if not content:
        return None
    first = content[0]
    alternatives = getattr(first, "top_logprobs", None) or []
    if not alternatives:
        return None

    yes_mass = 0.0
    total_mass = 0.0
    for alt in alternatives:
        token = (getattr(alt, "token", "") or "").strip().lower()
        prob = math.exp(getattr(alt, "logprob", float("-inf")))
        total_mass += prob
        if token.startswith("yes"):
            yes_mass += prob
    if total_mass <= 0:
        return None
    # Renormalise over the observed mass. The top-k mass rarely sums to 1.
    return yes_mass / total_mass


class Scorer:
    """Async disposition scorer with cache and retry.

    `api_key` comes from OPENAI_API_KEY or from the argument. Reading from
    cache needs no key. The `analyse` path runs from cache alone.
    """

    def __init__(
        self,
        *,
        cache_dir: str,
        legacy_cache_dir: str | None = None,
        model: str = JUDGE_MODEL,
        template: str = PROBE_TEMPLATE,
        api_key: str | None = None,
        concurrency: int = 64,
        max_retries: int = 4,
        threshold: float = 0.5,
    ):
        self.model = model
        self.template = template
        self.probe = probe_id(template, model)
        self.cache = ScoreCache(cache_dir, self.probe, model)
        self.legacy = LegacyCache(legacy_cache_dir) if legacy_cache_dir else None
        self.threshold = threshold
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
        self.stats = {"api": 0, "cache": 0, "legacy_cache": 0, "failed": 0}

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "no API key: set OPENAI_API_KEY to score new documents "
                    "(cached documents can be read without one)"
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def lookup(self, doc_id: str, trait: str, *, allow_legacy: bool = True) -> Judgment | None:
        hit = self.cache.get(doc_id, trait)
        if hit is not None:
            return hit
        if allow_legacy and self.legacy is not None:
            return self.legacy.get(doc_id, trait)
        return None

    def scored_doc_ids(self, *, allow_legacy: bool = True) -> set[str]:
        """Documents with at least one cached judgment.

        This method selects the analysis sample by what is scored. A
        position in a directory listing does not select it.
        """
        ids = self.cache.scan()
        if allow_legacy and self.legacy is not None:
            ids |= self.legacy.scan()
        return ids

    async def score_one(
        self, doc_id: str, text: str, trait: str, *, allow_legacy: bool = True
    ) -> Judgment:
        cached = self.lookup(doc_id, trait, allow_legacy=allow_legacy)
        if cached is not None:
            self.stats[cached.source] += 1
            return cached

        client = self._ensure_client()
        prompt = self.template.format(trait=trait, text=text[:MAX_TEXT_CHARS])

        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1,
                        temperature=0.0,
                        logprobs=True,
                        top_logprobs=8,
                    )
            except Exception as exc:  # noqa: BLE001 - provider errors have no stable type
                if attempt == self.max_retries - 1:
                    self.stats["failed"] += 1
                    return Judgment(doc_id, trait, None, None, self.model, self.probe)
                # Jittered backoff. The sleep sits outside the semaphore. A
                # slot held during a retry would otherwise idle.
                delay = (2 ** attempt) + random.random()
                await asyncio.sleep(delay)
                continue

            choice = response.choices[0]
            raw = (choice.message.content or "").strip().lower()
            logprob_content = getattr(getattr(choice, "logprobs", None), "content", None)
            p_yes = _p_yes_from_logprobs(logprob_content)
            if p_yes is None:
                # Fall back to the string match. The probe then runs against
                # a provider that returns no logprobs.
                p_yes = 1.0 if raw.startswith("yes") else 0.0

            judgment = Judgment(
                doc_id=doc_id,
                trait=trait,
                p_yes=p_yes,
                label=int(p_yes >= self.threshold),
                model=self.model,
                probe=self.probe,
            )
            self.cache.put(judgment)
            self.stats["api"] += 1
            return judgment

        self.stats["failed"] += 1
        return Judgment(doc_id, trait, None, None, self.model, self.probe)

    async def score_documents(
        self,
        documents,
        traits,
        *,
        allow_legacy: bool = True,
        progress_every: int = 200,
    ) -> list[Judgment]:
        """Score every (document, trait) pair. Report progress every N documents."""
        traits = list(traits)
        results: list[Judgment] = []

        for index, doc in enumerate(documents, start=1):
            tasks = [
                self.score_one(doc.id, doc.probe_text, trait, allow_legacy=allow_legacy)
                for trait in traits
            ]
            results.extend(await asyncio.gather(*tasks))
            if progress_every and index % progress_every == 0:
                print(
                    f"  scored {index}/{len(documents)} documents "
                    f"(api={self.stats['api']} cache={self.stats['cache']} "
                    f"legacy={self.stats['legacy_cache']} failed={self.stats['failed']})",
                    flush=True,
                )

        return results
