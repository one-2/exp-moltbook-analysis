"""Corpus loading.

Every downstream module reads `Document`, not Moltbook's JSON shape. The
same probe then runs against a human-authored corpus.

A prevalence figure needs a comparison rate. "54.2% of posts show desire for
self improvement" has no reference point on its own. The figure can describe
the agents or a permissive probe. Zhu et al. (2504.10286) score Mastodon
alongside Chirper.ai for this reason.

**Not run:** the human baseline. `load_jsonl` is the entry point for it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Document:
    """One unit of text the probe scores."""

    id: str
    text: str
    corpus: str
    author_id: str | None = None
    author_name: str | None = None
    created_at: str | None = None
    community: str | None = None
    upvotes: int | None = None
    downvotes: int | None = None
    comment_count: int | None = None
    parent_id: str | None = None
    title: str | None = None

    @property
    def score(self) -> int | None:
        if self.upvotes is None:
            return None
        return self.upvotes - (self.downvotes or 0)

    @property
    def probe_text(self) -> str:
        """The exact text the probe reads.

        The title and the body are concatenated when a title is present. The
        title changes what the probe reads. `include_title` is therefore an
        explicit loader argument. Its default is False.
        """
        if self.title:
            return f"{self.title}\n\n{self.text}"
        return self.text

    def to_json(self) -> dict:
        return asdict(self)


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_moltbook_posts(
    posts_dir: str,
    *,
    include_title: bool = False,
    limit: int | None = None,
    skip_empty: bool = True,
) -> list[Document]:
    """Load scraped Moltbook post files into Documents.

    Files are read in sorted filename order. `limit` therefore selects the
    same sample on any machine.
    """
    docs: list[Document] = []
    filenames = sorted(f for f in os.listdir(posts_dir) if f.endswith(".json"))

    for filename in filenames:
        if limit is not None and len(docs) >= limit:
            break
        with open(os.path.join(posts_dir, filename), "r") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                continue

        post = payload.get("post") or {}
        post_id = post.get("id")
        content = (post.get("content") or "").strip()
        if not post_id:
            continue
        if skip_empty and not content:
            continue

        author = post.get("author") or {}
        submolt = post.get("submolt") or {}

        docs.append(
            Document(
                id=post_id,
                text=content,
                corpus="moltbook",
                author_id=author.get("id"),
                author_name=author.get("name"),
                created_at=post.get("created_at"),
                community=submolt.get("name") if isinstance(submolt, dict) else submolt,
                upvotes=_as_int(post.get("upvotes")),
                downvotes=_as_int(post.get("downvotes")),
                comment_count=_as_int(post.get("comment_count")),
                title=(post.get("title") or None) if include_title else None,
            )
        )

    return docs


def load_moltbook_comments(posts_dir: str, *, limit: int | None = None) -> list[Document]:
    """Load comments. Moltbook embeds them in the post payloads.

    Each comment carries its parent post id. A stimulus/response design needs
    that id. Coppolillo et al. (2601.01090) use one to test whether a
    response inherits the traits of the post it answers.

    **Not scored:** all 143,955 comments in the 31 January 2026 scrape.
    """
    docs: list[Document] = []
    filenames = sorted(f for f in os.listdir(posts_dir) if f.endswith(".json"))

    for filename in filenames:
        if limit is not None and len(docs) >= limit:
            break
        with open(os.path.join(posts_dir, filename), "r") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                continue

        parent_id = (payload.get("post") or {}).get("id")
        for comment in payload.get("comments") or []:
            if not isinstance(comment, dict):
                continue
            comment_id = comment.get("id")
            content = (comment.get("content") or "").strip()
            if not comment_id or not content:
                continue
            author = comment.get("author") or {}
            docs.append(
                Document(
                    id=comment_id,
                    text=content,
                    corpus="moltbook_comments",
                    author_id=author.get("id"),
                    author_name=author.get("name"),
                    created_at=comment.get("created_at"),
                    upvotes=_as_int(comment.get("upvotes")),
                    downvotes=_as_int(comment.get("downvotes")),
                    parent_id=parent_id,
                )
            )

    return docs


def load_jsonl(path: str, corpus: str, *, limit: int | None = None) -> list[Document]:
    """Load a baseline corpus from JSONL.

    Required keys: id, text. Optional keys: author_id, author_name,
    created_at, community, upvotes, downvotes, title.

    This is the entry point for a human-authored comparison set, such as a
    Reddit sample. The same 48 dispositions then score under the same probe.
    """
    docs: list[Document] = []
    with open(path, "r") as handle:
        for line in handle:
            if limit is not None and len(docs) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if not text or not row.get("id"):
                continue
            docs.append(
                Document(
                    id=str(row["id"]),
                    text=text,
                    corpus=corpus,
                    author_id=row.get("author_id"),
                    author_name=row.get("author_name"),
                    created_at=row.get("created_at"),
                    community=row.get("community"),
                    upvotes=_as_int(row.get("upvotes")),
                    downvotes=_as_int(row.get("downvotes")),
                    title=row.get("title"),
                )
            )
    return docs


def write_jsonl(docs: Iterable[Document], path: str) -> int:
    count = 0
    with open(path, "w") as handle:
        for doc in docs:
            handle.write(json.dumps(doc.to_json()) + "\n")
            count += 1
    return count


def iter_batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
