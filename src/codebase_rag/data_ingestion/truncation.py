"""Measurement of how much of an ingest run exceeds the embedding model's window.

Content past the model's sequence limit is dropped when the vector is computed,
and nothing about the resulting vector says so. A retrieval miss caused by the
answer sitting in the discarded tail looks exactly like a miss caused by bad
ranking, so the share is measured at ingest and reported rather than left for
someone to go and audit by hand.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Chunks handed to the tokenizer at a time. Matches the ingestion embed batch,
# which is the other place the whole corpus is walked in slices.
DEFAULT_BATCH_SIZE = 100


@dataclass(frozen=True)
class TypeTruncation:
    """Truncation counts for a single file type."""

    file_type: str
    chunks: int
    over_limit: int

    @property
    def share(self) -> float:
        """Share of this type's chunks that exceed the limit, as a percentage."""
        return 100.0 * self.over_limit / self.chunks if self.chunks else 0.0


@dataclass(frozen=True)
class TruncationReport:
    """Per-file-type truncation for one ingest run."""

    limit: int
    chunks: int
    over_limit: int
    by_type: tuple[TypeTruncation, ...]

    @property
    def share(self) -> float:
        """Share of all chunks that exceed the limit, as a percentage."""
        return 100.0 * self.over_limit / self.chunks if self.chunks else 0.0


def measure_truncation(
    documents: Sequence[Any],
    count_tokens: Callable[[list[str]], list[int]],
    limit: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TruncationReport:
    """Count chunks exceeding ``limit`` tokens, overall and per file type.

    Every chunk is counted exactly once, so the report costs one tokenizer pass
    over the corpus. It goes in batches rather than in one call because the
    tokenizer materializes the token ids of everything handed to it at once,
    and a corpus large enough to be worth measuring is large enough for that to
    matter.
    """
    texts = [doc.page_content for doc in documents]
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        lengths.extend(count_tokens(texts[start : start + batch_size]))

    counts: dict[str, list[int]] = {}
    for doc, length in zip(documents, lengths, strict=True):
        file_type = doc.metadata.get("file_type") or Path(str(doc.metadata.get("source", ""))).suffix or "(none)"
        tallies = counts.setdefault(file_type, [0, 0])
        tallies[0] += 1
        if length > limit:
            tallies[1] += 1

    by_type = tuple(
        TypeTruncation(file_type=file_type, chunks=total, over_limit=over)
        # Worst share first: the point of the breakdown is which types are
        # affected, and an alphabetical list buries that under the harmless ones.
        for file_type, (total, over) in sorted(counts.items(), key=lambda kv: (-kv[1][1] / kv[1][0], -kv[1][0]))
    )

    return TruncationReport(
        limit=limit,
        chunks=len(texts),
        over_limit=sum(1 for length in lengths if length > limit),
        by_type=by_type,
    )


def format_truncation_report(report: TruncationReport) -> list[str]:
    """Render a report as log lines, one summary line plus the affected types.

    A clean run still produces a line. Silence would be indistinguishable from
    never having measured, which is the state this report exists to end.
    """
    if not report.chunks:
        return ["Truncation check: nothing to index, no chunks measured"]

    if not report.over_limit:
        return [
            (
                f"Truncation check: 0 of {report.chunks} chunks exceed the "
                f"{report.limit}-token embedding limit, nothing was truncated"
            )
        ]

    lines = [
        (
            f"Truncation check: {report.over_limit} of {report.chunks} chunks "
            f"({report.share:.2f}%) exceed the {report.limit}-token embedding limit "
            f"and lose their tail when embedded"
        )
    ]
    lines.extend(
        f"  {entry.file_type}: {entry.over_limit}/{entry.chunks} ({entry.share:.2f}%)"
        for entry in report.by_type
        if entry.over_limit
    )
    return lines
