"""Cursor-based pagination over ordered result sets."""

import base64
import json

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


class InvalidCursor(Exception):
    """Raised when a supplied cursor cannot be decoded."""


def encode_cursor(sort_value, tiebreak_id):
    """Encode a position in the result set as an opaque cursor.

    The cursor carries the tiebreak identifier as well as the sort value,
    because rows sharing a sort value would otherwise be skipped or repeated
    depending on which side of the page boundary they landed on.
    """
    return base64.urlsafe_b64encode(json.dumps([sort_value, tiebreak_id]).encode()).decode()


def decode_cursor(cursor):
    """Decode an opaque cursor back into a sort value and tiebreak identifier."""
    try:
        sort_value, tiebreak_id = json.loads(base64.urlsafe_b64decode(cursor))
    except (ValueError, TypeError) as exc:
        raise InvalidCursor(f"cursor is not decodable: {exc}") from exc
    return sort_value, tiebreak_id


def clamp_page_size(requested):
    """Bound a caller-supplied page size to the configured maximum."""
    if requested is None:
        return DEFAULT_PAGE_SIZE
    if requested < 1:
        raise ValueError(f"page size must be positive, got {requested}")
    return min(requested, MAX_PAGE_SIZE)


def paginate(rows, cursor=None, page_size=None):
    """Return one page of rows plus the cursor for the next page.

    Offset pagination is avoided deliberately: an insert before the current
    offset shifts every later row back by one, so a client walking the pages
    sees a row twice and never sees another.
    """
    size = clamp_page_size(page_size)
    start = 0
    if cursor is not None:
        sort_value, tiebreak_id = decode_cursor(cursor)
        start = next(
            (i + 1 for i, row in enumerate(rows) if (row["sort_value"], row["id"]) == (sort_value, tiebreak_id)),
            0,
        )

    page = rows[start : start + size]
    next_cursor = encode_cursor(page[-1]["sort_value"], page[-1]["id"]) if len(page) == size else None
    return page, next_cursor
