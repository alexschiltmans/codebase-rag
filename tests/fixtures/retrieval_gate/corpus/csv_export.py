"""Streaming CSV export for large result sets."""

import csv
import io

BOM_UTF8 = "\ufeff"
DEFAULT_CHUNK_ROWS = 1000


def quote_formula_safe(value):
    """Prefix a leading formula character so a spreadsheet treats the cell as text.

    A cell beginning with =, +, -, or @ is executed as a formula when the file is
    opened, which turns an exported field into code running on the reader's
    machine. Prefixing a single quote defuses it without changing the visible text.
    """
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in {"=", "+", "-", "@"} else text


def render_row(row, columns):
    """Render one record as a list of formula-safe cell values."""
    return [quote_formula_safe(row.get(column)) for column in columns]


def stream_csv(rows, columns, chunk_rows=DEFAULT_CHUNK_ROWS, write_bom=True):
    """Yield CSV text in chunks rather than building the whole export in memory.

    The byte order mark is emitted by default because Excel otherwise reads a
    UTF-8 export as the local code page and mangles every non-ASCII name.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    if write_bom:
        buffer.write(BOM_UTF8)
    writer.writerow(columns)

    for index, row in enumerate(rows, start=1):
        writer.writerow(render_row(row, columns))
        if index % chunk_rows == 0:
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

    remaining = buffer.getvalue()
    if remaining:
        yield remaining


def export_to_path(rows, columns, path):
    """Write a CSV export to disk, returning the number of bytes written."""
    written = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for chunk in stream_csv(rows, columns):
            written += handle.write(chunk)
    return written
