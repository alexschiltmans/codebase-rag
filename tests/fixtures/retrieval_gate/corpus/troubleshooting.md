# Troubleshooting guide

## Requests are being rejected with a quota error

The per-caller token bucket has been exhausted. Check the caller's retry-after
value: a caller retrying faster than the refill rate never accumulates enough
tokens to succeed and will stay rejected indefinitely.

If every caller is affected rather than one, the burst size is probably
configured too low for the traffic shape rather than any single caller
misbehaving.

## Thumbnails come out sideways

The source image carries an EXIF orientation tag that was not applied before
resizing. A viewer that honours the tag shows the original upright, which is
why the problem is invisible until the thumbnail is generated.

## Exported spreadsheets show formula errors

A field beginning with an equals sign was written without the text prefix, and
the spreadsheet is evaluating it. Confirm the export path is applying the
formula-safe quoting to every cell rather than only to fields it thinks are
user supplied.

## Non-ASCII names are mangled in exports

The byte order mark is missing from the export, so the spreadsheet is reading a
UTF-8 file as the local code page.

## Pagination is repeating rows

The cursor is carrying only the sort value and not the tiebreak identifier, so
rows sharing a sort value fall on both sides of the page boundary. This shows up
only when the data has duplicates in the sort column, which is why it commonly
reaches production.

## A migration will not apply

Its version sorts below one already recorded in the ledger. Two branches
numbered migrations independently and both merged. Renumber the later one above
the highest applied version.
