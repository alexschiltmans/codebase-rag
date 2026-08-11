# Changelog

## Unreleased

- Cursor pagination now carries a tiebreak identifier alongside the sort value,
  fixing repeated and skipped rows when the sort column contains duplicates.
- Thumbnail generation applies the EXIF orientation tag before resizing.

## 2.4.0

- Password hashes record their iteration count, and a login with a hash below
  current policy transparently upgrades it.
- The rate limiter refills continuously rather than on a fixed tick, closing the
  double-burst window at the boundary between two fixed windows.

## 2.3.1

- CSV exports prefix cells beginning with a formula character so spreadsheets
  treat them as text.
- CSV exports emit a byte order mark so non-ASCII fields survive being opened in
  Excel.

## 2.3.0

- Added the websocket broadcast hub with per-topic subscriber limits.
- Retry backoff applies full jitter rather than retrying on a fixed schedule.

## 2.2.0

- Migrations are forward-only and refuse to apply out of version order.
- The LRU cache reports its hit ratio and eviction count.
