"""Forward-only database schema migrations with an applied-version ledger."""

import logging

logger = logging.getLogger(__name__)

LEDGER_TABLE = "schema_migrations"


class MigrationOutOfOrder(Exception):
    """Raised when a pending migration sorts before one already applied."""


class Migration:
    """One forward schema change, identified by a monotonic version."""

    def __init__(self, version, description, statements):
        self.version = version
        self.description = description
        self.statements = statements


def ensure_ledger(connection):
    """Create the applied-version ledger if it does not exist."""
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (version INTEGER PRIMARY KEY, description TEXT NOT NULL)"
    )


def applied_versions(connection):
    """Versions already recorded in the ledger, ascending."""
    ensure_ledger(connection)
    return sorted(row[0] for row in connection.execute(f"SELECT version FROM {LEDGER_TABLE}"))


def pending(migrations, connection):
    """Migrations not yet applied, in version order.

    A pending migration whose version sorts below the highest applied one is
    refused rather than run: two branches numbering migrations independently
    would otherwise apply in an order neither branch was tested under.
    """
    already = applied_versions(connection)
    highest = already[-1] if already else 0
    outstanding = sorted((m for m in migrations if m.version not in already), key=lambda m: m.version)
    if outstanding and outstanding[0].version < highest:
        raise MigrationOutOfOrder(f"migration {outstanding[0].version} sorts below applied version {highest}")
    return outstanding


def migrate(migrations, connection):
    """Apply every pending migration inside a single transaction each."""
    applied = []
    for migration in pending(migrations, connection):
        logger.info("Applying migration %d: %s", migration.version, migration.description)
        for statement in migration.statements:
            connection.execute(statement)
        connection.execute(
            f"INSERT INTO {LEDGER_TABLE} (version, description) VALUES (?, ?)",
            (migration.version, migration.description),
        )
        connection.commit()
        applied.append(migration.version)
    return applied
