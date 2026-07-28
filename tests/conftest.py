"""Shared test fixtures and configuration."""

import logging

import pytest

from codebase_rag.config import Config


@pytest.fixture(autouse=True)
def reset_config_singleton() -> None:
    """Reset Config singleton before and after each test."""
    Config._instance = None
    yield
    Config._instance = None


@pytest.fixture(autouse=True)
def restore_root_logger_state():
    """Snapshot and restore root logger state around each test.

    Prevents tests from mutating logging config for subsequent tests.
    """
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    yield

    for handler in root_logger.handlers[:]:
        if handler not in original_handlers:
            handler.close()
            root_logger.removeHandler(handler)

    # setLevel(), not `.level =`: only the former clears the per-logger isEnabledFor cache,
    # so a direct assignment leaves children answering from a cache built at the test's level.
    root_logger.setLevel(original_level)


@pytest.fixture(autouse=True)
def restore_codebase_rag_logger_state():
    """Snapshot and restore the "codebase_rag" logger's handlers, level, and pipeline's
    module-global `_prior_level` around each test.

    Several `test_pipeline.py` tests call `setup_logging` directly without a matching
    `_teardown_logging`, which (correctly, by design, see fix-ingest-logging's design.md
    on the single-run-per-process limitation) leaves `_prior_level` captured and the
    logger's level pinned until a teardown call comes along. Without this fixture, that
    leak crosses test boundaries: whichever test runs first (order is randomized by
    pytest-randomly) permanently sets the level every later real `setup_logging` call
    restores to, instead of each test seeing a clean `NOTSET` baseline.
    """
    from codebase_rag.data_ingestion import pipeline as pipeline_module

    logger = logging.getLogger("codebase_rag")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_prior_level = pipeline_module._prior_level

    yield

    for handler in logger.handlers[:]:
        if handler not in original_handlers:
            handler.close()
            logger.removeHandler(handler)

    logger.setLevel(original_level)
    pipeline_module._prior_level = original_prior_level
