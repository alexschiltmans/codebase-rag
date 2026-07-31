"""Test dataset utilities for evaluation."""

import json
from pathlib import Path
from typing import Any

TEST_QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "What are the main features of the codebase?",
        "keywords": ["features", "main", "capabilities"],
        "sources": ["overview", "documentation/features"],
        "samples": [
            {
                "question": "What are the main features of the codebase?",
                "context": "This is a Python package for load modeling and network simulation.",
                "expected_answer": "This is a Python package for load modeling and network simulation.",
                "keywords": ["Python", "load modeling", "network", "simulation"],
            },
            {
                "question": "How do I set up a load model?",
                "context": "To set up a load model, initialize a LoadModeller with a network topology and providers.",
                "expected_answer": "Initialize a LoadModeller with a network topology and providers.",
                "keywords": ["LoadModeller", "network topology", "providers", "initialize"],
            },
        ],
    },
    {
        "question": "What is this codebase?",
        "keywords": ["codebase", "definition"],
        "sources": ["overview", "documentation"],
        "samples": [
            {
                "question": "What is this codebase?",
                "context": "This is a framework for power network analysis.",
                "expected_answer": "This is a framework for power network analysis.",
                "keywords": ["framework", "power network", "analysis"],
            }
        ],
    },
    {
        "name": "transformer_modeling",
        "question": "How do I model a transformer in codebase?",
        "keywords": ["transformer", "model", "component", "parameters"],
        "sources": ["components", "transformer"],
        "samples": [],
    },
    {
        "name": "distribution_network",
        "question": "Can I use codebase for distribution network analysis?",
        "keywords": ["distribution", "network", "unbalanced", "low voltage"],
        "sources": ["distribution", "networks"],
        "samples": [],
    },
]


def get_test_dataset() -> list[dict[str, Any]]:
    """Get the test dataset for evaluation.

    Returns:
        list[dict]: List of test question dictionaries.
    """
    dataset_path = Path(__file__).parent / "data" / "test_questions.json"

    if dataset_path.exists():
        try:
            with open(dataset_path) as f:
                loaded: list[dict[str, Any]] = json.load(f)
                return loaded
        except json.JSONDecodeError:
            # Fall through to return default dataset
            pass

    return TEST_QUESTIONS


def get_question_by_id(question_id: int) -> dict[str, Any]:
    """Get a specific test question by ID.

    Args:
        question_id: Index of the question to retrieve.

    Returns:
        dict: Test question dictionary.
    """
    if 0 <= question_id < len(TEST_QUESTIONS):
        return TEST_QUESTIONS[question_id]
    raise ValueError(f"Question ID {question_id} out of range (0-{len(TEST_QUESTIONS) - 1})")
