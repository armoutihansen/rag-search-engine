"""Shared pytest fixtures and configuration."""

import os
import pytest


@pytest.fixture(autouse=True)
def set_project_root(monkeypatch):
    """Ensure all tests run from the project root so relative paths resolve correctly."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(project_root)


SAMPLE_MOVIES = [
    {
        "id": 1,
        "title": "The Bear",
        "description": "A grizzly bear roams the wilderness of Canada.",
    },
    {
        "id": 2,
        "title": "Lion King",
        "description": "A lion cub grows up to rule the African savanna.",
    },
    {
        "id": 3,
        "title": "Finding Nemo",
        "description": "A clownfish searches the ocean to find his missing son.",
    },
    {
        "id": 4,
        "title": "Jungle Book",
        "description": "A boy raised by wolves and bears in the Indian jungle.",
    },
]
