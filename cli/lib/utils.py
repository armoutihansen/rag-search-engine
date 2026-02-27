"""Utility functions for the CLI."""
import json
from pathlib import Path


def load_movies_data(data_path: str = "./data/movies.json") -> list:
    """Load movie documents from JSON file.
    
    Args:
        data_path: Path to the movies JSON file
        
    Returns:
        List of movie document dictionaries
    """
    path = Path(data_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["movies"]
