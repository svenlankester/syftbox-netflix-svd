import json
import os

import numpy as np


def load_tv_vocabulary(tv_series_path):
    """
    Load TV series vocabulary from a JSON file.

    Args:
        tv_series_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary mapping TV series titles to item IDs.
    """
    if not os.path.isfile(tv_series_path):
        raise FileNotFoundError(
            f"TV series vocabulary file not found: {tv_series_path}"
        )

    with open(tv_series_path, "r") as f:
        tv_vocab = json.load(f)
    return tv_vocab


def load_imdb_ratings(imdb_ratings_path):
    """
    Load IMDB ratings from a NumPy file.

    Args:
        imdb_ratings_path (str): Path to the NumPy file.

    Returns:
        dict: A dictionary mapping normalized TV series titles to ratings.
    """
    if not os.path.isfile(imdb_ratings_path):
        raise FileNotFoundError(f"IMDB ratings file not found: {imdb_ratings_path}")

    imdb_data = np.load(imdb_ratings_path, allow_pickle=True).item()
    imdb_ratings = {
        normalize_string(title): float(rating)
        for title, rating in imdb_data.items()
        if rating
    }
    return imdb_ratings


def normalize_string(s):
    """
    Normalize a string by removing zero-width spaces and converting to lowercase.

    Args:
        s (str): The input string.

    Returns:
        str: The normalized string.
    """
    return s.replace("\u200b", "").lower()


def load_global_item_factors(path):
    """
    Load the global item factors matrix from the given path.

    Args:
        path (str): Path to the file.

    Returns:
        np.ndarray: Global item factors matrix.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Global item factors file not found: {path}")
    return np.load(path)


def save_global_item_factors(V, path):
    """
    Save the global item factors matrix to the given path.

    Args:
        V (np.ndarray): Global item factors matrix.
        path (str): Path to save the file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, V)
    print(f"Global item factors saved at: {path}")
