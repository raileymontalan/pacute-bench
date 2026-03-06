"""
Sampling Module

This module provides frequency-aware sampling functionality for creating
balanced linguistic datasets. It supports loading word frequency data,
adding frequency ranks to datasets, and performing weighted sampling
that balances between frequency-based and uniform random sampling.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
from .constants import (
    DEFAULT_FREQUENCY_FILE,
    DEFAULT_RANK_FILLNA,
    DEFAULT_FREQ_WEIGHT,
    DEFAULT_RANDOM_STATE
)
from .helpers import validate_dataframe_columns, validate_positive_integer, validate_probability


def load_frequency_data(freq_file_path: str = DEFAULT_FREQUENCY_FILE) -> pd.DataFrame:
    """
    Load word frequency data from a CSV file.
    
    The frequency file should contain at least two columns:
    - 'normalized': normalized word forms (lowercase)
    - 'rank': frequency rank (lower is more common)
    
    Args:
        freq_file_path: Path to the frequency CSV file (default: from constants)
        
    Returns:
        DataFrame containing word frequency information
        
    Raises:
        FileNotFoundError: If the frequency file doesn't exist
        ValueError: If required columns are missing
        
    Example:
        >>> freq_df = load_frequency_data()
        >>> print(freq_df.columns)
        Index(['normalized', 'rank', ...])
    """
    file_path = Path(freq_file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Frequency file not found: {freq_file_path}. "
            f"Please ensure the file exists at the specified path."
        )
    
    df = pd.read_csv(freq_file_path)
    
    # Validate required columns
    validate_dataframe_columns(df, ['normalized', 'rank'], "Frequency DataFrame")
    
    return df


def add_frequency_ranks(
    df: pd.DataFrame,
    freq_df: pd.DataFrame,
    word_column: str = 'normalized_word',
    rank_fillna: int = DEFAULT_RANK_FILLNA
) -> pd.DataFrame:
    """
    Add frequency rank information to a DataFrame by merging with frequency data.
    
    This function performs a case-insensitive merge between the input DataFrame
    and frequency data, adding a 'rank' column. Words not found in the frequency
    data are assigned a default high rank value.
    
    Args:
        df: Input DataFrame containing words to be ranked
        freq_df: Frequency DataFrame (from load_frequency_data)
        word_column: Column name in df containing the words (default: 'normalized_word')
        rank_fillna: Default rank for words not found in frequency data (default: 100000)
        
    Returns:
        DataFrame with added 'rank' column
        
    Raises:
        ValueError: If required columns are missing or rank_fillna is invalid
        
    Example:
        >>> freq_df = load_frequency_data()
        >>> words_df = pd.DataFrame({'normalized_word': ['ako', 'ikaw', 'siya']})
        >>> ranked_df = add_frequency_ranks(words_df, freq_df)
        >>> print(ranked_df[['normalized_word', 'rank']])
    """
    # Validate inputs
    validate_dataframe_columns(df, [word_column], "Input DataFrame")
    validate_dataframe_columns(freq_df, ['normalized', 'rank'], "Frequency DataFrame")
    validate_positive_integer(rank_fillna, "rank_fillna")
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Normalize to lowercase for matching
    df_copy['normalized_lower'] = df_copy[word_column].str.lower()
    
    # Merge with frequency data
    result = df_copy.merge(
        freq_df[['normalized', 'rank']],  # Only take needed columns
        left_on='normalized_lower',
        right_on='normalized',
        how='left'
    )
    
    # Fill missing ranks with default value
    result['rank'] = result['rank'].fillna(rank_fillna)
    
    # Clean up temporary columns
    result = result.drop(columns=['normalized_lower', 'normalized'])
    
    return result


def sample_by_frequency(
    df: pd.DataFrame,
    n_samples: int,
    freq_weight: float = DEFAULT_FREQ_WEIGHT,
    random_state: Optional[int] = DEFAULT_RANDOM_STATE
) -> pd.DataFrame:
    """
    Sample rows from a DataFrame using frequency-aware weighted sampling.
    
    This function combines frequency-based sampling with uniform random sampling
    using a weighted approach. The freq_weight parameter controls the balance:
    - freq_weight = 0.0: Pure uniform random sampling
    - freq_weight = 0.5: Balanced between frequency and random (default)
    - freq_weight = 1.0: Pure frequency-based sampling (strongly favors common words)
    
    The sampling weight for each word is calculated as:
        weight = freq_weight * (1 / (rank + 1)) + (1 - freq_weight) * (1 / n)
    
    Args:
        df: Input DataFrame with 'rank' column (use add_frequency_ranks first)
        n_samples: Number of samples to draw
        freq_weight: Weight for frequency-based sampling (0.0 to 1.0, default: 0.5)
        random_state: Random seed for reproducibility (default: 42, None for random)
        
    Returns:
        Sampled DataFrame (sample_weight column removed)
        
    Raises:
        ValueError: If 'rank' column is missing or parameters are invalid
        
    Example:
        >>> freq_df = load_frequency_data()
        >>> words_df = pd.DataFrame({'normalized_word': ['ako', 'ikaw', 'siya']})
        >>> ranked_df = add_frequency_ranks(words_df, freq_df)
        >>> sampled = sample_by_frequency(ranked_df, n_samples=2, freq_weight=0.7)
    """
    # Validate inputs
    if 'rank' not in df.columns:
        raise ValueError(
            "DataFrame must have 'rank' column. "
            "Use add_frequency_ranks() before calling sample_by_frequency()."
        )
    
    validate_positive_integer(n_samples, "n_samples")
    validate_probability(freq_weight, "freq_weight")
    
    if len(df) == 0:
        raise ValueError("Cannot sample from an empty DataFrame")
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate frequency-based weight: inverse of rank
    # Higher rank (less common) → lower weight
    df_copy['sample_weight'] = 1.0 / (df_copy['rank'] + 1)
    
    # Calculate uniform weight
    uniform_weight = 1.0 / len(df_copy)
    
    # Combine frequency-based and uniform weights
    df_copy['sample_weight'] = (
        freq_weight * df_copy['sample_weight'] +
        (1 - freq_weight) * uniform_weight
    )
    
    # Normalize weights to sum to 1
    df_copy['sample_weight'] = df_copy['sample_weight'] / df_copy['sample_weight'].sum()
    
    # Sample using weights
    actual_n_samples = min(n_samples, len(df_copy))
    sampled = df_copy.sample(
        n=actual_n_samples,
        weights='sample_weight',
        random_state=random_state,
        replace=False
    )
    
    # Remove the temporary weight column
    return sampled.drop(columns=['sample_weight'])


def sample_stratified_by_length(
    df: pd.DataFrame,
    n_samples: int,
    length_column: str = 'word_length',
    random_state: Optional[int] = DEFAULT_RANDOM_STATE
) -> pd.DataFrame:
    """
    Sample rows with stratification by word length to ensure balanced representation.
    
    This function ensures that the sample contains words of various lengths
    proportional to their distribution in the original dataset.
    
    Args:
        df: Input DataFrame
        n_samples: Total number of samples to draw
        length_column: Column name containing word length (default: 'word_length')
        random_state: Random seed for reproducibility (default: 42, None for random)
        
    Returns:
        Stratified sample DataFrame
        
    Raises:
        ValueError: If required columns are missing or parameters are invalid
        
    Example:
        >>> df = pd.DataFrame({'word': ['a', 'ab', 'abc'], 'word_length': [1, 2, 3]})
        >>> sampled = sample_stratified_by_length(df, n_samples=2)
    """
    validate_dataframe_columns(df, [length_column], "Input DataFrame")
    validate_positive_integer(n_samples, "n_samples")
    
    if len(df) == 0:
        raise ValueError("Cannot sample from an empty DataFrame")
    
    # Calculate proportions for each length
    length_counts = df[length_column].value_counts()
    length_proportions = length_counts / len(df)
    
    # Determine samples per length (proportional)
    samples_per_length = (length_proportions * n_samples).round().astype(int)
    
    # Adjust if rounding caused total to differ from n_samples
    total_samples = samples_per_length.sum()
    if total_samples < n_samples:
        # Add remaining samples to largest groups
        diff = n_samples - total_samples
        largest_groups = samples_per_length.nlargest(diff).index
        samples_per_length[largest_groups] += 1
    elif total_samples > n_samples:
        # Remove excess samples from largest groups
        diff = total_samples - n_samples
        largest_groups = samples_per_length.nlargest(diff).index
        samples_per_length[largest_groups] -= 1
    
    # Sample from each length group
    sampled_dfs = []
    for length, n in samples_per_length.items():
        if n > 0:
            length_df = df[df[length_column] == length]
            n_actual = min(n, len(length_df))
            sampled = length_df.sample(n=n_actual, random_state=random_state)
            sampled_dfs.append(sampled)
    
    return pd.concat(sampled_dfs, ignore_index=True)
