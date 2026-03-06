"""
Utility Functions Module

This module contains common utility functions shared across the pacute package,
including output formatting helpers and validation functions.
"""

from typing import Dict, Any, Optional


def prepare_mcq_outputs(
    text_en: str,
    text_tl: str,
    mcq_options: Dict[str, str],
    row: Optional[Dict[str, Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare formatted output for multiple-choice questions.
    
    This function creates a standardized output format for MCQ questions with
    bilingual text (English and Tagalog/Filipino) and multiple choice options.
    
    Args:
        text_en: English question text with format placeholders
        text_tl: Tagalog/Filipino question text with format placeholders
        mcq_options: Dictionary containing correct and incorrect options
        row: Optional row data for formatting (default: empty dict)
        kwargs: Optional additional keyword arguments for formatting (default: empty dict)
    
    Returns:
        Dictionary containing formatted prompts with bilingual text and MCQ options
        
    Example:
        >>> mcq_options = {"A": "option1", "B": "option2", "C": "option3", "D": "option4"}
        >>> result = prepare_mcq_outputs(
        ...     "What is {word}?",
        ...     "Ano ang {word}?",
        ...     mcq_options,
        ...     row={"word": "example"}
        ... )
    """
    if row is None:
        row = {}
    if kwargs is None:
        kwargs = {}
    
    outputs = {
        "prompts": [{
            "text_en": text_en.format(**row, **kwargs),
            "text_tl": text_tl.format(**row, **kwargs),
            "mcq_options": mcq_options,
        }],
    }
    return outputs


def prepare_gen_outputs(
    text_en: str,
    text_tl: str,
    label: str,
    row: Optional[Dict[str, Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare formatted output for generative questions.
    
    This function creates a standardized output format for generative questions
    with bilingual text (English and Tagalog/Filipino) and the correct answer.
    
    Args:
        text_en: English question text with format placeholders
        text_tl: Tagalog/Filipino question text with format placeholders
        label: The correct answer/label for the question
        row: Optional row data for formatting (default: empty dict)
        kwargs: Optional additional keyword arguments for formatting (default: empty dict)
    
    Returns:
        Dictionary containing formatted prompts with bilingual text and the correct label
        
    Example:
        >>> result = prepare_gen_outputs(
        ...     "Spell {word}",
        ...     "Baybay ang {word}",
        ...     "e x a m p l e",
        ...     row={"word": "example"}
        ... )
    """
    if row is None:
        row = {}
    if kwargs is None:
        kwargs = {}
    
    outputs = {
        "prompts": [{
            "text_en": text_en.format(**row, **kwargs),
            "text_tl": text_tl.format(**row, **kwargs),
        }],
        "label": label
    }
    return outputs


def validate_dataframe_columns(df, required_columns: list, df_name: str = "DataFrame") -> None:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name of the DataFrame for error messages (default: "DataFrame")
        
    Raises:
        ValueError: If any required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_positive_integer(value: int, param_name: str) -> None:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter for error messages
        
    Raises:
        ValueError: If value is not a positive integer
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{param_name} must be a positive integer, got {value}")


def validate_probability(value: float, param_name: str) -> None:
    """
    Validate that a value is a valid probability (between 0 and 1).
    
    Args:
        value: Value to validate
        param_name: Name of the parameter for error messages
        
    Raises:
        ValueError: If value is not between 0 and 1
    """
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{param_name} must be between 0 and 1, got {value}")
