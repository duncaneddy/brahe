"""
Utils Module

Utility functions for cache management, system configuration, and formatting.
"""

from brahe._brahe import (
    # Formatting
    format_time_string,
    # Cache Management
    get_brahe_cache_dir,
    get_brahe_cache_dir_with_subdir,
    get_celestrak_cache_dir,
    get_eop_cache_dir,
    get_max_threads,
    # Threading
    get_vectorization_length_threshold,
    set_ludicrous_speed,
    set_max_threads,
    set_num_threads,
    set_vectorization_length_threshold,
)

__all__ = [
    # Formatting
    "format_time_string",
    # Cache Management
    "get_brahe_cache_dir",
    "get_brahe_cache_dir_with_subdir",
    "get_celestrak_cache_dir",
    "get_eop_cache_dir",
    "get_max_threads",
    # Threading
    "get_vectorization_length_threshold",
    "set_ludicrous_speed",
    "set_max_threads",
    "set_num_threads",
    "set_vectorization_length_threshold",
]
