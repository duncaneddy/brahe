"""
Utils Module

Utility functions for cache management, system configuration, and formatting.
"""

from brahe._brahe import (
    # Cache Management
    get_brahe_cache_dir,
    get_brahe_cache_dir_with_subdir,
    get_eop_cache_dir,
    get_celestrak_cache_dir,
    # Threading
    set_num_threads,
    set_max_threads,
    set_ludicrous_speed,
    get_max_threads,
    # Formatting
    format_time_string,
)

__all__ = [
    # Cache Management
    "get_brahe_cache_dir",
    "get_brahe_cache_dir_with_subdir",
    "get_eop_cache_dir",
    "get_celestrak_cache_dir",
    # Threading
    "set_num_threads",
    "set_max_threads",
    "set_ludicrous_speed",
    "get_max_threads",
    # Formatting
    "format_time_string",
]
