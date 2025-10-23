import brahe


def set_cli_eop():
    """Initialize EOP data for CLI commands that require frame transformations."""
    brahe.initialize_eop()
