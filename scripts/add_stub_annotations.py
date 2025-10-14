#!/usr/bin/env python3
"""
Add type annotations to the pyo3-stubgen generated _brahe.pyi file.

This script parses the docstrings in the stub file and adds proper type annotations
to function signatures so that griffe/mkdocstrings can properly extract type information.
"""

import re
from pathlib import Path


def parse_function_types(docstring: str) -> tuple[list[tuple[str, str, bool]], str]:
    """
    Parse parameter and return types from a docstring.

    Returns:
        (params, return_type) where params is list of (name, type, is_optional)
    """
    params = []
    return_type = "None"

    lines = docstring.split("\n")
    in_args = False
    in_returns = False

    for line in lines:
        stripped = line.strip()

        if stripped == "Args:":
            in_args = True
            in_returns = False
            continue
        elif stripped.startswith("Returns:"):
            in_args = False
            in_returns = True
            continue
        elif stripped.endswith(":") and not stripped.startswith("("):
            in_args = False
            in_returns = False

        if in_args and stripped:
            # Match: param_name (type, optional): description
            match = re.match(r"(\w+)\s*\(([^)]+)\):", stripped)
            if match:
                param_name = match.group(1)
                type_str = match.group(2)

                is_optional = "optional" in type_str.lower()
                # Extract base type (before comma)
                base_type = type_str.split(",")[0].strip()

                # Map types to Python annotations
                if base_type == "float":
                    py_type = "float"
                elif base_type == "int":
                    py_type = "int"
                elif base_type == "str":
                    py_type = "str"
                elif base_type == "bool":
                    py_type = "bool"
                elif "numpy.ndarray" in base_type or "ndarray" in base_type:
                    py_type = "np.ndarray"
                elif base_type.startswith("list["):
                    py_type = base_type
                else:
                    # Custom types like Epoch, TimeSystem, etc.
                    py_type = base_type

                params.append((param_name, py_type, is_optional))

        elif in_returns and stripped:
            # Match: (type): description or type: description
            match = re.match(r"\(?([^):]+)\)?:", stripped)
            if match:
                ret_type_str = match.group(1).strip()

                # Map return types
                if ret_type_str == "float":
                    return_type = "float"
                elif ret_type_str == "int":
                    return_type = "int"
                elif ret_type_str == "str":
                    return_type = "str"
                elif ret_type_str == "bool":
                    return_type = "bool"
                elif "numpy.ndarray" in ret_type_str or "ndarray" in ret_type_str:
                    return_type = "np.ndarray"
                elif ret_type_str == "tuple":
                    # For tuples, keep generic for now
                    return_type = "tuple"
                elif ret_type_str.startswith("tuple["):
                    return_type = ret_type_str
                elif ret_type_str.startswith("list["):
                    return_type = ret_type_str
                else:
                    # Custom types
                    return_type = ret_type_str
                break

    return params, return_type


def add_annotations_to_stub(stub_path: Path):
    """Add type annotations to the stub file."""
    with open(stub_path, "r") as f:
        content = f.read()

    # Split into function blocks
    output = []
    output.append('"""Type stubs for brahe._brahe module."""\n')
    output.append("from typing import Any\n")
    output.append("import numpy as np\n")
    output.append("\n")

    # Parse function by function
    func_pattern = re.compile(
        r'^def (\w+)\((.*?)\):\s*\n\s*"""(.*?)"""', re.MULTILINE | re.DOTALL
    )

    for match in func_pattern.finditer(content):
        func_name = match.group(1)
        params_str = match.group(2)
        docstring = match.group(3)

        # Parse types from docstring
        param_types, return_type = parse_function_types(docstring)

        # Build parameter list with types
        param_list = [p.strip() for p in params_str.split(",") if p.strip()]
        typed_params = []

        for param in param_list:
            # Find matching parameter type
            matched = False
            for pname, ptype, is_optional in param_types:
                if pname == param:
                    if is_optional:
                        typed_params.append(f"{param}: {ptype} | None = None")
                    else:
                        typed_params.append(f"{param}: {ptype}")
                    matched = True
                    break

            if not matched:
                # No type info found, leave as-is
                typed_params.append(param)

        # Construct typed function signature
        output.append(
            f"def {func_name}({', '.join(typed_params)}) -> {return_type}: ...\n"
        )

    # Copy any remaining content (classes, constants, etc.)
    # For now, just add a note about classes
    output.append("\n# Note: Class definitions would be added here\n")
    output.append("# For now, classes are not fully typed in this stub\n")

    # Write output
    with open(stub_path, "w") as f:
        f.write("".join(output))

    print(f"✓ Added type annotations to {stub_path}")


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    stub_file = repo_root / "brahe" / "_brahe.pyi"

    if not stub_file.exists():
        print(f"Error: Stub file {stub_file} not found!")
        print("Run: .venv/bin/pyo3-stubgen brahe._brahe .")
        return 1

    add_annotations_to_stub(stub_file)
    return 0


if __name__ == "__main__":
    exit(main())
