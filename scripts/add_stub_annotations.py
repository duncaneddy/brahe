#!/usr/bin/env python3
"""
Generate comprehensive stub file from brahe._brahe module with full docstrings and type annotations.

This script extracts all docstrings, methods, and properties from the compiled PyO3 module and creates
a complete .pyi stub file for IDE support and documentation with proper type annotations.
"""

import inspect
import sys
import re
from pathlib import Path
from typing import Any


def parse_return_type_from_docstring(doc: str) -> str:
    """Extract return type from docstring Returns section."""
    if not doc:
        return "Any"

    # Look for Returns: section - match type including dots for numpy.ndarray, etc.
    returns_match = re.search(
        r"Returns:\s*\n\s*([\w.]+(?:\[[\w\[\], ]+\])?)", doc, re.MULTILINE
    )
    if returns_match:
        type_str = returns_match.group(1).strip()

        # Handle "A or B" union syntax from docstrings
        if " or " in type_str:
            types = [t.strip() for t in type_str.split(" or ")]
            mapped_types = []
            for t in types:
                if t == "numpy.ndarray" or t == "ndarray":
                    mapped_types.append("np.ndarray")
                elif t == "list":
                    mapped_types.append("List")
                elif t == "tuple":
                    mapped_types.append("Tuple")
                else:
                    mapped_types.append(t)
            return f"Union[{', '.join(mapped_types)}]"

        # Map common types
        type_map = {
            "str": "str",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "ndarray": "np.ndarray",
            "numpy.ndarray": "np.ndarray",  # Map numpy.ndarray to np.ndarray
            "Epoch": "Epoch",
            "Quaternion": "Quaternion",
            "RotationMatrix": "RotationMatrix",
            "EulerAngle": "EulerAngle",
            "EulerAxis": "EulerAxis",
            "OrbitalTrajectory": "OrbitTrajectory",  # Fix typo in docstring
            "tuple": "Tuple",
            "list": "List",
        }
        return type_map.get(type_str, type_str)

    return "Any"


def infer_return_type_from_name(name: str, doc: str) -> str:
    """Infer return type from method name and docstring."""
    # Check docstring for type hints first
    doc_type = parse_return_type_from_docstring(doc)
    if doc_type != "Any":
        return doc_type

    # Special dunder methods
    if name == "__setitem__":
        return "None"
    if name == "__getitem__":
        return "Any"
    if name == "__delitem__":
        return "None"
    if name == "__contains__":
        return "bool"
    if name == "__len__":
        return "int"
    if name == "__iter__":
        return "Any"  # Iterator type
    if name in ["__repr__", "__str__"]:
        return "str"

    # Special case: trajectory property returns OrbitTrajectory (not OrbitalTrajectory)
    if name == "trajectory":
        return "OrbitTrajectory"

    # Conversion methods
    if name.startswith("to_"):
        if "datetime" in name:
            return "Any"  # datetime object
        if "quaternion" in name:
            return "Quaternion"
        if "rotation_matrix" in name or "dcm" in name:
            return "RotationMatrix"
        if "euler" in name:
            return "EulerAngle"
        if "string" in name or "isostring" in name:
            return "str"
        if "array" in name or "vector" in name or "matrix" in name:
            return "np.ndarray"

    # Methods that return numpy arrays
    if any(
        x in name
        for x in [
            "interpolate",
            "state",
            "states",
            "epochs",
            "axis",
            "data",
            "vector",
            "matrix",
        ]
    ):
        return "np.ndarray"

    # Property-like methods
    if any(
        x in name
        for x in ["jd", "mjd", "gast", "gmst", "nanoseconds", "seconds", "angle"]
    ):
        return "float"
    if "date" in name:
        return "Tuple[int, ...]"
    if (
        "day" in name
        or "year" in name
        or "month" in name
        or "hour" in name
        or "minute" in name
    ):
        return "int"

    return "Any"


def parse_params_from_docstring(doc: str) -> list:
    """Extract parameters from Args or Arguments section of docstring.

    Returns:
        List of (param_name, param_type, is_optional) tuples
    """
    if not doc:
        return []

    # Look for Args: or Arguments: section - match everything after
    args_match = re.search(
        r"(?:Args|Arguments):(.*?)(?=\n(?:Returns:|Example:|Raises:|Note:)|$)",
        doc,
        re.DOTALL,
    )
    if not args_match:
        return []

    args_text = args_match.group(1)
    params = []

    # Parse each argument line
    for line in args_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match: "param_name (type): description" or "param_name (type, optional): description"
        param_match = re.match(r"(\w+)\s*\(([^)]+)\):", line)
        if param_match:
            param_name = param_match.group(1)
            param_type = param_match.group(2).strip()

            # Check if parameter is optional
            is_optional = "optional" in param_type.lower()
            if is_optional:
                # Remove "optional" from type string
                param_type = re.sub(
                    r",?\s*optional\s*", "", param_type, flags=re.IGNORECASE
                ).strip()

            # Handle "A or B" union syntax from docstrings
            if " or " in param_type:
                # Split and map each type
                types = [t.strip() for t in param_type.split(" or ")]
                mapped_types = []
                for t in types:
                    # Apply type mapping
                    if t == "numpy.ndarray" or t == "ndarray":
                        mapped_types.append("np.ndarray")
                    elif t == "list":
                        mapped_types.append("List")
                    elif t == "tuple":
                        mapped_types.append("Tuple")
                    else:
                        mapped_types.append(t)
                py_type = f"Union[{', '.join(mapped_types)}]"
            else:
                # Map types to Python type annotations
                type_map = {
                    "str": "str",
                    "int": "int",
                    "float": "float",
                    "bool": "bool",
                    "numpy.ndarray": "np.ndarray",
                    "ndarray": "np.ndarray",
                    "TimeSystem": "TimeSystem",
                    "AngleFormat": "AngleFormat",
                    "Epoch": "Epoch",
                    "list": "List",
                    "tuple": "Tuple",
                    "dict": "dict",
                    # AccessConstraint is a union of all constraint types
                    "AccessConstraint": "Union[ElevationConstraint, OffNadirConstraint, LocalTimeConstraint, LookDirectionConstraint, AscDscConstraint, ElevationMaskConstraint, ConstraintAll, ConstraintAny, ConstraintNot]",
                }

                py_type = type_map.get(param_type, param_type)

            # If param_type is empty or just whitespace, default to Any
            if not param_type or not param_type.strip():
                py_type = "Any"
            params.append((param_name, py_type, is_optional))

    return params


def parse_init_params_from_docstring(doc: str) -> str:
    """Extract __init__ parameters from class docstring Args section."""
    params = parse_params_from_docstring(doc)

    if not params:
        return "self, /, *args: Any, **kwargs: Any"

    param_strs = []
    for param_name, param_type, is_optional in params:
        if is_optional:
            param_strs.append(f"{param_name}: {param_type} = ...")
        else:
            param_strs.append(f"{param_name}: {param_type}")

    return "self, " + ", ".join(param_strs)


def parse_method_params_from_docstring(doc: str) -> str:
    """Extract method parameters from docstring Args section."""
    params = parse_params_from_docstring(doc)

    if not params:
        return "self"

    param_strs = []
    for param_name, param_type, is_optional in params:
        if is_optional:
            param_strs.append(f"{param_name}: {param_type} = ...")
        else:
            param_strs.append(f"{param_name}: {param_type}")

    return "self, " + ", ".join(param_strs)


def parse_classmethod_params_from_docstring(doc: str) -> str:
    """Extract classmethod parameters from docstring Args section."""
    params = parse_params_from_docstring(doc)

    if not params:
        return "cls"

    param_strs = []
    for param_name, param_type, is_optional in params:
        if is_optional:
            param_strs.append(f"{param_name}: {param_type} = ...")
        else:
            param_strs.append(f"{param_name}: {param_type}")

    return "cls, " + ", ".join(param_strs)


def format_docstring(doc: str, indent: int = 4) -> str:
    """Format a docstring with proper indentation."""
    if not doc:
        return " " * indent + '"""TODO: Add docstring"""'

    lines = doc.strip().split("\n")
    result = []
    result.append(" " * indent + '"""' + lines[0])
    for line in lines[1:]:
        result.append(" " * indent + line)
    result.append(" " * indent + '"""')
    return "\n".join(result)


def is_classmethod(cls: type, name: str, member: Any) -> bool:
    """Check if a member is a classmethod."""
    # For PyO3 classes, classmethods typically start with 'from_' or are constructors
    if name.startswith("from_"):
        return True
    # Check if it's a builtin_function_or_method (static/class methods in PyO3)
    if hasattr(member, "__self__"):
        return member.__self__ is cls
    return False


def generate_class_stub(name: str, cls: type) -> str:
    """Generate complete stub for a class with all methods and properties."""
    lines = []
    doc = inspect.getdoc(cls)

    lines.append(f"class {name}:")
    if doc:
        lines.append(format_docstring(doc))
    else:
        lines.append('    """TODO: Add docstring"""')
    lines.append("")

    # Track what we've added
    added_members = set()

    # First add __init__ if it exists
    if hasattr(cls, "__init__"):
        try:
            init_params = parse_init_params_from_docstring(doc)
            # If no parameters were found in docstring, use empty signature
            # This happens for classes without explicit #[new] in PyO3
            if init_params == "self, /, *args: Any, **kwargs: Any":
                init_params = "self"
            lines.append(f"    def __init__({init_params}) -> None:")
            lines.append('        """Initialize instance."""')
            lines.append("        ...")
            lines.append("")
            added_members.add("__init__")
        except Exception as e:
            print(f"Warning: Could not process {name}.__init__: {e}", file=sys.stderr)

    # Get all members and sort them
    all_members = []
    # Include common dunder methods for dict-like and container classes
    special_methods = [
        "__init__",
        "__new__",
        "__setitem__",
        "__getitem__",
        "__delitem__",
        "__contains__",
        "__len__",
        "__iter__",
        "__repr__",
        "__str__",
    ]
    for member_name in dir(cls):
        if member_name.startswith("_") and member_name not in special_methods:
            continue
        if member_name in added_members:
            continue

        try:
            member = getattr(cls, member_name)
            # Skip __new__ if it has the default generic docstring
            if member_name == "__new__":
                member_doc = inspect.getdoc(member)
                if member_doc and "Create and return a new object" in member_doc:
                    continue
            all_members.append((member_name, member))
        except Exception as e:
            print(f"Warning: Could not get {name}.{member_name}: {e}", file=sys.stderr)
            continue

    # Sort: classmethods first, then regular methods, then properties
    def sort_key(item):
        member_name, member = item
        if is_classmethod(cls, member_name, member):
            return (0, member_name)
        elif callable(member):
            return (1, member_name)
        else:
            return (2, member_name)

    all_members.sort(key=sort_key)

    # Process each member
    for member_name, member in all_members:
        try:
            member_doc = inspect.getdoc(member)

            # Check if it's a classmethod (PyO3 classmethods are often builtin_function_or_method)
            if is_classmethod(cls, member_name, member):
                return_type = (
                    name  # Constructor classmethods return instance of the class
                )
                # Parse parameters from docstring
                params_str = parse_classmethod_params_from_docstring(member_doc or "")
                lines.append("    @classmethod")
                lines.append(f"    def {member_name}({params_str}) -> {return_type}:")
                if member_doc:
                    lines.append(format_docstring(member_doc, indent=8))
                else:
                    lines.append('        """TODO: Add docstring"""')
                lines.append("        ...")
                lines.append("")

            # Regular methods
            elif callable(member):
                return_type = infer_return_type_from_name(member_name, member_doc or "")
                # Parse parameters from docstring
                params_str = parse_method_params_from_docstring(member_doc or "")

                # Special handling for dunder methods with known signatures
                if member_name == "__setitem__" and params_str == "self":
                    params_str = "self, key: str, value: Any"
                elif member_name == "__getitem__" and params_str == "self":
                    params_str = "self, key: str"
                elif member_name == "__delitem__" and params_str == "self":
                    params_str = "self, key: str"
                elif member_name == "__contains__" and params_str == "self":
                    params_str = "self, key: str"

                lines.append(f"    def {member_name}({params_str}) -> {return_type}:")
                if member_doc:
                    lines.append(format_docstring(member_doc, indent=8))
                else:
                    lines.append('        """TODO: Add docstring"""')
                lines.append("        ...")
                lines.append("")

            # Properties
            else:
                return_type = infer_return_type_from_name(member_name, member_doc or "")
                lines.append("    @property")
                lines.append(f"    def {member_name}(self) -> {return_type}:")
                if member_doc:
                    lines.append(format_docstring(member_doc, indent=8))
                else:
                    lines.append('        """TODO: Add docstring"""')
                lines.append("        ...")
                lines.append("")

        except Exception as e:
            print(
                f"Warning: Could not process {name}.{member_name}: {e}", file=sys.stderr
            )
            continue

    return "\n".join(lines)


def parse_function_params_from_docstring(doc: str) -> str:
    """Extract function parameters from docstring Args section."""
    params = parse_params_from_docstring(doc)

    if not params:
        return ""

    param_strs = []
    for param_name, param_type, is_optional in params:
        if is_optional:
            param_strs.append(f"{param_name}: {param_type} = ...")
        else:
            param_strs.append(f"{param_name}: {param_type}")

    return ", ".join(param_strs)


def generate_function_stub(name: str, func: Any) -> str:
    """Generate stub for a module-level function."""
    doc = inspect.getdoc(func)
    return_type = infer_return_type_from_name(name, doc or "")
    params_str = parse_function_params_from_docstring(doc or "")

    lines = []
    lines.append(f"def {name}({params_str}) -> {return_type}:")
    if doc:
        lines.append(format_docstring(doc))
    else:
        lines.append('    """TODO: Add docstring"""')
    lines.append("    ...")

    return "\n".join(lines)


def main():
    """Generate the stub file."""
    repo_root = Path(__file__).parent.parent
    stub_path = repo_root / "brahe" / "_brahe.pyi"

    # Import the module
    sys.path.insert(0, str(repo_root))
    try:
        import brahe._brahe as _brahe
    except ImportError as e:
        print(f"Error: Could not import brahe._brahe: {e}")
        print("Make sure the package is installed: uv pip install -e .")
        return 1

    output_lines = []

    # Header
    output_lines.append('"""Type stubs for brahe._brahe module - AUTO-GENERATED"""')
    output_lines.append("")
    output_lines.append("from typing import Any, List, Tuple, Optional, Union")
    output_lines.append("import numpy as np")
    output_lines.append("")

    # Process all module members
    module_members = {}
    for name in dir(_brahe):
        if name.startswith("_") and name != "__version__":
            continue
        try:
            member = getattr(_brahe, name)
            module_members[name] = member
        except Exception as e:
            print(f"Warning: Could not get {name}: {e}", file=sys.stderr)

    # Separate into classes, functions, and constants
    classes = {}
    functions = {}
    constants = {}

    for name, member in module_members.items():
        if inspect.isclass(member):
            classes[name] = member
        elif callable(member):
            functions[name] = member
        else:
            constants[name] = member

    # Generate classes first
    output_lines.append("# Classes")
    output_lines.append("")
    for name, cls in sorted(classes.items()):
        print(f"Processing class: {name}")
        try:
            stub = generate_class_stub(name, cls)
            output_lines.append(stub)
        except Exception as e:
            print(f"Error processing class {name}: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            output_lines.append(f"class {name}: ...")
            output_lines.append("")

    # Generate functions
    if functions:
        output_lines.append("# Functions")
        output_lines.append("")
        for name, func in sorted(functions.items()):
            print(f"Processing function: {name}")
            try:
                stub = generate_function_stub(name, func)
                output_lines.append(stub)
                output_lines.append("")
            except Exception as e:
                print(f"Error processing function {name}: {e}", file=sys.stderr)
                output_lines.append(
                    f"def {name}(*args: Any, **kwargs: Any) -> Any: ..."
                )
                output_lines.append("")

    # Generate constants
    if constants:
        output_lines.append("# Module constants")
        output_lines.append("")
        for name, value in sorted(constants.items()):
            print(f"Processing constant: {name}")
            if isinstance(value, str):
                output_lines.append(f"{name}: str")
            elif isinstance(value, (int, float)):
                output_lines.append(f"{name}: {type(value).__name__}")
            else:
                output_lines.append(f"{name}: Any")

    # Write output
    output = "\n".join(output_lines)
    with open(stub_path, "w") as f:
        f.write(output)

    print(f"\n✓ Generated stub file: {stub_path}")
    print(
        f"✓ Processed {len(classes)} classes, {len(functions)} functions, {len(constants)} constants"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
