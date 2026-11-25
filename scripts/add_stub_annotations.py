#!/usr/bin/env python3
"""
Generate comprehensive stub file from brahe._brahe module with full docstrings and type annotations.

This script extracts all docstrings, methods, and properties from the compiled PyO3 module and creates
a complete .pyi stub file for IDE support and documentation with proper type annotations.
"""

import ast
import inspect
import math
import re
import sys
import types
from pathlib import Path
from typing import Any

RETURN_TYPE_MAP = {
    "OrbitalTrajectory": "SOrbitTrajectory",  # Fix typo in docstring
    "tuple": "Tuple",
    "list": "List",
}

# One member of a documented type union: a dotted name with optional
# subscript, e.g. "float", "numpy.ndarray", "dict[str, list[Epoch]]".
TYPE_MEMBER = re.compile(r"[\w.]+(?:\[[\w.\[\], |]*\])?")


def parse_return_type_from_docstring(doc: str) -> str:
    """Extract return type from docstring Returns section.

    The type is the text before the first top-level colon on the first line
    of the Returns section. Unions may be written ``A | B``, ``A or B``,
    ``Optional[A]`` or ``Union[A, B]``; the description may wrap onto
    following lines.
    """
    if not doc:
        return "Any"

    returns_match = re.search(r"Returns:[ \t]*\n\s*(\S.*)", doc)
    if not returns_match:
        return "Any"
    head = split_top_level(returns_match.group(1), ":")
    if len(head) < 2:
        return "Any"
    type_str = head[0]

    separator = " or " if " or " in type_str else "|"
    members = split_top_level(type_str, separator)
    if not all(TYPE_MEMBER.fullmatch(member) for member in members):
        return "Any"
    mapped = [RETURN_TYPE_MAP.get(member, member) for member in members]
    py_type = (
        f"Union[{', '.join(mapped)}]" if separator == " or " else " | ".join(mapped)
    )
    # The stub imports numpy as ``np``.
    py_type = re.sub(r"(?<![\w.])(?:numpy\.)?ndarray\b", "np.ndarray", py_type)
    try:
        ast.parse(py_type, mode="eval")
    except SyntaxError:
        return "Any"
    return py_type


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

    # Special case: trajectory property returns SOrbitTrajectory (not OrbitalTrajectory)
    if name == "trajectory":
        return "SOrbitTrajectory"

    # Special case: metadata property returns a dict; without this, the
    # generic "data" substring heuristic below (meant for methods like
    # "state_data") would misclassify it as np.ndarray.
    if name == "metadata":
        return "dict"

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


ACCESS_CONSTRAINT_TYPE = "Union[ElevationConstraint, OffNadirConstraint, LocalTimeConstraint, LookDirectionConstraint, AscDscConstraint, ElevationMaskConstraint, ConstraintAll, ConstraintAny, ConstraintNot]"

PARAM_TYPE_MAP = {
    "numpy.ndarray": "np.ndarray",
    "ndarray": "np.ndarray",
    "list": "Sequence",
    "tuple": "Tuple",
    # AccessConstraint is a union of all constraint types
    "AccessConstraint": ACCESS_CONSTRAINT_TYPE,
}

# "optional" marks a defaulted parameter only as a trailing comma-separated
# token, e.g. "(int, optional)" or "(optional)", never inside "Optional[int]".
OPTIONAL_MARKER = re.compile(r"(?:^|,)\s*optional\s*$", re.IGNORECASE)


def split_top_level(type_str: str, separator: str) -> list:
    """Split a type string on a separator that is outside any brackets."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(type_str):
        char = type_str[i]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif depth == 0 and type_str.startswith(separator, i):
            parts.append(type_str[start:i].strip())
            i += len(separator)
            start = i
            continue
        i += 1
    parts.append(type_str[start:].strip())
    return parts


def map_param_type(param_type: str) -> str:
    """Map a docstring parameter type to a stub annotation."""
    if not param_type:
        return "Any"
    if " or " in param_type:
        types = split_top_level(param_type, " or ")
        py_type = f"Union[{', '.join(PARAM_TYPE_MAP.get(t, t) for t in types)}]"
    else:
        types = split_top_level(param_type, "|")
        py_type = " | ".join(PARAM_TYPE_MAP.get(t, t) for t in types)
    # The builtin predicate ``callable`` is not a type.
    return re.sub(r"\bcallable\b", "Callable[..., Any]", py_type)


def accepts_none(py_type: str) -> bool:
    """Return whether a stub annotation admits None at its outermost level."""
    if py_type == "Any":
        return True
    for member in split_top_level(py_type, "|"):
        if member == "None":
            return True
        match = re.fullmatch(r"(Optional|Union)\[(.*)\]", member)
        if match and (
            match.group(1) == "Optional"
            or "None" in split_top_level(match.group(2), ",")
        ):
            return True
    return False


def parse_params_from_docstring(doc: str) -> list:
    """Extract parameters from Args or Arguments section of docstring.

    Returns:
        List of (param_name, param_type, is_optional, is_keyword_only) tuples
    """
    if not doc:
        return []

    # The Args: or Arguments: section runs until the next unindented line,
    # which starts the following section (Returns:, Attributes:, ...).
    args_match = re.search(
        r"^(?:Args|Arguments):[ \t]*\n((?:[ \t]+\S.*\n?|[ \t]*\n)*)",
        doc,
        re.MULTILINE,
    )
    if not args_match:
        return []

    params = []
    current_param = None
    current_desc = []

    def finish_param():
        param_name, param_type, is_optional = current_param
        is_keyword_only = "(keyword-only)" in " ".join(current_desc)
        params.append(
            (param_name, map_param_type(param_type), is_optional, is_keyword_only)
        )

    for line in args_match.group(1).split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match: "param_name (type): description" or "param_name (type, optional): description"
        param_match = re.match(r"(\w+)\s*\(([^)]+)\):\s*(.*)", line)
        if param_match:
            if current_param is not None:
                finish_param()

            param_type = param_match.group(2).strip()
            is_optional = OPTIONAL_MARKER.search(param_type) is not None
            if is_optional:
                param_type = OPTIONAL_MARKER.sub("", param_type).strip()

            current_param = (param_match.group(1), param_type, is_optional)
            current_desc = [param_match.group(3).strip()]
        elif current_param is not None:
            # Continuation of previous parameter description
            current_desc.append(line)

    if current_param is not None:
        finish_param()

    # Python forbids a positional parameter without a default after one with a
    # default. The runtime (pyo3) signature always satisfies this, so if a
    # docstring omits "optional" on a later parameter, force a default rather
    # than emitting an unparseable stub.
    seen_optional = False
    for i, (param_name, py_type, is_optional, is_keyword_only) in enumerate(params):
        if is_keyword_only:
            continue
        if is_optional:
            seen_optional = True
        elif seen_optional:
            params[i] = (param_name, py_type, True, is_keyword_only)

    return params


def check_none_defaults(label: str, obj: Any, doc: str, errors: list) -> None:
    """Record parameters whose runtime default is None but whose docstring type omits None.

    Args:
        label (str): Qualified name used in error messages.
        obj (Any): Callable whose runtime signature supplies the defaults.
        doc (str): Docstring whose Args section supplies the parameter types.
        errors (list): List that error messages are appended to.
    """
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return
    doc_types = {
        name: py_type for name, py_type, _, _ in parse_params_from_docstring(doc)
    }
    for param in sig.parameters.values():
        py_type = doc_types.get(param.name)
        if param.default is not None or py_type is None:
            continue
        if not accepts_none(py_type):
            errors.append(
                f"{label}: parameter '{param.name}' defaults to None but its "
                f"docstring type '{py_type}' does not include None (use 'T | None')"
            )


def parse_init_params_from_docstring(doc: str) -> str:
    """Extract __init__ parameters from class docstring Args section."""
    params = parse_params_from_docstring(doc)

    if not params:
        return "self, /, *args: Any, **kwargs: Any"

    param_strs = []
    kw_only_added = False
    for param_name, param_type, is_optional, is_keyword_only in params:
        # Add * separator before first keyword-only parameter
        if is_keyword_only and not kw_only_added:
            param_strs.append("*")
            kw_only_added = True

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
    kw_only_added = False
    for param_name, param_type, is_optional, is_keyword_only in params:
        # Add * separator before first keyword-only parameter
        if is_keyword_only and not kw_only_added:
            param_strs.append("*")
            kw_only_added = True

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
    kw_only_added = False
    for param_name, param_type, is_optional, is_keyword_only in params:
        # Add * separator before first keyword-only parameter
        if is_keyword_only and not kw_only_added:
            param_strs.append("*")
            kw_only_added = True

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


def format_default(value: Any) -> str:
    """Render a runtime default value as stub source.

    Args:
        value (Any): Default value reported by ``inspect.signature``.

    Returns:
        str: The literal for None, bool, int, str and finite float values,
            otherwise ``...``.
    """
    if value is None or type(value) in (bool, int, str):
        return repr(value)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    return "..."


def format_signature_params(
    sig: inspect.Signature, doc: str, receiver: str | None
) -> str:
    """Render a parameter list from a runtime signature and docstring types.

    Parameter names, kinds and defaults come from the signature; annotations
    come from the docstring Args section, or ``Any`` when undocumented.

    Args:
        sig (inspect.Signature): Runtime signature of the callable.
        doc (str): Docstring whose Args section supplies the parameter types.
        receiver (str | None): Implicit first parameter (``self`` or ``cls``),
            or None for static methods and functions. A leading ``self``
            parameter in the signature is replaced by it.

    Returns:
        str: Comma-separated parameter list, without parentheses.
    """
    doc_types = {
        name: py_type for name, py_type, _, _ in parse_params_from_docstring(doc)
    }
    params = list(sig.parameters.values())
    if receiver is not None and params and params[0].name == "self":
        params = params[1:]

    parts = [] if receiver is None else [receiver]
    previous_kind = None
    star_emitted = False
    for param in params:
        if (
            previous_kind is inspect.Parameter.POSITIONAL_ONLY
            and param.kind is not inspect.Parameter.POSITIONAL_ONLY
        ):
            parts.append("/")
        annotation = doc_types.get(param.name, "Any")
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{param.name}: {annotation}")
            star_emitted = True
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{param.name}: {annotation}")
        else:
            if param.kind is inspect.Parameter.KEYWORD_ONLY and not star_emitted:
                parts.append("*")
                star_emitted = True
            text = f"{param.name}: {annotation}"
            if param.default is not inspect.Parameter.empty:
                text += f" = {format_default(param.default)}"
            parts.append(text)
        previous_kind = param.kind
    if previous_kind is inspect.Parameter.POSITIONAL_ONLY:
        parts.append("/")
    return ", ".join(parts)


def runtime_signature(obj: Any) -> inspect.Signature | None:
    """Return the runtime signature of a callable, or None if unavailable."""
    try:
        return inspect.signature(obj)
    except (ValueError, TypeError):
        return None


def member_kind(cls: type, name: str, member: Any) -> str:
    """Classify a class member for stub rendering.

    Args:
        cls (type): Class that owns the member.
        name (str): Attribute name of the member.
        member (Any): Value returned by ``getattr(cls, name)``.

    Returns:
        str: One of ``"attribute"`` (a class attribute holding an instance of
            ``cls``, such as an enum member), ``"classmethod"``,
            ``"staticmethod"``, ``"method"`` or ``"property"``.
    """
    if isinstance(member, cls):
        return "attribute"
    raw = inspect.getattr_static(cls, name, None)
    if isinstance(raw, staticmethod):
        return "staticmethod"
    if isinstance(raw, (classmethod, types.ClassMethodDescriptorType)):
        return "classmethod"
    if callable(member):
        return "method"
    return "property"


# Parameter lists for container-protocol methods whose docstrings document no
# arguments.
DUNDER_PARAMS = {
    "__setitem__": "self, key: str, value: Any",
    "__getitem__": "self, key: str",
    "__delitem__": "self, key: str",
    "__contains__": "self, key: str",
}


def generate_class_stub(name: str, cls: type, errors: list) -> str:
    """Generate complete stub for a class with all methods and properties."""
    lines = []
    doc = inspect.getdoc(cls)
    check_none_defaults(name, cls, doc or "", errors)

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
            sig = runtime_signature(cls)
            if sig is not None:
                init_params = format_signature_params(sig, doc or "", "self")
            else:
                # Fall back to docstring parsing if signature not available
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
        except Exception as e:  # noqa: BLE001 - isolate arbitrary failures of the wrapped implementation
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
        except Exception as e:  # noqa: BLE001 - isolate arbitrary failures of the wrapped implementation
            print(f"Warning: Could not get {name}.{member_name}: {e}", file=sys.stderr)
            continue

    # Sort: enum-like class attributes, class/static methods, regular
    # methods, then properties.
    kind_order = {
        "attribute": 0,
        "classmethod": 1,
        "staticmethod": 1,
        "method": 2,
        "property": 3,
    }
    kinds = {
        member_name: member_kind(cls, member_name, member)
        for member_name, member in all_members
    }
    all_members.sort(key=lambda item: (kind_order[kinds[item[0]]], item[0]))

    # Process each member
    for member_name, member in all_members:
        try:
            kind = kinds[member_name]
            if kind == "attribute":
                lines.append(f"    {member_name}: {name}")
                lines.append("")
                continue

            member_doc = inspect.getdoc(member)
            if callable(member):
                check_none_defaults(
                    f"{name}.{member_name}", member, member_doc or "", errors
                )
            sig = runtime_signature(member) if callable(member) else None

            if kind in ("classmethod", "staticmethod"):
                # Constructor class/static methods return an instance of the
                # class unless the docstring names another return type.
                doc_type = parse_return_type_from_docstring(member_doc or "")
                return_type = doc_type if doc_type != "Any" else name
                receiver = "cls" if kind == "classmethod" else None
                if sig is not None:
                    params_str = format_signature_params(
                        sig, member_doc or "", receiver
                    )
                elif kind == "classmethod":
                    params_str = parse_classmethod_params_from_docstring(
                        member_doc or ""
                    )
                else:
                    params_str = parse_function_params_from_docstring(member_doc or "")
                lines.append(f"    @{kind}")
                lines.append(f"    def {member_name}({params_str}) -> {return_type}:")
                if member_doc:
                    lines.append(format_docstring(member_doc, indent=8))
                else:
                    lines.append('        """TODO: Add docstring"""')
                lines.append("        ...")
                lines.append("")

            # Regular methods
            elif kind == "method":
                return_type = infer_return_type_from_name(member_name, member_doc or "")
                if member_name in DUNDER_PARAMS and not parse_params_from_docstring(
                    member_doc or ""
                ):
                    # Slot wrappers carry no parameter types; use the
                    # container protocol's conventional key/value types.
                    params_str = DUNDER_PARAMS[member_name]
                elif sig is not None:
                    params_str = format_signature_params(sig, member_doc or "", "self")
                else:
                    params_str = parse_method_params_from_docstring(member_doc or "")

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

        except Exception as e:  # noqa: BLE001 - isolate arbitrary failures of the wrapped implementation
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
    kw_only_added = False
    for param_name, param_type, is_optional, is_keyword_only in params:
        # Add * separator before first keyword-only parameter
        if is_keyword_only and not kw_only_added:
            param_strs.append("*")
            kw_only_added = True

        if is_optional:
            param_strs.append(f"{param_name}: {param_type} = ...")
        else:
            param_strs.append(f"{param_name}: {param_type}")

    return ", ".join(param_strs)


def generate_function_stub(name: str, func: Any, errors: list) -> str:
    """Generate stub for a module-level function."""
    doc = inspect.getdoc(func)
    check_none_defaults(name, func, doc or "", errors)
    return_type = infer_return_type_from_name(name, doc or "")
    sig = runtime_signature(func)
    if sig is not None:
        params_str = format_signature_params(sig, doc or "", None)
    else:
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
        from brahe import _brahe
    except ImportError as e:
        print(f"Error: Could not import brahe._brahe: {e}")
        print("Make sure the package is installed: uv pip install -e .")
        return 1

    output_lines = []
    errors = []

    # Header
    output_lines.append('"""Type stubs for brahe._brahe module - AUTO-GENERATED"""')
    output_lines.append("")
    output_lines.append(
        "from typing import Any, Callable, Iterable, List, Sequence, Tuple, Optional, Union"
    )
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
        except Exception as e:  # noqa: BLE001 - isolate arbitrary failures of the wrapped implementation
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
            stub = generate_class_stub(name, cls, errors)
            output_lines.append(stub)
        except Exception as e:  # noqa: BLE001 - isolate arbitrary failures of the wrapped implementation
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
                stub = generate_function_stub(name, func, errors)
                output_lines.append(stub)
                output_lines.append("")
            except Exception as e:  # noqa: BLE001 - isolate arbitrary failures of the wrapped implementation
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

    if errors:
        print(
            f"Error: {len(errors)} docstring parameter type(s) omit a None default:",
            file=sys.stderr,
        )
        for error in errors:
            print(f"    {error}", file=sys.stderr)
        return 1

    # Validate before writing: an unparseable stub is silently discarded by
    # tools like griffe, which then fall back to bare inspection of the
    # compiled module and lose every annotation.
    output = "\n".join(output_lines)
    try:
        ast.parse(output)
    except SyntaxError as e:
        print(
            f"Error: generated stub is not valid Python (line {e.lineno}): {e.msg}",
            file=sys.stderr,
        )
        if e.text:
            print(f"    {e.text.strip()}", file=sys.stderr)
        return 1

    with open(stub_path, "w") as f:
        f.write(output)

    print(f"\n✓ Generated stub file: {stub_path}")
    print(
        f"✓ Processed {len(classes)} classes, {len(functions)} functions, {len(constants)} constants"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
