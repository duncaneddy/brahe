#!/usr/bin/env python3
"""
Generate comprehensive stub file from brahe._brahe module with full docstrings and type annotations.

This script extracts all docstrings, methods, and properties from the compiled PyO3 module and creates
a complete .pyi stub file for IDE support and documentation with proper type annotations.
"""

import ast
import builtins
import functools
import importlib.util
import inspect
import math
import re
import sys
import traceback
import types
from pathlib import Path
from typing import Any

RETURN_TYPE_MAP = {
    "tuple": "Tuple",
    "list": "List",
}

# One member of a documented type union: a dotted name with optional
# subscript, e.g. "float", "numpy.ndarray", "dict[str, list[Epoch]]".
TYPE_MEMBER = re.compile(r"[\w.]+(?:\[[\w.\[\], |]*\])?")

# Docstring spellings of the numpy array type: "numpy.ndarray", "ndarray" and
# "array". The stub imports numpy as ``np``.
NDARRAY_SPELLING = re.compile(r"(?<![\w.])(?:(?:numpy\.)?ndarray|array)\b(?!\.)")


def normalize_ndarray(py_type: str) -> str:
    """Rewrite every docstring spelling of the numpy array type to ``np.ndarray``."""
    return NDARRAY_SPELLING.sub("np.ndarray", py_type)


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
    py_type = normalize_ndarray(py_type)
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
    return re.sub(r"\bcallable\b", "Callable[..., Any]", normalize_ndarray(py_type))


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
            parameter in the signature is replaced by it and keeps its
            positional-only marker.

    Returns:
        str: Comma-separated parameter list, without parentheses.
    """
    doc_types = {
        name: py_type for name, py_type, _, _ in parse_params_from_docstring(doc)
    }
    params = list(sig.parameters.values())
    previous_kind = None
    if receiver is not None and params and params[0].name == "self":
        previous_kind = params[0].kind
        params = params[1:]

    parts = [] if receiver is None else [receiver]
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


# Parameter lists for container-protocol methods of classes without a Rust
# binding definition.
DUNDER_PARAMS = {
    "__setitem__": "self, key: str, value: Any, /",
    "__getitem__": "self, key: str, /",
    "__delitem__": "self, key: str, /",
    "__contains__": "self, key: str, /",
}

RUST_BINDINGS_DIR = (
    Path(__file__).resolve().parent.parent / "crates" / "brahe-py" / "src"
)

RUST_SCALAR_TYPES = {
    "isize": "int",
    "usize": "int",
    "i8": "int",
    "i16": "int",
    "i32": "int",
    "i64": "int",
    "u8": "int",
    "u16": "int",
    "u32": "int",
    "u64": "int",
    "f32": "float",
    "f64": "float",
    "bool": "bool",
    "String": "str",
    "str": "str",
}

# Generic wrappers whose Python type is that of their last type argument.
RUST_TRANSPARENT_WRAPPERS = {"PyResult", "Bound", "Borrowed", "Py", "PyRef", "PyRefMut"}

# A pyclass attribute and the struct or enum it annotates. Any attributes
# between them, such as ``#[pyo3(name = "...")]``, are captured.
RUST_PYCLASS = re.compile(
    r"#\[pyclass\b(?P<attrs>.*?)\b(?:struct|enum)\s+(?P<rust>\w+)", re.DOTALL
)
RUST_PYCLASS_NAME = re.compile(r"\bname\s*=\s*\"(\w+)\"")

# Items that decide which type a slot method belongs to: a ``#[pymethods]``
# impl header, any other impl header, and a slot method definition.
RUST_ITEM = re.compile(
    r"#\[pymethods\]\s*(?:#\[[^\]]*\]\s*)*impl\s*(?:<[^>]*>\s*)?(?P<pymethods>\w+)"
    r"|^[ \t]*(?:unsafe\s+)?impl\b(?P<impl>)"
    r"|\bfn\s+(?P<slot>__(?:getitem|setitem|delitem|contains)__)\b",
    re.MULTILINE,
)


# A receiver written with an explicit type, such as ``slf: PyRef<'_, Self>``.
RUST_TYPED_RECEIVER = re.compile(
    r"&?\s*(?:(?:PyRef|PyRefMut|Bound|Borrowed|Py)\s*<\s*(?:'\w+\s*,\s*)?)?Self\s*>?"
)


def merge_slot_definitions(definitions: list) -> tuple:
    """Combine the definitions of one slot method into a single signature.

    A method defined more than once, for example in alternative
    ``#[cfg(...)]`` branches, keeps the annotations its definitions agree on
    and uses ``Any`` for the rest. The result does not depend on the order of
    the definitions.

    Args:
        definitions (list): ``(params, return_type)`` pairs, where ``params``
            lists ``(name, annotation)`` pairs.

    Returns:
        tuple: The merged ``(params, return_type)`` pair.
    """
    definitions = sorted(definitions)
    params, return_type = definitions[0]
    for other_params, other_return in definitions[1:]:
        if len(other_params) != len(params):
            params = [(name, "Any") for name, _ in params]
        else:
            params = [
                (name, py_type if py_type == other_type else "Any")
                for (name, py_type), (_, other_type) in zip(params, other_params)
            ]
        if other_return != return_type:
            return_type = "Any"
    return params, return_type


def split_rust_top_level(text: str) -> list:
    """Split Rust source text on commas outside any brackets.

    Args:
        text (str): Comma-separated Rust parameters or type arguments.

    Returns:
        list: Stripped, non-empty parts.
    """
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">)]" and not (char == ">" and text[i - 1 : i] == "-"):
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def rust_type_to_python(rust_type: str, class_names: dict) -> str:
    """Map a Rust binding parameter or return type to a stub annotation.

    Args:
        rust_type (str): Rust type as written in the binding source.
        class_names (dict): Rust pyclass type names mapped to Python class
            names, including ``Self`` for the type whose method is parsed.

    Returns:
        str: The Python annotation, or ``Any`` when the type has no specific
            Python equivalent.
    """
    text = re.sub(r"\s+", " ", rust_type).strip()
    text = re.sub(r"^&\s*(?:'\w+\s+)?(?:mut\s+)?", "", text)
    if text == "()":
        return "None"
    if text.startswith("(") and text.endswith(")"):
        members = [
            rust_type_to_python(member, class_names)
            for member in split_rust_top_level(text[1:-1])
        ]
        if not members or "Any" in members:
            return "Any"
        return f"tuple[{', '.join(members)}]"
    match = re.fullmatch(r"(?:\w+::)*(\w+)\s*(?:<(.*)>)?", text)
    if match is None:
        return "Any"
    head, args = match.groups()
    if args is None:
        if head in RUST_SCALAR_TYPES:
            return RUST_SCALAR_TYPES[head]
        if head == "PySlice":
            return "slice"
        if head in class_names:
            return class_names[head]
    if head.startswith(("PyArray", "PyReadonlyArray")):
        return "np.ndarray"
    type_args = [
        arg for arg in split_rust_top_level(args or "") if not arg.startswith("'")
    ]
    if not type_args:
        return "Any"
    if head in RUST_TRANSPARENT_WRAPPERS:
        return rust_type_to_python(type_args[-1], class_names)
    if head == "Result":
        return rust_type_to_python(type_args[0], class_names)
    if head == "Option":
        inner = rust_type_to_python(type_args[0], class_names)
        return "Any" if inner == "Any" else f"{inner} | None"
    return "Any"


def parse_rust_fn_signature(source: str, start: int) -> tuple[list, str]:
    """Read the parameters and return type of a Rust function definition.

    Args:
        source (str): Rust source text.
        start (int): Index just past the function name.

    Returns:
        tuple[list, str]: The ``(name, rust_type)`` pairs of the parameters
            other than the receiver and the ``Python`` token, and the return
            type text (``()`` when omitted).
    """
    open_paren = source.index("(", start)
    depth = 0
    for i in range(open_paren, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                close_paren = i
                break
    params = []
    for position, param in enumerate(
        split_rust_top_level(source[open_paren + 1 : close_paren])
    ):
        match = re.fullmatch(r"(?:mut\s+)?(\w+)\s*:(?!:)\s*(.+)", param, re.DOTALL)
        if match is None:
            continue
        name, rust_type = match.groups()
        if position == 0 and RUST_TYPED_RECEIVER.fullmatch(rust_type.strip()):
            continue
        if re.fullmatch(r"(?:\w+::)*Python\s*(?:<[^>]*>)?", rust_type.strip()):
            continue
        params.append((name, rust_type))
    tail = re.match(
        r"\s*(?:->\s*(.*?))?\s*(?:\bwhere\b|\{)", source[close_paren + 1 :], re.DOTALL
    )
    return_type = tail.group(1) if tail and tail.group(1) else "()"
    return params, return_type


def parse_rust_slot_signatures(sources: list) -> dict:
    """Extract container slot method signatures from Rust binding sources.

    Each ``#[pymethods] impl`` block is attributed to the Python class name of
    its type: the ``name`` given in ``#[pyclass(...)]`` or ``#[pyo3(...)]``,
    otherwise the Rust type name. Receiver and ``Python`` token parameters are
    omitted, and ``Self`` in other parameters names the class itself.

    Args:
        sources (list): Rust source texts, parsed together so that a pyclass
            defined in one source resolves in another.

    Returns:
        dict: Python class names mapped to ``{slot_name: (params,
            return_type)}``, where ``params`` lists ``(name, annotation)``
            pairs.
    """
    sources = [re.sub(r"//[^\n]*", "", source) for source in sources]
    class_names = {}
    for source in sources:
        for match in RUST_PYCLASS.finditer(source):
            name = RUST_PYCLASS_NAME.search(match.group("attrs"))
            rust = match.group("rust")
            class_names[rust] = name.group(1) if name else rust

    definitions = {}
    for source in sources:
        current = None
        for match in RUST_ITEM.finditer(source):
            if match.group("pymethods") is not None:
                current = match.group("pymethods")
            elif match.group("impl") is not None:
                current = None
            elif current is not None:
                params, return_type = parse_rust_fn_signature(source, match.end())
                py_class = class_names.get(current, current)
                names = {**class_names, "Self": py_class}
                definitions.setdefault(py_class, {}).setdefault(
                    match.group("slot"), []
                ).append(
                    (
                        [
                            (name, rust_type_to_python(rust_type, names))
                            for name, rust_type in params
                        ],
                        rust_type_to_python(return_type, names),
                    )
                )
    return {
        py_class: {
            slot: merge_slot_definitions(slot_definitions)
            for slot, slot_definitions in class_slots.items()
        }
        for py_class, class_slots in definitions.items()
    }


@functools.cache
def rust_slot_signatures() -> dict:
    """Return the container slot signatures defined in the Rust bindings."""
    sources = [
        path.read_text(encoding="utf-8")
        for path in sorted(RUST_BINDINGS_DIR.glob("*.rs"))
    ]
    return parse_rust_slot_signatures(sources)


def defining_class(cls: type, name: str) -> type | None:
    """Return the first class in ``cls.__mro__`` whose namespace defines ``name``."""
    return next((base for base in cls.__mro__ if name in vars(base)), None)


def rust_slots_of(cls: type, name: str) -> dict | None:
    """Return the Rust slot signatures of the class that defines member ``name``.

    Args:
        cls (type): Class whose member is rendered.
        name (str): Member name.

    Returns:
        dict | None: The slot signatures parsed for the defining class, or
            None when the Rust bindings define no slot method for it.
    """
    owner = defining_class(cls, name)
    if owner is None:
        return None
    return rust_slot_signatures().get(owner.__name__)


def is_builtin_inherited(cls: type, name: str) -> bool:
    """Return whether ``cls`` inherits ``name`` unchanged from a builtin type."""
    if name in vars(cls):
        return False
    owner = defining_class(cls, name)
    return owner is not None and owner.__module__ == "builtins"


def generate_class_stub(name: str, cls: type, errors: list) -> str:
    """Generate complete stub for a class with all methods and properties."""
    lines = []
    doc = inspect.getdoc(cls)
    check_none_defaults(name, cls, doc or "", errors)

    bases = [base.__name__ for base in cls.__bases__ if base is not object]
    lines.append(f"class {name}({', '.join(bases)}):" if bases else f"class {name}:")
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
        if is_builtin_inherited(cls, member_name):
            continue
        if member_name in DUNDER_PARAMS:
            rust_slots = rust_slots_of(cls, member_name)
            # PyO3 implements __setitem__ and __delitem__ with one slot, so
            # defining either exposes both; the undefined one raises
            # NotImplementedError.
            if rust_slots is not None and member_name not in rust_slots:
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
                # Constructor classmethods return an instance of the class
                # unless the docstring names another return type.
                return_type = parse_return_type_from_docstring(member_doc or "")
                if return_type == "Any" and kind == "classmethod":
                    return_type = name
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
                rust_slots = (
                    rust_slots_of(cls, member_name)
                    if member_name in DUNDER_PARAMS
                    else None
                )
                if rust_slots is not None:
                    # Slot wrappers carry no parameter types at runtime; the
                    # Rust definition supplies them.
                    slot_params, slot_return = rust_slots[member_name]
                    params_str = ", ".join(
                        [
                            "self",
                            *(f"{param}: {py_type}" for param, py_type in slot_params),
                            "/",
                        ]
                    )
                    if slot_return != "Any":
                        return_type = slot_return
                elif member_name in DUNDER_PARAMS and not parse_params_from_docstring(
                    member_doc or ""
                ):
                    # Slot wrappers without a Rust definition use the
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


class StubGenerationError(Exception):
    """Raised when the generated stub fails validation."""


def annotation_names(tree: ast.Module) -> tuple[set, set]:
    """Collect the names that annotations and class bases in a stub refer to.

    Args:
        tree (ast.Module): Parsed stub.

    Returns:
        tuple[set, set]: Names used directly (``Epoch`` in ``Epoch | None``)
            and names used only as the root of an attribute chain
            (``datetime`` in ``datetime.datetime``).
    """
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in [
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                args.vararg,
                args.kwarg,
            ]:
                if arg is not None and arg.annotation is not None:
                    annotations.append(arg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, ast.ClassDef):
            annotations.extend(node.bases)

    direct, roots = set(), set()
    for annotation in annotations:
        attribute_roots = {
            id(sub.value)
            for sub in ast.walk(annotation)
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
        }
        for sub in ast.walk(annotation):
            if isinstance(sub, ast.Name):
                (roots if id(sub) in attribute_roots else direct).add(sub.id)
    return direct, roots - direct


def defined_names(tree: ast.Module) -> set:
    """Return the names a stub defines or imports at module level."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def resolve_annotation_imports(tree: ast.Module, package: Any) -> tuple[list, list]:
    """Find imports for annotation names that a stub does not define.

    A class in the namespace of ``package`` is imported from the module that
    defines it (the namespace is read directly, so lazy module attributes are
    never triggered). A name used only as an attribute root (``polars`` in
    ``polars.DataFrame``) is imported as a module when it is importable.

    Args:
        tree (ast.Module): Parsed stub.
        package (Any): Package whose exported classes may be referenced.

    Returns:
        tuple[list, list]: Import statements to add, and names that could not
            be resolved.
    """
    known = defined_names(tree) | set(dir(builtins))
    direct, roots = annotation_names(tree)
    imports, unresolved = [], []
    for name in sorted(direct - known):
        obj = vars(package).get(name)
        if inspect.isclass(obj):
            imports.append(f"from {obj.__module__} import {name}")
        else:
            unresolved.append(name)
    for name in sorted(roots - known):
        if importlib.util.find_spec(name) is not None:
            imports.append(f"import {name}")
        else:
            unresolved.append(name)
    return imports, unresolved


def generate_stub(module: Any, package: Any) -> str:
    """Generate the complete, validated stub source for an extension module.

    Args:
        module (Any): Extension module to describe.
        package (Any): Package whose exported classes annotations may name.

    Returns:
        str: Stub source.

    Raises:
        StubGenerationError: If a docstring type omits a None default, the
            stub is not valid Python, or an annotation names something that
            is neither defined in the stub nor importable.
    """
    output_lines = []
    errors = []

    # Header
    output_lines.append('"""Type stubs for brahe._brahe module - AUTO-GENERATED"""')
    output_lines.append("")
    output_lines.append(
        "from typing import Any, Callable, Iterable, List, Sequence, Tuple, Optional, Union"
    )
    output_lines.append("import numpy as np")
    import_index = len(output_lines)
    output_lines.append("")

    # Process all module members
    module_members = {}
    for name in dir(module):
        if name.startswith("_") and name != "__version__":
            continue
        try:
            member = getattr(module, name)
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

    print(
        f"Processed {len(classes)} classes, {len(functions)} functions, "
        f"{len(constants)} constants"
    )

    if errors:
        raise StubGenerationError(
            f"{len(errors)} docstring parameter type(s) omit a None default:\n"
            + "\n".join(f"    {error}" for error in errors)
        )

    # Validate before writing: an unparseable stub is silently discarded by
    # tools like griffe, which then fall back to bare inspection of the
    # compiled module and lose every annotation.
    output = "\n".join(output_lines)
    try:
        tree = ast.parse(output)
    except SyntaxError as e:
        message = f"generated stub is not valid Python (line {e.lineno}): {e.msg}"
        if e.text:
            message += f"\n    {e.text.strip()}"
        raise StubGenerationError(message) from e

    imports, unresolved = resolve_annotation_imports(tree, package)
    if unresolved:
        raise StubGenerationError(
            "annotation name(s) neither defined in the stub nor importable: "
            + ", ".join(unresolved)
        )
    output_lines[import_index:import_index] = imports
    return "\n".join(output_lines)


def main():
    """Generate the stub file."""
    repo_root = Path(__file__).parent.parent
    stub_path = repo_root / "brahe" / "_brahe.pyi"

    # Import the module
    sys.path.insert(0, str(repo_root))
    try:
        import brahe
        from brahe import _brahe
    except ImportError as e:
        print(f"Error: Could not import brahe._brahe: {e}")
        print("Make sure the package is installed: uv pip install -e .")
        return 1

    try:
        output = generate_stub(_brahe, brahe)
    except StubGenerationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    with open(stub_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n✓ Generated stub file: {stub_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
