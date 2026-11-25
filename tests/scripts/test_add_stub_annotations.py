"""Tests for scripts/add_stub_annotations.py's stub rendering."""

import ast
import builtins
import importlib.util
import inspect
import textwrap
from pathlib import Path

import pytest

import brahe

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "add_stub_annotations.py"
_spec = importlib.util.spec_from_file_location("add_stub_annotations", _SCRIPT)
stubs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stubs)


class Switch:
    """Two-state switch."""

    def __init__(self, value=0):
        self.value = value


Switch.OFF = Switch(0)
Switch.ON = Switch(1)


class Config:
    """Configuration with keyword-only fields.

    Args:
        mode (Switch, optional): Operating mode.
        scale (float, optional): Scale factor.
        label (str | None, optional): Optional label.
    """

    def __init__(self, *, mode=..., scale=1.5, label=None):
        self.mode = mode

    @staticmethod
    def from_values(values):
        """Build a configuration from raw values.

        Args:
            values (list[int]): Raw values.

        Returns:
            Config: Configuration built from the values.
        """
        return Config()

    @staticmethod
    def checksum(values):
        """Checksum raw values.

        Args:
            values (list[int]): Raw values.
        """
        return sum(values)

    @classmethod
    def from_default(cls):
        """Build the default configuration.

        Returns:
            Config: Default configuration.
        """
        return cls()

    def scaled(self, factor, /, offset=0, *, clip=True):
        """Scale the configuration.

        Args:
            factor (float): Multiplicative factor.
            offset (int, optional): Additive offset.
            clip (bool, optional): Whether to clip.

        Returns:
            Config: Scaled configuration.
        """
        return self

    def offset_by(self, /, offset=0):
        """Offset the configuration.

        Args:
            offset (int, optional): Additive offset.

        Returns:
            Config: Offset configuration.
        """
        return self

    def reset(self, /):
        """Reset the configuration.

        Returns:
            Config: Reset configuration.
        """
        return self

    def combine(self, *others, **options):
        """Combine configurations.

        Args:
            others (Config): Configurations to combine.
            options (bool): Combination options.

        Returns:
            Config: Combined configuration.
        """
        return self

    @property
    def parent(self):
        """Parent configuration.

        Returns:
            Config | None: The parent configuration, or None when this
                configuration has no parent.
        """
        return None


def density(epoch, position, config=None, scale=...):
    """Compute a density.

    Args:
        epoch (Epoch): Evaluation epoch.
        position (numpy.ndarray): Position vector.
        config (Config | None, optional): Model configuration.
        scale (float, optional): Scale factor.

    Returns:
        float: Density.
    """
    return 0.0


def _class_stub(cls):
    errors = []
    stub = stubs.generate_class_stub(cls.__name__, cls, errors)
    assert errors == []
    ast.parse(stub)
    return stub


def test_format_default_renders_literals_and_ellipsis():
    assert stubs.format_default(None) == "None"
    assert stubs.format_default(True) == "True"
    assert stubs.format_default(3) == "3"
    assert stubs.format_default(-1.5) == "-1.5"
    assert stubs.format_default("km") == "'km'"
    assert stubs.format_default(...) == "..."
    assert stubs.format_default(float("inf")) == "..."
    assert stubs.format_default(Switch.ON) == "..."


def test_init_renders_keyword_only_marker_and_defaults():
    stub = _class_stub(Config)
    assert (
        "def __init__(self, *, mode: Switch = ..., scale: float = 1.5, "
        "label: str | None = None) -> None:" in stub
    )
    assert "Ellipsis" not in stub


def test_method_renders_positional_only_and_keyword_only_markers():
    stub = _class_stub(Config)
    assert (
        "def scaled(self, factor: float, /, offset: int = 0, *, clip: bool = True)"
        " -> Config:" in stub
    )


def test_method_renders_variadic_parameters():
    stub = _class_stub(Config)
    assert "def combine(self, *others: Config, **options: bool) -> Config:" in stub


def test_staticmethod_renders_without_cls():
    stub = _class_stub(Config)
    assert (
        "    @staticmethod\n    def from_values(values: list[int]) -> Config:" in stub
    )


def test_classmethod_renders_with_cls():
    stub = _class_stub(Config)
    assert "    @classmethod\n    def from_default(cls) -> Config:" in stub


def test_property_return_type_parses_wrapped_union():
    stub = _class_stub(Config)
    assert "    @property\n    def parent(self) -> Config | None:" in stub


def test_class_attributes_of_own_type_render_as_annotations():
    stub = _class_stub(Switch)
    assert "    OFF: Switch\n" in stub
    assert "    ON: Switch\n" in stub
    assert "def OFF" not in stub


def test_function_renders_none_and_ellipsis_defaults():
    errors = []
    stub = stubs.generate_function_stub("density", density, errors)
    assert errors == []
    ast.parse(stub)
    assert (
        "def density(epoch: Epoch, position: np.ndarray, "
        "config: Config | None = None, scale: float = ...) -> float:" in stub
    )


def test_parse_return_type_from_docstring_accepts_union_forms():
    doc = textwrap.dedent(
        """\
        Summary.

        Returns:
            {}: Description that wraps
                onto a second line.
        """
    )
    assert stubs.parse_return_type_from_docstring(doc.format("Epoch | None")) == (
        "Epoch | None"
    )
    assert stubs.parse_return_type_from_docstring(doc.format("Optional[Epoch]")) == (
        "Optional[Epoch]"
    )
    assert stubs.parse_return_type_from_docstring(
        doc.format("dict[str, tuple[float, float]]")
    ) == ("dict[str, tuple[float, float]]")
    assert stubs.parse_return_type_from_docstring(doc.format("float or None")) == (
        "Union[float, None]"
    )
    assert stubs.parse_return_type_from_docstring(doc.format("numpy.ndarray")) == (
        "np.ndarray"
    )
    assert stubs.parse_return_type_from_docstring(doc.format("The value of x")) == (
        "Any"
    )


def test_check_none_defaults_flags_type_without_none():
    def func(config=None):
        """Function.

        Args:
            config (Config, optional): Configuration.
        """

    errors = []
    stubs.generate_function_stub("func", func, errors)
    assert len(errors) == 1
    assert "'config' defaults to None" in errors[0]


def test_nrlmsise00_config_stub():
    stub = _class_stub(brahe.NRLMSISE00Config)
    assert "def __init__(self, *, f107_mean: NRLMSISE00Switch = ...," in stub
    assert "ap_mode: NRLMSISE00ApMode = ...," in stub
    assert "Ellipsis" not in stub
    assert "    @staticmethod\n    def from_switches(switches: list[int])" in stub


def test_nrlmsise00_switch_stub():
    stub = _class_stub(brahe.NRLMSISE00Switch)
    assert "    OFF: NRLMSISE00Switch\n" in stub
    assert "    ON: NRLMSISE00Switch\n" in stub
    assert "def OFF" not in stub


def test_atmospheric_model_stub():
    stub = _class_stub(brahe.AtmosphericModel)
    assert "    HARRIS_PRIESTER: AtmosphericModel\n" in stub
    assert "def nrlmsise00_config(self) -> NRLMSISE00Config | None:" in stub
    assert (
        "    @classmethod\n"
        "    def nrlmsise00(cls, config: NRLMSISE00Config | None = None)" in stub
    )


def test_density_nrlmsise00_stub():
    errors = []
    stub = stubs.generate_function_stub(
        "density_nrlmsise00", brahe.density_nrlmsise00, errors
    )
    assert errors == []
    assert "config: NRLMSISE00Config | None = None" in stub


def test_method_keeps_positional_only_receiver_marker():
    stub = _class_stub(Config)
    assert "def offset_by(self, /, offset: int = 0) -> Config:" in stub
    assert "def reset(self, /) -> Config:" in stub
    assert "def combine(self, *others: Config, **options: bool) -> Config:" in stub


def test_aem_to_json_string_keeps_positional_only_receiver():
    stub = _class_stub(brahe._brahe.AEM)
    assert "def to_json_string(self, /, uppercase_keys: bool = False) -> str:" in stub


def test_staticmethod_without_documented_return_is_any():
    stub = _class_stub(Config)
    assert "    @staticmethod\n    def checksum(values: list[int]) -> Any:" in stub


@pytest.fixture
def no_runtime_signatures(monkeypatch):
    def fail(obj, *args, **kwargs):
        raise ValueError(f"no signature found for {obj!r}")

    monkeypatch.setattr(stubs.inspect, "signature", fail)


def test_docstring_fallback_renders_class(no_runtime_signatures):
    stub = _class_stub(Config)
    assert (
        "def __init__(self, mode: Switch = ..., scale: float = ..., "
        "label: str | None = ...) -> None:" in stub
    )
    assert (
        "    @staticmethod\n    def from_values(values: list[int]) -> Config:" in stub
    )
    assert "    @classmethod\n    def from_default(cls) -> Config:" in stub
    assert (
        "def scaled(self, factor: float, offset: int = ..., clip: bool = ...)"
        " -> Config:" in stub
    )


def test_docstring_fallback_renders_function(no_runtime_signatures):
    def convert(value, scale=1.0, *, angle_format=None):
        """Convert a value.

        Args:
            value (float or array): Input value.
            scale (float, optional): Scale factor.
            angle_format (AngleFormat | None, optional): (keyword-only) Angle format.

        Returns:
            numpy.ndarray: Converted value.
        """

    errors = []
    stub = stubs.generate_function_stub("convert", convert, errors)
    assert errors == []
    assert (
        "def convert(value: Union[float, np.ndarray], scale: float = ..., *, "
        "angle_format: AngleFormat | None = ...) -> np.ndarray:" in stub
    )


def test_ndarray_spellings_normalize_to_np_ndarray():
    assert stubs.map_param_type("float or array") == "Union[float, np.ndarray]"
    assert stubs.map_param_type("list[numpy.ndarray]") == "list[np.ndarray]"
    assert stubs.map_param_type("ndarray | None") == "np.ndarray | None"
    assert stubs.map_param_type("np.ndarray") == "np.ndarray"
    assert stubs.map_param_type("array_like") == "array_like"


def test_resolve_annotation_imports():
    tree = ast.parse(
        textwrap.dedent(
            """\
            import numpy as np

            class Local: ...

            def f(a: Local, b: NAIFId | None, c: np.ndarray) -> datetime.datetime: ...
            def g(a: MissingType) -> missingmodule.Thing: ...
            """
        )
    )
    imports, unresolved = stubs.resolve_annotation_imports(tree, brahe)
    assert imports == ["from brahe.spice import NAIFId", "import datetime"]
    assert unresolved == ["MissingType", "missingmodule"]


def test_generate_stub_rejects_unresolved_names(monkeypatch):
    class Holder:
        """Holder.

        Returns:
            UndefinedThing: Never defined.
        """

        @property
        def thing(self):
            """Thing.

            Returns:
                UndefinedThing: Never defined.
            """

    class Module:
        pass

    Module.Holder = Holder
    with pytest.raises(stubs.StubGenerationError, match="UndefinedThing"):
        stubs.generate_stub(Module, brahe)


@pytest.fixture(scope="module")
def generated_stub():
    return stubs.generate_stub(brahe._brahe, brahe)


def test_generated_stub_annotation_names_resolve(generated_stub):
    tree = ast.parse(generated_stub)
    defined = stubs.defined_names(tree) | set(dir(builtins))
    direct, roots = stubs.annotation_names(tree)
    assert direct - defined == set()
    assert roots - defined == set()


def test_generated_stub_frame_center_body(generated_stub):
    assert "from brahe.spice import NAIFId" in generated_stub
    assert "def body(self) -> NAIFId | int | None:" in generated_stub
    assert inspect.isclass(brahe.NAIFId)


RUST_BINDINGS = textwrap.dedent(
    """\
    /// Indexed collection.
    #[pyclass(module = "brahe._brahe")]
    #[pyo3(name = "Items")]
    pub struct PyItems {
        parent: Py<PyAny>,
    }

    #[pyclass(name = "Entry")]
    #[derive(Clone)]
    pub struct PyEntry {}

    #[pyclass(module = "brahe._brahe")]
    pub struct Table {}

    impl PyItems {
        fn __contains__(&self, other: f64) -> bool {
            false
        }
    }

    #[pymethods]
    impl PyItems {
        fn __getitem__<'a>(&self, py: Python<'a>, index: isize) -> PyResult<PyEntry> {
            Ok(PyEntry {})
        }

        fn __delitem__(&self, py: Python, index: isize) -> PyResult<()> {
            Ok(())
        }

        fn __contains__(&self, item: &Bound<'_, PyAny>) -> PyResult<bool> {
            Ok(false)
        }
    }

    #[pymethods]
    impl Table {
        pub fn __getitem__(&self, key: String, py: Python) -> PyResult<f64> {
            Ok(format!("{}", key).len() as f64)
        }

        fn __setitem__(&mut self, key: &str, value: f64) {}

        fn __delitem__(&mut self, key: Selector) -> PyResult<()> {
            Ok(())
        }
    }

    #[pymethods]
    impl PyEntry {
        fn __getitem__(&self, index: Bound<'_, PySlice>) -> Vec<f64> {
            vec![]
        }

        fn __contains__(&self, cell: (usize, usize)) -> bool {
            true
        }
    }
    """
)


def test_parse_rust_slot_signatures():
    slots = stubs.parse_rust_slot_signatures([RUST_BINDINGS])
    assert slots == {
        "Items": {
            "__getitem__": ([("index", "int")], "Entry"),
            "__delitem__": ([("index", "int")], "None"),
            "__contains__": ([("item", "Any")], "bool"),
        },
        "Table": {
            "__getitem__": ([("key", "str")], "float"),
            "__setitem__": ([("key", "str"), ("value", "float")], "None"),
            "__delitem__": ([("key", "Any")], "None"),
        },
        "Entry": {
            "__getitem__": ([("index", "slice")], "Any"),
            "__contains__": ([("cell", "tuple[int, int]")], "bool"),
        },
    }


class Items:
    """Indexed collection."""

    def __getitem__(self, index, /):
        return index

    def __setitem__(self, index, value, /):
        raise NotImplementedError

    def __delitem__(self, index, /):
        pass

    def __contains__(self, item, /):
        return False


class Mapping:
    """Mapping without a Rust definition."""

    def __getitem__(self, key, /):
        return key


def test_slot_methods_render_rust_types(monkeypatch):
    monkeypatch.setattr(
        stubs,
        "rust_slot_signatures",
        lambda: stubs.parse_rust_slot_signatures([RUST_BINDINGS]),
    )
    stub = _class_stub(Items)
    assert "def __getitem__(self, index: int, /) -> Entry:" in stub
    assert "def __delitem__(self, index: int, /) -> None:" in stub
    assert "def __contains__(self, item: Any, /) -> bool:" in stub
    assert "__setitem__" not in stub
    assert "def __getitem__(self, key: str, /) -> Any:" in _class_stub(Mapping)


class SensorFault(Exception):
    """Sensor fault."""


class AnnotatedFault(Exception):
    """Fault that records notes."""

    def add_note(self, note):
        """Record a note.

        Args:
            note (str): Note text.
        """
        super().add_note(note)


def test_class_stub_skips_builtin_inherited_members():
    stub = _class_stub(SensorFault)
    assert "add_note" not in stub
    assert "with_traceback" not in stub
    stub = _class_stub(AnnotatedFault)
    assert "def add_note(self, note: str) -> Any:" in stub
    assert "with_traceback" not in stub


def _generated_class(stub, name):
    tree = ast.parse(stub)
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    return ast.get_source_segment(stub, node)


def test_generated_stub_index_slots_take_int(generated_stub):
    states = _generated_class(generated_stub, "OEMStates")
    assert "def __getitem__(self, index: int, /) -> OEMStateVector:" in states
    assert "def __delitem__(self, index: int, /) -> None:" in states
    assert "__setitem__" not in states
    rotation = _generated_class(generated_stub, "RotationMatrix")
    assert "def __getitem__(self, index: tuple[int, int], /) -> float:" in rotation
    trajectory = _generated_class(generated_stub, "Trajectory")
    assert "def __getitem__(self, index: int, /) -> np.ndarray:" in trajectory


def test_generated_stub_dict_slots_take_str(generated_stub):
    properties = _generated_class(generated_stub, "PropertiesDict")
    assert "def __getitem__(self, key: str, /) -> Any:" in properties
    assert "def __setitem__(self, key: str, value: Any, /) -> None:" in properties
    assert "def __delitem__(self, key: str, /) -> None:" in properties
    assert "def __contains__(self, key: str, /) -> bool:" in properties


def test_generated_stub_exceptions_omit_builtin_members(generated_stub):
    tree = ast.parse(generated_stub)
    exceptions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and issubclass(getattr(brahe._brahe, node.name), BaseException)
    ]
    assert {node.name for node in exceptions} >= {"BraheError", "PanicException"}
    for node in exceptions:
        methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
        assert methods.isdisjoint({"add_note", "with_traceback"}), node.name
