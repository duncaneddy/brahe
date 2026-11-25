"""Tests for scripts/add_stub_annotations.py's stub rendering."""

import ast
import importlib.util
import textwrap
from pathlib import Path

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
