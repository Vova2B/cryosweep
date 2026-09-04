import inspect
import types
import pytest
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS


@pytest.mark.parametrize("kind", BUILTIN_PLOTKINDS, ids=lambda k: k.key)
def test_every_builder_accepts_field_unit(kind):
    sig = inspect.signature(kind.series)
    assert "field_unit" in sig.parameters
    assert sig.parameters["field_unit"].default == "Oe"


@pytest.mark.parametrize("kind", BUILTIN_PLOTKINDS, ids=lambda k: k.key)
def test_empty_result_returns_empty_both_ways(kind):
    empty = types.SimpleNamespace(data={})
    assert kind.series(empty) == []
    assert kind.series(empty, field_unit="Oe") == []
    assert kind.series(empty, field_unit="T") == []
