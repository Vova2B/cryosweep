from cryosweep_core.registry import Registry, Need, build_default_registry
from cryosweep_core.gate import MetaGate

def test_default_registry_has_builtin_detectors():
    reg = build_default_registry()
    keys = reg.detector_keys()
    assert "heatcapacity" in keys and "resistivity" in keys

def test_external_probe_registers_with_zero_core_edits():
    # the extensibility invariant: a throwaway detector registers on a FRESH registry
    reg = Registry()
    class FooDetector:
        key = "foo"
        def matches(self, h, cols): return 1.0 if "foo" in cols else 0.0
        def axes(self, h, cols): return []
    reg.register_detector(FooDetector())
    assert "foo" in reg.detector_keys()

def test_metagate_omits_per_mol_without_molar_mass():
    gate = MetaGate(molar_mass=None, n_atoms=None, geometry_suspect=True)
    decision = gate.check(Need(key="molar_mass"))
    assert decision.satisfied is False
    assert decision.reason

def test_double_normalization_guard():
    gate = MetaGate(molar_mass=945.68, n_atoms=1.0, geometry_suspect=False)
    assert gate.may_normalize("hc_sample", already_normalized={"hc_sample"}) is False
    assert gate.may_normalize("moment", already_normalized=set()) is True
