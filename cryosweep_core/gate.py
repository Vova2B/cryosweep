from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GateDecision:
    satisfied: bool
    reason: str = ""
    remedy: dict | None = None

class MetaGate:
    def __init__(self, molar_mass=None, n_atoms=None, geometry_suspect=False):
        self.molar_mass = molar_mass
        self.n_atoms = n_atoms
        self.geometry_suspect = geometry_suspect

    def check(self, need) -> GateDecision:
        if need.key == "molar_mass" and self.molar_mass is None:
            return GateDecision(False, "no MOLWGHT in header",
                                {"flag": "--molar-mass", "example": "--molar-mass 945.68"})
        return GateDecision(True)

    def may_normalize(self, column, already_normalized) -> bool:
        return column not in already_normalized
