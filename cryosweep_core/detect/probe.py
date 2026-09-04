from __future__ import annotations

class _ByAppDetector:
    def __init__(self, key, byapp_token, fingerprint, strong_fingerprint=()):
        self.key = key
        self._tokens = [byapp_token.lower()] if isinstance(byapp_token, str) else [t.lower() for t in byapp_token]
        self._fp = fingerprint
        self._strong = tuple(s.lower() for s in strong_fingerprint)
    def matches(self, h, cols):
        score = 0.0
        if h.app and any(t in h.app.lower() for t in self._tokens):
            score += 0.8
        if any(any(f in c.lower() for c in cols) for f in self._fp):
            score += 0.2
        if self._strong and all(any(s in c.lower() for c in cols) for s in self._strong):
            score += 0.6
        return min(score, 1.0)
    def axes(self, h, cols):
        return ["temperature", "field"]

HeatCapacityDetector = _ByAppDetector("heatcapacity", "heatcapacity", ["samp hc"])
ResistivityDetector = _ByAppDetector("resistivity",
    ["resistivity", "actransport"],
    ["bridge 1 resistivity", "res. ch1 (ohm-cm)"],
    # Bare Origin "dc rho" exports carry no BYAPP token; a column that has BOTH "resistivity"
    # and "ohm-cm" (e.g. "Resistivity ... (mikroOhm-cm)") is specific enough to score >=0.5
    # without false-firing on QD ("...(Ohm-m)") or ACT ("Res. chN...") files.
    strong_fingerprint=["resistivity", "ohm-cm"])
VSMDetector = _ByAppDetector("vsm", "vsm", ["moment (emu)"],
    strong_fingerprint=["long moment (emu)", "long scan std dev"])
ACMSDetector = _ByAppDetector("acms", "acms", ["frequency (hz)", "amplitude (oe)"],
    strong_fingerprint=["m' (emu)", "m'' (emu)"])
# Thermal Transport Option (TTO). NOTE the fingerprints are RAW-name fragments (matches()
# lower-cases the raw column names; it never calls _norm), so "seebeck coef." is spelled
# WITHOUT the unit — the raw name carries a micro sign and would not contain "(v/k)".
TTODetector = _ByAppDetector("tto", "thermal_transport",
    ["conductivity (w/k-m)", "seebeck coef."],
    strong_fingerprint=["conductivity (w/k-m)", "figure of merit zt"])

def detect_probe_ranked(header, columns, registry):
    """Every detector's (score, key), best first.

    Detection has always computed this list and thrown away all but the winner. The
    runner-up matters for files whose BYAPP token and whose CONTENTS disagree -- e.g. a
    DC-mode ACMS file, which is an ACMS file by token but carries only DC magnetisation.
    """
    cols = set(columns)
    scored = [(d.matches(header, cols), d.key) for d in registry.detectors()]
    scored.sort(reverse=True)
    return scored


def detect_probe(header, columns, registry):
    scored = detect_probe_ranked(header, columns, registry)
    return scored[0] if scored else (0.0, None)
