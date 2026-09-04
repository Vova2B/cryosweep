import numpy as np
from cryosweep_core.fitting.heat_capacity import specific_heat_full, R

def test_full_model_matches_components():
    T = np.array([5.0, 50.0, 200.0])
    from cryosweep_core.fitting.heat_capacity import debye_heat_capacity, einstein_heat_capacity
    expect = (0.01 * T
              + debye_heat_capacity(T, 200.0, 3.0)
              + einstein_heat_capacity(T, 50.0, 1.0)
              + einstein_heat_capacity(T, 150.0, 2.0))
    got = specific_heat_full(T, 200.0, 3.0, 0.01, 50.0, 150.0, 1.0, 2.0)
    assert np.allclose(got, expect)

def test_full_model_dulong_petit_lattice_limit():
    # gamma=0 isolates the lattice; at high T, Cp -> 3R(n+m1+m2)
    T = np.array([5000.0])
    got = specific_heat_full(T, 200.0, 3.0, 0.0, 50.0, 150.0, 1.0, 2.0)[0]
    assert abs(got - 3 * R * (3.0 + 1.0 + 2.0)) < 0.05
