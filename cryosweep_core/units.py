"""Single source for unit conversions and cross-probe field-magnitude conventions.

One fact, one spelling: these constants exist because each of them used to be typed
independently in several modules (the Oe->T factor in six places, the zero-field
threshold in four), which is the defect class that shipped the fit-line-key bug.
Import THESE — never re-type the number.
"""

# Oersted per Tesla: B[T] = H[Oe] / OE_PER_T (CGS H-field <-> SI, mu0*H).
OE_PER_T = 10000.0

# |H| below this counts as a held "zero field": RRR ramp selection (resistivity, TTO),
# SC-screening preconditions (ACMS), and the setpoint display collapse to nominal 0.
ZERO_FIELD_OE = 50.0
