"""One physical field must get one label.

`setpoint_key` rounds to the nearest integer regardless of magnitude, so on a real file the
same 40 kOe ramp arrived as 40000.8870 -> 40001.0 and 39999.5860 -> 40000.0 and was plotted
as two separate M(T) curves. A coarser fixed bin does not fix this: any bin has edges, and
two nearby values can straddle one. Cluster the values actually present instead.
"""
from __future__ import annotations

import math

from cryosweep_core.grouping import cluster_field_setpoints


def test_real_file_medians_collapse_to_one_label():
    labels = cluster_field_setpoints([499.8600, 19999.8650, 40000.8870, 39999.5860])
    assert labels[2] == labels[3], labels          # the two 40 kOe blocks
    assert len({round(v, 6) for v in labels}) == 3, labels


def test_labels_are_round_setpoints_not_raw_medians():
    """The label reaches legends and CSV cells. Clustering must not turn 500 Oe into
    499.9 Oe -- group by cluster, label by setpoint_key(cluster median)."""
    labels = cluster_field_setpoints([499.8600, 19999.8650, 40000.8870, 39999.5860])
    assert labels == [500.0, 20000.0, 40000.0, 40000.0], labels


def test_distinct_fields_stay_distinct():
    labels = cluster_field_setpoints([100.0, 5000.0, 40000.0, 100000.0])
    assert len({round(v, 6) for v in labels}) == 4, labels


def test_straddling_a_round_number_does_not_split():
    """The exact case a bin-based fix would still get wrong."""
    labels = cluster_field_setpoints([39999.9, 40000.1])
    assert labels[0] == labels[1], labels


def test_small_fields_use_the_absolute_floor():
    """Below abs_floor a relative tolerance is meaningless -- 0.2 Oe and 0.9 Oe are both
    'about zero' and must not become separate curves."""
    labels = cluster_field_setpoints([0.2, 0.9])
    assert labels[0] == labels[1], labels


def test_order_is_preserved_and_length_matches():
    vals = [40000.9, 500.0, 39999.6]
    labels = cluster_field_setpoints(vals)
    assert len(labels) == len(vals)
    assert labels[0] == labels[2] and labels[1] != labels[0]


def test_non_finite_passes_through_as_nan():
    labels = cluster_field_setpoints([float("nan"), 500.0])
    assert math.isnan(labels[0]) and labels[1] == 500.0


def test_empty_and_all_non_finite():
    assert cluster_field_setpoints([]) == []
    out = cluster_field_setpoints([float("nan"), float("inf")])
    assert len(out) == 2 and all(math.isnan(v) for v in out), out


def test_a_dense_chain_does_not_collapse_everything():
    """Single-link clustering can chain. Values spaced just inside the tolerance SHOULD
    merge (they are one drifting hold); values spaced well outside must not."""
    merged = cluster_field_setpoints([1000.0, 1000.5, 1001.0])
    assert len({round(v, 6) for v in merged}) == 1, merged
    apart = cluster_field_setpoints([1000.0, 2000.0, 3000.0])
    assert len({round(v, 6) for v in apart}) == 3, apart
