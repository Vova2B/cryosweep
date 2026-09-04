from cryosweep_core.detect.probe import (HeatCapacityDetector, ResistivityDetector, VSMDetector,
                                    ACMSDetector, TTODetector)
from cryosweep_core.analyzers.mag import VSMAnalyzer
from cryosweep_core.analyzers.hc import HCAnalyzer
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
from cryosweep_core.analyzers.acms import ACMSAnalyzer
from cryosweep_core.analyzers.tto import TTOAnalyzer
from cryosweep_core.fitting.models import CurieWeissModel
from cryosweep_core.fitting.heat_capacity import DebyeLowTModel
from cryosweep_core.fitting.transport import LinearFitModel, PowerLawRhoModel
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS

BUILTIN_DETECTORS = [HeatCapacityDetector, ResistivityDetector, VSMDetector, ACMSDetector,
                     TTODetector]
BUILTIN_ANALYZERS = [("vsm", VSMAnalyzer()), ("heatcapacity", HCAnalyzer()),
                     ("resistivity", ResistivityAnalyzer()), ("hall", HallAnalyzer()),
                     ("hall_tdep", HallTempDepAnalyzer()), ("acms", ACMSAnalyzer()),
                     ("tto", TTOAnalyzer())]
BUILTIN_FITMODELS = [CurieWeissModel(), DebyeLowTModel(), LinearFitModel(), PowerLawRhoModel()]
