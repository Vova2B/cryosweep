from cryosweep_core.fitting.models import CurieWeissModel

def register_fitmodels(registry):
    registry.register_fitmodel(CurieWeissModel())
    return registry
