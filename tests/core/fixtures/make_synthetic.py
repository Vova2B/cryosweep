import numpy as np, pandas as pd

def ramp(seed=1):
    rng = np.random.default_rng(seed)
    T = np.linspace(2.2, 252.0, 300)
    H = 0.5 + 0.03 * rng.standard_normal(300)
    return pd.DataFrame({"temperature": T, "field": H})

def field_loop(seed=2):
    rng = np.random.default_rng(seed)
    H = np.concatenate([np.linspace(0, 90000, 100), np.linspace(90000, -90000, 200), np.linspace(-90000, 0, 100)])
    T = 300 + 0.01 * rng.standard_normal(H.size)
    return pd.DataFrame({"temperature": T, "field": H})

def mixed_two_block(seed=3):
    rng = np.random.default_rng(seed)
    T1 = np.linspace(2.0, 30.0, 150); H1 = 0.5 + 0.03 * rng.standard_normal(150)
    H2 = np.concatenate([np.linspace(0, 90000, 120), np.linspace(90000, -90000, 240), np.linspace(-90000, 0, 120)])
    T2 = 2.0 + 0.01 * rng.standard_normal(H2.size)
    return pd.DataFrame({"temperature": np.concatenate([T1, T2]), "field": np.concatenate([H1, H2])})
