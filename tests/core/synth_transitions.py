import numpy as np

def debye_like(T, plateau=78.0, theta=230.0):
    """Smooth Debye-shaped Cp(T): rises ~T^3, rolls to `plateau` — REAL curvature a rigid
    gamma*T+beta*T^3 cannot follow. Not drawn from any model under test."""
    x = np.asarray(T, float) / theta
    return plateau * (x ** 3 / (x ** 3 + 0.06)) * (1 + 0.15 * np.exp(-x))

def lam(T, Tc, amp=8.0, width=4.0):
    """Symmetric lambda-ish peak (exp of -|t|/w), sharp cusp at Tc."""
    T = np.asarray(T, float)
    return amp * np.exp(-np.abs(T - Tc) / width)

def rng(seed=7):
    return np.random.default_rng(seed)

def wide_null(n=150, noise=0.4, seed=7):
    T = np.linspace(2.0, 300.0, n)
    return T, debye_like(T) + rng(seed).normal(0.0, noise, n)

def afm_like(n=140, Tc=203.0, noise=0.4, seed=3):
    T = np.linspace(4.6, 262.0, n)
    return T, debye_like(T) + lam(T, Tc) + rng(seed).normal(0.0, noise, n)

def narrow_window(Tc=201.0, lo=181.8, hi=211.9, n=62, noise=0.3, seed=5):
    T = np.linspace(lo, hi, n)
    return T, debye_like(T) + lam(T, Tc) + rng(seed).normal(0.0, noise, n)

def broad_transition(n=150, Tc=40.0, noise=0.3, seed=11):
    """Genuinely broad 2nd-order anomaly (width 12 K) on real curvature — must SURVIVE."""
    T = np.linspace(2.0, 120.0, n)
    return T, debye_like(T, plateau=40.0, theta=140.0) + lam(T, Tc, amp=5.0, width=12.0) \
           + rng(seed).normal(0.0, noise, n)
