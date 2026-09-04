import numpy as np

# Known ground truth (Ohm-cm). Both channels metallic (rho increases with T).
RHO0_1, A1 = 3.0e-4, 1.0e-6     # ch1: rho = 3.0e-4 + 1e-6*T  -> rho(300)=6.0e-4
RHO0_2, A2 = 1.0e-4, 3.0e-6     # ch2: rho = 1.0e-4 + 3e-6*T  -> rho(300)=1.0e-3 (steeper)

_HEADER = """[Header]
TITLE,act_synth
BYAPP,ACTRANSPORT,2.0,1.1
INFO,1,SAMPLE1_CROSS_SECTION
[Data]
"""
_COLS = ("Comment,Time Stamp (sec),Status (code),Temperature (K),Magnetic Field (Oe),"
         "Res. ch1 (ohm-cm),Res. ch2 (ohm-cm)\n")

def write_act(path, n=150):
    T = np.linspace(2.0, 300.0, n)
    rows = []
    for t in T:
        r1 = RHO0_1 + A1 * t
        r2 = RHO0_2 + A2 * t
        rows.append(f",0,0,{t:.4f},0.0,{r1:.8e},{r2:.8e}\n")
    with open(path, "w") as f:
        f.write(_HEADER); f.write(_COLS); f.writelines(rows)
    return {"rho1_300": RHO0_1 + A1*300, "rho2_300": RHO0_2 + A2*300,
            "rrr1_raw": (RHO0_1 + A1*300)/(RHO0_1 + A1*2),
            "rrr2_raw": (RHO0_2 + A2*300)/(RHO0_2 + A2*2)}

if __name__ == "__main__":
    import pathlib
    print(write_act(pathlib.Path(__file__).parent / "act_synth.dat"))
