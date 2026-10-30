"""Independent check: Kabsch on conserved core labels, then CA distances for IMGT vs proposed pairings."""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze import load

CORES = {
    "V": [str(i) for i in list(range(18, 26)) + [39, 40, 41, 42, 43] + list(range(86, 92)) + list(range(100, 105)) + [118, 119]],
    "C": [str(i) for i in list(range(18, 26)) + [39, 40, 41, 42, 43] + list(range(100, 105)) + [118, 119]],
    "G": [str(i) for i in list(range(3, 14)) + list(range(21, 37))],
}


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return lambda X: (X - pc) @ R.T + qc


def check(t, X, Y, mapping, doms, core=None):
    core = core or CORES[t]
    A = [d for d in doms if d["cls"] == X]; B = [d for d in doms if d["cls"] == Y]
    imgt, prop = [], []
    for a, b in itertools.product(A, B):
        if a["id"] == b["id"]: continue
        la = {l: c for l, c in zip(a["lab"], a["ca"])}; lb = {l: c for l, c in zip(b["lab"], b["ca"])}
        cc = [l for l in core if l in la and l in lb]
        if len(cc) < 10: continue
        f = kabsch(np.array([la[l] for l in cc]), np.array([lb[l] for l in cc]))
        for pa, pb in mapping:
            if pa in la and pb in lb:
                x = f(la[pa][None])[0]
                imgt.append(np.linalg.norm(x - lb[pa]) if pa in lb else np.nan); prop.append(np.linalg.norm(x - lb[pb]))
    imgt, prop = np.array(imgt), np.array(prop)
    if len(imgt) == 0:
        return None
    ok = ~np.isnan(imgt)
    return len(prop), (np.median(imgt[ok]) if ok.any() else np.nan), np.median(prop), (np.mean(prop[ok] < imgt[ok]) if ok.any() else np.nan)


CLAIMS = [
    ("V", "V-LAMBDA", "V-KAPPA", [("8", "9"), ("9", "10")], "VL lambda A strand: lambda 8/9 = kappa 9/10"),
    ("V", "V-LAMBDA", "VH", [("83", "81"), ("84", "82")], "light-chain D strand: 83/84 = VH 81/82"),
    ("V", "V-KAPPA", "VH", [("83", "81"), ("84", "82")], "kappa D strand: 83/84 = VH 81/82"),
    ("V", "V-BETA", "VH", [("83", "82")], "TRBV D strand: 83 = VH 82"),
    ("V", "V-ALPHA", "V-KAPPA", [("81", "83"), ("82", "84")], "TRAV 81/82 = kappa 83/84"),
    ("V", "V-BETA", "V-KAPPA", [(str(i), str(i - 1)) for i in (66, 67, 68, 69, 70, 71, 72)] + [("74", "72")], "TRBV C'' strand 66-74 = kappa 65-72"),
    ("V", "V-KAPPA", "VH", [("27", "28"), ("28", "29"), ("29", "30")], "CDR1 base: light 27-29 = VH 28-30"),
    ("V", "V-BETA", "VH", [("27", "28"), ("28", "29")], "CDR1 base: TRBV 27-28 = VH 28-29"),
    ("C", "C-DELTA", "C-KAPPA", [(str(i), str(i - 1)) for i in range(86, 92)] + [("85", "85.1")], "TRDC E strand shifted +1"),
    ("C", "C-DELTA", "C-KAPPA", [("118", "119")], "TRDC G strand 118 = 119"),
    ("C", "C-GAMMA-1", "CH3", [("97", "95"), ("98", "96")], "TRGC F strand 97/98 = 95/96"),
    ("G", "G-ALPHA2", "G-ALPHA1", [("72A", "72")], "alpha2 helix kink: 72A = alpha1 72"),
    ("G", "G-ALPHA", "G-ALPHA2", [("51", "50"), ("59", "54"), ("60", "55"), ("62", "59")], "MHC-II alpha1 helix N-term vs MHC-I alpha2"),
    ("G", "G-BETA", "G-ALPHA2", [("84", "85"), ("87", "90")], "MHC-II beta1 helix C-term vs MHC-I alpha2"),
    ("G", "G-ALPHA1-LIKE", "G-ALPHA2", [(str(i), str(i - 1)) for i in (43, 44, 46, 48, 49, 50)], "alpha1-like 43-50 shift"),
]

if __name__ == "__main__":
    doms = load()
    # treat 12e8/1yjd CH1 separately to show the mouse IGHG1*03 reference error
    for d in doms:
        if d["cls"] == "CH1" and d["gene"] == "IGHG1*03" and d["species"] == "Mus musculus":
            d["cls"] = "CH1:mouse-IGHG1*03"
    for t, X, Y, mp, desc in CLAIMS + [("C", "CH1:mouse-IGHG1*03", "CH1", [(str(i), str(i + 4)) for i in range(1, 6)] + [("23", "27"), ("41", "45")], "mouse IGHG1*03 CH1 offset +4")]:
        r = check(t, X, Y, mp, doms)
        if r:
            n, mi, mp_, frac = r
            print(f"{desc:48s} n={n:5d}  median CA dist IMGT-pair={mi:5.2f} A  proposed-pair={mp_:5.2f} A  proposed closer in {frac*100:4.0f}%")
