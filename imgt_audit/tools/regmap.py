"""Register map: modal structural partner of each label of class X in class Y."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from analyze import load, lab_key
from classpair import classpair


def regmap(tag, pairs, positions, doms=None):
    doms = doms or load()
    stat, _ = classpair(tag, doms)
    hdr = "pos   " + " ".join(f"{(X[-5:]+">"+Y[-5:]):>15}" for X, Y in pairs)
    print(hdr)
    for p in positions:
        cells = []
        for X, Y in pairs:
            c = stat[(X, Y)].get(p)
            if not c:
                cells.append(f"{'':>15}"); continue
            n = sum(c.values())
            q, k = c.most_common(1)[0]
            mark = "" if q == p else ("*" if q != "-" else "")
            cells.append(f"{q+mark:>8} {k/n:4.2f}{'':2}")
        print(f"{p:>5} " + " ".join(cells))


if __name__ == "__main__":
    tag = sys.argv[1]
    pairs = [tuple(x.split(">")) for x in sys.argv[2].split(",")]
    lo, hi = sys.argv[3].split("-")
    positions = [str(i) for i in range(int(lo), int(hi) + 1)]
    if len(sys.argv) > 4:
        positions = sys.argv[4].split(",")
    regmap(tag, pairs, positions)
