import sys; sys.path[:0] = ["/tmp/pan/bench", "/tmp/pan", "/tmp/sabr_unified/tools", "/home/user/SAbR/src"]
import sys, time, json
sys.path.insert(0, "/tmp/pan/bench")
import numpy as np, pan, ulib, methods as M
doms, cache, _ = ulib.load_dataset(); byid = {d["id"]: d for d in doms}
M.load_templates([byid[t] for t in M.S["templates"]], cache)
test = [byid[i] for i in M.S["test"]][::6]
from sabr.model import encode
T = {}
def tick(k, f):
    t = time.perf_counter(); r = f(); T.setdefault(k, []).append(time.perf_counter() - t); return r
for d in test:
    ca, cbu = pan.geometry(d, cache); c = cache[(d["pdb"], d["chain"])]; a, b = d["rows"]; seq = "".join(c["seq"][a:b])
    X = ulib.dom_coords(d, cache)
    E = tick("MPNN encode (shared by M3/M5/M6)", lambda: encode(X))
    tick("M1 TM-align best template", lambda: M.tm_transfer(ca, seq, 1))
    seed = tick("M3 SAbR embedding (pan-trained)", lambda: M.emb_assign(E)[0])
    tick("M4 consensus-frame match (TM init)", lambda: M.frame_match(ca, cbu, "tm"))
    tick("M5a embedding seed + frame match (match only)", lambda: M.frame_match(ca, cbu, "seed", seed))
    tick("M5b fused DP (match only)", lambda: M.fused_match(ca, cbu, E, seed))
    tick("M7 adaptive consensus-frame matcher", lambda: M.adaptive_match(ca, cbu))
    tick("M2 TM-align top-5 vote", lambda: M.tm_transfer(ca, seq, 5))
    tick("single TM-align (reference)", lambda: __import__("tmtools").tm_align(ca, M.FRAME.xyz, "A" * len(ca), "A" * len(M.FRAME.xyz)))
out = {k: 1000 * float(np.median(v)) for k, v in T.items()}
for k, v in out.items(): print(f"{k:48s} {v:8.1f} ms (median, n={len(test)})")
json.dump(out, open("/tmp/pan/bench/speed.json", "w"), indent=1)
