"""Download structures, embed every chain with SAbR's encoder, attach structural-scheme labels."""
import gzip, json, os, sys, time, warnings
from multiprocessing import Pool
import numpy as np
import requests

sys.path.insert(0, "/home/user/SAbR/src")
from Bio.PDB import MMCIFParser
from sabr.structure import extract_chain
from sabr.model import encode

W = "/tmp/sabr_unified"
RENUM = "/home/user/SAbR/imgt_audit/results_large/renumbered.json.gz"
warnings.filterwarnings("ignore")


def fetch(pdb):
    p = f"{W}/cif/{pdb}.cif"
    if not os.path.exists(p):
        for attempt in range(4):
            try:
                r = requests.get(f"https://files.rcsb.org/download/{pdb}.cif", timeout=60)
                if r.status_code == 200:
                    tmp = f"{p}.{os.getpid()}.tmp"
                    open(tmp, "w").write(r.text); os.replace(tmp, p); break
            except Exception:
                time.sleep(2 ** attempt)
    return p if os.path.exists(p) else None


def chain_job(args):
    pdb, chain = args
    out = f"{W}/emb/{pdb}_{chain}.npz"
    if os.path.exists(out):
        return pdb, chain, "cached"
    path = fetch(pdb)
    if path is None:
        return pdb, chain, "nofile"
    try:
        st = MMCIFParser(QUIET=True).get_structure(pdb, path)
        for m in list(st)[1:]:
            st.detach_child(m.id)
        data = extract_chain(st, chain, None)
    except Exception as e:
        return pdb, chain, f"extract-fail {e}"
    try:
        emb = encode(data.coords)
    except Exception as e:
        return pdb, chain, f"encode-fail {e}"
    ids = np.array([f"{n}{i}" for n, i in data.residue_ids])
    np.savez_compressed(out, emb=emb.astype(np.float32), coords=data.coords.astype(np.float32), ids=ids,
                        seq=np.array(list(data.sequence)), gaps=np.array(sorted(data.gap_indices), int))
    return pdb, chain, f"ok {len(ids)}"


if __name__ == "__main__":
    recs = [r for r in json.load(gzip.open(RENUM, "rt")) if r["domtype"] in "VCF"]
    chains = sorted({(r["pdb"], r["chain"]) for r in recs})
    with Pool(4) as pool:
        for i, (pdb, ch, msg) in enumerate(pool.imap_unordered(chain_job, chains)):
            if not msg.startswith(("ok", "cached")) or i % 100 == 0:
                print(i, pdb, ch, msg, flush=True)
    print("done", len(chains))
