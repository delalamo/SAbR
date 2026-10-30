"""Large-scale threading of DGA numbering onto structures (parallel, resumable)."""
import json, os, sys, traceback
from multiprocessing import Pool
import gemmi
sys.path.insert(0, os.path.dirname(__file__))
import dga
from thread import fetch, one, MIN_SW, MIN_OVERLAP

BIG = os.path.join(os.path.dirname(__file__), "..", "big")
OUT = os.path.join(BIG, "dom"); os.makedirs(OUT, exist_ok=True)


def process(job):
    pdb = job["pdb"]
    dst = os.path.join(OUT, f"{pdb}.json")
    if os.path.exists(dst):
        return pdb, "cached"
    log, out = [], []
    try:
        path = fetch(pdb)
        if path is None:
            json.dump(dict(domains=[], log=["MISSING"]), open(dst, "w")); return pdb, "missing"
        st = gemmi.make_structure_from_block(gemmi.cif.read(path)[0])
        st.setup_entities()
        model = st[0]
        for ent in st.entities:
            if ent.entity_type != gemmi.EntityType.Polymer or ent.polymer_type != gemmi.PolymerType.PeptideL:
                continue
            chains = [ch.name for ch in model if any(r.subchain in ent.subchains for r in ch)]
            if not chains:
                continue
            want = (job["entities"] and ent.name in job["entities"]) or (job["chains"] and set(chains) & set(job["chains"]))
            if not want:
                continue
            chain_name = sorted(set(chains) & set(job["chains"]))[0] if job["chains"] and set(chains) & set(job["chains"]) else chains[0]
            sub = next(r.subchain for r in model[chain_name] if r.subchain in ent.subchains)
            seq = "".join(one(gemmi.Entity.first_mon(m)) for m in ent.full_sequence)
            if len(seq) < 60:
                continue
            coords = {}
            for r in model.get_subchain(sub):
                if r.label_seq is None:
                    continue
                atoms = {}
                for a in r:
                    if a.name in ("N", "CA", "C", "O", "CB") and a.name not in atoms:
                        atoms[a.name] = [round(a.pos.x, 3), round(a.pos.y, 3), round(a.pos.z, 3)]
                if "CA" in atoms:
                    coords[r.label_seq - 1] = dict(atoms=atoms, auth=f"{r.seqid.num}{r.seqid.icode.strip()}")
            for t in job["types"]:
                doms = dga.parse(dga.query(f"{pdb}_{chain_name}", seq, t))
                for d in doms:
                    if d["sw"] < MIN_SW or d["overlap"] < MIN_OVERLAP:
                        continue
                    if "error" in d:
                        log.append(f"DGA-INCONSISTENT {pdb} {chain_name} {t} {d['label']} {d['gene']} {d['error']}"); continue
                    ungapped = d["gapped"].replace(".", "")
                    start = seq.find(ungapped)
                    if start < 0:
                        log.append(f"NOTFOUND {pdb} {chain_name} {d['label']}"); continue
                    residues, k = [], start
                    for lab, aa in zip(d["labels"], d["gapped"]):
                        if aa == ".":
                            continue
                        c = coords.get(k)
                        if c is not None:
                            residues.append(dict(imgt=lab, aa=aa, auth=c["auth"], atoms=c["atoms"]))
                        k += 1
                    out.append(dict(pdb=pdb, chain=chain_name, entity=ent.name, domtype=t, label=d["label"], gene=d["gene"],
                                    species=d["species"], sw=d["sw"], ident=d["ident"], resolution=st.resolution,
                                    n_seq=len(ungapped), start=start, fasta_disagrees=d["fasta_disagrees"],
                                    src=job["src"], residues=residues))
    except Exception as e:
        log.append("ERROR " + repr(e) + " " + traceback.format_exc().splitlines()[-2])
    json.dump(dict(domains=out, log=log), open(dst, "w"))
    return pdb, f"{len(out)} domains {len(log)} log"


if __name__ == "__main__":
    jobs = json.load(open(os.path.join(BIG, "jobs.json")))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for i, (pdb, msg) in enumerate(pool.imap_unordered(process, jobs)):
            print(i, pdb, msg, flush=True)
