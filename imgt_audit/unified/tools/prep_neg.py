"""Negative (non-Ig) chains from clearly non-Ig Pfam families, embedded with SAbR's encoder."""
import json, os, sys, time, warnings
from multiprocessing import Pool
import numpy as np, requests
sys.path.insert(0, "/home/user/SAbR/src")
from Bio.PDB import MMCIFParser
from sabr.structure import extract_chain
from sabr.model import encode
import gemmi
warnings.filterwarnings("ignore")
W = "/tmp/sabr_unified"
FAMS = {"globin": "PF00042", "TIM": "PF00121", "kinase": "PF00069", "lysozyme": "PF00062", "GFP": "PF01353",
        "trypsin": "PF00089", "SDR": "PF00106", "ferritin": "PF00210", "thioredoxin": "PF00085",
        "legume_lectin": "PF00139", "cupin": "PF00190", "WD40": "PF00400", "TNF": "PF00229", "PDZ": "PF00595",
        "SH3": "PF00018", "SH2": "PF00017", "ubiquitin": "PF00240", "ankyrin": "PF12796", "TPR": "PF00515",
        "MHC_groove": "PF00129", "lipocalin": "PF00061", "galectin": "PF00337", "RRM": "PF00076",
        "C-type_lectin": "PF00059", "streptavidin": "PF01382", "beta-lactamase": "PF00144"}


def reps(pfam, n=8):
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_polymer_entity_annotation.annotation_id", "operator": "exact_match", "value": pfam}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 2.5}}]},
        "return_type": "polymer_entity",
        "request_options": {"group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": 50,
                                         "ranking_criteria_type": {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}},
                            "group_by_return_type": "representatives", "paginate": {"start": 0, "rows": n}}}
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=120)
    return [x["identifier"] for x in r.json().get("result_set", [])] if r.status_code == 200 else []


def job(args):
    fam, ent = args
    pdb, eid = ent.split("_"); pdb = pdb.lower()
    out = f"{W}/neg/{fam}_{pdb}_{eid}.npz"
    if os.path.exists(out):
        return out
    p = f"{W}/neg/{pdb}.cif"
    if not os.path.exists(p):
        r = requests.get(f"https://files.rcsb.org/download/{pdb}.cif", timeout=60)
        if r.status_code != 200:
            return None
        tmp = f"{p}.{os.getpid()}"; open(tmp, "w").write(r.text); os.replace(tmp, p)
    try:
        st = gemmi.read_structure(p); st.setup_entities()
        ent_obj = next(e for e in st.entities if e.name == eid)
        chain = next(ch.name for ch in st[0] if any(res.subchain in ent_obj.subchains for res in ch))
        bs = MMCIFParser(QUIET=True).get_structure(pdb, p)
        for m in list(bs)[1:]:
            bs.detach_child(m.id)
        data = extract_chain(bs, chain, None)
        emb = encode(data.coords)
    except Exception as e:
        return None
    np.savez_compressed(out, emb=emb.astype(np.float32))
    return out


if __name__ == "__main__":
    items = [(f, e) for f, pf in FAMS.items() for e in reps(pf)]
    print("entities", len(items), flush=True)
    with Pool(4) as pool:
        res = pool.map(job, items, chunksize=2)
    print("embedded", sum(r is not None for r in res))
