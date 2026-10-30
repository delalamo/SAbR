"""Thread DGA-derived IMGT numbering onto PDB structures; writes domains.json."""
import json, os, sys
import gemmi, requests
sys.path.insert(0, os.path.dirname(__file__))
import dga
from entries import ENTRIES

ROOT = os.path.join(os.path.dirname(__file__), "..")
CIF = os.path.join(ROOT, "cif"); os.makedirs(CIF, exist_ok=True)
MIN_SW, MIN_OVERLAP = 120, 50


def fetch(pdb):
    p = os.path.join(CIF, f"{pdb}.cif")
    if not os.path.exists(p):
        r = requests.get(f"https://files.rcsb.org/download/{pdb}.cif", timeout=60)
        if r.status_code != 200:
            return None
        open(p, "w").write(r.text)
    return p


def one(code):
    t = gemmi.find_tabulated_residue(code)
    if t is None:
        return "X"
    c = t.one_letter_code.upper()
    return c if c.isalpha() and c != " " else "X"


def main():
    out = []
    for pdb, types in ENTRIES.items():
        path = fetch(pdb)
        if path is None:
            print("MISSING", pdb); continue
        doc = gemmi.cif.read(path)
        st = gemmi.make_structure_from_block(doc[0])
        st.setup_entities()
        res_txt = st.resolution
        model = st[0]
        for ent in st.entities:
            if ent.entity_type != gemmi.EntityType.Polymer or ent.polymer_type != gemmi.PolymerType.PeptideL:
                continue
            seq = "".join(one(gemmi.Entity.first_mon(m)) for m in ent.full_sequence)
            if len(seq) < 60:
                continue
            sub = ent.subchains[0]
            poly = model.get_subchain(sub)
            chain_name = poly[0].subchain and next(ch.name for ch in model if any(r.subchain == sub for r in ch))
            coords = {}
            for r in poly:
                if r.label_seq is None:
                    continue
                atoms = {}
                for a in r:
                    if a.name in ("N", "CA", "C", "O", "CB") and a.name not in atoms:
                        atoms[a.name] = [round(a.pos.x, 3), round(a.pos.y, 3), round(a.pos.z, 3)]
                if "CA" in atoms:
                    coords[r.label_seq - 1] = dict(atoms=atoms, auth=f"{r.seqid.num}{r.seqid.icode.strip()}", aa=one(r.name))
            for t in types:
                try:
                    doms = dga.parse(dga.query(f"{pdb}_{chain_name}", seq, t))
                except Exception as e:
                    print("DGA-FAIL", pdb, chain_name, t, e); continue
                for d in doms:
                    if "error" in d:
                        if d["sw"] >= MIN_SW and d["overlap"] >= MIN_OVERLAP:
                            print("DGA-INCONSISTENT", pdb, chain_name, t, d["label"], d["gene"], d["sw"], d["error"])
                        continue
                    if d["sw"] < MIN_SW or d["overlap"] < MIN_OVERLAP:
                        continue
                    ungapped = d["gapped"].replace(".", "")
                    start = seq.find(ungapped)
                    if start < 0:
                        print("NOTFOUND", pdb, chain_name, d["label"]); continue
                    residues, k = [], start
                    for lab, aa in zip(d["labels"], d["gapped"]):
                        if aa == ".":
                            continue
                        c = coords.get(k)
                        if c is not None:
                            residues.append(dict(imgt=lab, aa=aa, auth=c["auth"], atoms=c["atoms"]))
                        k += 1
                    rec = dict(pdb=pdb, chain=chain_name, domtype=t, label=d["label"], gene=d["gene"],
                               species=d["species"], sw=d["sw"], ident=d["ident"], resolution=res_txt,
                               n_seq=len(ungapped), fasta_disagrees=d["fasta_disagrees"], residues=residues)
                    out.append(rec)
                    print(f"{pdb} {chain_name} {t} {d['label']:12s} {d['gene']:18s} sw={d['sw']} id={d['ident']} obs={len(residues)}/{len(ungapped)}{' FASTA-DISAGREES' if d['fasta_disagrees'] else ''}")
    json.dump(out, open(os.path.join(ROOT, "domains.json"), "w"))
    print("domains:", len(out))


if __name__ == "__main__":
    main()
