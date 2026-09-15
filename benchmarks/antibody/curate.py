"""Rebuild the curated fixtures from independently downloaded source files.

Usage: python -m benchmarks.antibody.curate SOURCE_DIR --retrieved YYYY-MM-DD
No SAbR encoder, alignment or numbering code is used to create the labels.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_dir", type=Path)
parser.add_argument("--retrieved", required=True)
args = parser.parse_args()
SOURCE = args.source_dir
OUT = Path(__file__).parent
SELECTIONS = [
    ("12e8", "H", "conventional"),
    ("12e8", "L", "conventional"),
    ("1mel", "A", "vhh_long_cdr3"),
    ("1lmk", "A", "diabody_vh_vl"),
    ("1moe", "A", "diabody_vl_vh"),
    ("6nou", "A", "scfv_vh_vl"),
    ("5kve", "L", "scfv_vl_vh"),
    ("1ubq", "A", "negative"),
    ("1lyz", "A", "negative"),
]


class Backbone(Select):
    def accept_residue(self, r):
        return r.id[0] == " "

    def accept_atom(self, a):
        return a.name in ("N", "CA", "C")


manifest = {
    "schema_version": 1,
    "scheme": "imgt",
    "retrieved": args.retrieved,
    "annotation_source": (
        "SAbDab 2.1.1, independent IMGT annotations (CC BY 4.0)"
    ),
    "cases": [],
}
for pdb, chain, category in SELECTIONS:
    name = f"{pdb}_{chain}"
    path = SOURCE / f"{pdb}_raw.pdb"
    s = PDBParser(QUIET=True).get_structure(name, path)
    for c in list(s[0]):
        if c.id != chain:
            s[0].detach_child(c.id)
    io = PDBIO()
    io.set_structure(s)
    fixture = OUT / "fixtures" / f"{name}.pdb"
    io.save(str(fixture), Backbone())
    fixture.write_text(
        "\n".join(line.rstrip() for line in fixture.read_text().splitlines())
        + "\n"
    )
    ss = PDBParser(QUIET=True).get_structure(
        name, OUT / "fixtures" / f"{name}.pdb"
    )
    observed = "".join(seq1(r.resname) for r in ss[0][chain])
    case = {
        "id": name,
        "category": category,
        "fixture": f"fixtures/{name}.pdb",
        "chain": chain,
        "pdb_id": pdb.upper(),
        "source_url": f"https://files.rcsb.org/download/{pdb.upper()}.pdb",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(
            (OUT / "fixtures" / f"{name}.pdb").read_bytes()
        ).hexdigest(),
        "options": {},
        "positive": category != "negative",
    }
    labels = []
    if case["positive"]:
        annpath = SOURCE / f"{pdb}_annotations.json"
        a = json.loads(annpath.read_text())
        polymers = {p["sabdab_auth_asym_id"]: p for p in a["polymer_instances"]}
        segs = {}
        for ab in a["antibody_instances"]:
            for key in ("heavy_like_segment", "light_like_segment"):
                seg = ab.get(key)
                if (
                    seg
                    and polymers[seg["sabdab_auth_asym_id"]]["pdb_auth_asym_id"]
                    == chain
                ):
                    segs[seg["id"]] = seg
        segs = sorted(segs.values(), key=lambda x: x["query_start"])
        full = polymers[segs[0]["sabdab_auth_asym_id"]]["sequence"]
        cifpath = SOURCE / f"{pdb}_raw.cif"
        cif = MMCIF2Dict(str(cifpath))
        keys = [
            "auth_asym_id",
            "auth_seq_id",
            "pdbx_PDB_ins_code",
            "label_seq_id",
            "pdbx_PDB_model_num",
        ]
        seqids = {}
        for auth_chain, num, ins, label, model in zip(
            *(cif["_atom_site." + key] for key in keys)
        ):
            if auth_chain == chain and model == "1" and label not in (".", "?"):
                residue_id = (int(num), "" if ins in (".", "?") else ins)
                index = int(label) - 1
                assert residue_id not in seqids or seqids[residue_id] == index
                seqids[residue_id] = index
        # Use deposited sequence indices, not sequence alignment: e.g. a
        # missing terminal S in SS must not move the remaining S by one.
        mapping = {
            seqids[(r.id[1], r.id[2].strip())]: row
            for row, r in enumerate(ss[0][chain])
        }
        assert len(mapping) == len(observed)
        assert all(full[i] == observed[j] for i, j in mapping.items()), name
        case["residue_mapping_url"] = (
            f"https://files.rcsb.org/download/{pdb.upper()}.cif"
        )
        case["residue_mapping_sha256"] = hashlib.sha256(
            cifpath.read_bytes()
        ).hexdigest()
        for domain, seg in enumerate(segs):
            numbered = [(pos, aa) for pos, aa in seg["numbering"] if aa != "-"]
            assert "".join(aa for _, aa in numbered) == seg["sequence"]
            assert (
                full[seg["query_start"] : seg["query_end"] + 1]
                == seg["sequence"]
            )
            for offset, ((num, ins), aa) in enumerate(numbered):
                row = mapping.get(seg["query_start"] + offset)
                if row is not None:
                    labels.append(
                        [
                            row,
                            domain,
                            seg["type"],
                            num,
                            ins.strip(),
                            aa,
                            ss[0][chain].child_list[row].id[1],
                            ss[0][chain].child_list[row].id[2].strip(),
                            seg["query_start"] + offset + 1,
                        ]
                    )
        labels.sort()
        case.update(
            {
                "labels": f"fixtures/{name}.csv",
                "expected_types": "".join(seg["type"] for seg in segs),
                "annotation_url": (
                    "https://sabdab.opig.stats.ox.ac.uk/api/frontend/pdb/"
                    f"pdb_0000{pdb}"
                ),
                "annotation_sha256": hashlib.sha256(
                    annpath.read_bytes()
                ).hexdigest(),
                "annotation_updated_at": a["updated_at"],
                "segments": [
                    {
                        "id": seg["id"],
                        "chain": seg["sabdab_auth_asym_id"],
                        "query_start": seg["query_start"],
                        "query_end": seg["query_end"],
                        "sequence": seg["sequence"],
                    }
                    for seg in segs
                ],
            }
        )
        if len(segs) > 1:
            case["options"]["scfv"] = True
        if pdb in ("6nou", "5kve"):
            case["options"]["dangerously_allow_structural_gaps"] = True
        with (OUT / case["labels"]).open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(
                [
                    "row",
                    "domain",
                    "chain_type",
                    "imgt_position",
                    "insertion_code",
                    "amino_acid",
                    "source_residue_number",
                    "source_insertion_code",
                    "label_seq_id",
                ]
            )
            writer.writerows(labels)
        case["labels_sha256"] = hashlib.sha256(
            (OUT / case["labels"]).read_bytes()
        ).hexdigest()
    manifest["cases"].append(case)
    print(name, len(observed), len(labels), case.get("expected_types"))
# Controlled terminal deletions of a deposited VH, retaining inherited labels.
for name, start, stop in [
    ("12e8_H_n_truncated", 10, 120),
    ("12e8_H_c_truncated", 0, 110),
]:
    base = copy.deepcopy(manifest["cases"][0])
    base.update(
        id=name,
        category="truncated",
        row_slice=[start, stop],
        derived_from="12e8_H",
    )
    manifest["cases"].append(base)
# Same gapped scFvs under the safe default policy, separately from accuracy.
for original in list(manifest["cases"]):
    if original["pdb_id"] not in ("6NOU", "5KVE"):
        continue
    case = copy.deepcopy(original)
    case.update(
        id=original["id"] + "_default_gaps",
        category="gap_policy",
        expected_rejection="structural gap",
    )
    case["options"].pop("dangerously_allow_structural_gaps")
    manifest["cases"].append(case)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
