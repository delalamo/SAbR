"""Biopython adapters for anisotropic tensors and unresolved residues."""

import numpy as np
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser

# Use PDB/Biopython's ANISOU order, not mmCIF's upper-triangle row order.
_COMPONENTS = ("U[1][1]", "U[2][2]", "U[3][3]", "U[1][2]", "U[1][3]", "U[2][3]")
_PREFIX = "_atom_site_anisotrop."


def _tensor(values):
    tensor = np.asarray(values, dtype=np.float64)
    if tensor.shape != (6,) or not np.isfinite(tensor).all():
        raise ValueError(
            "Anisotropic displacement tensors need six finite values."
        )
    return tensor


def _atom_id(value):
    # MMCIFParser converts numerical serials to ints, retaining other strings.
    try:
        return int(value)
    except ValueError:
        return value


class AnisouPDBIO(PDBIO):
    """Write each tensor immediately after its corresponding ATOM/HETATM."""

    def _get_atom_line(self, atom, *args, **kwargs):
        line = super()._get_atom_line(atom, *args, **kwargs)
        if atom.get_anisou() is None:
            return line
        values = np.rint(_tensor(atom.get_anisou()) * 10000)
        if np.any(values < -999999) or np.any(values > 9999999):
            raise ValueError(
                "PDB cannot represent this ANISOU tensor; use mmCIF output."
            )
        fields = "".join(f"{int(value):7d}" for value in values)
        return line + "ANISOU" + line[6:28] + fields + line[70:]


class AnisouMMCIFIO(MMCIFIO):
    """Add sparse anisotropic rows using the atom IDs assigned by MMCIFIO."""

    def _save_structure(self, out_file, select, preserve_atom_numbering):
        self._selected_atoms = [
            atom
            for model in self.structure
            if select.accept_model(model)
            for chain in model
            if select.accept_chain(chain)
            for residue in chain.get_unpacked_list()
            if select.accept_residue(residue)
            for atom in residue.get_unpacked_list()
            if select.accept_atom(atom)
        ]
        super()._save_structure(out_file, select, preserve_atom_numbering)

    def _save_dict(self, out_file):
        rows = [
            (index, _tensor(atom.get_anisou()))
            for index, atom in enumerate(self._selected_atoms)
            if atom.get_anisou() is not None
        ]
        if rows:
            self.dic[_PREFIX + "id"] = [
                self.dic["_atom_site.id"][index] for index, _ in rows
            ]
            self.dic[_PREFIX + "type_symbol"] = [
                self.dic["_atom_site.type_symbol"][index] for index, _ in rows
            ]
            for component, name in enumerate(_COMPONENTS):
                self.dic[_PREFIX + name] = [
                    str(float(tensor[component])) for _, tensor in rows
                ]
        super()._save_dict(out_file)


class AnisouMMCIFParser(MMCIFParser):
    """Read tensors by atom ID and retain unresolved polymer metadata."""

    def get_structure(self, structure_id, filename):
        structure = super().get_structure(structure_id, filename)
        # Match PDBParser's REMARK 465 representation. Author identifiers are
        # required because they also identify residues in the parsed structure.
        prefix = "_pdbx_unobs_or_zero_occ_residues."
        data = self._mmcif_dict
        names = ("auth_asym_id", "auth_seq_id", "auth_comp_id")
        records = []
        if (
            self.auth_chains
            and self.auth_residues
            and all(prefix + name in data for name in names)
        ):
            count = len(data[prefix + names[0]])

            def value(name, index, default):
                return data.get(prefix + name, [default] * count)[index]

            for index in range(count):
                if value("polymer_flag", index, "Y") != "Y":
                    continue
                # Zero occupancy is distinct from absent coordinates.
                if value("occupancy_flag", index, "1") != "1":
                    continue
                try:
                    number = int(value("auth_seq_id", index, "?"))
                    model = value("PDB_model_num", index, "?")
                    model = None if model in ("?", ".") else int(model)
                except ValueError:
                    continue
                insertion = value("PDB_ins_code", index, "?")
                records.append(
                    {
                        "chain": value("auth_asym_id", index, "?"),
                        "ssseq": number,
                        "res_name": value("auth_comp_id", index, "UNK"),
                        "insertion": (
                            None if insertion in ("?", ".") else insertion
                        ),
                        "model": model,
                    }
                )
        structure.header["missing_residues"] = records
        return structure

    def _build_structure(self, structure_id):
        data = self._mmcif_dict
        if not any(_PREFIX + name in data for name in _COMPONENTS):
            return super()._build_structure(structure_id)
        keys = [_PREFIX + name for name in ("id", *_COMPONENTS)]
        if not all(key in data for key in keys):
            raise ValueError(
                "Incomplete mmCIF anisotropic displacement category."
            )
        tensors = {}
        for identifier, *values in zip(
            *(data[key] for key in keys), strict=True
        ):
            identifier = _atom_id(identifier)
            if identifier in tensors:
                raise ValueError(
                    f"Duplicate anisotropic atom ID: {identifier}."
                )
            tensors[identifier] = _tensor(values)

        # The upstream parser assigns tensors by row index and uses a different
        # component order. Let it parse atoms, then attach the tensors by ID.
        self._mmcif_dict = {
            key: value
            for key, value in data.items()
            if not key.startswith(_PREFIX)
        }
        try:
            super()._build_structure(structure_id)
        finally:
            self._mmcif_dict = data
        atoms = {}
        for model in self._structure_builder.get_structure():
            for chain in model:
                for residue in chain.get_unpacked_list():
                    for atom in residue.get_unpacked_list():
                        identifier = atom.serial_number
                        if identifier in atoms:
                            raise ValueError(
                                f"Ambiguous mmCIF atom ID: {identifier}."
                            )
                        atoms[identifier] = atom
        for identifier, tensor in tensors.items():
            if identifier not in atoms:
                raise ValueError(f"Unknown anisotropic atom ID: {identifier}.")
            atoms[identifier].set_anisou(tensor)
