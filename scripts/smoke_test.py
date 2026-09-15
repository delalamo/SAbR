"""Exercise an installed release's CLI and check its full IMGT numbering.

Run with the artifact's Python: python -I scripts/smoke_test.py tests/data.
No test dependencies or source-tree imports are needed in the runtime image.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

import sabr


def main() -> None:
    package_path = Path(sabr.__file__).resolve()
    if not package_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected an installed package, got {package_path}")
    data = Path(sys.argv[1]).resolve()
    executable = Path(sys.executable).with_name("sabr")
    subprocess.run([str(executable), "--help"], check=True)
    subprocess.run([str(executable), "--version"], check=True)
    expected = json.loads((data / "numbering_baseline.json").read_text())
    expected = [
        (number, code.strip(), amino_acid)
        for number, code, amino_acid in expected["H"]["schemes"]["imgt"][
            "numbered"
        ]
    ]
    parser = PDBParser(QUIET=True)
    original = parser.get_structure("original", data / "test_heavy_chain.pdb")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "numbered.pdb"
        subprocess.run(
            [
                str(executable),
                "-i",
                str(data / "test_heavy_chain.pdb"),
                "-c",
                "F",
                "-o",
                str(output),
            ],
            cwd=directory,
            check=True,
        )
        numbered = parser.get_structure("numbered", output)
        actual = [
            (residue.id[1], residue.id[2].strip(), seq1(residue.resname))
            for residue in numbered[0]["F"]
            if not residue.id[0].strip()
        ]
        if actual != expected:
            raise RuntimeError("CLI output does not match the IMGT baseline")
        np.testing.assert_array_equal(
            [atom.coord for atom in numbered.get_atoms()],
            [atom.coord for atom in original.get_atoms()],
        )
    print(f"Validated {len(actual)} residue IDs and unchanged coordinates.")


if __name__ == "__main__":
    main()
