# Python API

The public API is intentionally small:

- `RenumberOptions`
- `RenumberResult`
- `renumber_file`
- `renumber_structure`

```python
from Bio.PDB import PDBParser

from sabr import RenumberOptions, renumber_structure

parser = PDBParser(QUIET=True)
structure = parser.get_structure("antibody", "input.pdb")

renumbered = renumber_structure(
    structure,
    chain_id="H",
    options=RenumberOptions.from_values(
        numbering_scheme="imgt",
        chain_type="auto",
        reference_chain_type="auto",
    ),
)
```

Library code raises `SAbRError` subclasses for user-facing failures such as
missing chains, invalid output formats, alignment failures, and numbering
failures. The CLI converts those errors into Click messages.
