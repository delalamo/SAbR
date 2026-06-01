# Quickstart

Renumber one chain from the command line:

```bash
sabr -i input.pdb -c H -o output.cif --chain-type H
```

Use mmCIF output when long CDR insertions may require multi-character insertion
codes. By default, `--chain-type auto` tries the labelled `H`, `K`, and `L`
reference embeddings and passes the best-scoring label to ANARCI. Passing
`--chain-type H`, `--chain-type K`, or `--chain-type L` restricts alignment to
that labelled embedding and passes the same label to ANARCI.

Use SAbR from Python:

```python
from pathlib import Path

from sabr import RenumberOptions, renumber_file

result = renumber_file(
    input_path=Path("input.pdb"),
    chain_id="H",
    output_path=Path("output.cif"),
    options=RenumberOptions.from_values(chain_type="H", overwrite=True),
)

print(result.detected_chain_type.value)
```

The output file preserves non-target chains. If a residue range is supplied,
residues outside the range are preserved unchanged.
