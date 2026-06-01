# Output Formats

SAbR writes PDB (`.pdb`) and mmCIF (`.cif`) files.

Use PDB output when assigned insertion codes are single characters. PDB cannot
represent multi-character insertion codes.

Use mmCIF output for long CDR loops or any structure where ANARCI assigns
multi-character insertion codes. If PDB output is requested for such a result,
SAbR raises an error explaining that `.cif` output is required.

Non-target chains are preserved. When `--residue-range START END` is supplied,
only standard residues in the selected chain whose original numeric residue
number is inside the inclusive range are renumbered; all other residues are
copied unchanged.
