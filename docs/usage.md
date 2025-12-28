# Usage Guide

This guide covers how to use SAbR for antibody structure renumbering.

## Basic Usage

The simplest way to renumber an antibody structure:

```bash
sabr -i input.pdb -c A -o output.pdb
```

This will:

1. Read the structure from `input.pdb`
2. Extract chain A
3. Generate MPNN embeddings for the chain
4. Align embeddings against the unified reference
5. Apply IMGT numbering (default)
6. Write the renumbered structure to `output.pdb`

## Command-Line Options

### Required Options

| Option                   | Description                                     |
| ------------------------ | ----------------------------------------------- |
| `-i, --input-pdb FILE`   | Input structure file (PDB or mmCIF format)      |
| `-c, --input-chain TEXT` | Chain identifier to renumber (single character) |
| `-o, --output FILE`      | Output structure file (.pdb or .cif)            |

### Numbering Scheme

```bash
-n, --numbering-scheme [imgt|chothia|kabat|martin|aho|wolfguy]
```

Available schemes:

- **IMGT**: International ImMunoGeneTics (default)
- **Chothia**: Structure-based CDR definitions
- **Kabat**: Sequence variability-based
- **Martin**: Enhanced Chothia
- **Aho**: Cross-species comparisons
- **Wolfguy**: Specialized applications

### Chain Type

```bash
-t, --chain-type [H|K|L|heavy|kappa|lambda|auto]
```

Specify the chain type for ANARCI numbering:

- `H` or `heavy`: Heavy chain
- `K` or `kappa`: Kappa light chain
- `L` or `lambda`: Lambda light chain
- `auto`: Automatically detect (default)

!!! tip
Specify the chain type manually when known, as heavy and light chains have similar structures that can be confused.

### Advanced Options

| Option                                | Description                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `--extended-insertions`               | Enable extended insertion codes (AA, AB, ..., ZZ) for very long CDR loops. Requires mmCIF output. |
| `--disable-deterministic-renumbering` | Use raw alignment output without loop corrections                                                 |
| `--max-residues INTEGER`              | Maximum residues to process (0 = all)                                                             |
| `--overwrite`                         | Overwrite output file if it exists                                                                |
| `-v, --verbose`                       | Enable verbose logging                                                                            |

## Examples

### Renumber with Chothia scheme

```bash
sabr -i antibody.pdb -c H -o antibody_chothia.pdb -n chothia
```

### Renumber with explicit chain type

```bash
sabr -i fab.pdb -c L -o fab_imgt.pdb -t kappa
```

### Handle long CDR3 loops with mmCIF

```bash
sabr -i nanobody.cif -c A -o nanobody_imgt.cif --extended-insertions
```

### Verbose output for debugging

```bash
sabr -i input.pdb -c A -o output.pdb -v
```

## Practical Considerations

### Truncate to Fv Region

It is recommended to truncate the query structure to contain only the Fv (variable fragment) region before running SAbR. The aligner may sometimes align variable region beta-strands to those in the constant region.

### Single-Chain scFvs

When running scFvs (single-chain variable fragments), run each variable domain independently. SAbR currently struggles with scFvs because:

1. Domain assignment for canonical numbering is ambiguous
2. The aligner may incorrectly align across both domains

See [issue #2](https://github.com/delalamo/SAbR/issues/2) for details.

### Missing Residues

The CDR numbering algorithm uses the same approach as IMGT and does not account for missing residues. If a residue is missing due to disorder or heterogeneity, other residues in the CDR may be misnumbered.

## Python API

SAbR can also be used programmatically:

```python
from sabr import mpnn_embeddings, softaligner, aln2hmm, edit_pdb, util
from ANARCI import anarci

# Load structure and generate embeddings
input_data = mpnn_embeddings.from_pdb("input.pdb", "A")

# Align against reference
aligner = softaligner.SoftAligner()
result = aligner(input_data)

# Convert to state vector
states, start, end, first_row = aln2hmm.alignment_matrix_to_state_vector(
    result.alignment
)

# Apply ANARCI numbering
chain_type = util.detect_chain_type(result.alignment)
anarci_out, _, _ = anarci.number_sequence_from_alignment(
    states,
    sequence,
    scheme="imgt",
    chain_type=chain_type,
)

# Write renumbered structure
edit_pdb.thread_alignment(
    "input.pdb", "A", anarci_out, "output.pdb", 0, len(anarci_out), first_row
)
```

See the [API Reference](api/cli.md) for detailed documentation.
