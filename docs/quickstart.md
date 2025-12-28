# Quick Start

Get up and running with SAbR in 5 minutes.

## Basic Renumbering

The simplest way to renumber an antibody structure:

```bash
sabr -i input.pdb -c A -o output.pdb
```

This reads chain A from `input.pdb`, applies IMGT numbering (default), and writes the result to `output.pdb`.

## Step-by-Step Example

### 1. Prepare Your Structure

Ensure your PDB file contains an antibody variable domain. For best results:

- Truncate to the Fv (variable fragment) region
- Use a single chain per run

### 2. Run SAbR

```bash
# Renumber heavy chain with IMGT scheme
sabr -i antibody.pdb -c H -o antibody_imgt.pdb -n imgt

# Renumber light chain with Chothia scheme
sabr -i antibody.pdb -c L -o antibody_chothia.pdb -n chothia
```

### 3. Verify Output

Open the output file to verify the renumbering. Key positions to check:

- **Cys23**: Conserved cysteine (should be present)
- **Trp41**: Conserved tryptophan
- **Cys104**: Second conserved cysteine
- **CDR positions**: Should follow the scheme's definitions

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `-n` | Numbering scheme | `-n chothia` |
| `-t` | Chain type | `-t heavy` |
| `-v` | Verbose output | `-v` |
| `--overwrite` | Overwrite existing output | `--overwrite` |

## Example Commands

```bash
# Heavy chain with Kabat numbering
sabr -i fab.pdb -c H -o fab_kabat.pdb -n kabat

# Kappa light chain (explicit type)
sabr -i fab.pdb -c L -o fab_imgt.pdb -t kappa

# Verbose output for debugging
sabr -i nanobody.pdb -c A -o nanobody.pdb -v --overwrite

# mmCIF format with extended insertions for long CDRs
sabr -i structure.cif -c A -o output.cif --extended-insertions
```

## Troubleshooting

### "Chain not found"

Verify your chain ID with:

```bash
grep "^ATOM" input.pdb | cut -c22 | sort -u
```

### Misaligned CDR loops

Try specifying the chain type explicitly:

```bash
sabr -i input.pdb -c A -o output.pdb -t heavy
```

### Apple Silicon errors

Set the JAX platform:

```bash
export JAX_PLATFORMS=cpu
sabr -i input.pdb -c A -o output.pdb
```

## Next Steps

- [Usage Guide](usage.md): Detailed options and examples
- [Numbering Schemes](numbering_schemes.md): Learn about IMGT, Chothia, Kabat, etc.
- [API Reference](api/cli.md): Use SAbR programmatically
