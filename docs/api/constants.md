# Constants Module

The constants module defines constants used throughout the SAbR package.

## Type Aliases

```python
AnarciAlignment = List[Tuple[Tuple[int, str], str]]
```

Type alias for ANARCI alignment output. Each element is a tuple of `((residue_number, insertion_code), amino_acid)`.

## Neural Network Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `EMBED_DIM` | 64 | Dimension of MPNN embeddings |
| `N_MPNN_LAYERS` | 3 | Number of layers in the MPNN model |

## IMGT Numbering

| Constant | Value | Description |
|----------|-------|-------------|
| `IMGT_MAX_POSITION` | 128 | Maximum position in IMGT numbering |
| `DEFAULT_TEMPERATURE` | 1e-4 | Default alignment temperature for SoftAlign |

### Framework Regions

```python
IMGT_FRAMEWORKS = {
    "FW1": [1, 2, ..., 26],
    "FW2": [39, 40, ..., 55],
    "FW3": [66, 67, ..., 104],
    "FW4": [118, 119, ..., 128],
}
```

### CDR Loops

```python
IMGT_LOOPS = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}
```

### CDR Anchors

Framework anchor positions used for CDR renumbering:

```python
CDR_ANCHORS = {
    "CDR1": (23, 40),   # Cys23 and position 40
    "CDR2": (54, 66),   # Position 54 and 66
    "CDR3": (104, 118), # Position 104 and 118
}
```

## FR1 Region Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `FR1_ANCHOR_START_COL` | 5 | 0-indexed column for IMGT position 6 |
| `FR1_ANCHOR_END_COL` | 11 | 0-indexed column for IMGT position 12 |
| `FR1_POSITION_10_COL` | 9 | 0-indexed column for IMGT position 10 |
| `FR1_KAPPA_RESIDUE_COUNT` | 7 | Kappa chains have 7 residues in positions 6-12 |

## DE Loop Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DE_LOOP_START_COL` | 78 | 0-indexed column for IMGT position 79 |
| `DE_LOOP_END_COL` | 83 | 0-indexed column for IMGT position 84 |
| `DE_LOOP_HEAVY_THRESHOLD` | 5 | Number of residues indicating heavy chain (>= 5) |

## FR3 Position Constants

| Constant | Value |
|----------|-------|
| `FR3_POS81_COL` | 80 |
| `FR3_POS82_COL` | 81 |
| `FR3_POS83_COL` | 82 |
| `FR3_POS84_COL` | 83 |

## C-Terminus Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `C_TERMINUS_ANCHOR_POSITION` | 124 | 0-indexed position for IMGT position 125 |

## Amino Acid Mapping

```python
AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}
```
