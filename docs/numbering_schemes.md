# Numbering Schemes

SAbR supports multiple antibody numbering schemes through ANARCI.

## Overview

Antibody numbering schemes provide a standardized way to identify residue positions within antibody variable domains. Different schemes have evolved for different purposes.

| Scheme  | CDR Definition   | Best For                         |
| ------- | ---------------- | -------------------------------- |
| IMGT    | Structural       | General use, standardization     |
| Chothia | Structural       | CDR loop structure analysis      |
| Kabat   | Sequence         | Sequence variability analysis    |
| Martin  | Enhanced Chothia | Combining structure and sequence |
| Aho     | Structural       | Cross-species comparisons        |
| Wolfguy | Custom           | Specific applications            |

## IMGT

The **International ImMunoGeneTics** (IMGT) numbering scheme is the current standard for antibody numbering.

**Key features:**

- Fixed positions for conserved residues (Cys23, Cys104, Trp41, etc.)
- CDR insertions placed in the middle of CDR regions
- Consistent numbering across all chain types
- Maximum position: 128

**CDR definitions (IMGT):**

| CDR  | Positions |
| ---- | --------- |
| CDR1 | 27-38     |
| CDR2 | 56-65     |
| CDR3 | 105-117   |

```bash
sabr -i input.pdb -c A -o output.pdb -n imgt
```

## Chothia

The **Chothia** numbering scheme is based on structural analysis of antibody loops.

**Key features:**

- CDR definitions based on structural hypervariability
- Commonly used in structural biology
- Insertions placed at specific positions within loops

**CDR definitions (Chothia):**

| Chain | CDR1    | CDR2    | CDR3     |
| ----- | ------- | ------- | -------- |
| Heavy | H26-H32 | H52-H56 | H95-H102 |
| Light | L24-L34 | L50-L56 | L89-L97  |

```bash
sabr -i input.pdb -c A -o output.pdb -n chothia
```

## Kabat

The **Kabat** numbering scheme is the oldest widely-used scheme, based on sequence variability.

**Key features:**

- CDR definitions based on sequence variability
- Well-established with extensive literature
- Popular in the pharmaceutical industry

**CDR definitions (Kabat):**

| Chain | CDR1     | CDR2    | CDR3     |
| ----- | -------- | ------- | -------- |
| Heavy | H31-H35B | H50-H65 | H95-H102 |
| Light | L24-L34  | L50-L56 | L89-L97  |

```bash
sabr -i input.pdb -c A -o output.pdb -n kabat
```

## Martin

The **Martin** (Enhanced Chothia) numbering scheme combines elements of Chothia and Kabat.

```bash
sabr -i input.pdb -c A -o output.pdb -n martin
```

## Aho

The **Aho** numbering scheme was designed for cross-species antibody comparisons.

```bash
sabr -i input.pdb -c A -o output.pdb -n aho
```

## Wolfguy

The **Wolfguy** numbering scheme is a specialized scheme.

```bash
sabr -i input.pdb -c A -o output.pdb -n wolfguy
```

## Choosing a Scheme

| Use Case                   | Recommended Scheme |
| -------------------------- | ------------------ |
| General purpose            | IMGT               |
| CDR loop conformations     | Chothia            |
| Legacy/pharmaceutical data | Kabat              |
| Hybrid analysis            | Martin             |
| Cross-species comparison   | Aho                |

## Chain Type Detection

SAbR automatically detects the chain type from the alignment:

1. **DE loop occupancy**: Heavy chains have 6 residues in positions 79-84, while light chains have 4 (skipping 81-82).

2. **Position 10**: For light chains, kappa chains have position 10 occupied while lambda chains do not.

Override automatic detection:

```bash
sabr -i input.pdb -c H -o output.pdb -t heavy
sabr -i input.pdb -c L -o output.pdb -t kappa
```
