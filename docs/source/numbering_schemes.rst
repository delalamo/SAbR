Numbering Schemes
=================

SAbR supports multiple antibody numbering schemes through ANARCI. This page
describes each scheme and when to use it.

Overview
--------

Antibody numbering schemes provide a standardized way to identify residue
positions within antibody variable domains. Different schemes have evolved
for different purposes and communities.

+------------+------------------+----------------------------------------+
| Scheme     | CDR Definition   | Best For                               |
+============+==================+========================================+
| IMGT       | Structural       | General use, standardization           |
+------------+------------------+----------------------------------------+
| Chothia    | Structural       | CDR loop structure analysis            |
+------------+------------------+----------------------------------------+
| Kabat      | Sequence         | Sequence variability analysis          |
+------------+------------------+----------------------------------------+
| Martin     | Enhanced Chothia | Combining structure and sequence info  |
+------------+------------------+----------------------------------------+
| Aho        | Structural       | Cross-species comparisons              |
+------------+------------------+----------------------------------------+
| Wolfguy    | Custom           | Specific applications                  |
+------------+------------------+----------------------------------------+

IMGT
----

The **International ImMunoGeneTics** (IMGT) numbering scheme is the current
standard for antibody numbering. It uses a unique numbering system based on
structural conservation.

Key features:

- Fixed positions for conserved residues (Cys23, Cys104, Trp41, etc.)
- CDR insertions placed in the middle of CDR regions
- Consistent numbering across all chain types
- Position 128 maximum

CDR definitions (IMGT):

- **CDR1**: Positions 27-38
- **CDR2**: Positions 56-65
- **CDR3**: Positions 105-117

Usage::

   sabr -i input.pdb -c A -o output.pdb -n imgt

Chothia
-------

The **Chothia** numbering scheme is based on structural analysis of
antibody loops. It was designed to identify canonical CDR loop
conformations.

Key features:

- CDR definitions based on structural hypervariability
- Commonly used in structural biology
- Insertions placed at specific positions within loops

CDR definitions (Chothia):

- **CDR1-H**: H26-H32
- **CDR2-H**: H52-H56
- **CDR3-H**: H95-H102
- **CDR1-L**: L24-L34
- **CDR2-L**: L50-L56
- **CDR3-L**: L89-L97

Usage::

   sabr -i input.pdb -c A -o output.pdb -n chothia

Kabat
-----

The **Kabat** numbering scheme is the oldest widely-used scheme, based on
sequence variability analysis. It remains popular in the pharmaceutical
industry.

Key features:

- CDR definitions based on sequence variability
- Well-established with extensive literature
- May not align well with structural hypervariability

CDR definitions (Kabat):

- **CDR1-H**: H31-H35B
- **CDR2-H**: H50-H65
- **CDR3-H**: H95-H102
- **CDR1-L**: L24-L34
- **CDR2-L**: L50-L56
- **CDR3-L**: L89-L97

Usage::

   sabr -i input.pdb -c A -o output.pdb -n kabat

Martin
------

The **Martin** (Enhanced Chothia) numbering scheme combines elements of
Chothia and Kabat numbering. It refines Chothia's structural definitions
with additional sequence-based considerations.

Usage::

   sabr -i input.pdb -c A -o output.pdb -n martin

Aho
---

The **Aho** numbering scheme was designed for cross-species antibody
comparisons. It provides consistent numbering across diverse antibody
sequences.

Usage::

   sabr -i input.pdb -c A -o output.pdb -n aho

Wolfguy
-------

The **Wolfguy** numbering scheme is a specialized scheme for specific
applications.

Usage::

   sabr -i input.pdb -c A -o output.pdb -n wolfguy

Choosing a Scheme
-----------------

- **IMGT**: Recommended for most purposes. It's the current standard and
  provides consistent, well-defined positions.

- **Chothia**: Use when analyzing CDR loop conformations or canonical
  structures.

- **Kabat**: Use when working with legacy data or pharmaceutical industry
  partners who use this scheme.

- **Martin**: Use when you need a hybrid approach between structural and
  sequence-based definitions.

Chain Type Detection
--------------------

SAbR automatically detects the chain type (heavy, kappa, or lambda) from
the alignment. The detection uses:

1. **DE loop occupancy**: Heavy chains have 6 residues in positions 79-84,
   while light chains have only 4 residues (skipping 81-82).

2. **Position 10**: For light chains, kappa chains have position 10
   occupied while lambda chains do not.

You can override automatic detection with the ``-t`` flag::

   sabr -i input.pdb -c H -o output.pdb -t heavy
   sabr -i input.pdb -c L -o output.pdb -t kappa
