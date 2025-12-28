Usage Guide
===========

This guide covers how to use SAbR for antibody structure renumbering.

Basic Usage
-----------

The simplest way to renumber an antibody structure::

   sabr -i input.pdb -c A -o output.pdb

This will:

1. Read the structure from ``input.pdb``
2. Extract chain A
3. Generate MPNN embeddings for the chain
4. Align embeddings against the unified reference
5. Apply IMGT numbering (default)
6. Write the renumbered structure to ``output.pdb``

Command-Line Options
--------------------

Required Options
~~~~~~~~~~~~~~~~

``-i, --input-pdb FILE``
   Input structure file in PDB or mmCIF format.

``-c, --input-chain TEXT``
   Chain identifier to renumber (single character, e.g., "A", "H", "L").

``-o, --output FILE``
   Output structure file. Use ``.pdb`` extension for PDB format or
   ``.cif`` extension for mmCIF format.

Numbering Scheme
~~~~~~~~~~~~~~~~

``-n, --numbering-scheme [imgt|chothia|kabat|martin|aho|wolfguy]``
   The antibody numbering scheme to apply. Default: ``imgt``

   Available schemes:

   - **IMGT**: International ImMunoGeneTics information system
   - **Chothia**: Chothia numbering
   - **Kabat**: Kabat numbering
   - **Martin**: Martin (Enhanced Chothia) numbering
   - **Aho**: Aho numbering
   - **Wolfguy**: Wolfguy numbering

Chain Type
~~~~~~~~~~

``-t, --chain-type [H|K|L|heavy|kappa|lambda|auto]``
   Specify the chain type for ANARCI numbering. Default: ``auto``

   - ``H`` or ``heavy``: Heavy chain
   - ``K`` or ``kappa``: Kappa light chain
   - ``L`` or ``lambda``: Lambda light chain
   - ``auto``: Automatically detect from DE loop occupancy

   **Recommendation**: Specify the chain type manually when known, as heavy
   and light chains have similar structures that can be confused.

Advanced Options
~~~~~~~~~~~~~~~~

``--extended-insertions``
   Enable extended insertion codes (AA, AB, ..., ZZ, AAA, etc.) for
   antibodies with very long CDR loops. Requires mmCIF output format
   (``.cif`` extension).

   Standard PDB format only supports single-character insertion codes
   (A-Z, max 26 insertions per position).

``--disable-deterministic-renumbering``
   Disable deterministic renumbering corrections for loop regions.
   By default, corrections are applied for:

   - Light chain FR1 positions 7-10
   - DE loop positions 80-85 (all chains)
   - CDR loops (CDR1, CDR2, CDR3)

   Use this flag to get raw alignment output without corrections.

``--max-residues INTEGER``
   Maximum number of residues to process from the chain. If 0 (default),
   process all residues. Useful for truncating long chains.

``--overwrite``
   Overwrite the output file if it already exists.

``-v, --verbose``
   Enable verbose logging to see detailed alignment information.

Examples
--------

Renumber with Chothia scheme::

   sabr -i antibody.pdb -c H -o antibody_chothia.pdb -n chothia

Renumber with explicit chain type::

   sabr -i fab.pdb -c L -o fab_imgt.pdb -t kappa

Handle long CDR3 loops with mmCIF::

   sabr -i nanobody.cif -c A -o nanobody_imgt.cif --extended-insertions

Verbose output for debugging::

   sabr -i input.pdb -c A -o output.pdb -v

Practical Considerations
------------------------

Truncate to Fv Region
~~~~~~~~~~~~~~~~~~~~~

It is recommended to truncate the query structure to contain only the Fv
(variable fragment) region before running SAbR. The aligner may sometimes
align variable region beta-strands to those in the constant region.

Single-Chain scFvs
~~~~~~~~~~~~~~~~~~

When running scFvs (single-chain variable fragments), it is recommended
to run each variable domain independently. SAbR currently struggles with
scFvs because:

1. Domain assignment for canonical numbering is ambiguous
2. The aligner may incorrectly align across both domains

See `issue #2 <https://github.com/delalamo/SAbR/issues/2>`_ for details.

Missing Residues
~~~~~~~~~~~~~~~~

The CDR numbering algorithm uses the same approach as IMGT and does not
account for missing residues. If a residue is missing due to disorder or
heterogeneity, other residues in the CDR may be misnumbered.

Python API
----------

SAbR can also be used programmatically. See the :doc:`API Reference <api/cli>`
for detailed documentation.

Basic example::

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
